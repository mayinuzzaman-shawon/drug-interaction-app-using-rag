import streamlit as st
import os
import pandas as pd
import warnings
import logging
from typing import List, Tuple

from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# Setup environment (Original API Key & Password needs to be replaced with your own API Key & Password)
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
os.environ["NEO4J_URI"] = "YOUR_URI"
os.environ["NEO4J_USERNAME"] = "YOUR_USERNAME"
os.environ["NEO4J_PASSWORD"] = "YOUR_PASSWORD"

# Loading External Knowledge Base
@st.cache_data
def load_data():
    df = pd.read_csv("drug_reaction.csv")
    return df

df = load_data()
documents = [Document(page_content=f"Drug: {row['Drug']}\nReaction: {row['Reaction']}") for _, row in df.iterrows()]

# Setting Up LLM and Graph 
llm = ChatOpenAI(model="gpt-4o", temperature=0)
graph = Neo4jGraph()
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

# Vector index
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Entity extraction model
class Entities(BaseModel):
    names: List[str] = Field(..., description="Names of drugs or reactions mentioned")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert on drug reactions. Extract drugs and reactions mentioned."),
    ("human", "Extract from this input: {question}")
])
entity_chain = entity_prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    query = " AND ".join([f"{word}~2" for word in words])
    return query

def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:3})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
"""
    return final_data


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Condense Question Prompt follow-ups
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be standalone.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

_search_query = RunnableBranch(
    (RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
     RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
     | CONDENSE_QUESTION_PROMPT
     | llm
     | StrOutputParser()
    ),
    RunnableLambda(lambda x: x["question"])
)

# Final prompt and chain
qa_prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
Use clear medical explanations if needed, be concise.
Answer:""")

chain = (
    RunnableParallel({
        "context": _search_query | retriever,
        "question": RunnablePassthrough(),
    })
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("💊 Drug Interaction Identifier with GraphRAG")

question = st.text_input("Ask a drug interaction question:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Submit") and question:
    result = chain.invoke({"question": question, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((question, result))
    st.markdown("### 💬 Response")
    st.write(result)

    st.markdown("### 🕘 Conversation History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**System:** {a}")
