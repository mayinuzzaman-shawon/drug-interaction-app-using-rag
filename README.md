# Drug Interaction Identification Using Graph RAG Pipeline

This project aims to identify potential drug interactions and their related adverse effects using a **LangChain-based RAG (Retriever-Augmented Generation)** pipeline, leveraging **Neo4j** for graph-based storage and querying, and **OpenAI's GPT model** for natural language processing.

---

## 🚀 **Features**

- **Drug Interaction Identification**: Leverages LangChain to identify relationships between drugs and adverse reactions.
- **Graph-based Retrieval**: Utilizes Neo4j to store and query drug-reaction data efficiently.
- **Powered by OpenAI Models**: Uses OpenAI's GPT-4o model to provide natural language insights on drug interactions.
- **Interactive UI**: A simple Streamlit-based web interface to interact with the system and ask questions about drug interactions.

---

## 🛠️ **Technologies Used**

- **LangChain**: Framework for building LLM-powered pipelines.
- **Neo4j**: Graph database for storing drug-reaction relationships.
- **OpenAI GPT**: For natural language understanding and response generation.
- **Streamlit**: Web framework to create the UI for interacting with the drug interaction identification system.

---

## 💡 **Project Overview**

The system allows users to input questions about specific drug interactions, and the backend queries a Neo4j graph database to retrieve results by leberaging graph rag technique. The results are then processed using GPT-4o model to provide answers to the user's query in a conversational format.

---

## 🔧 **Installation and Setup**

### 1. **Clone the Repository**

```bash
git clone https://github.com/your-username/drug-interaction-identification.git
cd drug-interaction-identification

