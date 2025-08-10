# 📄🤖 DocuMind – "Ask. Understand. Summarize."

![LangChain](https://img.shields.io/badge/LangChain-%2300BFFF.svg?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-009688.svg?style=for-the-badge&logo=vector&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-%2300A67E.svg?style=for-the-badge&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Prompt Engineering](https://img.shields.io/badge/Prompt%20Engineering-8A2BE2.svg?style=for-the-badge&logo=json&logoColor=white)
![Information Retrieval](https://img.shields.io/badge/Information%20Retrieval-4682B4.svg?style=for-the-badge)
![Chunking](https://img.shields.io/badge/Chunking-FF8C00.svg?style=for-the-badge)
![Vector Search](https://img.shields.io/badge/Vector%20Search-6A5ACD.svg?style=for-the-badge)

---

## 🚀 Overview
**DocuMind** is an advanced **Retrieval-Augmented Generation (RAG)** application designed to make working with documents smarter, faster, and more interactive.

With **DocuMind**, you can:
- 🔍 **Ask** questions directly from your uploaded documents
- 🧠 **Understand** complex context using intelligent retrieval
- ✍ **Summarize** lengthy files into concise, actionable insights

---

## 🛠️ Tech Stack & Features
- **LangChain** – for LLM orchestration and chain building
- **FAISS Vector Store** – high-performance semantic search
- **OpenAI LLMs & Embeddings** – for deep semantic understanding
- **Prompt Engineering** – well-structured prompt templates
- **JSON-based Prompt Format** – for flexible, reusable task definitions
- **Document Chunking** – split large documents for efficient processing
- **Information Retrieval** – accurate, context-rich answers
- **Streamlit** – intuitive and interactive user interface

---

## 📜 Example Prompt JSON Format
```json
{
  "task": "summarization",
  "context": "<document_text_chunk>",
  "instructions": "Summarize in bullet points",
  "output_format": "plain_text"
}

---

## ⚡ Workflow
- 1 Upload Document (PDF, TXT, etc.)

Chunking & Embeddings – document is split and converted into vector embeddings

FAISS Search – retrieve the most relevant chunks based on user query

LLM Response Generation – context is passed to OpenAI LLM via LangChain

Display Results – answers or summaries rendered in Streamlit UI


