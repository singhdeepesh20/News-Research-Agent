<div align="center">

# News Research Agent

### AI-Powered Multi-Source News Analysis & Question Answering System

</div>

---

## Overview

**News Research Agent** is an AI-driven research tool designed to analyze and extract insights from multiple news articles in real time.

Built using **LangChain, Groq LLaMA-3, HuggingFace embeddings, and FAISS**, the system enables users to **ask intelligent questions across multiple sources** and receive **context-aware answers with source attribution**.

The goal is to reduce information overload and transform raw news into **actionable insights**.

---

## Key Features

* 🌐 **Multi-URL Processing** → Analyze up to 3 news articles simultaneously
* 🧹 **Smart Content Filtering** → Removes ads, cookie popups, and irrelevant text
* 🤖 **AI-Powered Q&A** → Natural language querying using LLaMA-3 (70B)
* 📊 **Relevant Context Display** → Top 3 most relevant chunks before answers
* 🔗 **Source Attribution** → Direct links to original articles
* 🎨 **Interactive UI** → Clean and engaging Streamlit interface

---

## System Architecture

```
URLs Input → Content Loader → Text Cleaning
        ↓
   Chunking (Text Splitter)
        ↓
 Embeddings (HuggingFace)
        ↓
   FAISS Vector Store
        ↓
User Query → Retrieval → Groq LLM → Answer + Sources
```

---

## Tech Stack

* **Frontend**: Streamlit
* **LLM**: Groq API (LLaMA 3 – 70B)
* **Framework**: LangChain
* **Embeddings**: HuggingFace Sentence Transformers
* **Vector Store**: FAISS
* **Document Loader**: UnstructuredURLLoader
* **Environment Management**: .env for API keys

---

## Project Structure

```
News-Research-Agent/
│
├── app.py
├── faiss_store_hf/
├── faiss_store_metadata.pkl
├── requirements.txt
├── .env
└── README.md
```

---

## Usage

1. Paste up to **3 news article URLs** in the sidebar
2. Click **"Process URLs"** to load and index content
3. Enter your query in natural language
4. View:

   * Top relevant text chunks
   * AI-generated answer
   * Source references

---

## Example

**Question:**

> What economic reforms were discussed in the articles?

**Answer:**
The articles highlight fiscal policy changes, increased capital expenditure, a focus on manufacturing, and initiatives to boost digital transactions.

**Sources:**

* Article 1
* Article 2

---

## Engineering Approach

### Retrieval-Augmented Generation (RAG)

* Combines document retrieval with LLM reasoning
* Ensures context-grounded and reliable answers

### Data Cleaning Pipeline

* Removes noisy web elements (ads, cookies, prompts)
* Improves embedding quality and retrieval accuracy

### Efficient Retrieval

* FAISS enables fast similarity search
* Top-k retrieval ensures relevant context selection

---

## What This Project Demonstrates

* End-to-end LLM application development
* Multi-source document processing and reasoning
* Practical implementation of RAG pipelines
* Real-world AI system for information extraction

---

## Future Improvements

* Support for more than 3 URLs
* Real-time news API integration
* Hybrid retrieval (BM25 + dense embeddings)
* Summarization dashboards
* Deployment via Streamlit Cloud or Docker

---

## Author

**Deepesh Singh**
AI & Agentic Systems Builder

🌐 LinkedIn: [https://www.linkedin.com/in/contactdeepeshsingh/](https://www.linkedin.com/in/contactdeepeshsingh/)
💻 GitHub: [https://github.com/singhdeepesh20](https://github.com/singhdeepesh20)

---

<div align="center">

### "Turning information overload into actionable intelligence."

</div>

