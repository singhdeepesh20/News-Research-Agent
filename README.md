# News-Research-Agent

An AI-powered research tool built with **Streamlit**, **LangChain**, **Groq LLaMA-3**, and **HuggingFace Embeddings** that helps you analyze and answer questions from multiple news articles.  
It fetches article content from URLs, processes it into clean chunks, stores them in a FAISS vector database, and lets you ask intelligent questions with accurate source references.

---

## 🚀 Features

- **Multi-URL Article Loader** – Load up to 3 news article URLs at once.
- **Smart Content Filtering** – Removes boilerplate junk like ads, cookie notices, and login prompts.
- **AI-Powered Search & QA** – Uses `llama3-70b-8192` from Groq for natural language answers.
- **Top 3 Relevant Chunks Display** – Shows the most relevant text sections before answering.
- **Source Attribution** – Links directly to the articles used for each answer.
- **Beautiful UI** – Gradient headings and background images for an engaging experience.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: [Groq API](https://groq.com/) – LLaMA 3 (70B)
- **Embeddings**: [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Document Loader**: LangChain `UnstructuredURLLoader`
- **Environment Variables**: `.env` file for API keys

---

## 📂 Project Structure

📁 News-Research-Agent
│
├── app.py # Main Streamlit app
├── faiss_store_hf/ # FAISS vector store
├── faiss_store_metadata.pkl # Vector store metadata
├── requirements.txt # Python dependencies
├── .env # API keys
└── README.md # Project documentation

---
💡 Usage

Paste up to 3 news article URLs in the sidebar.

Click "Process URLs" to load, filter, and store the content in FAISS.

Type your question in the input box.

View:

Top 3 relevant chunks

AI-generated answer

Sources list

📜 Example

Question:

What economic reforms were discussed in the articles?

Answer:

The articles highlight major fiscal policy changes including increased capital expenditure, a focus on manufacturing, and steps to boost digital transactions.

Sources:

Article 1

Article 2


---

## 📂 Project Structure
