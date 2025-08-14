import os
import pickle as pkl
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS

# Load API Keys 
import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]


if not GROQ_API_KEY or not HF_TOKEN:
    st.error("Please set GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN in your .env file.")
    st.stop()

#  Streamlit Page Config 
st.set_page_config(
    page_title="News Research Agent",
    page_icon="📉",
    layout="wide",
)

# Add Background Images
main_bg = "https://i.ibb.co/ccRBvfz7/Screenshot-2025-08-14-at-12-14-51-PM.png"


sidebar_bg = ""

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{main_bg}");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }}
    [data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("{sidebar_bg}");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }}
    .stTextInput>div>div>input {{
        background-color: rgba(0,0,0,0.5);
        color: #ffffff;
    }}
    .stButton>button {{
        background-color: #333333;
        color: #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App Title 
st.markdown("""
<h1 style="
    background: -webkit-linear-gradient(#ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align:center;
">
News Research Agent
</h1>
""", unsafe_allow_html=True)

# Sidebar 
st.sidebar.title("News Articles URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# LLM & Embeddings 
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0.0,
    max_tokens=500
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

#Junk Filter Function
def is_junk_chunk(chunk_text):
    text_lower = chunk_text.lower()

    # Rule 1: Too short
    if len(chunk_text.split()) < 30:
        return True

    # Rule 2: Contains boilerplate
    boilerplate_keywords = [
        "login", "sign-up", "subscribe", "advertisement", "cookie",
        "follow us", "download app", "price alerts", "fixed deposits",
        "credit score", "watchlist", "logout"
    ]
    if any(keyword in text_lower for keyword in boilerplate_keywords):
        return True

    # Rule 3: Too many non-alphabetic characters
    non_alpha_ratio = sum(1 for c in chunk_text if not c.isalpha() and not c.isspace()) / len(chunk_text)
    if non_alpha_ratio > 0.4:
        return True

    return False

#URL Processing
if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
        st.stop()

    with st.spinner("Loading and processing data..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        raw_chunks = text_splitter.split_documents(data)

        # Apply junk filter
        docs = [chunk for chunk in raw_chunks if not is_junk_chunk(chunk.page_content)]
        st.info(f"✅ Filtered out {len(raw_chunks) - len(docs)} junk chunks. {len(docs)} clean chunks remain.")

        # Store in FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_store_hf")

        with open("faiss_store_metadata.pkl", "wb") as f:
            pkl.dump({"info": "HuggingFace FAISS index"}, f)

    st.success("✅ Data processed and saved!")

#Query Section 
query = st.text_input("Ask a question about the articles:")
if query:
    try:
        vectorstore = FAISS.load_local(
            "faiss_store_hf",
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)

        # Show top 3 relevant chunks
        st.subheader(" Top 3 Relevant Chunks")
        for idx, doc in enumerate(docs, start=1):
            st.markdown(f"**Chunk {idx}:**\n{doc.page_content}\n\n---")

        # QA Chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=retriever
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("💡 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources:")
            for src in sources.split("\n"):
                st.write(src)

    except Exception as e:
        st.error(f"Error: {e}")

