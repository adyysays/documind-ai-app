# app.py
import os
import time
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from rag_pipeline import (
    split_text,
    setup_rag_chain,
    get_embeddings,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONALITY,
)
import tiktoken
import pandas as pd
import docx
from pypdf import PdfReader

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DocuMind AI", layout="wide")

# --- FILE PARSING FUNCTION ---
def parse_uploaded_file(uploaded_file):
    text = ""
    try:
        file_name = uploaded_file.name
        if file_name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif file_name.endswith('.csv'):
            text = pd.read_csv(uploaded_file).to_string()
        elif file_name.endswith('.xlsx'):
            text = pd.read_excel(uploaded_file).to_string()
        elif file_name.endswith('.txt'):
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            st.error("Unsupported file format.")
            return None
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None
    return text

# --- HELPER FUNCTIONS ---
@st.cache_resource
def get_pinecone_client():
    if not all([os.environ.get("PINECONE_API_KEY"), os.environ.get("PINECONE_ENVIRONMENT")]):
        st.error("Missing Pinecone credentials. Please set them in your .env file.")
        return None
    return Pinecone()

def setup_pinecone_index():
    pc = get_pinecone_client()
    if pc and PINECONE_INDEX_NAME not in pc.list_indexes().names():
        with st.spinner(f"Creating new Pinecone index: **{PINECONE_INDEX_NAME}**..."):
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONALITY,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        st.toast("Pinecone index created successfully!")

def estimate_cost(text: str, model_type: str) -> tuple[int, float]:
    try:
        tokens = tiktoken.get_encoding("cl100k_base").encode(text)
        token_count = len(tokens)
        cost_map = {"embedding": 0.10, "llm": 0.35} # Per 1M tokens
        cost = (token_count / 1_000_000) * cost_map.get(model_type, 0.0)
        if model_type == "rerank": cost = 0.001 # Per 1k reranks
        return token_count, cost
    except Exception:
        return 0, 0.0

def ingest_text(text, title):
    embeddings = get_embeddings()
    if not embeddings: return
    with st.spinner(f"Ingesting '{title}'..."):
        start_time = time.time()
        documents = split_text(text, title)
        if documents:
            PineconeVectorStore.from_documents(
                documents, embeddings, index_name=PINECONE_INDEX_NAME
            )
            total_time = time.time() - start_time
            token_count, cost = estimate_cost("".join(d.page_content for d in documents), "embedding")
            st.success(f"Ingested **{len(documents)}** chunks from '{title}'.")
            st.info(f"**Time Taken:** {total_time:.2f}s | **Cost Estimate:** ${cost:.6f}")
            st.session_state.ingested = True
        else:
            st.error("Text processing failed. No documents were generated.")

# --- MAIN APP UI ---
st.title("DocuMind AI")
st.markdown("Ingest knowledge by pasting text or uploading a file, then ask questions about it.")

# Initialize services and session state
setup_pinecone_index()
rag_chain = setup_rag_chain()
if "ingested" not in st.session_state:
    st.session_state.ingested = False

col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Ingest Knowledge")
    tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
    with tab1:
        source_title_text = st.text_input("Source Title", "Pasted Document")
        input_text = st.text_area("Paste text here", height=280)
        if st.button("Ingest Text", disabled=len(input_text) < 100):
            ingest_text(input_text, source_title_text)
    with tab2:
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'txt', 'csv', 'xlsx'])
        if st.button("Ingest File", disabled=not uploaded_file):
            parsed_text = parse_uploaded_file(uploaded_file)
            if parsed_text:
                ingest_text(parsed_text, uploaded_file.name)

with col2:
    st.subheader("Step 2: Ask a Question")
    query = st.text_input("Enter your question", "What is the main idea of this document?")
    if st.button("Get Answer", disabled=not st.session_state.ingested or not query):
        if not rag_chain:
            st.error("RAG chain not initialized. Check API keys and setup.")
        else:
            with st.spinner("Searching for answers..."):
                start_time = time.time()
                try:
                    result = rag_chain.invoke({"question": query})
                    answer, docs = result.get("answer", ""), result.get("docs", [])
                    total_time = time.time() - start_time
                    st.success("Answer Generated!")
                    st.markdown(f"**Answer:** {answer}")
                    st.divider()
                    st.markdown("#### Cited Sources:")
                    for i, doc in enumerate(docs):
                        with st.expander(f"**Source [{i+1}]**: {doc.metadata.get('title')}"):
                            st.write(doc.page_content)
                    _, cost = estimate_cost(query, "llm")
                    st.sidebar.subheader("Request Analysis")
                    st.sidebar.metric("Response Time (s)", f"{total_time:.2f}")
                    st.sidebar.metric("Est. Query Cost ($)", f"{cost:.6f}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")