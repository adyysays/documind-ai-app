# rag_pipeline.py

import os
import hashlib
import asyncio
from typing import List
from dotenv import load_dotenv
import streamlit as st

# LangChain components
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

# Universal fix for the "no current event loop" error in threads
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- CONFIGURATION ---
PINECONE_INDEX_NAME = "gemini-rag-index"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
RERANKER_MODEL_NAME = "rerank-english-v3.0" # Updated model
EMBEDDING_DIMENSIONALITY = 768

# --- MODEL INITIALIZATION (CACHED) ---
@st.cache_resource
def get_embeddings():
    """Initializes and returns the Gemini embeddings model, cached for efficiency."""
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Missing GOOGLE_API_KEY. Please set it in your .env file and restart.")
        return None
    try:
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, task_type="retrieval_document")
    except Exception as e:
        st.error(f"Error initializing Gemini embeddings: {e}")
        return None

def split_text(text: str, source_title: str) -> List[Document]:
    """Chunks text and adds metadata."""
    if not text:
        return []
    source_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={
                "source": source_id,
                "title": source_title,
                "section": f"Chunk {i+1}/{len(chunks)}",
                "position": i + 1,
            },
        )
        for i, chunk in enumerate(chunks)
    ]

@st.cache_resource
def setup_rag_chain():
    """Sets up the main RAG chain, cached for efficiency."""
    embeddings = get_embeddings()
    if not embeddings:
        return None

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    base_retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 10}
    )
    cohere_reranker = CohereRerank(model=RERANKER_MODEL_NAME, top_n=3, user_agent="DocuMind-AI")
    retriever = ContextualCompressionRetriever(
        base_compressor=cohere_reranker, base_retriever=base_retriever
    )
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=0.1, convert_system_message_to_human=True)
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs_with_citations(docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            citation = f"[{i+1}]"
            content = doc.page_content.replace('\n', ' ')
            metadata = f"Source: {doc.metadata.get('title', 'N/A')}"
            formatted.append(f"{citation} {content}\n(Info: {metadata})")
        return "\n\n".join(formatted)

    def format_and_retrieve(question: str):
        docs = retriever.invoke(question)
        return {"context": format_docs_with_citations(docs), "question": question, "original_docs": docs}

    return (
        RunnablePassthrough.assign(
            context_and_docs=lambda x: format_and_retrieve(x["question"])
        )
        | {
            "context": lambda x: x["context_and_docs"]["context"],
            "question": lambda x: x["question"],
            "original_docs": lambda x: x["context_and_docs"]["original_docs"],
        }
        | {"answer": prompt | llm | StrOutputParser(), "docs": lambda x: x["original_docs"]}
    )