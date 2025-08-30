#  Gemini-Powered RAG Application with Reranking

This is a high-performance Retrieval-Augmented Generation (RAG) application that allows users to ingest text, store it in a cloud vector database, and ask questions. The system uses a powerful retriever-reranker pipeline to ensure answers are accurate and grounded in the provided source text, complete with citations.

**[Live URL]()** <- *Add your Streamlit Cloud URL here after deploying*
**[Resume/Portfolio Link]()** <- https://drive.google.com/file/d/1xBFb6QYY701hjgarJn13T1sU0GVCh366/view?usp=drive_link

---

## 🏛️ Architecture Diagram

The application follows a standard RAG pipeline, enhanced with a reranking step for improved accuracy.



```mermaid
graph TD
    A[User Pastes Text] --> B{Text Processing};
    B --> C[Chunking (1000/150 overlap)];
    C --> D[Generate Embeddings (Gemini)];
    D --> E[Upsert to Pinecone Vector DB];

    F[User Asks Question] --> G{Query Processing};
    G --> H[Generate Query Embedding (Gemini)];
    H --> I[Retrieve Top-K Chunks (Pinecone MMR, k=10)];
    I --> J[Rerank Chunks (Cohere Rerank, top_n=3)];
    J --> K[Format Context with Citations];
    K --> L[Prompt LLM (Gemini 1.5 Flash)];
    L --> M[Generate Grounded Answer];
    M --> N[Display Answer & Sources to User];

    subgraph "Ingestion"
        A
        B
        C
        D
        E
    end

    subgraph "Querying"
        F
        G
        H
        I
        J
        K
        L
        M
        N
    end

    style E fill:#2962FF,stroke:#FFF,stroke-width:2px,color:#FFF
    style J fill:#FF6D00,stroke:#FFF,stroke-width:2px,color:#FFF
    style L fill:#00BFA5,stroke:#FFF,stroke-width:2px,color:#FFF
```

---

##  System Configuration

* **Frontend:** Streamlit
* **Vector Database:** Pinecone (Serverless)
* **Embedding Model:** Google `models/text-embedding-004` (768 dimensions)
* **LLM:** Google `gemini-1.5-flash-latest`
* **Reranker:** `cohere-rerank-english-v2.0`

### Index and Retrieval Configuration

* **Pinecone Index Name:** `gemini-rag-index`
* **Dimensionality:** 768
* **Metric:** `cosine`
* **Upsert Strategy:** Documents are chunked and upserted in batches. A unique hash of the source text is used as a `source` ID and namespace to logically separate documents.
* **Retrieval Strategy:**
    1.  **Retriever:** Maximal Marginal Relevance (MMR) is used to fetch an initial diverse set of **k=10** documents.
    2.  **Reranker:** `CohereRerank` processes these 10 documents and returns the **top 3** most semantically relevant documents to the query. This two-stage process significantly improves signal-to-noise ratio before prompting the LLM.

### Chunking Strategy

* **Method:** `RecursiveCharacterTextSplitter`
* **Chunk Size:** 1,000 characters
* **Chunk Overlap:** 150 characters (15% overlap)
* **Metadata Stored:** `source` (unique ID of parent text), `title`, `section`, `position` (chunk number).

---

##  Quick Start & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/gemini-rag-app.git](https://github.com/your-username/gemini-rag-app.git)
    cd gemini-rag-app
    ```

2.  **Create a Virtual Environment & Install Dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables:**
    * Copy the example file: `cp .env.example .env`
    * Open the `.env` file and add your API keys from Google, Pinecone, and Cohere.

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

---

##  Remarks and Trade-offs

* **Speed vs. Cost:** Using a powerful reranker like Cohere adds a small amount of latency and cost to each query, but the increase in answer quality is substantial. The initial retrieval (`k=10`) is a trade-off: higher `k` gives the reranker more to work with but increases the retrieval workload.
* **Provider Limits:** The free tiers for Google Gemini, Pinecone, and Cohere are generous but have rate limits. For a high-traffic application, you would need to move to paid plans. This app is designed to stay well within free-tier limits for personal use.
* **Parallel Chunking:** The user's request for "parallel chunking" was interpreted as a desire for overall speed. For a single text input, the primary performance bottlenecks are network I/O for embedding and upserting, not the CPU-bound chunking task. This implementation prioritizes speed by using Pinecone's efficient batch `upsert` method, which is the most effective optimization for this use case.
* **No Answer Cases:** The prompt explicitly instructs the LLM to state when it does not have enough information to answer, which it handles gracefully. The UI also prevents querying before a document is ingested.

### Next Steps

* **File Uploads:** Implement `st.file_uploader` to support PDF, DOCX, and TXT files.
* **Chat History:** Store the conversation history in `st.session_state` to allow for follow-up questions.
* **Advanced Chunking:** Explore semantic chunking or agentic chunking for more context-aware document splitting.
* **Scalability:** For larger-scale use, move the ingestion logic to a background worker (e.g., Celery) to avoid blocking the UI.