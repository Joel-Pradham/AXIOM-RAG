# AXIOM-RAG — Retrieval-Augmented Generation System

AXIOM-RAG is a hybrid Retrieval-Augmented Generation (RAG) system that answers queries using both local document knowledge and external web search. It dynamically routes queries to the most relevant source and generates grounded, context-aware responses.

---

## Overview

The system is designed to improve answer reliability by combining:

* Local document retrieval (vector + keyword search)
* External web search (fallback when local data is insufficient)
* LLM-based reasoning and response generation

It supports document ingestion, query routing, and persistent session handling.

---

## Features

### Dynamic Query Routing

* Classifies queries as:

  * Document-based (local retrieval)
  * General knowledge (web search)
* Routes to the appropriate pipeline automatically

### Hybrid Retrieval

* Dense retrieval using embeddings (Cohere + FAISS)
* Sparse retrieval using BM25
* Combined results improve accuracy and relevance

### Web Search Integration

* Uses DuckDuckGo search as fallback
* Ensures answers even when local documents lack information

### Document Processing

* Supports PDF, TXT, and Markdown files
* Chunking-based ingestion for efficient retrieval

### Persistent Context

* Stores session history using SQLite
* Maintains conversational continuity across queries

---

## Tech Stack

### Core Framework

* LangChain (RAG pipeline construction)
* LangGraph (stateful agent workflows)

### LLM & Embeddings

* Groq API (fast inference)
* Cohere (text embeddings)

### Retrieval & Search

* FAISS (vector similarity search)
* BM25 (keyword-based retrieval)
* DuckDuckGo Search (external data source)

### Backend

* FastAPI (Python API framework)
* Uvicorn (ASGI server)

### Storage

* SQLite (session and context storage)

### Frontend

* HTML, CSS, JavaScript (lightweight UI)

---

## System Workflow

1. User submits query
2. Query router determines intent:

   * Local document query → retrieval pipeline
   * General query → web search pipeline
3. Retrieval layer:

   * Dense (FAISS) + Sparse (BM25)
4. Context is passed to LLM
5. LLM generates final response
6. Session is stored for continuity

---

## Installation

### Prerequisites

* Python 3.9+
* API keys (Groq, Cohere)

---

### Clone the repository

```bash id="r2x1hf"
git clone https://github.com/your-username/axiom-rag.git
cd axiom-rag
```

---

### Setup environment

```bash id="8l1m2o"
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

### Environment variables

Create a `.env` file:

```env id="3k9fsl"
GROQ_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

---

### Run the application

```bash id="z6k1qa"
uvicorn src.main:app --reload
```

Access:

```id="t1p9wd"
http://127.0.0.1:8000
```

---

## Project Structure

```id="u6lm2n"
axiom-rag/
├── src/
│   ├── main.py
│   ├── routes/
│   ├── retrieval/
│   ├── agents/
│   └── utils/
├── data/
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Improve query classification accuracy
* Add re-ranking layer for retrieval results
* Support multi-document reasoning
* Deploy scalable version with vector DB (e.g., ChromaDB)

---

## Author

Joel Pradham
AI/ML Engineer | Backend Developer

---

## License

MIT License
