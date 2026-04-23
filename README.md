---
title: Axiom 
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
AXIOM-RAG: Advanced Agentic Retrieval Architecture

> "It's not who I am underneath, but what I do that defines me."

**AXIOM-RAG** is an intelligent, high-performance Retrieval-Augmented Generation (RAG) system built with a premium,It acts as an adaptive knowledge agent, instantly routing queries between heavily grounded local documents and real-time global web search.

---

What It Does

### AXIOM-RAG is a state-aware analytical engine designed for relentless fact-retrieval and direct, professional outputs.

- **Dynamic Query Routing:** Intelligently distinguishes between document-specific queries and general knowledge, routing to either local vector stores or DuckDuckGo web search.
- **Hybrid Search Engine:** Combines dense retrieval (Cohere embeddings + FAISS) with sparse keyword retrieval (BM25) for unparalleled accuracy.
- **Resilient Upload Pipeline:** Supports robust chunked uploads for large documents (PDF, TXT, MD) within strict serverless constraints.
- **Persistent Context:** Integrates SQLite-backed telemetry to maintain conversation history and agentic state across sessions.

---

### Tech Stack

### Intelligence Core
- **LangGraph & LangChain:** Orchestrating the agentic state machine and RAG pipelines.
- **Groq API:** Blistering-fast LLM inference for relevance grading and generation.
- **Cohere:** State-of-the-art vector embeddings.

###Search & Memory
- **FAISS (CPU):** Lightning-fast dense vector similarity search.
- **Rank-BM25:** Precise sparse keyword matching.
- **DuckDuckGo Search:** Automated fallback mechanism for external intelligence.
- **SQLite:** Lightweight, rock-solid session and telemetry tracking.

### Backend Infrastructure
- **FastAPI & Uvicorn:** High-concurrency Python ASGI server capable of handling chunked streaming uploads.
- **HTML/CSS/JS (Vanilla):** A sleek, glassmorphism-inspired dark mode frontend with dynamic transitions.

---

## How to Run

Deploying AXIOM-RAG requires minimal configuration, designed to run flawlessly on local environments or serverless platforms like Hugging Face Spaces.

### 1. Configure the Environment
Ensure you have your API keys ready. Create a `.env` file in the root directory:

```env
GROQ_API_KEY="your_groq_api_key"
COHERE_API_KEY="your_cohere_api_key"
```
*(Optionally include `OPENAI_API_KEY` if utilizing alternative models).*

### 2. Initialize the Environment
Open a terminal in the project root and set up your virtual Python environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the System
Spin up the backend API and serve the static frontend:

```bash
uvicorn src.main:app --reload
```

*The AXIOM terminal will be live at `http://127.0.0.1:8000`. Access the interface to begin querying the intelligence stream.*

---

## Author

**Joel Pradham**  
AI/ML Engineer | Backend Developer

---

*Designed and Engineered to achieve the stated outcome. No compromises.*
