"""
store.py — Shared vectorstore singleton.

Single process-wide FAISS instance shared between:
  - DataIngestionPipeline  (writes chunks)
  - RAGTutorRetriever      (reads during QA)

The `save_local` call is kept lightweight — only called explicitly
at the end of a full ingestion pass, not after every batch.
"""

import os
from langchain_core.documents import Document

_vectorstore = None
_embeddings  = None


def _build_embeddings():
    """
    Initialise embeddings in priority order:
      1. Cohere  — primary; used at ingestion time, so FAISS index must use same model
      2. OpenAI  — secondary
      3. HuggingFace (all-MiniLM-L6-v2) — local fallback, no API key needed

    IMPORTANT: changing the embedding model after ingestion causes a dimension /
    semantic mismatch and produces garbage retrieval results.  Always use the same
    model that built the index.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    # 1. Cohere (preferred — matches the ingestion-time embedding space)
    if "COHERE_API_KEY" in os.environ:
        try:
            from langchain_cohere import CohereEmbeddings
            _embeddings = CohereEmbeddings(model="embed-english-v3.0")
            print("[store] Using Cohere embeddings (embed-english-v3.0).")
            return _embeddings
        except Exception as e:
            print(f"[store] Cohere embeddings failed: {e} — falling back.")

    # 2. OpenAI
    if "OPENAI_API_KEY" in os.environ:
        try:
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings()
            print("[store] Using OpenAI embeddings.")
            return _embeddings
        except Exception as e:
            print(f"[store] OpenAI embeddings failed: {e} — falling back.")

    # 3. Local HuggingFace fallback
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[store] WARNING: Using local HuggingFace embeddings (fallback). "
          "Retrieval quality will be reduced if the index was built with Cohere.")
    return _embeddings


def _persist_dir() -> str:
    return "/tmp/faiss_store" if os.environ.get("VERCEL") == "1" else "./faiss_store"


def get_vectorstore():
    """Return the shared vectorstore, lazily initialised."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings  = _build_embeddings()
    persist_dir = _persist_dir()

    try:
        from langchain_community.vectorstores import FAISS

        if os.path.isdir(persist_dir):
            try:
                _vectorstore = FAISS.load_local(
                    persist_dir, embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[store] Loaded existing FAISS index from {persist_dir}")
                # Test dimensionality
                try:
                    # Test search to verify dimension
                    _vectorstore.similarity_search("test", k=1)
                except Exception as eval_e:
                    raise Exception(f"Index failed dimension test: {eval_e}")
                return _vectorstore
            except Exception as e:
                print(f"[store] Could not load existing index ({e}); wiping and starting fresh.")
                import shutil
                shutil.rmtree(persist_dir)

        _vectorstore = FAISS.from_texts(["Knowledge base initialised."], embeddings)
        print("[store] Created new in-memory FAISS index.")

    except ImportError:
        from langchain_core.vectorstores import InMemoryVectorStore
        _vectorstore = InMemoryVectorStore(embeddings)
        print("[store] FAISS unavailable; using InMemoryVectorStore.")

    return _vectorstore


def add_documents_batched(chunks, batch_size: int = 90, persist: bool = False) -> int:
    """
    Add document chunks in batches sized for Cohere's embed API (max 96/call).
    
    Args:
        chunks:     List of Document objects.
        batch_size: Texts per embedding API call. Default 90 (near Cohere's 96 limit).
        persist:    If True, saves FAISS index to disk after all batches complete.
                    Set to True only at the end of a full ingestion pass to avoid
                    repeated disk writes mid-ingestion.
    """
    global _vectorstore
    vs    = get_vectorstore()
    added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        try:
            vs.add_documents(batch)
            added += len(batch)
        except Exception as e:
            print(f"[store] Batch {i}–{i+len(batch)} failed: {e}")
            if "dimension" in str(e).lower() or i == 0:
                print(f"[store] Fatal error adding docs. Wiping index & retrying...")
                import shutil
                if os.path.exists(_persist_dir()):
                    shutil.rmtree(_persist_dir())
                
                try:
                    from langchain_community.vectorstores import FAISS
                    embeddings = _build_embeddings()
                    _vectorstore = FAISS.from_texts(["Knowledge base initialised."], embeddings)
                    vs = _vectorstore
                    vs.add_documents(batch)
                    added += len(batch)
                    print("[store] Retry successful.")
                except Exception as inner_e:
                    print(f"[store] Retry failed: {inner_e}")

    if persist:
        _save_index()

    return added


def save_index():
    """Persist the current FAISS index to disk. Call this once after full ingestion."""
    _save_index()


def _save_index():
    try:
        vs = get_vectorstore()
        if hasattr(vs, "save_local"):
            vs.save_local(_persist_dir())
            print(f"[store] FAISS index saved to {_persist_dir()}")
    except Exception as e:
        print(f"[store] Could not save index: {e}")
