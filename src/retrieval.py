"""
retrieval.py — Hybrid dense + sparse retriever backed by the shared vectorstore.

Uses src.store.get_vectorstore() so it always reads from the same FAISS index
that ingestion.py writes to — no more disconnected instances.
"""

from typing import List
from langchain_core.documents import Document


class RAGTutorRetriever:

    def __init__(self, persist_directory: str = None, all_documents: List[Document] = None):
        """
        persist_directory is kept for backwards-compatibility but is now
        ignored — the shared store manages persistence internally.
        """
        import os
        from src.store import get_vectorstore

        self.vectorstore = get_vectorstore()
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})

        # ── Sparse (BM25) retriever — built from whatever is in the store ────
        self.sparse_retriever = None
        try:
            from langchain_community.retrievers import BM25Retriever

            documents = []
            if hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "_dict"):
                documents = list(self.vectorstore.docstore._dict.values())
            elif hasattr(self.vectorstore, "store"):
                documents = list(self.vectorstore.store.values())

            if documents:
                self.sparse_retriever = BM25Retriever.from_documents(documents)
                self.sparse_retriever.k = 15
        except Exception as e:
            print(f"[retrieval] BM25 init skipped: {e}")

        # ── Cohere cross-encoder reranker (optional) ─────────────────────────
        self.compressor = None
        if "COHERE_API_KEY" in os.environ:
            try:
                from langchain_cohere import CohereRerank
                from langchain_classic.retrievers import ContextualCompressionRetriever
                compressor = CohereRerank(top_n=6, model="rerank-english-v3.0")
                self.final_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.dense_retriever,
                )
            except Exception as e:
                print(f"[retrieval] Cohere reranker unavailable: {e}")
                self.final_retriever = None
        else:
            self.final_retriever = None

    # ── Method called after a new upload to refresh BM25 ─────────────────────
    def refresh_sparse_retriever(self):
        """Re-build BM25 index after new documents are added to the store."""
        try:
            from langchain_community.retrievers import BM25Retriever

            documents = []
            if hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "_dict"):
                documents = list(self.vectorstore.docstore._dict.values())
            elif hasattr(self.vectorstore, "store"):
                documents = list(self.vectorstore.store.values())

            if documents:
                self.sparse_retriever = BM25Retriever.from_documents(documents)
                self.sparse_retriever.k = 15
                print(f"[retrieval] BM25 refreshed with {len(documents)} documents.")
        except Exception as e:
            print(f"[retrieval] BM25 refresh failed: {e}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Dense + sparse hybrid retrieval with deduplication and length filtering.
        Falls back gracefully if any component fails.
        """
        MIN_CHUNK_LEN = 80
        MAX_RESULTS   = 8

        # Dense
        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception as e:
            print(f"[retrieval] Dense retrieval failed: {e}")
            dense_docs = []

        # Sparse
        sparse_docs = []
        if self.sparse_retriever:
            try:
                sparse_docs = self.sparse_retriever.invoke(query)
            except Exception:
                sparse_docs = []

        # Merge + deduplicate by content hash
        seen, merged = set(), []
        for doc in dense_docs + sparse_docs:
            h = hash(doc.page_content.strip())
            if h not in seen and len(doc.page_content.strip()) >= MIN_CHUNK_LEN:
                seen.add(h)
                merged.append(doc)

        # Rerank if Cohere available
        if self.final_retriever:
            try:
                return self.final_retriever.invoke(query)[:MAX_RESULTS]
            except Exception as e:
                print(f"[retrieval] Reranker failed, using merged: {e}")

        return merged[:MAX_RESULTS]
