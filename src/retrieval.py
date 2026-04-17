"""
retrieval.py — Hybrid dense + sparse retriever backed by the shared vectorstore.

Key fix: The Cohere reranker now receives the MERGED hybrid (dense+sparse) candidates,
not a fresh dense-only query. This ensures BM25 keyword matches are not discarded
before reranking, significantly improving chunk alignment.
"""

import os
from typing import List
from langchain_core.documents import Document


class RAGTutorRetriever:

    def __init__(self, persist_directory: str = None, all_documents: List[Document] = None):
        from src.store import get_vectorstore

        self.vectorstore    = get_vectorstore()
        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 20}   # Retrieve more candidates for reranking
        )

        # ── BM25 sparse retriever ─────────────────────────────────────────────
        self.sparse_retriever = None
        self._build_sparse()

        # ── Cohere cross-encoder reranker ─────────────────────────────────────
        self.reranker = None
        if "COHERE_API_KEY" in os.environ:
            try:
                from langchain_cohere import CohereRerank
                # top_n=8 — keep more chunks for the LLM to work with
                self.reranker = CohereRerank(top_n=8, model="rerank-english-v3.0")
                print("[retrieval] Cohere reranker initialised (top_n=8).")
            except Exception as e:
                print(f"[retrieval] Cohere reranker unavailable: {e}")

    def _build_sparse(self):
        """Build or rebuild BM25 index from current vectorstore contents."""
        try:
            from langchain_community.retrievers import BM25Retriever

            documents = self._get_all_docs()
            if documents:
                self.sparse_retriever   = BM25Retriever.from_documents(documents)
                self.sparse_retriever.k = 20
                print(f"[retrieval] BM25 built with {len(documents)} documents.")
            else:
                print("[retrieval] No documents yet — BM25 skipped.")
        except Exception as e:
            print(f"[retrieval] BM25 init skipped: {e}")

    def _get_all_docs(self) -> List[Document]:
        """Extract all Document objects from the FAISS docstore."""
        try:
            if hasattr(self.vectorstore, "docstore") and \
               hasattr(self.vectorstore.docstore, "_dict"):
                return list(self.vectorstore.docstore._dict.values())
            if hasattr(self.vectorstore, "store"):
                return list(self.vectorstore.store.values())
        except Exception:
            pass
        return []

    def doc_count(self) -> int:
        """Return number of docs in the vectorstore (used for routing decisions)."""
        try:
            docs = self._get_all_docs()
            # Subtract 1 for the bootstrap sentinel document
            return max(0, len(docs) - 1)
        except Exception:
            return 0

    def refresh_sparse_retriever(self):
        """Re-build BM25 index after new documents are added."""
        self._build_sparse()

    def retrieve(self, query: str) -> List[Document]:
        """
        Hybrid dense + sparse retrieval with Cohere reranking.

        Fix: The reranker receives MERGED (dense+BM25) candidates — not a fresh
        dense-only run — so BM25 keyword matches survive into the final ranked set.
        """
        MIN_CHUNK_LEN = 80

        # ── Dense ─────────────────────────────────────────────────────────────
        try:
            dense_docs = self.dense_retriever.invoke(query)
        except Exception as e:
            print(f"[retrieval] Dense retrieval failed: {e}")
            dense_docs = []

        # ── Sparse (BM25) ─────────────────────────────────────────────────────
        sparse_docs = []
        if self.sparse_retriever:
            try:
                sparse_docs = self.sparse_retriever.invoke(query)
            except Exception:
                pass

        # ── Merge + deduplicate ────────────────────────────────────────────────
        # Dense results first (higher baseline relevance), then BM25 additions.
        seen, merged = set(), []
        for doc in dense_docs + sparse_docs:
            h = hash(doc.page_content.strip())
            if h not in seen and len(doc.page_content.strip()) >= MIN_CHUNK_LEN:
                seen.add(h)
                merged.append(doc)

        if not merged:
            return []

        # ── Cohere rerank on the MERGED candidates ─────────────────────────────
        if self.reranker and len(merged) > 1:
            try:
                # CohereRerank.compress_documents reranks any list of Documents
                reranked = self.reranker.compress_documents(merged, query)
                print(f"[retrieval] Reranked {len(merged)} → {len(reranked)} docs")
                return reranked
            except Exception as e:
                print(f"[retrieval] Reranker failed, using merged: {e}")

        # Fallback: return top-8 hybrid results
        return merged[:8]
