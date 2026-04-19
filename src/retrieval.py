"""
retrieval.py — AXIOM Ultimate Retrieval Engine.

Full pipeline:
  - Dense FAISS retrieval (top-20 per query)
  - Sparse BM25 retrieval (top-20 per query)
  - Reciprocal Rank Fusion (RRF) for principled score merging
  - Cohere cross-encoder reranking (top-10 from merged pool)
  - retrieve_multi() for multi-query parallel retrieval from graph
"""

import os
from typing import List, Dict
from langchain_core.documents import Document

# Sentinel text injected when the vectorstore is first created (store.py)
_SENTINEL = "Knowledge base initialised."


class RAGTutorRetriever:

    DENSE_K  = 20   # candidates per query from FAISS
    SPARSE_K = 20   # candidates per query from BM25
    RRF_K    = 60   # RRF constant (higher = smoother merging)
    FINAL_N  = 10   # chunks to return after reranking

    def __init__(self, persist_directory: str = None):
        from src.store import get_vectorstore

        self.vectorstore     = get_vectorstore()
        self.dense_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.DENSE_K}
        )
        self.sparse_retriever = None
        self._build_sparse()

        # Cohere reranker — enabled when key is available
        self.reranker = None
        if "COHERE_API_KEY" in os.environ:
            try:
                from langchain_cohere import CohereRerank
                self.reranker = CohereRerank(
                    top_n=self.FINAL_N,
                    model="rerank-english-v3.0",
                )
                print(f"[retrieval] Cohere reranker ready (top_n={self.FINAL_N}).")
            except Exception as e:
                print(f"[retrieval] Cohere reranker unavailable: {e}")

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _build_sparse(self):
        """Build BM25 index from all documents in the vectorstore."""
        try:
            from langchain_community.retrievers import BM25Retriever
            docs = self._get_all_docs()
            # Filter out the sentinel initialisation document
            real_docs = [d for d in docs if d.page_content.strip() != _SENTINEL]
            if real_docs:
                self.sparse_retriever   = BM25Retriever.from_documents(real_docs)
                self.sparse_retriever.k = self.SPARSE_K
                print(f"[retrieval] BM25 built — {len(real_docs)} real docs.")
            else:
                self.sparse_retriever = None
                print("[retrieval] BM25 skipped — no real documents yet.")
        except Exception as e:
            print(f"[retrieval] BM25 init failed: {e}")

    def refresh_sparse_retriever(self):
        self._build_sparse()

    def _get_all_docs(self) -> List[Document]:
        """
        Extract all documents from the FAISS vectorstore.

        LangChain FAISS stores documents in an InMemoryDocstore backed by a
        plain dict ( docstore._dict ).  We also handle the index_to_docstore_id
        path for robustness across LangChain versions.
        """
        try:
            vs = self.vectorstore

            # Primary path — InMemoryDocstore._dict (most LangChain versions)
            if hasattr(vs, "docstore"):
                docstore = vs.docstore
                if hasattr(docstore, "_dict") and docstore._dict:
                    return list(docstore._dict.values())

                # Fallback: iterate via index_to_docstore_id mapping
                if hasattr(vs, "index_to_docstore_id") and hasattr(docstore, "search"):
                    docs = []
                    for doc_id in vs.index_to_docstore_id.values():
                        doc = docstore.search(doc_id)
                        if isinstance(doc, Document):
                            docs.append(doc)
                    if docs:
                        return docs

            # Last resort — some InMemoryVectorStore variants expose .store
            if hasattr(vs, "store"):
                return list(vs.store.values())

        except Exception as e:
            print(f"[retrieval] _get_all_docs error: {e}")

        return []

    def doc_count(self) -> int:
        """Return number of REAL (non-sentinel) documents in the store."""
        try:
            all_docs  = self._get_all_docs()
            real_docs = [d for d in all_docs
                         if d.page_content.strip() != _SENTINEL]
            return len(real_docs)
        except Exception:
            return 0

    # ── Core retrieval ────────────────────────────────────────────────────────

    def _dense(self, query: str) -> List[Document]:
        try:
            results = self.dense_retriever.invoke(query)
            # Strip sentinel from dense results
            return [d for d in results if d.page_content.strip() != _SENTINEL]
        except Exception as e:
            print(f"[retrieval] Dense failed: {e}")
            return []

    def _sparse(self, query: str) -> List[Document]:
        if not self.sparse_retriever:
            return []
        try:
            return self.sparse_retriever.invoke(query)
        except Exception as e:
            print(f"[retrieval] BM25 failed: {e}")
            return []

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(ranked_lists: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Merge multiple ranked document lists using Reciprocal Rank Fusion.
        Score(d) = Σ 1 / (k + rank_i(d))  for each list i that contains d.
        Higher scores = better combined ranking.
        """
        scores: Dict[int, float]    = {}
        docs:   Dict[int, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                doc_id = hash(doc.page_content.strip())
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    docs[doc_id]   = doc
                scores[doc_id] += 1.0 / (k + rank)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [docs[i] for i in sorted_ids]

    # ── Single-query retrieve ─────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """Dense + BM25 with RRF merge and Cohere reranking."""
        MIN_LEN = 80
        dense  = self._dense(query)
        sparse = self._sparse(query)

        merged = self._rrf_merge([dense, sparse])
        merged = [d for d in merged if len(d.page_content.strip()) >= MIN_LEN]

        if not merged:
            return []

        if self.reranker and len(merged) > 1:
            try:
                reranked = self.reranker.compress_documents(merged, query)
                print(f"[retrieval] {len(merged)} → reranked → {len(reranked)} docs")
                return reranked
            except Exception as e:
                print(f"[retrieval] Reranker failed, using RRF: {e}")

        return merged[:self.FINAL_N]

    # ── Multi-query retrieve (called by graph for maximum coverage) ───────────

    def retrieve_multi(self, queries: List[str], original_question: str) -> List[Document]:
        """
        Run dense+sparse for each query variant, merge all candidates with RRF,
        then rerank the full candidate pool against the ORIGINAL question.
        This ensures comprehensive document coverage before the LLM generation.
        """
        MIN_LEN = 80

        # Collect ranked lists per query
        all_dense  = []
        all_sparse = []
        for q in queries:
            all_dense.append(self._dense(q))
            all_sparse.append(self._sparse(q))

        # RRF merge across all query dense results, then all sparse results
        dense_merged  = self._rrf_merge(all_dense)
        sparse_merged = self._rrf_merge(all_sparse)
        # Final RRF between the two modalities
        candidates = self._rrf_merge([dense_merged, sparse_merged])
        candidates = [d for d in candidates if len(d.page_content.strip()) >= MIN_LEN]

        n_candidates = len(candidates)
        print(f"[retrieval] Multi-query: {len(queries)} queries -> "
              f"{n_candidates} unique candidates")

        if not candidates:
            return []

        # Rerank against ORIGINAL question (not expansions)
        if self.reranker and n_candidates > 1:
            try:
                reranked = self.reranker.compress_documents(
                    candidates, original_question
                )
                print(f"[retrieval] Reranked {n_candidates} → {len(reranked)} docs")
                return reranked
            except Exception as e:
                print(f"[retrieval] Reranker failed, using RRF result: {e}")

        return candidates[:self.FINAL_N]
