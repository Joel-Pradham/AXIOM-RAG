from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings, CohereRerank
from typing import List
from langchain_core.documents import Document

class RAGTutorRetriever:
    def __init__(self, persist_directory: str = "./chroma_db", all_documents: List[Document] = None):
        import os
        if "OPENAI_API_KEY" in os.environ:
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                pass
            self.embeddings = OpenAIEmbeddings()
        elif "COHERE_API_KEY" in os.environ:
            from langchain_cohere import CohereEmbeddings
            self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        else:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 1. Base Dense Retriever — expanded to k=15 for maximum recall
        self.vectorstore = Chroma(
            collection_name="educational_material",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # 2. Sparse (Keyword) Retriever
        if all_documents is not None:
            self.sparse_retriever = BM25Retriever.from_documents(all_documents)
            self.sparse_retriever.k = 15
        else:
            try:
                docs = self.vectorstore.get()
                if docs and len(docs.get("documents", [])) > 0:
                    documents = [Document(page_content=doc, metadata=meta) for doc, meta in zip(docs['documents'], docs['metadatas'])]
                    self.sparse_retriever = BM25Retriever.from_documents(documents)
                    self.sparse_retriever.k = 15
                else:
                    self.sparse_retriever = None
            except Exception as e:
                print(f"BM25 Initialization fell back: {e}")
                self.sparse_retriever = None

        # 3. No ensemble — we merge manually to prevent the EnsembleRetriever
        #    from silently collapsing 20 → 3 results.
        #    Cross-Encoder Reranker (only if Cohere key present)
        import os
        if "COHERE_API_KEY" in os.environ:
            self.compressor = CohereRerank(top_n=6, model="rerank-english-v3.0")
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.dense_retriever
            )
        else:
            self.final_retriever = None

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using dense + sparse search, deduplicate, 
        filter garbage chunks, and return top 8 unique results.
        """
        MIN_CHUNK_LENGTH = 80  # Ignore fragments (headers, footers, page numbers)
        MAX_RESULTS = 8

        # Get dense results
        dense_docs = self.dense_retriever.invoke(query)
        
        # Get sparse results if available
        sparse_docs = []
        if self.sparse_retriever:
            try:
                sparse_docs = self.sparse_retriever.invoke(query)
            except Exception:
                sparse_docs = []

        # Merge and deduplicate by content hash
        seen = set()
        merged = []
        for doc in dense_docs + sparse_docs:
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen and len(doc.page_content.strip()) >= MIN_CHUNK_LENGTH:
                seen.add(content_hash)
                merged.append(doc)

        # If Cohere reranker is available, use it; otherwise return top merged
        if self.final_retriever:
            try:
                return self.final_retriever.invoke(query)[:MAX_RESULTS]
            except Exception:
                pass
        
        return merged[:MAX_RESULTS]
