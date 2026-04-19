import os
from dotenv import load_dotenv
load_dotenv()

for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"]:
    if k in os.environ:
        os.environ[k] = os.environ[k].strip().strip('"').strip("'")

print("=== API Keys ===")
print("GROQ_API_KEY:   ", "YES" if "GROQ_API_KEY"   in os.environ else "MISSING")
print("COHERE_API_KEY: ", "YES" if "COHERE_API_KEY" in os.environ else "MISSING")
cohere_key = os.environ.get("COHERE_API_KEY", "")
print("COHERE key len: ", len(cohere_key))
print("COHERE repr:    ", repr(cohere_key[:20]))

print()
print("=== Store init ===")
from src.store import get_vectorstore
vs = get_vectorstore()
print("Vectorstore type:", type(vs).__name__)

print()
print("=== Retriever init ===")
from src.retrieval import RAGTutorRetriever
r = RAGTutorRetriever()
print("doc_count():", r.doc_count())
print("BM25 retriever:", "Ready" if r.sparse_retriever else "None (no docs yet — expected)")
print("Cohere reranker:", "Ready" if r.reranker else "None (no key)")

print()
print("=== ALL SYSTEMS OK ===")
