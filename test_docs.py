import os
from dotenv import load_dotenv
load_dotenv()

from src.retrieval import RAGTutorRetriever

r = RAGTutorRetriever()
print("doc count:", r.doc_count())
docs = r._get_all_docs()
print("all docs len:", len(docs))
