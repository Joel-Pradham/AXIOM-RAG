import sys
import os

from dotenv import load_dotenv
load_dotenv()

from src.retrieval import RAGTutorRetriever

retriever = RAGTutorRetriever()
print("Doc count:", retriever.doc_count())
docs = retriever.retrieve_multi(["what are the topics in the docuument"], "what are the topics in the docuument")
print("Retrieved docs:", len(docs))
