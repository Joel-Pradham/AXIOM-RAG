import os
from dotenv import load_dotenv
load_dotenv()

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

reranker = CohereRerank(top_n=10, model="rerank-english-v3.0")

docs = [
    Document(page_content="The recipe for chocolate cake requires 2 cups of sugar, flour, and baking powder."),
    Document(page_content="Dogs are known to be man's best friend. They are loyal and protective.")
]

reranked = reranker.compress_documents(docs, "Explain the financial crisis of 2008")
print(f"Reranked length for completely irrelevant query: {len(reranked)}")
