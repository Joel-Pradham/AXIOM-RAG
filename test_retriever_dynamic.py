import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = FAISS.from_texts(["Knowledge base initialised."], hf)
retriever = vs.as_retriever(search_kwargs={"k": 5})

print("Before add:", [d.page_content for d in retriever.invoke("physics")])

vs.add_documents([Document(page_content="Quantum physics is weird.")])
print("After add:", [d.page_content for d in retriever.invoke("physics")])
