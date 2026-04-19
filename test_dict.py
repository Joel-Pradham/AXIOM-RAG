import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = FAISS.from_texts(["Knowledge base initialised."], hf)

print("Before add dict size:", len(vs.docstore._dict))
vs.add_documents([Document(page_content="Quantum physics is weird.")])
print("After add dict size:", len(vs.docstore._dict))

print("Is index_to_docstore_id present?", hasattr(vs, "index_to_docstore_id"))
