import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from src.socratic_graph import SocraticTutor

tutor = SocraticTutor()

# 1. Ask a question
q = "what are the topics in the document"
r = tutor.run(q)

print(f"Internal thought: {r['internal_thought_process']}")
print(f"Answer: {r['answer'][:200]}...")
