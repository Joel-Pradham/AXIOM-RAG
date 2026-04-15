import os
from dotenv import load_dotenv
load_dotenv()

from src.socratic_graph import SocraticTutor
from src.db import add_interaction, get_history

t = SocraticTutor()

# 1. Base query
q1 = "What is data cleaning?"
print(f"User: {q1}")
r1 = t.run(q1)
print(f"Bot: {r1['answer'][:100]}...\n")

# Store in history
add_interaction("test_session", q1, r1)

# 2. Follow-up query
hist = get_history("test_session")
q2 = "explain it in detail"
print(f"User (Follow-up): {q2}")
r2 = t.run(q2, chat_history=hist)
print(f"Bot Internal Thought: {r2['internal_thought_process']}")
print(f"Bot: {r2['answer'][:500]}...\n")
