import sqlite3
import os
import json
import re
from datetime import datetime


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace for clean LLM context."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # &nbsp; etc.
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:600]  # Hard cap — no single turn bloats the context

if os.environ.get("VERCEL") == "1":
    DB_PATH = "/tmp/telemetry.db"
else:
    DB_PATH = "chroma_db/telemetry.db"

def init_db():
    try:
        if not os.environ.get("VERCEL") == "1" and not os.path.exists("chroma_db"):
            os.makedirs("chroma_db", exist_ok=True)
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_query TEXT NOT NULL,
                bot_response TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[db] init failed: {e}")

def add_interaction(session_id: str, query: str, response: dict):
    """
    Response is a dictionary typically containing `internal_thought_process` and `answer`.
    Wrapped in try/except so DB failures never crash the chat endpoint.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO interactions (session_id, user_query, bot_response) VALUES (?, ?, ?)",
            (session_id, query, json.dumps(response))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[db] Failed to save interaction: {e}")

def get_history(session_id: str, limit: int = 4) -> str:
    """
    Fetch the historical context for the provided session ID.
    Returns a formatted string that can be directly passed into the LLM context.
    Limit applies to the number of recent turn pairs.
    """
    if not session_id:
        return ""
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Fetch from newest to oldest up to limit, then reverse to chronological order
    cursor.execute(
        "SELECT user_query, bot_response FROM interactions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    )
    results = cursor.fetchall()
    conn.close()
    
    if not results:
        return ""
        
    history_lines = []
    total_len = 0
    MAX_HISTORY_CHARS = 4000
    
    # Process from newest to oldest to guarantee we keep the most recent context
    for q, r_str in results:
        try:
            r_dict = json.loads(r_str)
            raw_ans = r_dict.get("answer", "")
        except Exception:
            raw_ans = r_str

        clean_ans = _strip_html(raw_ans)
        q_clean = q[:400] # cap enormously long user queries
        
        turn_str = f"User: {q_clean}\nTutor: {clean_ans}"
        if total_len + len(turn_str) > MAX_HISTORY_CHARS and history_lines:
            break
            
        history_lines.append(turn_str)
        total_len += len(turn_str)

    # Reverse to chronological order (oldest -> newest)
    history_lines.reverse()
    return "\n\n".join(history_lines)

# Initialize database mapping automatically on load
init_db()
