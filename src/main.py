import os
import time
import shutil
from dotenv import load_dotenv

load_dotenv()

# Sanitize Windows command-line quoting artifacts
for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"]:
    if k in os.environ:
        os.environ[k] = os.environ[k].strip('"').strip("'")

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.socratic_graph import SocraticTutor

app = FastAPI(title="AXIOM RAG API")

# Mount the static directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request Model
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

# Lazy load tutor to avoid breaking on startup if API key is missing
tutor = None
demo_mode = False

@app.on_event("startup")
async def startup_event():
    global tutor, demo_mode
    if "COHERE_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ and "GROQ_API_KEY" not in os.environ:
        print("WARNING: Neither COHERE, OPENAI, nor GROQ API keys found. Entering Missing-Key Mode.")
        demo_mode = True
    else:
        try:
            print("Initializing Socratic Graph...")
            tutor = SocraticTutor()
        except Exception as e:
            print(f"Failed to initialize SocraticTutor: {e}")
            demo_mode = True

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global tutor, demo_mode
    from src.db import get_history, add_interaction
    
    if demo_mode:
        import time
        time.sleep(0.5)  # simulate latency
        return {
            "internal_thought_process": "CRITICAL ERROR: Language Model disconnected.",
            "answer": "[SYSTEM ALERT] I cannot process this request because I am missing an LLM API key. Please stop the backend server, set `OPENAI_API_KEY` in your terminal shell, and run `uvicorn src.main:app --reload` again.",
            "citations": []
        }
        
    try:
        # Pass conversation history dynamically
        history = get_history(request.session_id)
        response = tutor.run(request.query, history)
        
        # Log this interaction to the database
        add_interaction(request.session_id, request.query, response)
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global demo_mode
    if demo_mode:
        import asyncio
        await asyncio.sleep(1.5)
        return {"filename": file.filename, "num_chunks": 12}
        
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        from src.ingestion import DataIngestionPipeline
        pipeline = DataIngestionPipeline()
        num_chunks = pipeline.ingest_file(
            file_path, 
            course_name="User Uploaded Docs", 
            chapter_number=1, 
            concept_tags=["Upload", "Context"]
        )
        
        # Files are kept in temp_uploads for potential re-ingestion
        
        return {"filename": file.filename, "num_chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
