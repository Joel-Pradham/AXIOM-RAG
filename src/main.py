import os
import sys
import shutil
import uuid

# Force UTF-8 output on all platforms (prevents crashes on Windows cp1252 terminals)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    print(f"Warning: could not load .env file: {e}")

# Sanitize env var quoting/whitespace artifacts
for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"]:
    if k in os.environ:
        os.environ[k] = os.environ[k].strip('"').strip("'").strip()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.socratic_graph import SocraticTutor

app = FastAPI(title="AXIOM RAG API")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ─────────────────────────────────────────────────────────────────────
# Vercel enforces a 4.5 MB body limit per request.
# Chunks must be below this. We accept up to 4 MB per chunk,
# and allow a reassembled file up to 50 MB.
CHUNK_SIZE_BYTES  = 4 * 1024 * 1024   # 4 MB per upload chunk
MAX_FILE_MB       = 50                  # 50 MB max reassembled file
STREAM_READ_SIZE  = 256 * 1024          # 256 KB read buffer


# ── Request models ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"


# ── App state ──────────────────────────────────────────────────────────────────
tutor       = None
demo_mode   = False
debug_error = "None"


@app.on_event("startup")
async def startup_event():
    global tutor, demo_mode, debug_error

    try:
        from src.store import get_vectorstore
        get_vectorstore()
        print("[startup] Shared vectorstore ready.")
    except Exception as e:
        print(f"[startup] Vectorstore init warning: {e}")

    has_key = any(k in os.environ for k in ["COHERE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"])
    if not has_key:
        print("WARNING: No API keys found. Entering Demo Mode.")
        demo_mode = True
        return

    try:
        print("[startup] Initialising SocraticTutor...")
        tutor = SocraticTutor()
        print("[startup] SocraticTutor ready.")
    except Exception as e:
        import traceback
        debug_error = traceback.format_exc()
        print(f"[startup] SocraticTutor init failed: {e}")
        demo_mode = True


# ── Static ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


# ── Chat ───────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    global tutor, demo_mode, debug_error
    from src.db import get_history, add_interaction

    if demo_mode:
        return {
            "internal_thought_process": f"demo_mode=True | error={debug_error}",
            "answer": (
                "<p><strong>[AXIOM — Backend Alert]</strong><br>"
                "Running in fallback mode — API keys not detected or tutor failed to start.<br>"
                "Ensure <code>GROQ_API_KEY</code> and <code>COHERE_API_KEY</code> "
                "are set in Vercel environment variables.</p>"
            ),
            "citations": [],
        }

    try:
        history  = get_history(request.session_id)
        response = tutor.run(request.query, history)
        add_interaction(request.session_id, request.query, response)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Upload helpers ─────────────────────────────────────────────────────────────
def _tmp_base() -> str:
    return "/tmp/uploads" if os.environ.get("VERCEL") == "1" else "temp_uploads"

def _chunk_dir(upload_id: str) -> str:
    return os.path.join(_tmp_base(), upload_id)

ALLOWED_EXT = (".pdf", ".txt", ".md")


# ── CHUNK UPLOAD endpoint ──────────────────────────────────────────────────────
@app.post("/api/upload/chunk")
async def upload_chunk(
    file:         UploadFile = File(...),
    upload_id:    str = Form(...),
    chunk_index:  int = Form(...),
    total_chunks: int = Form(...),
    filename:     str = Form(...),
):
    """
    Receive one slice of a large file.  The browser splits the file into
    ≤4 MB pieces and posts them here individually, each safely under
    Vercel's 4.5 MB per-request body limit.

    Chunks are written to /tmp/uploads/<upload_id>/chunk_<N>.

    This endpoint returns immediately after saving the chunk — no heavy
    processing — so it always completes well within the 10-second limit.
    """
    # Validate filename extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXT):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}"
        )

    chunk_dir = _chunk_dir(upload_id)
    os.makedirs(chunk_dir, exist_ok=True)

    # Store original filename so finalize knows it
    meta_path = os.path.join(chunk_dir, "meta_filename.txt")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            f.write(os.path.basename(filename))

    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}")
    bytes_written = 0

    try:
        with open(chunk_path, "wb") as out:
            while True:
                data = await file.read(STREAM_READ_SIZE)
                if not data:
                    break
                bytes_written += len(data)
                if bytes_written > CHUNK_SIZE_BYTES + 100_000:  # 100 KB tolerance
                    out.close()
                    os.remove(chunk_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Chunk {chunk_index} exceeds maximum size of 4 MB."
                    )
                out.write(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chunk: {e}")

    return {
        "chunk_index":    chunk_index,
        "total_chunks":   total_chunks,
        "bytes_received": bytes_written,
        "status":         "chunk_saved",
    }


# ── FINALIZE endpoint ──────────────────────────────────────────────────────────
@app.post("/api/upload/finalize")
async def finalize_upload(upload_id: str = Form(...), total_chunks: int = Form(...)):
    """
    Called after all chunks have been uploaded.  This endpoint:
      1. Verifies all expected chunks exist
      2. Reassembles them into a single file
      3. Runs the ingestion pipeline
      4. Refreshes the BM25 sparse retriever on the live tutor
      5. Cleans up temp files

    Optimised to complete within Vercel's 10-second execution window for
    documents up to ~100 pages / ~15 MB.
    """
    chunk_dir = _chunk_dir(upload_id)

    # ── 1. Check all chunks are present ───────────────────────────────────────
    missing = []
    for i in range(total_chunks):
        cp = os.path.join(chunk_dir, f"chunk_{i:05d}")
        if not os.path.exists(cp):
            missing.append(i)

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing chunks: {missing}. Please retry the upload."
        )

    # ── 2. Read original filename ─────────────────────────────────────────────
    meta_path = os.path.join(chunk_dir, "meta_filename.txt")
    try:
        with open(meta_path) as f:
            safe_name = os.path.basename(f.read().strip())
    except Exception:
        safe_name = f"upload_{upload_id[:8]}.pdf"

    # ── 3. Reassemble ─────────────────────────────────────────────────────────
    final_path = os.path.join(_tmp_base(), safe_name)
    total_bytes = 0

    try:
        with open(final_path, "wb") as out:
            for i in range(total_chunks):
                cp = os.path.join(chunk_dir, f"chunk_{i:05d}")
                with open(cp, "rb") as chunk_f:
                    while True:
                        buf = chunk_f.read(256 * 1024)
                        if not buf:
                            break
                        total_bytes += len(buf)
                        if total_bytes > MAX_FILE_MB * 1_048_576:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Reassembled file exceeds {MAX_FILE_MB} MB limit "
                                    f"({total_bytes / 1_048_576:.1f} MB so far). "
                                    "Please split into smaller files."
                                )
                            )
                        out.write(buf)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reassembly failed: {e}")

    # ── 4. Ingest ─────────────────────────────────────────────────────────────
    try:
        from src.ingestion import ingest_file
        num_chunks = ingest_file(
            final_path,
            course_name="User Uploaded Docs",
            chapter_number=1,
            concept_tags=["Upload"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    # ── 5. Refresh BM25 retriever ─────────────────────────────────────────────
    if tutor is not None:
        try:
            tutor.retriever.refresh_sparse_retriever()
        except Exception as e:
            print(f"[finalize] BM25 refresh warning: {e}")

    # ── 6. Cleanup ────────────────────────────────────────────────────────────
    try:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        os.remove(final_path)
    except Exception:
        pass

    return {
        "filename":   safe_name,
        "num_chunks": num_chunks,
        "size_mb":    round(total_bytes / 1_048_576, 2),
        "message":    f"✅ '{safe_name}' ingested — {num_chunks} indexed chunks ready.",
    }


# ── Legacy single-shot upload (kept for small files < 4 MB) ───────────────────
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Single-shot upload for files under 4 MB.
    Larger files must use the /api/upload/chunk + /api/upload/finalize pair.
    """
    fname = file.filename or "upload.bin"
    if not any(fname.lower().endswith(ext) for ext in ALLOWED_EXT):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXT)}"
        )

    temp_dir  = _tmp_base()
    os.makedirs(temp_dir, exist_ok=True)
    safe_name = os.path.basename(fname)
    file_path = os.path.join(temp_dir, safe_name)

    bytes_written = 0
    try:
        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(STREAM_READ_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > 4 * 1_048_576:
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds 4 MB. Use chunked upload for larger files."
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    try:
        from src.ingestion import ingest_file
        num_chunks = ingest_file(file_path, course_name="User Uploaded Docs",
                                 chapter_number=1, concept_tags=["Upload"])
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if tutor is not None:
        try:
            tutor.retriever.refresh_sparse_retriever()
        except Exception:
            pass

    try:
        os.remove(file_path)
    except Exception:
        pass

    return {
        "filename":   safe_name,
        "num_chunks": num_chunks,
        "size_mb":    round(bytes_written / 1_048_576, 2),
        "message":    f"✅ '{safe_name}' ingested — {num_chunks} indexed chunks ready.",
    }
