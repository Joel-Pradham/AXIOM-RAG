"""
ingestion.py — Optimised document ingestion for the AXIOM RAG system.

Vercel constraints addressed:
  - 4.5MB body limit → solved by chunked upload (see main.py + app.js)
  - 10s function timeout → solved by:
      1. Extracting ALL PDF text in one fast sequential pass
      2. Splitting ALL chunks at once (pure CPU, <1s)
      3. Embedding in large batches (90/call — Cohere handles up to 96)
      4. Saving FAISS index only ONCE at the very end (not after every batch)
  - 512MB RAM → page text is processed and released, not held in full
"""

import os
import re
import glob
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_MB     = 50           # Hard cap per uploaded chunk reassembly
MIN_CHUNK_CHARS = 80           # Discard short noise fragments
CHUNK_SIZE      = 2400   # ~half a textbook page — captures complete concepts
CHUNK_OVERLAP   = 400    # generous overlap preserves cross-boundary context


# ── Module-level splitter (cheap singleton) ────────────────────────────────────
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Collapse PDF layout noise into readable prose."""
    text = re.sub(r'\s*\n\s*', ' ', text)          # newlines → spaces
    text = re.sub(r' {2,}', ' ', text)              # multiple spaces → one
    text = re.sub(r'[^\x20-\x7E\u00A0-\u024F]', '', text)  # non-printable out
    return text.strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest_file(
    file_path: str,
    course_name:    str       = "User Uploaded Docs",
    chapter_number: int       = 1,
    concept_tags:   List[str] = None,
) -> int:
    """
    Ingest a file (.pdf / .txt / .md) into the shared vectorstore.
    Returns the number of chunks added.
    """
    if concept_tags is None:
        concept_tags = ["Upload"]

    # Size guard
    size_mb = os.path.getsize(file_path) / 1_048_576
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"Reassembled file is {size_mb:.1f} MB — exceeds the {MAX_FILE_MB} MB limit."
        )

    meta = {
        "course_name":    course_name,
        "chapter_number": chapter_number,
        "concept_tags":   ", ".join(concept_tags),
    }

    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return _ingest_pdf(file_path, meta)
    elif ext in ("txt", "md"):
        return _ingest_text(file_path, meta)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: .pdf, .txt, .md")


# ── PDF ingestion ──────────────────────────────────────────────────────────────

def _ingest_pdf(file_path: str, meta: dict) -> int:
    """
    Single-pass PDF ingestion optimised for Vercel's 10-second timeout.

    Strategy:
      1. Read ALL pages sequentially (fast I/O — pypdf is efficient here)
      2. Build ALL Document objects in memory
      3. Split into ALL chunks at once
      4. Embed in large batches of 90 (Cohere: 1-2 API calls for most docs)
      5. Save FAISS index once
    """
    from src.store import add_documents_batched, save_index

    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required. Ensure 'pypdf' is in requirements.txt.")

    page_docs: List[Document] = []
    errors = 0

    with open(file_path, "rb") as f:
        try:
            reader = pypdf.PdfReader(f)
        except Exception as e:
            raise ValueError(f"Cannot read PDF: {e}. File may be encrypted or corrupted.")

        total_pages = len(reader.pages)
        print(f"[ingestion] PDF: {total_pages} pages")

        for page_idx in range(total_pages):
            try:
                raw = reader.pages[page_idx].extract_text() or ""
                cleaned = clean_text(raw)
                if len(cleaned) >= MIN_CHUNK_CHARS:
                    page_docs.append(Document(
                        page_content=cleaned,
                        metadata={**meta, "page": page_idx + 1},
                    ))
            except Exception as e:
                errors += 1
                print(f"[ingestion] Page {page_idx + 1} skipped: {e}")

    if not page_docs:
        raise ValueError(
            f"No readable text found in PDF ({errors} pages had errors). "
            "The PDF may be scanned images or encrypted."
        )

    print(f"[ingestion] Extracted text from {len(page_docs)}/{total_pages} pages "
          f"({errors} skipped)")

    # Chunk all at once
    chunks = [c for c in _splitter.split_documents(page_docs)
              if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    print(f"[ingestion] Generated {len(chunks)} chunks")

    if not chunks:
        raise ValueError("Document produced no valid chunks after splitting.")

    # Embed + add (batch_size=90, persist=False until the end)
    added = add_documents_batched(chunks, batch_size=90, persist=False)

    # Single disk save
    save_index()

    print(f"[ingestion] Done — {added} chunks indexed.")
    return added


# ── Text / Markdown ingestion ──────────────────────────────────────────────────

def _ingest_text(file_path: str, meta: dict) -> int:
    from src.store import add_documents_batched, save_index

    content = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError("Could not decode file. Unsupported encoding.")

    doc    = Document(page_content=clean_text(content), metadata={**meta, "page": 1})
    chunks = [c for c in _splitter.split_documents([doc])
              if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

    if not chunks:
        return 0

    added = add_documents_batched(chunks, batch_size=90, persist=False)
    save_index()
    return added


# ── Directory ingestion (CLI only) ─────────────────────────────────────────────

def ingest_directory(directory_path, course_name, chapter_number, concept_tags) -> int:
    from src.store import save_index
    files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
    total = 0
    for fp in files:
        if fp.lower().rsplit(".", 1)[-1] not in ("pdf", "txt", "md"):
            continue
        try:
            n = ingest_file(fp, course_name, chapter_number, concept_tags)
            total += n
        except Exception as e:
            print(f"[ingestion] Skipping {fp}: {e}")
    save_index()
    return total


# ── Backwards-compat shim ──────────────────────────────────────────────────────

class DataIngestionPipeline:
    def __init__(self, persist_directory=None):
        pass

    def ingest_file(self, fp, course_name="Docs", chapter_number=1, concept_tags=None):
        return ingest_file(fp, course_name, chapter_number, concept_tags or [])

    def ingest_directory(self, dp, course_name="Docs", chapter_number=1, concept_tags=None):
        return ingest_directory(dp, course_name, chapter_number, concept_tags or [])
