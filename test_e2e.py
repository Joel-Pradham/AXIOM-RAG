"""
Full end-to-end pipeline test:
  1. Create a minimal but realistic PDF with known content
  2. Ingest it into FAISS
  3. Query it via the full SocraticTutor pipeline
  4. Verify the answer references the known content
"""
import os
from dotenv import load_dotenv
load_dotenv()

for k in ["GROQ_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"]:
    if k in os.environ:
        os.environ[k] = os.environ[k].strip().strip('"').strip("'")

import tempfile, textwrap

# ── Step 1: Create a test PDF ─────────────────────────────────────────────────
print("=== Step 1: Creating test PDF ===")
try:
    import pypdf
    from pypdf import PdfWriter
    # Use reportlab if available, else create a minimal PDF manually
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        import io
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        c.setFont("Helvetica", 12)
        content = (
            "Data Cleaning Definition:\n"
            "Data cleaning, also known as data cleansing or scrubbing, is the process of "
            "detecting and correcting corrupt or inaccurate records from a dataset. "
            "It involves identifying incomplete, incorrect, inaccurate, irrelevant, or "
            "missing parts of the data and then replacing, modifying, or deleting this "
            "dirty data.\n\n"
            "Common techniques in data cleaning include handling missing values using "
            "imputation or deletion, removing duplicate records, standardising data formats, "
            "correcting structural errors, filtering outliers, and validating data against "
            "a set of predefined rules.\n\n"
            "The goal of data cleaning is to improve the overall quality of data, making "
            "it suitable for downstream analytics and machine learning tasks."
        )
        y = 750
        for line in textwrap.wrap(content, width=90):
            c.drawString(50, y, line)
            y -= 20
        c.save()
        buf.seek(0)
        pdf_bytes = buf.read()
        use_reportlab = True
    except ImportError:
        use_reportlab = False

    if use_reportlab:
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_pdf.write(pdf_bytes)
        tmp_pdf.close()
        pdf_path = tmp_pdf.name
        print(f"Created test PDF (reportlab): {pdf_path}")
    else:
        # Minimal hand-crafted PDF
        content_str = (
            "Data cleaning, also known as data cleansing or scrubbing, is the process of "
            "detecting and correcting corrupt or inaccurate records from a dataset. "
            "Techniques include handling missing values, removing duplicates, standardising "
            "formats, correcting structural errors, filtering outliers, and validating data."
        )
        pdf_raw = (
            "%PDF-1.4\n"
            "1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
            "2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
            "3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
            "  /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>>\nendobj\n"
        )
        stream = f"BT /F1 12 Tf 50 750 Td ({content_str[:200]}) Tj ET"
        pdf_raw += (
            f"4 0 obj\n<</Length {len(stream)}>>\nstream\n{stream}\nendstream\nendobj\n"
            "5 0 obj\n<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\nendobj\n"
            "xref\n0 6\n"
            "0000000000 65535 f\r\n"
            "0000000009 00000 n\r\n"
            "0000000058 00000 n\r\n"
            "0000000115 00000 n\r\n"
            "0000000266 00000 n\r\n"
            "0000000350 00000 n\r\n"
            "trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n420\n%%EOF"
        )
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="w")
        tmp_pdf.write(pdf_raw)
        tmp_pdf.close()
        pdf_path = tmp_pdf.name
        print(f"Created test PDF (minimal): {pdf_path}")

except Exception as e:
    print(f"PDF creation error: {e}")
    # Use a text file as fallback
    content_str = (
        "Data cleaning, also known as data cleansing, is the process of detecting and "
        "correcting corrupt or inaccurate records. Techniques include handling missing values "
        "with imputation, removing duplicate records, standardising data formats, correcting "
        "structural errors, filtering outliers, and validating against predefined rules. "
        "The goal is to ensure high data quality for analytics and machine learning."
    )
    tmp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    tmp_txt.write(content_str * 5)  # repeat to exceed MIN_CHUNK_CHARS
    tmp_txt.close()
    pdf_path = tmp_txt.name
    print(f"Created text fallback: {pdf_path}")

# ── Step 2: Ingest ─────────────────────────────────────────────────────────────
print("\n=== Step 2: Ingesting document ===")
from src.ingestion import ingest_file
try:
    n = ingest_file(pdf_path, course_name="Test Course", chapter_number=1, concept_tags=["test"])
    print(f"Ingested {n} chunks")
except Exception as e:
    print(f"Ingestion error: {e}")
    import traceback; traceback.print_exc()
    exit(1)

# ── Step 3: Verify retrieval ───────────────────────────────────────────────────
print("\n=== Step 3: Testing retrieval ===")
from src.retrieval import RAGTutorRetriever
r = RAGTutorRetriever()
print(f"doc_count after ingestion: {r.doc_count()}")
docs = r.retrieve("What is data cleaning?")
print(f"Retrieved {len(docs)} docs")
if docs:
    print("Top doc preview:", docs[0].page_content[:200])
else:
    print("WARNING: No docs retrieved!")

# ── Step 4: Full pipeline query ────────────────────────────────────────────────
print("\n=== Step 4: Full pipeline query ===")
from src.socratic_graph import SocraticTutor
tutor = SocraticTutor()
result = tutor.run("What is data cleaning and what are its main techniques?")
print("Route taken:", result.get("internal_thought_process", "")[:120])
print("Citations:", len(result.get("citations", [])))
answer = result.get("answer", "")
print("Answer length:", len(answer))
print("Answer preview:", answer[:400])

# Check answer actually references the PDF content
keywords = ["cleaning", "missing", "duplicate", "imputation", "outlier"]
hits = [kw for kw in keywords if kw.lower() in answer.lower()]
print(f"\nKeyword hits ({len(hits)}/{len(keywords)}): {hits}")
if len(hits) >= 3:
    print("\n✅ PASS: LLM correctly answered from PDF content")
else:
    print("\n❌ FAIL: Answer doesn't reference expected PDF content")

# Cleanup
os.unlink(pdf_path)
