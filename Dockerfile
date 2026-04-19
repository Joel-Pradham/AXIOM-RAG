FROM python:3.12-slim

# Install sqlite3 as well just in case, though it's built-in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user matching UID 1000 required by HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# We will just copy the main requirements.txt which we unified
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ensure upload/db directories exist with right permissions for user 1000
RUN mkdir -p /app/temp_uploads /app/chroma_db /app/faiss_store

# Copy source with ownership
COPY --chown=user . .

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120"]
