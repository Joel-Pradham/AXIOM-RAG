FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — uses Railway requirements (includes uvicorn + faiss-cpu)
COPY requirements-railway.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Persistent directories
# On Railway: mount a volume at /app/faiss_store for durable FAISS storage
RUN mkdir -p /app/temp_uploads /app/faiss_store

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 120"]
