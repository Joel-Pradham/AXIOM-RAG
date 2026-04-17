# AXIOM RAG — Railway Deployment Guide

## Why Railway vs Vercel?

| Problem | Vercel | Railway |
|---------|--------|---------|
| Body size limit | ❌ 4.5 MB max | ✅ No limit |
| Execution timeout | ❌ 10 seconds | ✅ No timeout |
| Persistent RAM | ❌ Stateless (FAISS lost on cold start) | ✅ Long-running process |
| Persistent disk | ❌ /tmp is ephemeral | ✅ Mount a volume |
| Cost | Free | ~$5/month (or free starter credit) |

---

## One-Time Setup (5 minutes)

### 1. Install Railway CLI
```bash
npm install -g @railway/cli
railway login
```

### 2. Create project
```bash
# From your project root:
railway init
# Select "Empty project" when prompted
```

### 3. Set environment variables
```bash
railway variables set GROQ_API_KEY=your_key_here
railway variables set COHERE_API_KEY=your_key_here
```

### 4. Add a Persistent Volume (keeps FAISS index across redeploys)
- Go to railway.app → your project → "New Volume"
- Mount path: `/app/faiss_store`
- Size: 1 GB (free tier)

### 5. Deploy
```bash
railway up
```
Railway detects the `Dockerfile` automatically.

---

## Redeploy after code changes
```bash
git push  # If you connected GitHub to Railway (recommended)
# OR
railway up  # Manual push
```

---

## Connect GitHub for auto-deploys (recommended)
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Select your `edu` repository
4. Add env vars in Settings → Variables
5. Add volume at `/app/faiss_store`
6. Every `git push` auto-deploys ✅

---

## Environment Variables Required

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | LLM inference (llama-3.3-70b) |
| `COHERE_API_KEY` | Embeddings + reranking |

---

## Architecture on Railway

```
Browser ──► Railway (uvicorn, persistent)
              │
              ├── /api/chat         → LangGraph 3-way routing
              ├── /api/upload/chunk → Save chunk to temp_uploads/
              └── /api/upload/finalize → Reassemble + ingest → FAISS
                                                                  │
                                                              /app/faiss_store
                                                           (persistent volume)
```

Unlike Vercel where every request is a fresh Lambda, Railway runs uvicorn as a **single persistent process**. The FAISS vectorstore stays in RAM between requests — uploaded documents are always available until the next redeploy, at which point they're reloaded from the persistent volume.
