# RAG Branch — Field & Flower with Retrieval-Augmented Generation

This branch adds a RAG pipeline to the Field & Flower starter bot while keeping every other part of the stack — STT, TTS, tools, Twilio wiring, deployment — identical to `main`.

## What changed

| File | Status | Notes |
|---|---|---|
| `server/rag.py` | **New** | RAG pipeline: embeddings, in-memory Qdrant, cross-encoder reranker |
| `server/bot-gpt-rag.py` | **New** | Drop-in for `bot-gpt.py` — same tools + `RAGContextProcessor` injected |
| `server/pyproject.toml` | **Updated** | Added `qdrant-client`, `sentence-transformers`, `langchain-*` |
| `server/.env.example` | **Updated** | Notes — no new API keys needed |
| Everything else | **Unchanged** | `bot-gpt.py`, `bot-nemotron.py`, `mock_backend.py`, `Dockerfile`, etc. |

## How the RAG integration works

```
Caller speech
     │
     ▼
 Gradium STT
     │  (transcription)
     ▼
 User Aggregator
     │  LLMMessagesFrame
     ▼
 ┌─────────────────────────────────────────────┐
 │  RAGContextProcessor  (new in this branch)  │
 │                                             │
 │  1. Extract latest user message             │
 │  2. Embed it with ALBERT-small              │
 │  3. Cosine search in Qdrant (top-20)        │
 │  4. Re-rank with MiniLM cross-encoder       │
 │  5. Inject top-5 results as a system msg    │
 └─────────────────────────────────────────────┘
     │  augmented LLMMessagesFrame
     ▼
 OpenAI GPT-4.1  ←── sees catalog context + normal system prompt
     │
     ▼
 Gradium TTS → caller
```

The RAG context is injected **per-turn**, right before the LLM sees the user's message, so recommendations are always grounded in the actual catalog.

## Running locally

```bash
cd server
cp .env.example .env
# Edit .env — same keys as before (OPENAI_API_KEY + GRADIUM_API_KEY)

uv sync          # installs new RAG dependencies
uv run bot-gpt-rag.py
```

Open http://localhost:7860 and click **Connect**.

> **First launch note:** `sentence-transformers` will download the ALBERT-small embedding model (~45 MB) and the MiniLM cross-encoder (~80 MB) from HuggingFace on first run. Subsequent starts use the local cache and are fast.

## Deploying to Pipecat Cloud

The deployment flow is the same as `main`. The only difference is the entry point:

```bash
# Upload secrets (unchanged)
pc cloud secrets set flower-bot-secrets --file .env

# Deploy — Pipecat Cloud builds the Docker image and starts bot-gpt-rag.py
pc cloud deploy
```

Update `pcc-deploy.toml` if needed to point at the new entry point:

```toml
[bot]
command = "uv run bot-gpt-rag.py"
```

## RAG configuration

All knobs are at the top of `server/rag.py`:

```python
EMBEDDING_MODEL = "paraphrase-albert-small-v2"   # ~45 MB, fast
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE  = 20   # candidates fetched before reranking
TOP_K_FINAL     = 5    # results returned to the LLM
```

To use a larger embedding model (e.g. `all-MiniLM-L6-v2`) or Qdrant Cloud instead of in-memory, edit `initialize_rag()` in `rag.py`.

## Extending RAG to external data

`rag.py` currently embeds `BOUQUETS` from `mock_backend.py`. To add more data (e.g. care instructions, FAQs, seasonal notes):

1. Create a list of `Document` objects in `_build_vector_store()`.
2. Pass them to `Qdrant.from_documents()` alongside the bouquet docs.
3. The retriever picks up whatever is in the collection automatically.
