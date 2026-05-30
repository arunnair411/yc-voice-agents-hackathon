"""RAG (Retrieval-Augmented Generation) pipeline for Field & Flower.

Provides semantic search over the flower shop's catalog and order history
using Qdrant as a vector store, HuggingFace embeddings, and a cross-encoder
reranker for improved retrieval relevance.

The module exposes:
  - ``get_rag_context(query)`` — async function that retrieves the most
    relevant catalog snippets for a given natural-language query.
  - ``initialize_rag()`` — call once at startup to build/populate the
    in-memory Qdrant collection from mock_backend.BOUQUETS.

Usage::

    from rag import initialize_rag, get_rag_context

    await initialize_rag()          # once, at startup
    context = await get_rag_context("flowers for a birthday")

Dependencies (add to pyproject.toml or requirements.txt)::

    qdrant-client>=1.9.0
    sentence-transformers>=3.0.0
    langchain-qdrant>=0.1.0
    langchain-huggingface>=0.0.3
"""

import asyncio
import json
from functools import lru_cache
from typing import List

from loguru import logger

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when initialize_rag() is called so the rest
# of the bot still imports cleanly even if these packages are missing.
# ---------------------------------------------------------------------------

_vector_store = None
_reranker = None
_embeddings = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "field-and-flower"
EMBEDDING_MODEL = "paraphrase-albert-small-v2"   # ~45 MB, fast
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 20   # candidates to fetch before reranking
TOP_K_FINAL = 5       # results returned after reranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bouquet_to_document_text(name: str, info: dict) -> str:
    """Render a bouquet dict as a human-readable string for embedding."""
    occasions = ", ".join(info.get("occasions", []))
    special = " [ON SPECIAL TODAY]" if info.get("on_special") else ""
    stock = "In stock" if info.get("in_stock") else "Out of stock"
    return (
        f"Bouquet: {name.title()}\n"
        f"Description: {info['description']}\n"
        f"Price: ${info['price']:.2f}\n"
        f"Occasions: {occasions}\n"
        f"Availability: {stock}{special}"
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

async def initialize_rag() -> None:
    """Build the in-memory Qdrant collection from BOUQUETS in mock_backend.

    Safe to call multiple times — re-uses the existing collection if it is
    already populated.  Call this once at bot startup before the first
    ``get_rag_context`` invocation.
    """
    global _vector_store, _reranker, _embeddings

    if _vector_store is not None:
        logger.debug("RAG already initialized, skipping.")
        return

    logger.info("Initializing RAG pipeline...")

    # Run CPU-heavy model loading off the event loop so it doesn't block audio.
    loop = asyncio.get_event_loop()

    def _load_models():
        from langchain_huggingface import HuggingFaceEmbeddings
        from sentence_transformers import CrossEncoder

        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        reranker = CrossEncoder(RERANKER_MODEL)
        return emb, reranker

    _embeddings, _reranker = await loop.run_in_executor(None, _load_models)
    logger.info("Embedding + reranker models loaded.")

    # Build in-memory Qdrant collection
    def _build_vector_store():
        from langchain_community.vectorstores import Qdrant
        from langchain_core.documents import Document
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        from mock_backend import BOUQUETS

        # In-memory Qdrant — no server required during the hackathon.
        client = QdrantClient(":memory:")

        docs = []
        for name, info in BOUQUETS.items():
            text = _bouquet_to_document_text(name, info)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "name": name,
                        "price": info["price"],
                        "in_stock": info["in_stock"],
                        "on_special": info.get("on_special", False),
                        "occasions": info.get("occasions", []),
                    },
                )
            )

        vs = Qdrant.from_documents(
            docs,
            _embeddings,
            location=":memory:",
            collection_name=COLLECTION_NAME,
        )
        return vs

    _vector_store = await loop.run_in_executor(None, _build_vector_store)
    logger.info(f"RAG collection '{COLLECTION_NAME}' ready with {len(_vector_store.get()['ids'])} documents.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def get_rag_context(query: str) -> str:
    """Retrieve the most relevant bouquet information for *query*.

    Returns a formatted string ready to be injected into the LLM's system
    prompt or as a tool result.  Returns an empty string if RAG is not yet
    initialized.

    Args:
        query: Natural-language query from the caller, e.g.
               "something nice for my mom's birthday under fifty dollars".
    """
    if _vector_store is None:
        logger.warning("get_rag_context called before initialize_rag(). Returning empty context.")
        return ""

    loop = asyncio.get_event_loop()

    def _retrieve():
        # Step 1: Vector similarity search — retrieve top-K candidates
        candidates = _vector_store.similarity_search(query, k=TOP_K_RETRIEVE)

        if not candidates:
            return ""

        # Step 2: Cross-encoder reranking over candidates
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = _reranker.predict(pairs)

        # Sort by descending score, take top final results
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:TOP_K_FINAL]]

        # Format as context block
        lines = ["=== Relevant Catalog Information ==="]
        for doc in top_docs:
            lines.append(doc.page_content)
            lines.append("")  # blank line separator
        return "\n".join(lines).strip()

    try:
        return await loop.run_in_executor(None, _retrieve)
    except Exception as exc:
        logger.error(f"RAG retrieval failed: {exc}")
        return ""
