"""RAG (Retrieval-Augmented Generation) pipeline for HealthLine.

Provides semantic search over the clinic's NON-DIAGNOSTIC knowledge base
(policies, appointment prep, medication general guidance, insurance/billing
FAQ) using Qdrant as an in-memory vector store, HuggingFace embeddings, and a
cross-encoder reranker for improved retrieval relevance.

This mirrors the flower-shop ``rag.py`` but is scoped to the healthcare
knowledge base so the two demos stay independent. Nothing here returns
clinical advice — diagnosis, dosing changes, and symptom evaluation must
always be routed to a registered nurse.

The module exposes:
  - ``initialize_health_rag()`` — call once at startup to build the
    in-memory collection from healthcare_backend.KNOWLEDGE_BASE.
  - ``get_health_context(query)`` — async retrieval of the most relevant
    knowledge-base snippets for a natural-language patient question.

Dependencies (already in pyproject via the flower RAG)::

    qdrant-client>=1.9.0
    sentence-transformers>=3.0.0
    langchain-qdrant>=0.1.0
    langchain-huggingface>=0.0.3
"""

import asyncio

from loguru import logger

# ---------------------------------------------------------------------------
# Module state — populated lazily by initialize_health_rag().
# ---------------------------------------------------------------------------

_vector_store = None
_reranker = None
_embeddings = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "healthline-kb"
EMBEDDING_MODEL = "paraphrase-albert-small-v2"   # ~45 MB, fast
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 10   # candidates to fetch before reranking
TOP_K_FINAL = 3       # results returned after reranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_to_document_text(entry: dict) -> str:
    """Render a knowledge-base entry as human-readable text for embedding."""
    return (
        f"Topic: {entry['title']}\n"
        f"Category: {entry.get('category', 'general')}\n"
        f"Information: {entry['content']}"
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

async def initialize_health_rag() -> None:
    """Build the in-memory Qdrant collection from KNOWLEDGE_BASE.

    Safe to call multiple times — re-uses the existing collection if it is
    already populated. Call once at bot startup before the first
    ``get_health_context`` invocation.
    """
    global _vector_store, _reranker, _embeddings

    if _vector_store is not None:
        logger.debug("Health RAG already initialized, skipping.")
        return

    logger.info("Initializing HealthLine RAG pipeline...")

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

    def _build_vector_store():
        from langchain_community.vectorstores import Qdrant
        from langchain_core.documents import Document

        from healthcare_backend import KNOWLEDGE_BASE

        docs = []
        for entry in KNOWLEDGE_BASE:
            text = _entry_to_document_text(entry)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "title": entry["title"],
                        "category": entry.get("category", "general"),
                    },
                )
            )

        vs = Qdrant.from_documents(
            docs,
            _embeddings,
            location=":memory:",
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Health RAG collection '{COLLECTION_NAME}' built with {len(docs)} documents.")
        return vs

    _vector_store = await loop.run_in_executor(None, _build_vector_store)
    logger.info(f"Health RAG collection '{COLLECTION_NAME}' ready.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

async def get_health_context(query: str) -> str:
    """Retrieve the most relevant knowledge-base information for *query*.

    Returns a formatted string ready to be injected as a tool result. Returns
    an empty string if RAG is not yet initialized or nothing matches.

    Args:
        query: Natural-language patient question, e.g. "do I need to fast
               before my blood test" or "what's your cancellation policy".
    """
    if _vector_store is None:
        logger.warning("get_health_context called before initialize_health_rag(). Returning empty context.")
        return ""

    loop = asyncio.get_event_loop()

    def _retrieve():
        candidates = _vector_store.similarity_search(query, k=TOP_K_RETRIEVE)
        if not candidates:
            return ""

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = _reranker.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:TOP_K_FINAL]]

        lines = ["=== Relevant Clinic Information ==="]
        for doc in top_docs:
            lines.append(doc.page_content)
            lines.append("")
        return "\n".join(lines).strip()

    try:
        return await loop.run_in_executor(None, _retrieve)
    except Exception as exc:
        logger.error(f"Health RAG retrieval failed: {exc}")
        return ""
