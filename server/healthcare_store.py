"""JSON-backed data store for the HealthLine voice agent (no-RAG variant).

Design goals
------------
- **Fast, no model loading.** The RAG variant loaded a ~45 MB embedding
  model plus a cross-encoder reranker (and torch) before each call could
  start, which made calls slow to connect and could leave the line silent.
  This module replaces all of that with a small JSON file and lightweight
  keyword scoring — nothing to download, nothing to warm up.
- **Loaded once per call session.** ``load_data()`` reads the JSON a single
  time at the start of a call; every tool then reads from that in-memory dict.
- **Write conclusions back to disk.** ``persist_call_results()`` reloads the
  current file, merges this call's deltas (medication adherence, refill
  decrements, an appended call-log entry), and writes atomically. A process
  lock plus reload-before-write keeps concurrent calls from clobbering each
  other's writes.

The on-disk file (``healthcare_data.json``) is the single source of truth:
patients, departments, a keyword-tagged knowledge base, and a ``call_logs``
array that accumulates the outcome of every call.
"""

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from loguru import logger

DATA_PATH = Path(__file__).parent / "healthcare_data.json"

# Serializes write-backs within this process so concurrent calls don't
# clobber each other. Each writer also reloads from disk before merging,
# so it only ever overwrites the specific records it touched.
_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data(path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Load the full dataset from disk. Call once at the start of a call.

    Returns a fresh dict (safe to mutate in-memory during the call). If the
    file is missing or corrupt, returns an empty-but-valid skeleton so the
    bot still runs rather than crashing the call.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            f"Loaded healthcare data: {len(data.get('patients', {}))} patients, "
            f"{len(data.get('knowledge_base', []))} KB entries."
        )
        return data
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Could not load {path}: {exc}. Using empty dataset.")
        return {"patients": {}, "departments": {}, "knowledge_base": [], "call_logs": []}


# ---------------------------------------------------------------------------
# Knowledge-base lookup (keyword scoring — the RAG replacement)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def search_knowledge(data: dict, query: str, top_k: int = 2) -> list[dict]:
    """Return the best-matching knowledge-base entries for *query*.

    Scoring is deliberately simple and fast: each entry earns points for
    curated keyword phrases that appear in the query, plus a smaller bonus
    for token overlap between the query and the entry's title/content. No
    embeddings, no network, no model load.

    Returns a list of ``{"title", "category", "content", "score"}`` dicts,
    highest score first, with score > 0 only.
    """
    kb = data.get("knowledge_base", [])
    if not query or not kb:
        return []

    q_lower = query.lower()
    q_tokens = set(_tokenize(query))

    scored = []
    for entry in kb:
        score = 0.0

        # Curated keyword phrases are the strongest signal.
        for kw in entry.get("keywords", []):
            if kw.lower() in q_lower:
                # Multi-word phrase matches count for more than single words.
                score += 3.0 if " " in kw else 2.0

        # Token overlap with title (medium) and content (light) as a backstop.
        title_tokens = set(_tokenize(entry.get("title", "")))
        content_tokens = set(_tokenize(entry.get("content", "")))
        score += 1.5 * len(q_tokens & title_tokens)
        score += 0.25 * len(q_tokens & content_tokens)

        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "title": e["title"],
            "category": e.get("category", "general"),
            "content": e["content"],
            "score": round(s, 2),
        }
        for s, e in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Patient lookup helpers
# ---------------------------------------------------------------------------

def find_patient_by_phone(data: dict, phone: str | None) -> dict | None:
    if not phone:
        return None
    return data.get("patients", {}).get(phone)


def find_patient_by_name(data: dict, name: str) -> dict | None:
    """Loose name search — all provided words must appear in a stored name."""
    words = [w for w in name.strip().lower().split() if w]
    if len(words) < 2:
        return None
    for patient in data.get("patients", {}).values():
        stored = patient["name"].lower()
        if all(w in stored for w in words):
            return patient
    return None


# ---------------------------------------------------------------------------
# Write-back (persist call conclusions to disk)
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON to *path* atomically (temp file + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # Clean up the temp file if anything went wrong before the rename.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def persist_call_results(
    call_record: dict,
    patient_updates: dict | None = None,
    path: Path | str = DATA_PATH,
) -> bool:
    """Merge this call's results into the on-disk JSON and save atomically.

    Reloads the current file first (so concurrent calls don't clobber each
    other), then:
      - applies ``patient_updates`` (keyed by MRN) to the matching patient,
        e.g. updated medication ``refills_remaining`` / ``last_taken`` and
        an appended ``adherence_log`` entry;
      - appends ``call_record`` to ``call_logs``.

    Args:
        call_record: A dict describing the call (timestamp, from_number,
            patient_mrn, events, conclusion).
        patient_updates: Optional ``{mrn: {...changes...}}``. Supported keys
            per patient: ``medications`` (full replacement list) and
            ``adherence_log`` (list to extend).
        path: Override the data file (tests).

    Returns True on success, False on failure (failures are logged, never
    raised, so a write-back problem can't drop the call).
    """
    path = Path(path)
    try:
        with _write_lock:
            # Reload freshest state from disk before merging.
            data = load_data(path)

            if patient_updates:
                patients = data.setdefault("patients", {})
                # Patients are keyed by phone; index by MRN for update.
                by_mrn = {p.get("mrn"): p for p in patients.values()}
                for mrn, changes in patient_updates.items():
                    target = by_mrn.get(mrn)
                    if not target:
                        logger.warning(f"persist: no patient with MRN {mrn}; skipping update.")
                        continue
                    if "medications" in changes:
                        target["medications"] = changes["medications"]
                    if "adherence_log" in changes:
                        target.setdefault("adherence_log", []).extend(changes["adherence_log"])

            data.setdefault("call_logs", []).append(call_record)
            _atomic_write(path, data)
            logger.info(
                f"Persisted call results to {path.name} "
                f"(call_logs now {len(data['call_logs'])})."
            )
            return True
    except Exception as exc:
        logger.error(f"Failed to persist call results: {exc}")
        return False
