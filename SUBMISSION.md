# HealthLine — AI-Powered Healthcare Voice Agent

> YC Voice Agents Hackathon submission

---

## 1. What is this?

**HealthLine** is a voice agent that handles inbound patient calls for a medical clinic — the kind of work that today burns dozens of hours a week from front-desk staff and often leaves patients on hold.

A patient calls in. The bot handles the full interaction end-to-end:

- Screens for medical emergencies upfront (directs to 9-1-1 before anything else)
- Looks up the caller by phone number and offers to confirm their identity by name
- Verifies identity with name + date of birth before sharing any protected health information
- Asks what they need, then routes: prescription refill, appointment scheduling, medication adherence check-in, lab result access, billing, or live nurse transfer
- Answers general clinic questions (policies, hours, insurance accepted, refill timing) from a fast in-memory knowledge base — no hallucinations on factual clinic info
- Handles no-response and unclear speech gracefully: rephrases once, falls back to a DTMF keypad menu on the second failure, and offers a nurse transfer on the third
- Writes call conclusions (adherence records, refill decrements, a structured call log) back to a persistent JSON store so data survives across calls

The domain is real. Clinics spend enormous amounts of staff time on calls that are routine, repetitive, and scriptable. A voice agent that handles these reliably — and escalates appropriately — directly reduces cost and improves patient access.

---

## 2. Demo video

**[FILL IN — link to your ≤60 second demo video]**

Suggested content: one full patient call showing caller ID recognition → identity verification → prescription refill → "anything else?" → goodbye. Keep it under 60 seconds — one clean golden-path call is better than a tour of features.

---

## 3. How we used Cekura, Nemotron, and Pipecat

### Pipecat

Pipecat is the orchestration layer for everything. The pipeline is:

```
Twilio / WebRTC → Nemotron STT → LLM context aggregator → Nemotron LLM → Gradium TTS → output
```

We used Pipecat's:
- **`SileroVADAnalyzer`** with tuned parameters (`confidence=0.75`, `stop_secs=1.0`) — healthcare callers pause more than average, and speakerphone/background noise from a clinical environment makes the defaults too aggressive
- **`FilterIncompleteUserTurnStrategies`** with `MinWordsUserTurnStartStrategy(min_words=2)` — prevents single-word noise bursts from opening a turn mid-sentence
- **`FunctionCallParams` / direct function registration** for all tool calls — every action the bot takes (identity verification, refill, appointment scheduling, medication check-in, nurse transfer) is a typed Python function the LLM calls directly
- **`on_client_disconnected`** event to trigger atomic JSON write-back of the call's conclusions

### Nemotron

We built two versions of the healthcare bot — one on GPT-4.1 (`bot-gpt.py`-style), one on the NVIDIA-hosted Nemotron stack:
- **STT**: Nemotron Speech Streaming (WebSocket, 16 kHz)
- **LLM**: Nemotron-3-Super-120B-A12B via the hackathon vLLM endpoint

The Nemotron path uses the same tool definitions and system prompt as the GPT-4.1 path — we used Cekura to measure performance differences between the two.

### Cekura

We used Cekura to evaluate and improve the healthcare bot's performance across the call flow stages.

**What we were trying to accomplish:**
- Test that the emergency check fires correctly and never skips to other steps
- Test that identity verification correctly rejects mismatched names/DOBs and doesn't leak PHI before verification
- Test the no-response robustness path (voice → DTMF fallback → nurse transfer)
- Measure how often the bot asks multiple questions in a single turn (a key UX failure for elderly callers)
- Compare GPT-4.1 vs. Nemotron-3-Super on task completion rate

**Results:**

| Scenario | Before tuning | After tuning |
|---|---|---|
| Emergency check fires first (every call) | [FILL IN]% | [FILL IN]% |
| Identity verified before PHI disclosed | [FILL IN]% | [FILL IN]% |
| DTMF fallback triggered correctly on 2nd no-response | [FILL IN]% | [FILL IN]% |
| Multi-question turn rate (lower is better) | [FILL IN]% | [FILL IN]% |
| Overall task completion rate | [FILL IN]% | [FILL IN]% |

The biggest improvements came from [FILL IN — e.g. "tightening the system prompt's one-question-at-a-time rule" / "adding explicit step labels to the no-response tool"]. Cekura let us catch [FILL IN — e.g. "the bot was sometimes skipping the calling-for-self step when caller ID matched"] without needing to manually run dozens of test calls.

---

## 4. What we built during the hackathon

Everything in this repo under `healthapp_json` was built during the hackathon. Specifically:

**Starting point (from the hackathon starter repo):**
- `bot-gpt.py` — flower shop voice agent (GPT-4.1 + Gradium)
- `bot-nemotron.py` — flower shop voice agent (Nemotron stack)
- Pipeline infrastructure, Twilio wiring, Pipecat Cloud deploy config

**Built during the hackathon:**

| What | Where |
|------|-------|
| Full healthcare call flow (emergency → identity → reason → service → transfer) | `server/bot-healthcare.py` |
| Patient data store with medications, appointments, departments | `server/healthcare_data.json` |
| JSON data layer: load-once-per-call, keyword KB search, atomic write-back with per-call mutex | `server/healthcare_store.py` |
| Medication adherence check-in tool (did you take it, was it on time) | `bot-healthcare.py: record_medication_taken` |
| No-response robustness: per-step retry counter → voice retry → DTMF fallback → nurse transfer | `bot-healthcare.py: record_no_response` |
| Identity verification with exact DOB matching and 2-word name requirement | `bot-healthcare.py: verify_identity` |
| Call conclusion write-back (refills decremented, adherence logged, call_log entry appended) | `bot-healthcare.py: _persist_results` |
| RAG variant (explored and benchmarked, then replaced — see below) | `healthapp_withrag` branch |

**What we tried and moved away from:**
We built a full RAG pipeline first (`healthapp_withrag` branch) using Qdrant + HuggingFace embeddings + a cross-encoder reranker over a clinic knowledge base. It worked, but loading the embedding model and reranker on call start caused a ~20 second silent gap before the bot could speak — a non-starter for a real healthcare call. We replaced it with an in-memory keyword scoring approach (`healthcare_store.search_knowledge`) that is instantaneous, has no cold start, and performed equally well on the 10-entry KB we're using. We kept the RAG branch intact for comparison.

---

## 5. Tool feedback

### Pipecat

**What worked well:**
- The pipeline abstraction is clean and the composability is real — swapping Nemotron STT for Gradium STT required exactly one line change, everything else stayed identical
- `FunctionCallParams` + `register_direct_function` is a great pattern — tools are just typed async Python functions, easy to write and easy to test independently of the pipeline
- `LLMContextAggregatorPair` handles the tricky stateful turn management so we didn't have to

**Could be better:**
- The `SileroVAD` tuning parameters (`confidence`, `min_volume`, `start_secs`, `stop_secs`) are not well documented in terms of what they actually map to perceptually — we had to tune by ear. A table of "typical values for noisy vs. quiet environments" in the docs would save time
- `FilterIncompleteUserTurnStrategies` + `MinWordsUserTurnStartStrategy` is a lot of nesting to get to a simple "wait for 2 words" behavior. A convenience parameter would help
- The silent-call-on-startup behavior (if a module import fails silently, the call connects but nothing happens) makes debugging startup errors harder than it needs to be — a startup health check that logs loudly before accepting the first call would help

### Nemotron (NVIDIA feedback)

**What worked well:**
- The 120B model is genuinely impressive for function calling — it reliably maps patient intent to the right tool call without needing elaborate prompt engineering
- Streaming latency was acceptable for most turns; the TTFT felt competitive with GPT-4.1 for short outputs

**Could be better:**
- [FILL IN — e.g. "Tool call accuracy dropped on multi-step scenarios (e.g. identity verification followed immediately by a refill request in the same turn). The model sometimes tried to call place_refill before verify_identity resolved."]
- [FILL IN — e.g. "The model occasionally generated partial JSON in tool call arguments on long parameter lists, causing a parse failure. GPT-4.1 never did this."]
- [FILL IN — e.g. "Nemotron Speech Streaming occasionally dropped the first syllable of an utterance after a long pause — particularly noticeable when an elderly caller paused to think. This caused the STT to miss words like 'I' at the start of a sentence."]
- Latency on the hosted endpoint was [FILL IN — good/acceptable/inconsistent] and [FILL IN — was/wasn't] suitable for real-time voice

### Cekura

**What worked well:**
- [FILL IN — e.g. "The scenario generation was surprisingly good — it produced edge cases we hadn't thought of, like a caller who gives their married name but the record has their maiden name."]
- [FILL IN — e.g. "The transcript scoring was easy to interpret and actionable — we could see exactly which turn the bot went wrong."]
- The Pipecat integration worked with minimal configuration

**Bugs / improvement suggestions:**
- [FILL IN — any bugs you hit]
- [FILL IN — e.g. "Scenario generation for healthcare was biased toward happy-path callers; we had to manually write the no-response and confused-caller scenarios"]
- [FILL IN — e.g. "Re-running a specific failed scenario individually (without re-running the full suite) wasn't obvious in the UI"]

---

## 6. Live link

**[FILL IN — Twilio phone number or Pipecat Cloud URL, or remove this section if not deployed]**

To call the bot: dial `[FILL IN]` from any phone. You will be greeted as a new patient unless your number is in the test patient list (ask us during the demo).

---

## Running it yourself

```bash
git clone https://github.com/arunnair411/yc-voice-agents-hackathon
cd yc-voice-agents-hackathon
git checkout healthapp_json
cd server
cp .env.example .env
# Fill in OPENAI_API_KEY, GRADIUM_API_KEY (and NVIDIA_* if using Nemotron path)
uv sync
uv run bot-healthcare.py
# Open http://localhost:7860 and click Connect
```

To test with a known patient (gets caller-ID recognition), add your number to `healthcare_data.json` under `patients`.

---

*Built at the YC Voice Agents Hackathon, May 2026.*
