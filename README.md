Cake order — Voice Ordering Bot
YC Voice Agents Hackathon submission



https://github.com/arunnair411/yc-voice-agents-hackathon/tree/main


1. What is this?
It is a voice agent for a neighborhood cake shop. A customer calls in and the bot handles the full ordering experience — no hold music, no phone trees, no staff needed for routine orders.
The caller can:
Ask what's available today, filtered by occasion ("something for my mom's birthday") or deals ("anything on special?")
Get bouquet descriptions and prices read back naturally
Add items to an order, hear a summary, and confirm
Give delivery details — recipient, address, date — one at a time
Place the order and receive a confirmation number, all by voice
What makes it real: the bot recognizes returning customers by phone number, greets them by name, and offers to reorder their last bouquet as a shortcut. All customer data — order history, preferences, past purchases — lives in a JSON record that is loaded once at the start of every call and written back at the end with the new order appended. No database, no latency, no cold starts. The next call picks up exactly where the last one left off.
The bot runs on the full NVIDIA stack (Nemotron Speech Streaming STT + Nemotron-3-Super-120B LLM) with Gradium TTS, orchestrated with Pipecat, and deployed via Pipecat Cloud to a real Twilio phone number.

2. Demo video
[https://drive.google.com/file/d/1BgyUizkRiqfWBGjASC6d_A07pzCeuqgn/view?usp=sharing ](https://photos.google.com/share/AF1QipPWIl5X1N5If7p-nliUFVKTqMg0_6pf1ZOo35mXFjS8yJU5FT6H0WHoeCgsDu-Tvg?key=QUFxVEwtb0NPeFBXTTFCUGhBVTFuWUNsOEI4VS1R&pli=1)


3. How we used Cekura, Nemotron, and Pipecat
Pipecat
Pipecat is the full orchestration layer:
Twilio / WebRTC → Nemotron STT → LLM context aggregator → Nemotron LLM → Gradium TTS → caller
What we used and tuned:
SileroVADAnalyzer with tuned params (confidence=0.8, min_volume=0.7, start_secs=0.3) — background noise on speakerphone calls was triggering false turns; Cekura tests helped us find the right thresholds
FilterIncompleteUserTurnStrategies + MinWordsUserTurnStartStrategy(min_words=3) — while the bot is speaking, requires 3 words before treating input as a real interruption; eliminates most accidental barge-ins from ambient noise
FunctionCallParams + register_direct_function for all 7 tools: list_bouquets, check_availability, add_to_order, get_order_summary, set_delivery_details, place_order, end_call
on_client_connected to greet and kick off the conversation; on_client_disconnected to trigger the JSON write-back
Nemotron
STT: NVIDIA Nemotron Speech Streaming over WebSocket at 16 kHz
LLM: Nemotron-3-Super-120B-A12B on the hackathon AWS endpoint
We ran the same bot on GPT-4.1 (baseline) and Nemotron-3-Super with identical system prompts and tools, and used Cekura to measure task completion rates across both.
Cekura
What we were testing:
Does the bot apply the right occasion filter (list_bouquets(occasion="birthday")) when the caller mentions an occasion, rather than reading 15 bouquets?
Does it collect delivery details one field at a time — name, then address, then date — never in one breath?
Does it only call place_order after the customer has explicitly confirmed — never before?
Does VAD tuning reduce false interruptions from background noise?

Results:
Before tuning
0%
After tuning
40%

Other evals


Biggest improvements was in unavailable scenario. Cekura caught it without needing to manually run dozens of test calls.

4. What we built during the hackathon
Starting point (hackathon starter):
bot-gpt.py — cake shop bot on GPT-4.1 + Gradium
bot-nemotron.py — same bot on NVIDIA stack
mock_backend.py — Python dict of 15 bouquets + 2 known customers
Built during the hackathon:
What
Where
JSON data store: single shop_data.json file replacing the Python dict
server/shop_data.json
Load-once-per-call pattern: data loaded once at call start, all tools read from memory
run_bot() in bot
Returning customer recognition by phone number with last-order shortcut
lookup_customer_by_phone tool
Order write-back: placed orders appended to customer history in JSON at call end
_persist_order()
VAD robustness tuning — iterated using Cekura eval results
VADParams in bot
Cekura evaluation scenarios

Key design decision — JSON over a database:
We replaced the hardcoded Python dict with a JSON file that persists across calls. Every call loads it once (a single file read, ~1ms), reads from memory for the entire call, and writes the new order back on disconnect. A returning customer's second call already knows what they ordered last time. No ORM, no connection pool, no cold start. For a demo-scale shop this is the right tradeoff — if it were a real shop with thousands of customers you'd swap in a database behind the same load/persist interface.

5. Tool feedback
Pipecat — what worked well
STT and LLM are genuinely plug-and-play — swapping Nemotron for GPT-4.1 is one import line, everything else is identical
FunctionCallParams + register_direct_function makes tools easy to write and test independently of the pipeline
LLMContextAggregatorPair handles the stateful turn management cleanly — we didn't have to think about it
Pipecat — could be better
SileroVADAnalyzer params (confidence, min_volume, start_secs, stop_secs) have no perceptual documentation — had to tune by ear and Cekura tests. A reference table for "quiet desktop vs. speakerphone vs. noisy environment" would save hours
When a module import fails silently at startup, the call connects but the bot says nothing — very hard to debug. A startup health check that fails loudly before accepting the first call would help
FilterIncompleteUserTurnStrategies(start=[MinWordsUserTurnStartStrategy(min_words=3)]) is a lot of nesting for a common pattern; a shorthand on LLMUserAggregatorParams would be cleaner
Nemotron — what worked well
Function calling accuracy was strong — "something for a funeral, not too expensive" reliably maps to list_bouquets(occasion="sympathy") without extra prompt engineering
TTFT on short outputs felt competitive with GPT-4.1


Run it yourself
git clone https://github.com/arunnair411/yc-voice-agents-hackathon
cd yc-voice-agents-hackathon && git checkout main
cd server && cp .env.example .env
# Fill in GRADIUM_API_KEY and NVIDIA_* keys (or OPENAI_API_KEY for GPT-4.1)
uv sync
uv run bot-nemotron.py    # Nemotron stack
# or: uv run bot-gpt.py  # GPT-4.1 baseline
# Open http://localhost:7860 → Connect
Add your phone number to shop_data.json under customers to test the returning-customer recognition flow.

Built at the YC Voice Agents Hackathon, May 2026.
