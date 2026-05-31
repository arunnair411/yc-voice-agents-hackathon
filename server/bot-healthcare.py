#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HealthLine — healthcare voice agent (JSON-backed, no-RAG variant).

Why no RAG: the RAG variant loaded a ~45 MB embedding model plus a
cross-encoder reranker (and torch) before a call could start. That made calls
slow to connect and could leave the line silent while models warmed up. This
variant replaces all of that with a small JSON dataset loaded ONCE per call
session and answered with fast in-memory keyword scoring. Conclusions from the
call (medication adherence, refill decrements, a call-log entry) are written
back to the JSON file when the call ends.

Call flow:
  1. Emergency check — if medical emergency, instruct caller to call 911 first.
  2. Caller ID lookup — match phone to a known patient.
  3. Calling for self / someone else.
  4. Identity verification — name + exact date of birth before any PHI.
  5. Reason for call — free voice OR DTMF fallback menu (1–6, 0 repeats).
  6. Service flows: refills, appointments, medication check-in, clinic info.
  7. Warm transfer to a registered nurse when needed.

Robustness:
  - Every question tracks a no-response / unclear-response counter.
  - After 2 failed attempts at voice, the bot switches to DTMF instructions.
  - After 3 total failures on any step, the bot offers to transfer to a nurse.

Pipeline: Nemotron Speech Streaming STT → Nemotron-3-Super-120B LLM → Gradium TTS

Run::

    uv run bot-healthcare.py
"""

import os
import random
from datetime import date

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, FunctionCallResultProperties, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.runner.types import (
    DailyRunnerArguments,
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from healthcare_store import (
    find_patient_by_name,
    find_patient_by_phone,
    load_data,
    persist_call_results,
    search_knowledge,
)
from nemotron_llm import VLLMOpenAILLMService
from nvidia_stt import NVidiaWebSocketSTTService

load_dotenv(override=True)


async def get_call_info(call_sid: str) -> dict:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        return {}
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"
    try:
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        async with aiohttp.ClientSession() as http_session:
            async with http_session.get(url, auth=auth) as response:
                if response.status != 200:
                    return {}
                data = await response.json()
                return {"from_number": data.get("from"), "to_number": data.get("to")}
    except Exception as e:
        logger.error(f"Error fetching call info: {e}")
        return {}


async def run_bot(
    transport: BaseTransport,
    from_number: str | None = None,
    audio_in_sample_rate: int = 16000,
    audio_out_sample_rate: int = 24000,
):
    logger.info("Starting healthcare bot (JSON-backed)")

    # Load the dataset ONCE for this call session. Everything below reads from
    # this in-memory dict — no per-turn disk reads, no model loading.
    data = load_data()

    # Per-call session state
    session: dict = {
        "verified_patient": None,   # reference into data["patients"][phone]
        "calling_for_self": None,
        "caller_name": None,
        "selected_area": None,
        "no_response_counts": {},
        "transfer_requested": False,
        # Accumulated for write-back at end of call:
        "events": [],               # human-readable log of what happened
        "persisted": False,         # guard against double write-back
    }

    departments = data.get("departments", {})

    def _log_event(text: str) -> None:
        logger.info(f"[call event] {text}")
        session["events"].append(text)

    # -----------------------------------------------------------------------
    # Tools
    # -----------------------------------------------------------------------

    async def check_emergency(params: FunctionCallParams, is_emergency: bool) -> None:
        """Call this immediately at the start of every call to confirm whether
        this is a medical emergency. If the caller says yes, is_emergency=True.
        If they say it's not an emergency, is_emergency=False and the call
        continues normally.

        Args:
            is_emergency: True if the caller indicates this is an emergency.
        """
        if is_emergency:
            _log_event("Caller reported a medical emergency; directed to 911.")
            await params.result_callback({
                "action": "end_after_911",
                "message": (
                    "This is a medical emergency — please hang up and call 911 immediately. "
                    "If you cannot call 911, stay on the line and I will connect you to emergency services."
                ),
            })
        else:
            await params.result_callback({"action": "continue"})

    async def lookup_patient_by_phone(params: FunctionCallParams) -> None:
        """Look up whether the calling phone number matches a patient on file.
        Call this automatically after the emergency check. Returns the patient
        name if found, so the bot can ask 'Is this [name]?' instead of asking
        from scratch.
        """
        patient = find_patient_by_phone(data, from_number)
        if patient:
            await params.result_callback({
                "found": True,
                "name": patient["name"],
                "area": patient["area"],
                "provider": patient["provider"],
            })
        else:
            await params.result_callback({"found": False})

    async def verify_identity(
        params: FunctionCallParams,
        provided_name: str,
        provided_dob: str,
    ) -> None:
        """Verify a patient's identity by matching their stated name and date
        of birth against records. Must be called before any PHI is disclosed.

        Args:
            provided_name: Full name stated by the caller.
            provided_dob: Date of birth stated by the caller (any spoken format,
                e.g. "March fourteenth nineteen seventy-eight" or "03/14/1978").
        """
        # Prefer the phone-matched record; fall back to a name search.
        patient = find_patient_by_phone(data, from_number)
        if not patient:
            patient = find_patient_by_name(data, provided_name)

        if not patient:
            await params.result_callback({
                "verified": False,
                "reason": "No patient record found under that name.",
            })
            return

        def _digits(s: str) -> str:
            return "".join(c for c in s if c.isdigit())

        stored_digits = _digits(patient["dob"])  # YYYYMMDD, e.g. "19780314"
        provided_digits = _digits(provided_dob)

        # Name: at least two words, all of which appear in the stored name.
        provided_words = [w for w in provided_name.strip().lower().split() if w]
        stored_lower = patient["name"].lower()
        name_ok = len(provided_words) >= 2 and all(w in stored_lower for w in provided_words)

        # DOB must match exactly (all 8 digits).
        dob_ok = len(provided_digits) == 8 and provided_digits == stored_digits

        if name_ok and dob_ok:
            session["verified_patient"] = patient
            _log_event(f"Identity verified for {patient['name']} ({patient['mrn']}).")
            await params.result_callback({
                "verified": True,
                "patient_name": patient["name"],
                "mrn": patient["mrn"],
                "area": patient["area"],
                "provider": patient["provider"],
            })
        else:
            await params.result_callback({
                "verified": False,
                "reason": "Name or date of birth did not match our records.",
            })

    async def set_calling_for_self(
        params: FunctionCallParams,
        for_self: bool,
        caller_name: str | None = None,
    ) -> None:
        """Record whether the caller is calling on behalf of themselves or
        another person.

        Args:
            for_self: True if calling for themselves, False if for someone else.
            caller_name: If for_self is False, the name of the person calling
                (not the patient). Optional.
        """
        session["calling_for_self"] = for_self
        session["caller_name"] = caller_name
        await params.result_callback({"ok": True, "for_self": for_self})

    async def record_no_response(params: FunctionCallParams, step: str) -> None:
        """Record that the caller did not respond or gave an unclear response
        to the current step. Returns how many failures have occurred and what
        action to take next: 'retry_voice', 'switch_to_dtmf', or 'offer_transfer'.

        Args:
            step: Short label for the current step, e.g. 'emergency_check',
                'identity_verification', 'reason_for_call', 'dob_collection'.
        """
        counts = session["no_response_counts"]
        counts[step] = counts.get(step, 0) + 1
        n = counts[step]

        if n == 1:
            action = "retry_voice"
            hint = "Rephrase the question more simply and try again."
        elif n == 2:
            action = "switch_to_dtmf"
            hint = (
                "Switch to keypad instructions. For example: "
                "'Let's try a different way. Press 1 for yes, press 2 for no.' "
                "Or give the numbered menu for reason-for-call."
            )
        else:
            action = "offer_transfer"
            hint = "Offer to connect the caller to a patient services representative."

        await params.result_callback({"failures": n, "action": action, "hint": hint})

    async def get_reason_menu(params: FunctionCallParams) -> None:
        """Return the standard reason-for-call menu text. Call this when the
        caller is unclear about why they're calling, or when switching to DTMF
        fallback. The LLM should read this menu aloud.
        """
        await params.result_callback({
            "menu": (
                "Press 1 for a prescription refill. "
                "Press 2 to schedule or confirm an appointment. "
                "Press 3 to check in on your medications. "
                "Press 4 for lab or test results. "
                "Press 5 for billing or insurance questions. "
                "Press 6 to speak with a registered nurse. "
                "Press 0 to hear these options again."
            ),
            "options": {
                "1": "prescription_refill",
                "2": "appointment",
                "3": "medication_checkin",
                "4": "lab_results",
                "5": "billing",
                "6": "nurse_transfer",
                "0": "repeat_menu",
            },
        })

    async def request_prescription_refill(
        params: FunctionCallParams,
        medication_name: str,
    ) -> None:
        """Submit a prescription refill request for the verified patient.

        Args:
            medication_name: Name of the medication to refill, as stated by
                the caller.
        """
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({
                "ok": False,
                "reason": "Patient identity not yet verified.",
            })
            return

        med = next(
            (m for m in patient["medications"]
             if medication_name.lower() in m["name"].lower()),
            None,
        )
        if not med:
            await params.result_callback({
                "ok": False,
                "reason": f"No active prescription for '{medication_name}' found on file.",
            })
            return

        if med["refills_remaining"] == 0:
            _log_event(f"Refill requested for {med['name']} — no refills left, sent for provider auth.")
            await params.result_callback({
                "ok": False,
                "reason": (
                    f"No refills remaining for {med['name']}. "
                    "The request will be sent to your provider for authorization."
                ),
                "action": "provider_auth_required",
                "medication": med["name"],
            })
            return

        # Mutates the in-memory patient record; persisted to JSON at call end.
        med["refills_remaining"] -= 1
        ref_id = f"RX-{random.randint(100000, 999999)}"
        _log_event(f"Refill {ref_id} for {med['name']} ({med['refills_remaining']} left).")
        await params.result_callback({
            "ok": True,
            "refill_id": ref_id,
            "medication": med["name"],
            "dose": med["dose"],
            "refills_remaining_after": med["refills_remaining"],
            "ready_in": "24 to 48 hours",
        })

    async def get_upcoming_appointments(params: FunctionCallParams) -> None:
        """Retrieve upcoming appointments for the verified patient."""
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({
                "ok": False,
                "reason": "Patient identity not yet verified.",
            })
            return
        appts = patient.get("appointments", [])
        await params.result_callback({
            "ok": True,
            "appointments": appts,
            "count": len(appts),
        })

    async def schedule_appointment(
        params: FunctionCallParams,
        department: str,
        preferred_date: str,
        preferred_time: str | None = None,
    ) -> None:
        """Schedule a new appointment for the verified patient.

        Args:
            department: Department name or key (e.g. 'cardiology', 'primary care').
            preferred_date: Preferred date in the caller's own words.
            preferred_time: Preferred time in the caller's own words. Optional.
        """
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({
                "ok": False,
                "reason": "Patient identity not yet verified.",
            })
            return

        conf = f"APT-{random.randint(100000, 999999)}"
        _log_event(f"Appointment {conf} requested in {department} for {preferred_date}.")
        await params.result_callback({
            "ok": True,
            "confirmation": conf,
            "department": department,
            "requested_date": preferred_date,
            "requested_time": preferred_time or "morning",
            "note": "Scheduling team will call back within one business day to confirm.",
        })

    async def medication_checkin(params: FunctionCallParams) -> None:
        """Return the patient's current medication list with last-taken info
        so the bot can ask whether they took each medication on time.
        """
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({
                "ok": False,
                "reason": "Patient identity not yet verified.",
            })
            return
        await params.result_callback({
            "ok": True,
            "medications": patient["medications"],
        })

    async def record_medication_taken(
        params: FunctionCallParams,
        medication_name: str,
        taken_today: bool,
        taken_on_time: bool | None = None,
    ) -> None:
        """Record that the patient has (or has not) taken a specific medication
        today. The result is written back to the JSON record at call end.

        Args:
            medication_name: Name of the medication.
            taken_today: True if taken today, False if missed.
            taken_on_time: True if taken at the correct scheduled time. None if unknown.
        """
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({"ok": False})
            return

        # Update last_taken on the matching med, and append an adherence entry.
        med = next(
            (m for m in patient["medications"]
             if medication_name.lower() in m["name"].lower()),
            None,
        )
        canonical_name = med["name"] if med else medication_name
        if med and taken_today:
            med["last_taken"] = str(date.today())

        entry = {
            "date": str(date.today()),
            "medication": canonical_name,
            "taken_today": taken_today,
            "taken_on_time": taken_on_time,
        }
        patient.setdefault("adherence_log", []).append(entry)
        _log_event(
            f"Adherence: {canonical_name} taken={taken_today} on_time={taken_on_time}."
        )
        await params.result_callback({
            "ok": True,
            "recorded": True,
            "medication": canonical_name,
            "taken_today": taken_today,
        })

    async def lookup_clinic_info(params: FunctionCallParams, question: str) -> None:
        """Look up general, non-diagnostic clinic information to answer a
        patient's question — clinic policies, appointment prep, prescription
        processing times, insurance/billing, lab result access, telehealth, and
        general medication guidance.

        This searches the in-memory knowledge base with fast keyword scoring —
        no models, no delay. Read the returned information back in your own
        words. If nothing matches or the question is clinical (diagnosis, dosing
        changes, evaluating symptoms), do NOT guess — offer to transfer to a
        registered nurse instead.

        Args:
            question: The patient's question in natural language.
        """
        matches = search_knowledge(data, question, top_k=2)
        if not matches:
            await params.result_callback({
                "found": False,
                "note": (
                    "No matching clinic information. If this is a clinical "
                    "question, offer to transfer to a registered nurse."
                ),
            })
            return
        await params.result_callback({
            "found": True,
            "information": [{"title": m["title"], "content": m["content"]} for m in matches],
        })

    async def transfer_to_nurse(
        params: FunctionCallParams,
        department: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Initiate a warm transfer to a registered nurse or patient services
        representative. Call this when the caller requests to speak to someone,
        when the bot cannot resolve the issue, or after 3 failed interaction
        attempts.

        Args:
            department: Department to route to. Defaults to the patient's
                registered department or 'general'.
            reason: Brief reason for the transfer, spoken to the nurse. Optional.
        """
        verified = session.get("verified_patient")
        dept_key = (
            department
            or (verified["area"] if verified else None)
            or "general"
        ).lower()

        dept = departments.get(dept_key) or departments.get("general", {
            "display_name": "Patient Services",
            "nurse_line": "",
            "hours": "",
        })
        session["transfer_requested"] = True
        _log_event(f"Transfer to {dept['display_name']} — {reason or 'general inquiry'}.")

        await params.result_callback({
            "ok": True,
            "department": dept["display_name"],
            "nurse_line": dept.get("nurse_line", ""),
            "hours": dept.get("hours", ""),
            "reason": reason or "General patient inquiry",
            "message": (
                f"Transferring you to {dept['display_name']}. Please hold for a moment."
            ),
        })

    async def end_call(params: FunctionCallParams) -> None:
        """End the call. Only call this AFTER you have said goodbye in the
        same turn.
        """
        logger.info("end_call invoked")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    tool_functions = [
        check_emergency,
        lookup_patient_by_phone,
        verify_identity,
        set_calling_for_self,
        record_no_response,
        get_reason_menu,
        request_prescription_refill,
        get_upcoming_appointments,
        schedule_appointment,
        medication_checkin,
        record_medication_taken,
        lookup_clinic_info,
        transfer_to_nurse,
        end_call,
    ]
    tools = ToolsSchema(standard_tools=tool_functions)

    def _persist_results() -> None:
        """Write this call's conclusions back to the JSON file. Idempotent —
        guarded so it runs at most once even if called from multiple hooks.
        """
        if session["persisted"]:
            return
        session["persisted"] = True

        verified = session.get("verified_patient")
        patient_updates = {}
        if verified:
            patient_updates[verified["mrn"]] = {
                "medications": verified["medications"],
                "adherence_log": verified.get("adherence_log", []),
            }

        call_record = {
            "timestamp": str(date.today()),
            "from_number": from_number,
            "patient_mrn": verified["mrn"] if verified else None,
            "calling_for_self": session.get("calling_for_self"),
            "transferred": session.get("transfer_requested", False),
            "events": session["events"],
        }
        persist_call_results(call_record, patient_updates or None)

    # -----------------------------------------------------------------------
    # System prompt
    # -----------------------------------------------------------------------

    system_instruction = f"""You are a voice agent for HealthLine Medical Center — a calm, professional, and empathetic automated attendant that handles inbound patient calls 24/7.

Today is {date.today().strftime('%A, %B %d, %Y')}.

## Call flow — follow this order strictly

### Step 1: Emergency check
Your VERY FIRST line, always, before anything else:
"Thank you for calling HealthLine Medical Center. If this is a medical emergency, please hang up and call 9-1-1 now."
Then ask: "Is this a medical emergency?"
- If yes → call check_emergency(is_emergency=True), then read the result message and end the call.
- If no → call check_emergency(is_emergency=False) and continue.

### Step 2: Caller ID lookup
Immediately call lookup_patient_by_phone (no need to say anything while doing this).
- If a patient is found → say: "Is this [name]?" and ask them to confirm.
- If not found → ask for their full name.

### Step 3: Calling for self or someone else
Ask: "Are you calling for yourself today, or on behalf of someone else?"
Call set_calling_for_self with the answer.

### Step 4: Identity verification
Before sharing any medical information, verify identity:
- Ask for date of birth (you already have their name from step 2/3).
- Call verify_identity(provided_name=..., provided_dob=...).
- If verification fails: allow one retry with a different phrasing.
  After 2 failures, say you cannot share protected health information over the phone
  and offer to transfer to patient services.

### Step 5: Reason for call
Ask: "What can I help you with today?"
Accept a free voice response. Map it to one of:
  - prescription_refill → call request_prescription_refill
  - appointment (schedule/confirm/cancel) → call get_upcoming_appointments or schedule_appointment
  - medication_checkin → call medication_checkin
  - lab_results → transfer to the lab department
  - billing → transfer to the billing department
  - general question (policies, appointment prep, insurance accepted, refill
    timing, telehealth, accessing lab results, general medication guidance)
    → call lookup_clinic_info and answer from the returned information
  - nurse / unclear → call transfer_to_nurse

If the caller is unclear after 1 attempt, call get_reason_menu and read the numbered options.
If a caller presses or says a digit (1–6), map it to the menu options returned by get_reason_menu.

### Step 6: Service flows

**Prescription refill:**
- Ask which medication needs a refill.
- Call request_prescription_refill.
- Read back the result: refill ID, expected ready time, or explain if provider auth is needed.
- Ask: "Is there anything else I can help you with today?"

**Appointments:**
- Call get_upcoming_appointments and read any upcoming appointments.
- Offer to schedule a new one if needed; call schedule_appointment.
- Ask: "Is there anything else I can help you with today?"

**Medication check-in:**
- Call medication_checkin to get their medication list.
- For each medication: "Did you take your [name] today?" then "Was it on time?"
- Call record_medication_taken for each answer.
- Offer encouragement if they missed a dose. If they report side effects or concern,
  offer to transfer to a nurse.
- Ask: "Is there anything else I can help you with today?"

**Transfer to nurse:**
- Call transfer_to_nurse with the appropriate department.
- Read the message from the result aloud, then say a brief warm handoff line and call end_call.

## Answering general questions (knowledge base)
For "how does X work" or "what's your policy on Y" questions, use lookup_clinic_info
and answer from its result in your own words. NEVER answer clinical questions
(diagnosis, whether to change a dose, evaluating symptoms) from the knowledge base —
for those, and whenever lookup_clinic_info returns nothing, offer to transfer to a
registered nurse. General policy questions can be answered without identity
verification; a patient's specific records require verification first.

## Handling no-response / unclear input
Whenever the caller does not respond, says something unintelligible, or gives an answer
that doesn't make sense for the current question:
1. Call record_no_response(step="<current step label>").
2. Follow the action in the response:
   - "retry_voice" → rephrase the question simply and try again.
   - "switch_to_dtmf" → say "Let's try a different way," then give keypad instructions.
     For yes/no: "Press 1 for yes, press 2 for no." For reason-for-call: call
     get_reason_menu and read the numbered options.
   - "offer_transfer" → say you're having trouble understanding and call
     transfer_to_nurse, then end_call.

## Tone and style rules
- Calm, warm, professional — like a real medical office receptionist.
- One question at a time. Never ask for name + DOB + reason in the same breath.
- Keep responses short (1–2 sentences) except when reading menus or appointment summaries.
- Spell out numbers in medical context: "nine-one-one" not "911".
- Never say "Absolutely!", "Great!", "Perfect!" — go straight to the point.
- No bullet points, no emojis — everything is spoken aloud. Use contractions.
- Never disclose another patient's information.
- If the caller mentions suicidal thoughts, self-harm, or a mental health crisis:
  immediately say "I hear you. I'm connecting you with our mental health line right now,"
  then call transfer_to_nurse(department="mental health") and end_call.

## Ending a call
When the caller has no more needs: say "Take care, and feel better soon," then call end_call.
Never call end_call without a spoken goodbye in the same turn.
"""

    # Speech-to-Text — Nemotron Speech Streaming STT over WebSocket (16-bit PCM,
    # 16 kHz mono, matching the WebRTC input path). Override via NVIDIA_ASR_URL.
    stt = NVidiaWebSocketSTTService(
        url=os.getenv("NVIDIA_ASR_URL", "ws://192.168.7.228:8081"),
        strip_interim_prefix=True,
    )

    # LLM — Nemotron-3-Super-120B served by vLLM (OpenAI-compatible chat
    # completions at /v1). Reasoning ("thinking") is OFF by default for
    # low-latency voice; enable with NEMOTRON_ENABLE_THINKING=true only if the
    # vLLM server runs a reasoning parser (else chain-of-thought is spoken).
    enable_thinking = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"
    llm = VLLMOpenAILLMService(
        api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),  # vLLM ignores unless --api-key set
        base_url=os.getenv("NEMOTRON_LLM_URL", "http://192.168.7.228:8000/v1"),
        settings=VLLMOpenAILLMService.Settings(
            model=os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super"),
            system_instruction=system_instruction,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}},
        ),
    )

    # TTS — calm, professional voice
    tts = GradiumTTSService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumTTSService.Settings(
            voice=os.getenv("GRADIUM_VOICE_ID", "_6Aslh2DxfmnRLmP"),
        ),
    )

    for fn in tool_functions:
        llm.register_direct_function(fn)

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Default VAD + turn-taking, matching the Nemotron bot. Krisp
            # denoising runs on Pipecat Cloud, so no babble-robustness tuning is
            # needed here — and a min_words gate could strand a short backchannel
            # that lands on a tool result, leaving the bot silent.
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=FilterIncompleteUserTurnStrategies(),
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_in_sample_rate=audio_in_sample_rate,
            audio_out_sample_rate=audio_out_sample_rate,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected — starting healthcare call")
        context.add_message({
            "role": "user",
            "content": "A patient just called in. Begin the call following the call flow in your instructions.",
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected — persisting call results")
        _persist_results()
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    from_number: str | None = None
    transport_overrides: dict = {}

    if os.environ.get("ENV") != "local":
        from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
        krisp_filter = KrispVivaFilter()
    else:
        krisp_filter = None

    match runner_args:
        case DailyRunnerArguments():
            # Pipecat Cloud starts a session into a Daily room (the path Cekura's
            # automated testing uses). Join that room as the bot. Uses run_bot's
            # default 16 kHz in / 24 kHz out, same as the WebRTC path.
            transport = DailyTransport(
                runner_args.room_url,
                runner_args.token,
                "HealthLine",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_in_filter=krisp_filter,
                    audio_out_enabled=True,
                ),
            )

        case SmallWebRTCRunnerArguments():
            webrtc_connection: SmallWebRTCConnection = runner_args.webrtc_connection
            transport = SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=TransportParams(
                    audio_in_enabled=True,
                    audio_in_filter=krisp_filter,
                    audio_out_enabled=True,
                ),
            )

        case WebSocketRunnerArguments():
            transport_overrides["audio_in_sample_rate"] = 8000
            transport_overrides["audio_out_sample_rate"] = 8000

            _, call_data = await parse_telephony_websocket(runner_args.websocket)
            call_info = await get_call_info(call_data["call_id"])
            if call_info:
                from_number = call_info.get("from_number")
                logger.info(f"Healthcare call from: {from_number}")

            serializer = TwilioFrameSerializer(
                stream_sid=call_data["stream_id"],
                call_sid=call_data["call_id"],
                account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            )
            transport = FastAPIWebsocketTransport(
                websocket=runner_args.websocket,
                params=FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_in_filter=krisp_filter,
                    audio_out_enabled=True,
                    add_wav_header=False,
                    serializer=serializer,
                ),
            )

        case _:
            logger.error(f"Unsupported runner arguments: {type(runner_args)}")
            return

    await run_bot(transport, from_number=from_number, **transport_overrides)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
