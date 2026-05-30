#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HealthLine — healthcare voice agent.

Call flow:
  1. Emergency check — if medical emergency, instruct caller to hang up and
     call 911 before anything else.
  2. Caller ID lookup — if phone matches a known patient, offer to confirm
     identity rather than asking from scratch.
  3. Identity verification — confirm name + date of birth (or MRN).
  4. Reason for call — free voice OR DTMF fallback menu:
       1 = Prescription refill
       2 = Appointment scheduling
       3 = Medication check-in / adherence
       4 = Lab / test results
       5 = Billing question
       6 = Speak to a nurse
       0 = Repeat menu
  5. Service-specific flows (refills, appointments, med check-in).
  6. Warm transfer to registered nurse if needed.

Robustness:
  - Every question tracks a no-response / unclear-response counter.
  - After 2 failed attempts at voice, the bot switches to DTMF instructions.
  - After 3 total failures on any step, the bot offers to transfer to a nurse.

Pipeline: Gradium STT → OpenAI GPT-4.1 LLM → Gradium TTS

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
from pipecat.audio.vad.vad_analyzer import VADParams
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
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gradium.stt import GradiumSTTService
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from healthcare_backend import DEPARTMENTS, PATIENTS

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
    logger.info("Starting healthcare bot")

    # Per-call session state
    session: dict = {
        "verified_patient": None,       # patient dict once identity is confirmed
        "calling_for_self": None,       # True/False
        "caller_name": None,            # name if calling for someone else
        "selected_area": None,          # department key
        "no_response_counts": {},       # step -> int, tracks retries per step
        "transfer_requested": False,
    }

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
        Call this automatically after emergency check. Returns patient name if
        found, so the bot can ask 'Is this [name]?' instead of asking from scratch.
        """
        patient = PATIENTS.get(from_number or "")
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
        Accept flexible DOB formats — just pass what the caller said.

        Args:
            provided_name: Full name stated by the caller.
            provided_dob: Date of birth stated by the caller (any spoken format,
                e.g. "March fourteenth nineteen seventy-eight" or "03/14/1978").
        """
        # Try phone-based lookup first, then fall back to name search
        patient = PATIENTS.get(from_number or "")
        if not patient:
            # Search by name
            for p in PATIENTS.values():
                if provided_name.strip().lower() in p["name"].lower():
                    patient = p
                    break

        if not patient:
            await params.result_callback({
                "verified": False,
                "reason": "No patient record found under that name.",
            })
            return

        # Normalize DOB: extract digits and compare loosely
        def _digits(s: str) -> str:
            return "".join(c for c in s if c.isdigit())

        stored_digits = _digits(patient["dob"])  # e.g. "19780314"
        provided_digits = _digits(provided_dob)

        # Accept if all 8 digits match, or if the 4-digit year + month/day appear
        # Full-name match: all words provided must appear in the stored name
        provided_words = provided_name.strip().lower().split()
        stored_lower = patient["name"].lower()
        name_ok = len(provided_words) >= 2 and all(w in stored_lower for w in provided_words)

        # DOB must match exactly (all 8 digits: YYYYMMDD)
        dob_ok = len(provided_digits) == 8 and provided_digits == stored_digits

        if name_ok and dob_ok:
            session["verified_patient"] = patient
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

        med["refills_remaining"] -= 1
        ref_id = f"RX-{random.randint(100000, 999999)}"
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
        """Record that the patient has (or has not) taken a specific medication today.

        Args:
            medication_name: Name of the medication.
            taken_today: True if taken today, False if missed.
            taken_on_time: True if taken at the correct scheduled time. None if unknown.
        """
        patient = session.get("verified_patient")
        if not patient:
            await params.result_callback({"ok": False})
            return

        logger.info(
            f"Medication adherence: patient={patient['mrn']} "
            f"med={medication_name} taken={taken_today} on_time={taken_on_time}"
        )
        await params.result_callback({
            "ok": True,
            "recorded": True,
            "medication": medication_name,
            "taken_today": taken_today,
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
            reason: Brief reason for the transfer, spoken to the nurse.
                Optional.
        """
        dept_key = (
            department
            or (session["verified_patient"]["area"] if session["verified_patient"] else None)
            or "general"
        ).lower()

        dept = DEPARTMENTS.get(dept_key, DEPARTMENTS["general"])
        session["transfer_requested"] = True

        await params.result_callback({
            "ok": True,
            "department": dept["display_name"],
            "nurse_line": dept["nurse_line"],
            "hours": dept["hours"],
            "reason": reason or "General patient inquiry",
            "message": (
                f"Transferring you to {dept['display_name']}. "
                f"Please hold for a moment."
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
        transfer_to_nurse,
        end_call,
    ]
    tools = ToolsSchema(standard_tools=tool_functions)

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
Accept free voice response. Map it to one of:
  - prescription_refill → call request_prescription_refill
  - appointment (schedule/confirm/cancel) → call get_upcoming_appointments or schedule_appointment
  - medication_checkin → call medication_checkin
  - lab_results → transfer to lab department
  - billing → transfer to billing department
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
- Read the message from the result aloud.
- Say a brief warm handoff line, then call end_call.

### Handling no-response / unclear input
Whenever the caller does not respond, says something unintelligible, or gives an answer
that doesn't make sense for the current question:
1. Call record_no_response(step="<current step label>").
2. Follow the action in the response:
   - "retry_voice" → rephrase the question simply and try again.
   - "switch_to_dtmf" → say: "Let's try a different way." Then give keypad instructions.
     For yes/no: "Press 1 for yes, press 2 for no."
     For reason-for-call: call get_reason_menu and read the numbered options.
   - "offer_transfer" → say: "I'm having trouble understanding. Let me connect you with
     a patient services representative." then call transfer_to_nurse and end_call.

## Tone and style rules
- Calm, warm, professional — like a real medical office receptionist.
- One question at a time. Never ask for name + DOB + reason in the same breath.
- Keep responses short (1–2 sentences) except when reading back menus or appointment summaries.
- Spell out numbers in medical context: "nine-one-one" not "911", "ten milligrams" not "10mg".
- Never say "Absolutely!", "Great!", "Perfect!" — go straight to the point.
- No bullet points, no emojis — everything is spoken aloud.
- Always use contractions. Fragments are fine.
- Never disclose another patient's information.
- If the caller mentions suicidal thoughts, self-harm, or a mental health crisis:
  immediately say "I hear you. I'm connecting you with our mental health line right now."
  then call transfer_to_nurse(department="mental health") and end_call.

## Ending a call
When the caller has no more needs: say "Take care, and feel better soon." then call end_call.
Never call end_call without a spoken goodbye in the same turn.
"""

    # STT
    stt = GradiumSTTService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumSTTService.Settings(language=Language.EN),
    )

    # LLM
    llm = OpenAIResponsesLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIResponsesLLMService.Settings(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            system_instruction=system_instruction,
        ),
    )

    # TTS — use a calm, professional voice
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
            # Healthcare calls often have background noise (hospital environments,
            # speakerphone, elderly callers). These VAD settings are tuned to be
            # tolerant of that while still detecting real speech reliably.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.75,   # slightly lower than default — don't miss soft speech
                    min_volume=0.6,
                    start_secs=0.4,    # require speech to persist before opening a turn
                    stop_secs=1.0,     # wait longer before closing turn — patients may pause
                ),
            ),
            user_turn_strategies=FilterIncompleteUserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=2)],
            ),
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
        logger.info("Client disconnected")
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
