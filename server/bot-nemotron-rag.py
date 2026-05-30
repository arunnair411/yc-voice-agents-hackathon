#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Field & Flower — flower shop voice ordering bot with RAG (rag branch, Nemotron variant).

Drop-in replacement for bot-nemotron.py that augments the LLM with
retrieval-augmented generation (RAG). Before the LLM responds to each caller
turn, a semantic search over the bouquet catalog fetches the most relevant
items and injects them as additional context into the shared LLMContext.
Everything else — NVIDIA STT, Nemotron LLM, Gradium TTS, tools, Twilio
wiring — is identical to the original bot-nemotron.py.

Pipeline:
    NVIDIA WebSocket STT → RAGContextProcessor → Nemotron-3-Super-120B LLM → Gradium TTS

Run with::

    uv run bot-nemotron-rag.py
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
from pipecat.frames.frames import (
    EndTaskFrame,
    FunctionCallResultProperties,
    LLMContextFrame,
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import (
    RunnerArguments,
    SmallWebRTCRunnerArguments,
    WebSocketRunnerArguments,
)
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_start.min_words_user_turn_start_strategy import (
    MinWordsUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import FilterIncompleteUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from mock_backend import BOUQUETS, KNOWN_CUSTOMERS
from nemotron_llm import VLLMOpenAILLMService
from nvidia_stt import NVidiaWebSocketSTTService
from rag import get_rag_context, initialize_rag

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# RAG context processor
# ---------------------------------------------------------------------------

class RAGContextProcessor(FrameProcessor):
    """Intercepts LLMContextFrame, runs a RAG lookup against the bouquet
    catalog, and injects the retrieved snippets as a system message directly
    into the shared LLMContext before passing the frame downstream to the LLM.

    Using LLMContextFrame (not the deprecated LLMMessagesFrame) and mutating
    the shared context object in-place is the correct pattern for Pipecat 1.3+.
    """

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            context = frame.context  # shared, mutable LLMContext

            # Extract the latest user message to use as the RAG query.
            user_text = ""
            for msg in reversed(context.get_messages()):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_text = content
                    elif isinstance(content, list):
                        user_text = " ".join(
                            p.get("text", "") for p in content if isinstance(p, dict)
                        )
                    break

            if user_text:
                rag_context = await get_rag_context(user_text)
                if rag_context:
                    logger.debug(f"RAG injecting {len(rag_context)} chars of catalog context.")

                    # Find the index of the last user message so we can insert
                    # the RAG system message immediately before it — this keeps
                    # the context ordering natural for the LLM.
                    messages = context.get_messages()
                    insert_at = len(messages)
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            insert_at = i
                            break

                    context.add_message(
                        {
                            "role": "system",
                            "content": (
                                "The following catalog information was retrieved specifically "
                                "for the caller's latest message. Use it as a helpful hint "
                                "when recommending bouquets, but always verify stock with "
                                "check_availability before committing.\n\n" + rag_context
                            ),
                        },
                        index=insert_at,
                    )

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Twilio helper (unchanged from bot-nemotron.py)
# ---------------------------------------------------------------------------

async def get_call_info(call_sid: str) -> dict:
    """Fetch call information from Twilio REST API using aiohttp."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.warning("Missing Twilio credentials, cannot fetch call info")
        return {}

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"

    try:
        auth = aiohttp.BasicAuth(account_sid, auth_token)
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=auth) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Twilio API error ({response.status}): {error_text}")
                    return {}
                data = await response.json()
                return {
                    "from_number": data.get("from"),
                    "to_number": data.get("to"),
                }
    except Exception as e:
        logger.error(f"Error fetching call info from Twilio: {e}")
        return {}


# ---------------------------------------------------------------------------
# Bot logic
# ---------------------------------------------------------------------------

async def run_bot(
    transport: BaseTransport,
    from_number: str | None = None,
    audio_in_sample_rate: int = 16000,
    audio_out_sample_rate: int = 24000,
):
    """Main bot logic — identical to bot-nemotron.py, plus RAGContextProcessor."""
    logger.info("Starting RAG Nemotron bot")

    # Initialise the RAG pipeline (no-op if already done).
    await initialize_rag()

    # Per-call order state.
    order: dict = {"items": [], "delivery": None}

    # --- Tools (identical to bot-nemotron.py) --------------------------------

    async def list_bouquets(
        params: FunctionCallParams,
        occasion: str | None = None,
        specials_only: bool = False,
    ) -> None:
        """List bouquets available today. Optionally filter by occasion or by
        what's currently on special.

        Use this when the caller asks what's available, mentions a specific
        occasion ("it's for my mom's birthday", "for Valentine's Day", "for a
        funeral"), or asks about specials/deals. Sold-out bouquets are
        automatically excluded from results.

        Args:
            occasion: Lowercase occasion to filter by. Common values:
                "birthday", "anniversary", "valentine's day", "mother's day",
                "sympathy", "wedding", "graduation", "thank you", "get well",
                "new baby", "housewarming", "christmas", "easter", "just
                because". Pass the canonical short form ("birthday", not "mom's
                birthday"). Omit to return the full catalog.
            specials_only: If True, only return bouquets currently on special.
        """
        results = []
        for name, info in BOUQUETS.items():
            if not info["in_stock"]:
                continue
            if specials_only and not info.get("on_special", False):
                continue
            if occasion is not None:
                occ = occasion.strip().lower()
                tags = [o.lower() for o in info.get("occasions", [])]
                if not any(occ in tag or tag in occ for tag in tags):
                    continue
            results.append({"name": name, **info})

        if not results and (occasion is not None or specials_only):
            await params.result_callback(
                {
                    "bouquets": [],
                    "note": (
                        "No bouquets match those filters. Tell the caller you don't have "
                        "anything specifically for that, and offer to browse the full "
                        "catalog or try a different angle."
                    ),
                }
            )
            return

        await params.result_callback({"bouquets": results})

    async def check_availability(params: FunctionCallParams, bouquet_name: str) -> None:
        """Check whether a specific bouquet is in stock today.

        Args:
            bouquet_name: The name of the bouquet to check, lowercase.
        """
        item = BOUQUETS.get(bouquet_name.lower())
        if not item:
            await params.result_callback(
                {"available": False, "reason": f"We don't carry a bouquet called '{bouquet_name}'."}
            )
            return
        if not item["in_stock"]:
            await params.result_callback(
                {"available": False, "reason": f"{bouquet_name} is sold out today."}
            )
            return
        await params.result_callback({"available": True, "price": item["price"]})

    async def add_to_order(
        params: FunctionCallParams, bouquet_name: str, quantity: int = 1
    ) -> None:
        """Add a bouquet to the customer's order. Only call this after the
        customer has confirmed they want this bouquet.

        Args:
            bouquet_name: The name of the bouquet to add, lowercase.
            quantity: How many of this bouquet to add. Defaults to 1.
        """
        item = BOUQUETS.get(bouquet_name.lower())
        if not item:
            await params.result_callback(
                {"ok": False, "reason": f"We don't carry a bouquet called '{bouquet_name}'."}
            )
            return
        if not item["in_stock"]:
            await params.result_callback(
                {"ok": False, "reason": f"{bouquet_name} is sold out today."}
            )
            return
        order["items"].append(
            {"bouquet": bouquet_name.lower(), "quantity": quantity, "price": item["price"]}
        )
        await params.result_callback({"ok": True, "items": order["items"]})

    async def get_order_summary(params: FunctionCallParams) -> None:
        """Read back the current order: items, quantities, and running total."""
        total = sum(line["price"] * line["quantity"] for line in order["items"])
        await params.result_callback(
            {"items": order["items"], "total": round(total, 2), "delivery": order["delivery"]}
        )

    async def set_delivery_details(
        params: FunctionCallParams,
        recipient_name: str,
        address: str,
        delivery_date: str,
    ) -> None:
        """Capture delivery details for the order.

        Args:
            recipient_name: Name of the person receiving the flowers.
            address: Delivery street address.
            delivery_date: Requested delivery date, in the customer's own words
                (e.g. "Friday", "May 20th"). No parsing required.
        """
        order["delivery"] = {
            "recipient_name": recipient_name,
            "address": address,
            "delivery_date": delivery_date,
        }
        await params.result_callback({"ok": True, "delivery": order["delivery"]})

    async def place_order(params: FunctionCallParams) -> None:
        """Finalize the order. Only call this after the customer has confirmed
        the items AND delivery details."""
        if not order["items"]:
            await params.result_callback({"ok": False, "reason": "No items in the order yet."})
            return
        if not order["delivery"]:
            await params.result_callback({"ok": False, "reason": "Missing delivery details."})
            return
        total = sum(line["price"] * line["quantity"] for line in order["items"])
        confirmation = f"FLW-{random.randint(100000, 999999)}"
        logger.info(f"Order placed: {confirmation} total=${total:.2f} order={order}")
        await params.result_callback(
            {
                "ok": True,
                "confirmation_number": confirmation,
                "total": round(total, 2),
                "eta": "within 2 business days",
            }
        )

    async def end_call(params: FunctionCallParams) -> None:
        """End the call. Only call this AFTER you have said goodbye to the
        customer in the same turn. The pipeline will flush any queued speech
        and then hang up."""
        logger.info("end_call invoked — pushing EndTaskFrame upstream")
        await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        await params.result_callback(
            {"ok": True}, properties=FunctionCallResultProperties(run_llm=False)
        )

    tool_functions = [
        list_bouquets,
        check_availability,
        add_to_order,
        get_order_summary,
        set_delivery_details,
        place_order,
        end_call,
    ]
    tools = ToolsSchema(standard_tools=tool_functions)

    # --- System instruction (unchanged from bot-nemotron.py) ----------------

    customer = KNOWN_CUSTOMERS.get(from_number or "")
    if customer:
        caller_context = (
            f"This caller is a returning customer (caller ID matched). On file: "
            f"name {customer['name']}, last order the {customer['last_order']} bouquet. "
            'Greet them generically: "Welcome back to Field & Flower! How can I help '
            'today?" Do not use their name or mention their last order in the greeting; '
            "that comes across as surveilling. Once they say they want flowers, you "
            "can offer their last order as a helpful shortcut, framed as record-keeping: "
            f'"I have you down for the {customer["last_order"]} last time, want that '
            'again or something different?" Always give them the alternative.'
        )
    else:
        caller_context = (
            "You're talking to a new customer. Introduce the shop briefly and ask how you can help."
        )

    system_instruction = (
        "You are a friendly order-taker for Field & Flower, a neighborhood flower shop. "
        "Help callers pick a bouquet and arrange delivery. Use the tools to look up "
        "bouquets, check stock, add items, capture delivery details, and place the order. "
        "Confirm the full order before calling place_order.\n\n"
        # RAG-specific note:
        "You will sometimes receive a block of 'Relevant Catalog Information' as a system "
        "message retrieved from our catalog database. Use it as a helpful hint when "
        "recommending bouquets, but always verify stock with check_availability before "
        "committing.\n\n"
        "Talk like a real shop clerk on the phone — not a chatbot:\n"
        "- Keep it to 1–2 short sentences per turn. Longer only when listing options or "
        "doing the final order read-back.\n"
        "- Ask ONE thing at a time. Don't ask for name, address, and date in one breath — "
        "ask for the name, wait, then the next.\n"
        '- Skip filler openers like "Absolutely!", "That sounds lovely!", "Perfect!", '
        '"I\'d be happy to" — go straight to the point.\n'
        "- Describe bouquets plainly. \"A dozen red roses with baby's breath, sixty-five "
        'dollars." Not "a classic, romantic bouquet showing love and appreciation."\n'
        "- When listing bouquets, ALWAYS lead with the bouquet's name. Format: "
        '"<Name> — <description>, <price>." For example: "Spring Sunshine — yellow tulips '
        'and daffodils, forty-five dollars." The name is how the caller refers back to it.\n'
        "- When the caller mentions an occasion (birthday, Mother's Day, anniversary, "
        "sympathy, etc.) or asks about specials/deals, pass those as filters to "
        'list_bouquets (occasion="..." or specials_only=True) instead of reading the '
        "full catalog. Don't list 15 bouquets when 3 are relevant.\n"
        "- The catalog has many options — when listing, name at most 4 or 5 at a time. "
        "If the caller doesn't bite, offer to share more.\n"
        "- Don't restate what the customer just said back to them, except in the final "
        "order confirmation.\n"
        "- Use contractions. Fragments are fine.\n\n"
        "Responses are spoken aloud. No bullet points, no emojis. Read prices in words "
        '("forty-five dollars", not "$45.00").\n\n'
        "When the order is placed and the customer has no more requests, or when they say "
        'goodbye: say a short closing line (e.g. "Thanks, have a great day!") AND call '
        "end_call in the same turn. Never call end_call without saying goodbye first.\n\n"
        f"Today is {date.today().strftime('%A, %B %d, %Y')}. Use this when the caller "
        'gives a relative delivery date like "this Friday" or "next Tuesday".\n\n'
        f"Caller context: {caller_context}"
    )

    # --- Services (identical to bot-nemotron.py) ----------------------------

    stt = NVidiaWebSocketSTTService(
        url=os.getenv("NVIDIA_ASR_URL", "ws://192.168.7.228:8081"),
        strip_interim_prefix=True,
    )

    enable_thinking = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"
    llm = VLLMOpenAILLMService(
        api_key=os.getenv("NEMOTRON_LLM_API_KEY", "EMPTY"),
        base_url=os.getenv("NEMOTRON_LLM_URL", "http://192.168.7.228:8000/v1"),
        settings=VLLMOpenAILLMService.Settings(
            model=os.getenv("NEMOTRON_LLM_MODEL", "nvidia/nemotron-3-super"),
            system_instruction=system_instruction,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}},
        ),
    )

    tts = GradiumTTSService(
        api_key=os.environ["GRADIUM_API_KEY"],
        settings=GradiumTTSService.Settings(
            voice=os.getenv("GRADIUM_VOICE_ID", "Eu9iL_CYe8N-Gkx_"),
        ),
    )

    for fn in tool_functions:
        llm.register_direct_function(fn)

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(confidence=0.8, min_volume=0.7, start_secs=0.3),
            ),
            user_turn_strategies=FilterIncompleteUserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=3)],
            ),
        ),
    )

    # --- Pipeline: RAGContextProcessor inserted between user_aggregator and llm ---
    rag_processor = RAGContextProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            rag_processor,      # <-- RAG injection point
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

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
        logger.info("Client connected")
        context.add_message(
            {
                "role": "user",
                "content": "A customer just called. Greet them, 'This is Field & Flower, your local flower shop. How can I help you today?'",
            }
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


# ---------------------------------------------------------------------------
# Entry points (identical structure to bot-nemotron.py)
# ---------------------------------------------------------------------------

async def bot(runner_args: RunnerArguments):
    """Main bot entry point."""
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
                logger.info(f"Call from: {from_number} to: {call_info.get('to_number')}")

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
            logger.error(f"Unsupported runner arguments type: {type(runner_args)}")
            return

    await run_bot(transport, from_number=from_number, **transport_overrides)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
