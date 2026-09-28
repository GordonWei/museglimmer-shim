"""OpenAI-compatible /v1/chat/completions shim for Muse Glimmer (MLX).

Muse Glimmer is a very new architecture that LM Studio's bundled MLX runtime
(mlx-llm-mac-arm64-apple-metal-advsimd@1.11.0, confirmed the latest available
on both stable and beta channels as of 2026-09-20) doesn't support yet:

    No module named 'mlx_vlm.speculative.drafters.muse_glimmer'

The raw `mlx_vlm` pip package (0.7.1) does support it. This process loads the
model ONCE at startup and keeps it resident in memory, then serves plain
OpenAI-shaped HTTP so any existing OpenAI-compatible client (victoria-gateway's
pkg/model.OpenAIClient, Clawdbot's `openai-completions` provider) can talk to
it exactly like it talks to LM Studio today — no client-side code changes.

Run via the museglimmer-shim.sh management script (launchd-backed), not
directly, in normal operation.
"""

import asyncio
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_vlm import generate, load, stream_generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("museglimmer-shim")

MODEL_ID = os.environ.get("MUSEGLIMMER_MODEL", "mlx-community/Muse-Glimmer-30B-4bit")

# MLX generation is not safe to call concurrently from multiple threads
# against the same model/GPU state — serialize requests. This is a personal
# single-user shim (victoria-gateway + Clawdbot, not a public service), so
# one-request-at-a-time is an acceptable tradeoff for correctness over
# throughput.
_generate_lock = threading.Lock()
_state: dict = {}

# Reject instead of queue when a generation is already running. With the
# lock above, a second caller just waits — and one Clawdbot request (~22k
# prompt tokens of tool definitions) holds the model for 5+ minutes, so
# victoria-gateway's alert summary sits in that queue until its own timeout
# and only then falls back to the cloud. A fast 503 lets it fall back right
# away (it treats 5xx/429 as "backend unavailable"; 4xx would be a hard
# error). Set MUSEGLIMMER_REJECT_WHEN_BUSY=0 to get the old queueing back.
REJECT_WHEN_BUSY = os.environ.get("MUSEGLIMMER_REJECT_WHEN_BUSY", "1") != "0"
BUSY_RETRY_AFTER_S = 30

# Hard ceiling on one generation. Under memory pressure this model has run at
# 1-4 tok/s, and a caller that has long since given up still holds the lock
# (and gets everyone else a 503) until the generation finishes. Checked between
# tokens, so it can't cut a single long prefill short.
MAX_GENERATION_S = float(os.environ.get("MUSEGLIMMER_MAX_GENERATION_S", "900"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading %s (this can take a while the first time / after a reboot)...", MODEL_ID)
    start = time.time()
    model, processor = load(MODEL_ID)
    _state["model"] = model
    _state["processor"] = processor
    log.info("model loaded in %.1fs", time.time() - start)
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


# Muse Glimmer's chat template speaks its own tool-calling dialect, NOT the
# Harmony-style "to=self<|message|>" JSON format this shim originally
# assumed. Confirmed by calling the tokenizer's own apply_chat_template with
# a `tools=` argument (2026-09-20): when tools are declared, the template
# injects instructions for an XML-ish <atem:function_calls> block, and the
# model routes each generated segment with a "to=<recipient><|message|>"
# prefix — "self" for private reasoning (never shown to the user), "user"
# for the real visible reply, or a tool name when it's invoking a tool.
#
# The root cause of the original leak bug: this shim never read the
# request's `tools` field at all, so the model never received the
# <atem:function_calls> syntax instructions and improvised a garbled hybrid
# of its own Harmony instincts and whatever informal tool descriptions
# leaked in through the system prompt text — which is what showed up
# verbatim in the user's chat.
_TO_RECIPIENT_RE = re.compile(r"to=(\S+?)<\|message\|>", re.DOTALL)
_INVOKE_RE = re.compile(r'<atem:invoke\s+name="([^"]+)">(.*?)</atem:invoke>', re.DOTALL)
_PARAM_RE = re.compile(r'<atem:parameter\s+name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL)


def _parse_function_calls(segment_text: str) -> list[dict]:
    """Extract tool calls from an <atem:function_calls> block. The template's
    own instructions say the output "is not expected to be valid XML and is
    parsed with regular expressions" — so this does the same, rather than
    trying a strict XML parse that a slightly malformed generation would
    break."""
    calls = []
    for name, body in _INVOKE_RE.findall(segment_text):
        arguments = {p_name: p_value.strip() for p_name, p_value in _PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": arguments})
    return calls


def _parse_model_output(text: str) -> tuple[str, list[dict]]:
    """Split raw Muse Glimmer output into (visible_content, tool_calls).

    "self"-routed segments (private reasoning) are always dropped. A
    segment containing an <atem:function_calls> block is parsed as tool
    calls regardless of its nominal "to=" recipient (the <atem:invoke
    name=...> value is authoritative). Everything else is visible text.
    A generation with no "to=" routing at all (the common case for a plain
    reply when the model doesn't need a tool) is returned as-is.
    """
    matches = list(_TO_RECIPIENT_RE.finditer(text))
    if not matches:
        return text.strip(), []

    visible_parts: list[str] = []
    tool_calls: list[dict] = []
    for i, m in enumerate(matches):
        recipient = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end]
        segment = re.sub(r"<\|eom\|>.*$", "", segment, flags=re.DOTALL).strip()
        segment = re.sub(r"<\|start\|>\s*assistant\s*$", "", segment).strip()

        if recipient == "self":
            continue
        calls = _parse_function_calls(segment)
        if calls:
            tool_calls.extend(calls)
        else:
            visible_parts.append(segment)

    return "\n".join(p for p in visible_parts if p).strip(), tool_calls


def _to_mlx_messages(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Convert OpenAI-shaped messages into what the tokenizer's chat
    template + mlx_vlm's generate() expect.

    Content that's already a plain string passes through unchanged. A
    multimodal content list (mixing {"type": "text", ...} and
    {"type": "image_url", ...} items, the standard OpenAI vision shape)
    gets each image_url replaced with a bare {"type": "image"} placeholder
    — confirmed by direct testing that Muse Glimmer's chat template
    inserts the real <|patch|> image token itself from just that
    placeholder, given no url/data. The actual image sources (URLs, file
    paths, or data: URIs — all three are handled by mlx_vlm's own
    load_image, so they're passed through unmodified) are collected
    separately for generate()'s `image=` argument, not embedded in the
    prompt text.
    """
    flattened = []
    images: list[str] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            flattened.append({"role": m["role"], "content": content})
            continue

        parts = []
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type in ("image_url", "image", "input_image"):
                url = part.get("image_url")
                if isinstance(url, dict):
                    url = url.get("url")
                if not isinstance(url, str):
                    url = part.get("url") if isinstance(part.get("url"), str) else None
                if url:
                    images.append(url)
                    parts.append({"type": "image"})
        flattened.append({"role": m["role"], "content": parts})
    return flattened, images


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


def _build_prompt(body: dict) -> tuple[str, list[str]]:
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")
    tools = body.get("tools") or None

    processor = _state["processor"]
    tokenizer = getattr(processor, "tokenizer", processor)

    flat_messages, images = _to_mlx_messages(messages)
    # Bypass mlx_vlm.prompt_utils.apply_chat_template here and call the
    # underlying tokenizer directly with `tools=` — this is what actually
    # gets Muse Glimmer's chat template to inject the <atem:function_calls>
    # syntax instructions (see _parse_model_output's doc comment above).
    # (`enable_thinking=False` was tried here too, on the theory that it'd
    # suppress the "to=self" reasoning preamble — confirmed by direct
    # testing that this template doesn't reference that variable at all, so
    # it's a silent no-op. Left out rather than kept as dead cargo-culted
    # code.)
    prompt = tokenizer.apply_chat_template(flat_messages, tools=tools, add_generation_prompt=True, tokenize=False)
    _log_prompt_breakdown(tokenizer, prompt, flat_messages, tools)
    return prompt, images


# Prefill is the slow part here (~60 tok/s under memory pressure), so a big
# prompt is minutes of silence before the first token. When one shows up, log
# where the tokens actually went, so there's something to trim.
_BREAKDOWN_THRESHOLD_TOKENS = 8000


def _log_prompt_breakdown(tokenizer, prompt: str, messages: list[dict], tools) -> None:
    def count(text: str) -> int:
        return len(tokenizer.encode(text))

    total = count(prompt)
    if total < _BREAKDOWN_THRESHOLD_TOKENS:
        return
    by_role: dict[str, int] = {}
    for m in messages:
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        by_role[m.get("role", "?")] = by_role.get(m.get("role", "?"), 0) + count(text or "")
    tool_sizes = sorted(
        ((t.get("function", t).get("name", "?"), count(json.dumps(t, ensure_ascii=False))) for t in (tools or [])),
        key=lambda x: -x[1],
    )
    log.info(
        "large prompt: %d tokens; messages by role %s; %d tools = %d tokens; tools %s",
        total,
        by_role,
        len(tool_sizes),
        sum(n for _, n in tool_sizes),
        tool_sizes,
    )


class _Generation:
    """What one generation produced, plus why it stopped early (if it did)."""

    def __init__(self, text: str, last, stopped: str | None):
        self.text = text
        self.prompt_tokens = last.prompt_tokens if last is not None else 0
        self.generation_tokens = last.generation_tokens if last is not None else 0
        self.total_tokens = self.prompt_tokens + self.generation_tokens
        self.stopped = stopped  # None, "client_gone" or "time_limit"


def _generate_once(
    model, processor, prompt: str, images: list[str], max_tokens: int, temperature: float, cancel: threading.Event
) -> _Generation:
    # Always token-by-token, even for non-streaming requests: it's the only
    # place we get control back between tokens, which is what lets a caller
    # that disconnected (or a generation that ran past MAX_GENERATION_S)
    # release the lock without killing a thread mid-Metal-call.
    with _generate_lock:
        t0 = time.time()
        deadline = t0 + MAX_GENERATION_S
        # Each chunk's .text is only the incremental delta for that step
        # (verified directly: the LAST chunk alone is a fragment like
        # " instruction:", not the full reply) — has to be concatenated to
        # reconstruct the full generation. Only the last chunk's stats fields
        # (generation_tokens etc.) are cumulative.
        text = ""
        last = None
        stopped = None
        for chunk in stream_generate(
            model, processor, prompt, image=images or None, max_tokens=max_tokens, temperature=temperature
        ):
            text += chunk.text
            last = chunk
            if cancel.is_set():
                stopped = "client_gone"
                break
            if time.time() > deadline:
                stopped = "time_limit"
                break
        elapsed = time.time() - t0
        if last is not None:
            log.info(
                "generated %d tokens in %.1fs (%.1f tok/s), prompt_tokens=%d, images=%d%s",
                last.generation_tokens,
                elapsed,
                last.generation_tps,
                last.prompt_tokens,
                len(images),
                f", stopped early: {stopped}" if stopped else "",
            )
        return _Generation(text, last, stopped)


async def _run_generation(model, processor, prompt, images, max_tokens, temperature, cancel, request=None):
    """Run one generation in a worker thread; flag it cancelled if the caller goes away."""
    task = asyncio.ensure_future(
        anyio.to_thread.run_sync(_generate_once, model, processor, prompt, images, max_tokens, temperature, cancel)
    )
    try:
        while not task.done():
            # Non-streaming clients don't cancel the handler when they hang
            # up, so poll for it; streaming ones are handled in _body().
            if request is not None and await request.is_disconnected():
                log.info("client disconnected, stopping generation")
                cancel.set()
            await asyncio.wait({task}, timeout=1)
        return task.result()
    finally:
        if not task.done():
            cancel.set()


async def _generate_with_empty_retry(
    model, processor, prompt: str, images: list[str], max_tokens: int, temperature: float, cancel, request=None
):
    result = await _run_generation(model, processor, prompt, images, max_tokens, temperature, cancel, request)
    content, tool_calls = _parse_model_output(result.text)

    # Muse Glimmer always opens with a "to=self" reasoning segment before
    # its real "to=user" answer or a tool call, and how long that reasoning
    # runs is genuinely stochastic (same non-determinism already documented
    # in victoria-gateway's summarizer for this model family) — confirmed by
    # direct testing: an identical trivial prompt at the same temperature
    # sometimes wraps up in a few dozen tokens, sometimes still hasn't
    # reached "to=user" after 400+. When the whole max_tokens budget gets
    # spent on "self" chatter with nothing left for the real answer, this
    # shim would otherwise silently return an empty reply with
    # finish_reason=length, indistinguishable from a genuine empty
    # response. One retry (same bet victoria-gateway's
    # chatWithEmptyReplyRetry already makes) is cheap insurance against
    # exactly that. Not worth it if the first try was stopped early.
    if not content and not tool_calls and result.stopped is None:
        log.info("empty reply after self-reasoning consumed the token budget, retrying once")
        result = await _run_generation(model, processor, prompt, images, max_tokens, temperature, cancel, request)
        content, tool_calls = _parse_model_output(result.text)

    return result, content, tool_calls


def _build_message(content: str, tool_calls: list[dict]) -> tuple[dict, str]:
    message: dict = {"role": "assistant"}
    finish_reason = "stop"
    if tool_calls:
        message["content"] = content or None
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
            }
            for i, call in enumerate(tool_calls)
        ]
        finish_reason = "tool_calls"
    else:
        message["content"] = content
    return message, finish_reason


# How often to emit an SSE heartbeat while waiting for a slow generation.
# Clawdbot (and presumably other OpenAI-client implementations) treat a
# provider as "idle" — and abort + retry — if a streaming response goes too
# long without any bytes at all, regardless of the fetch-level
# `timeoutSeconds` configured for the provider (confirmed against real
# traffic: a plain, non-streaming JSON response from this shim was getting
# killed at a hardcoded ~120s ceiling even with timeoutSeconds set far
# higher, because from the client's point of view a slow single JSON blob
# looks identical to a stalled connection — zero bytes until the very end).
# A cheap heartbeat frame every few seconds is standard practice for
# gateways fronting slow, non-incrementally-parseable generations for
# exactly this reason.
_HEARTBEAT_INTERVAL_S = 10


async def _stream_chat_completions(
    model, processor, prompt: str, images: list[str], max_tokens: int, temperature: float
):
    cancel = threading.Event()

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()

    base = {"id": "museglimmer-shim", "object": "chat.completion.chunk", "model": MODEL_ID}

    async def _generate_with_heartbeats():
        task = asyncio.ensure_future(
            anyio.to_thread.run_sync(_generate_once, model, processor, prompt, images, max_tokens, temperature, cancel)
        )
        try:
            while not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=_HEARTBEAT_INTERVAL_S)
                except asyncio.TimeoutError:
                    yield None
        finally:
            # Client hung up mid-stream (Starlette cancels this generator):
            # stop the worker between tokens so it releases the lock.
            if not task.done():
                log.info("client disconnected, stopping generation (stream)")
                cancel.set()
        yield await task

    async def _body():
        try:
            async for frame in _frames():
                yield frame
        finally:
            # Belt and braces: when Starlette cancels this generator the inner
            # one isn't guaranteed to be closed right away, so flag it here too.
            # Harmless after a normal finish (nothing is generating any more).
            cancel.set()

    async def _frames():
        yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

        result = None
        async for item in _generate_with_heartbeats():
            if item is None:
                yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
            else:
                result = item
        content, tool_calls = _parse_model_output(result.text)
        if not content and not tool_calls and result.stopped is None:
            log.info("empty reply after self-reasoning consumed the token budget (stream), retrying once")
            async for item in _generate_with_heartbeats():
                if item is None:
                    yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
                else:
                    result = item
            content, tool_calls = _parse_model_output(result.text)

        message, finish_reason = _build_message(content, tool_calls)
        if result.stopped == "time_limit":
            finish_reason = "length"
        delta = {k: v for k, v in message.items() if k != "role"}
        yield sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_body(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Best-effort check, not a reservation: two requests arriving in the same
    # instant can both pass it, and the lock still serializes them.
    if REJECT_WHEN_BUSY and _generate_lock.locked():
        log.info("busy, rejecting request with 503")
        return JSONResponse(
            {"error": {"message": "model is busy with another request", "type": "server_busy"}},
            status_code=503,
            headers={"Retry-After": str(BUSY_RETRY_AFTER_S)},
        )
    body = await request.json()
    prompt, images = _build_prompt(body)
    model = _state["model"]
    processor = _state["processor"]

    max_tokens = int(body.get("max_tokens") or 4096)
    temperature = float(body.get("temperature") or 0.2)

    if body.get("stream"):
        return await _stream_chat_completions(model, processor, prompt, images, max_tokens, temperature)

    result, content, tool_calls = await _generate_with_empty_retry(
        model, processor, prompt, images, max_tokens, temperature, threading.Event(), request
    )
    if result.stopped == "time_limit":
        # 5xx so callers with a fallback (victoria-gateway) move on to it.
        return JSONResponse(
            {"error": {"message": f"generation exceeded {MAX_GENERATION_S:.0f}s", "type": "timeout"}},
            status_code=504,
        )
    message, finish_reason = _build_message(content, tool_calls)

    return JSONResponse(
        {
            "id": "museglimmer-shim",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generation_tokens,
                "total_tokens": result.total_tokens,
            },
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": "model" in _state}
