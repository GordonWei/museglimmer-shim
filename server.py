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
    return prompt, images


def _generate_once(model, processor, prompt: str, images: list[str], max_tokens: int, temperature: float):
    with _generate_lock:
        t0 = time.time()
        result = generate(
            model,
            processor,
            prompt,
            image=images or None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        elapsed = time.time() - t0
        log.info(
            "generated %d tokens in %.1fs (%.1f tok/s), prompt_tokens=%d, images=%d",
            result.generation_tokens,
            elapsed,
            result.generation_tps,
            result.prompt_tokens,
            len(images),
        )
        return result


async def _generate_with_empty_retry(
    model, processor, prompt: str, images: list[str], max_tokens: int, temperature: float
):
    result = await anyio.to_thread.run_sync(_generate_once, model, processor, prompt, images, max_tokens, temperature)
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
    # exactly that.
    if not content and not tool_calls:
        log.info("empty reply after self-reasoning consumed the token budget, retrying once")
        result = await anyio.to_thread.run_sync(
            _generate_once, model, processor, prompt, images, max_tokens, temperature
        )
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
    result_holder: dict = {}

    def _run_stream():
        with _generate_lock:
            t0 = time.time()
            # Each chunk's .text is only the incremental delta for that
            # step (verified directly: the LAST chunk alone is a fragment
            # like " instruction:", not the full reply) — has to be
            # concatenated to reconstruct the full generation. Only the
            # last chunk's stats fields (generation_tokens etc.) are
            # cumulative.
            full_text = ""
            last = None
            for chunk in stream_generate(
                model, processor, prompt, image=images or None, max_tokens=max_tokens, temperature=temperature
            ):
                full_text += chunk.text
                last = chunk
            elapsed = time.time() - t0
            if last is not None:
                log.info(
                    "generated %d tokens in %.1fs (%.1f tok/s), prompt_tokens=%d",
                    last.generation_tokens,
                    elapsed,
                    last.generation_tps,
                    last.prompt_tokens,
                )
            result_holder["result"] = last
            result_holder["text"] = full_text

    def sse(obj: dict) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()

    base = {"id": "museglimmer-shim", "object": "chat.completion.chunk", "model": MODEL_ID}

    async def _body():
        yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

        task = asyncio.ensure_future(anyio.to_thread.run_sync(_run_stream))
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=_HEARTBEAT_INTERVAL_S)
            except asyncio.TimeoutError:
                yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
        await task

        raw_text = result_holder.get("text", "")
        content, tool_calls = _parse_model_output(raw_text)
        if not content and not tool_calls:
            log.info("empty reply after self-reasoning consumed the token budget (stream), retrying once")
            result_holder.clear()
            retry_task = asyncio.ensure_future(anyio.to_thread.run_sync(_run_stream))
            while not retry_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(retry_task), timeout=_HEARTBEAT_INTERVAL_S)
                except asyncio.TimeoutError:
                    yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": None}]})
            await retry_task
            raw_text = result_holder.get("text", "")
            content, tool_calls = _parse_model_output(raw_text)

        message, finish_reason = _build_message(content, tool_calls)
        delta = {k: v for k, v in message.items() if k != "role"}
        yield sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        yield b"data: [DONE]\n\n"

    return StreamingResponse(_body(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    prompt, images = _build_prompt(body)
    model = _state["model"]
    processor = _state["processor"]

    max_tokens = int(body.get("max_tokens") or 4096)
    temperature = float(body.get("temperature") or 0.2)

    if body.get("stream"):
        return await _stream_chat_completions(model, processor, prompt, images, max_tokens, temperature)

    result, content, tool_calls = await _generate_with_empty_retry(
        model, processor, prompt, images, max_tokens, temperature
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
