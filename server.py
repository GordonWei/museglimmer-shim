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

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template

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


def _to_mlx_history(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-shaped {role, content: str} messages into the
    {role, content: [{"type": "text", "text": ...}]} shape mlx_vlm's
    apply_chat_template expects (see mlx_vlm/chat.py's add_to_history)."""
    history = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        history.append({"role": m["role"], "content": content})
    return history


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    max_tokens = int(body.get("max_tokens") or 4096)
    temperature = float(body.get("temperature") or 0.2)

    model = _state["model"]
    processor = _state["processor"]

    history = _to_mlx_history(messages)
    prompt = apply_chat_template(processor, model.config, history, num_images=0)

    def _run():
        with _generate_lock:
            t0 = time.time()
            result = generate(
                model,
                processor,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            elapsed = time.time() - t0
            log.info(
                "generated %d tokens in %.1fs (%.1f tok/s), prompt_tokens=%d",
                result.generation_tokens,
                elapsed,
                result.generation_tps,
                result.prompt_tokens,
            )
            return result

    # generate() is a blocking call (holds the GIL-adjacent MLX/Metal work) —
    # run it in FastAPI's threadpool so the event loop isn't blocked for
    # other endpoints (health checks, /v1/models) while a summary is cooking.
    result = await anyio.to_thread.run_sync(_run)

    return JSONResponse(
        {
            "id": "museglimmer-shim",
            "object": "chat.completion",
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason or "stop",
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
