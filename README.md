# museglimmer-shim

An OpenAI-compatible `/v1/chat/completions` server for [Muse Glimmer](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit) on Apple Silicon, for when your local model runner's MLX build is a step behind the model you want to run.

## Why this exists

LM Studio bundles its own MLX runtime, and as of 2026-09 the latest available build on both the stable and beta channels (`mlx-llm-mac-arm64-apple-metal-advsimd@1.11.0`) doesn't yet support Muse Glimmer's architecture. Trying to load it fails with:

```
No module named 'mlx_vlm.speculative.drafters.muse_glimmer'
```

The `mlx_vlm` Python package on PyPI (not LM Studio's bundled fork) supports it. This project loads the model once, in-process, via `mlx_vlm`'s `load()`/`generate()` API, and serves it over a small FastAPI app shaped exactly like the OpenAI Chat Completions API — so any existing OpenAI-compatible client keeps working unchanged, just pointed at a different host/port.

## What it is not

`mlx_vlm` is MLX-based, which means Apple Silicon + Metal only. This does **not** run on Linux, and it does **not** run inside a Docker container on macOS either — Docker Desktop for Mac runs a Linux VM with no Metal GPU passthrough. If you need this on a Kubernetes cluster, the nodes would have to be Apple Silicon Macs, which is not a configuration RKE2/most homelab clusters have. Run it as a native macOS process (see below).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

Ad hoc, in the foreground:

```bash
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8091
```

As a persistent, auto-restarting macOS service via `launchd`:

```bash
./museglimmer-shim.sh start     # generates the plist for this checkout's path and loads it
./museglimmer-shim.sh status    # launchctl state + a live /health check
./museglimmer-shim.sh logs      # tail stdout/stderr
./museglimmer-shim.sh restart
./museglimmer-shim.sh stop
```

`museglimmer-shim.sh` generates `~/Library/LaunchAgents/com.wei.museglimmer-shim.plist` from `com.wei.museglimmer-shim.plist.template` at `start`/`restart` time, substituting in this checkout's actual path — the template isn't tied to any one machine or username.

The model is chosen by the `MUSEGLIMMER_MODEL` environment variable (defaults to `mlx-community/Muse-Glimmer-30B-4bit`); swapping models doesn't require a code change, just an env var and a restart.

## API

- `POST /v1/chat/completions` — `{model, messages, max_tokens, temperature, tools, stream}` in, standard OpenAI chat-completion shape out (`choices[].message.content` / `.tool_calls`, or SSE chunks when `stream: true`).
  - **Tool calling**: pass `tools` in OpenAI's function-calling shape and this shim gets Muse Glimmer's chat template to inject its own tool-calling syntax (an XML-ish `<atem:function_calls>` block, not the OpenAI/Anthropic convention — this shim parses it and translates it into a standard `tool_calls` response). Without `tools` declared, the model still emits its internal routing tokens (`to=self<|message|>...`) but has no idea what syntax to use if it decides to call something anyway, producing garbled output — always pass `tools` if the caller might want the model to use one.
  - **Images**: pass OpenAI-shaped vision content (`{"type": "image_url", "image_url": {"url": "..."}}` — a `data:` URI, a plain URL, or a local file path all work) mixed into a message's `content` list. Muse Glimmer is a genuine vision-language model (confirmed via its `config.json`'s `vision_config`/`image_token_id`), not text-only — it sees the actual image, not a caption from some other model.
  - **Streaming matters more than it looks like it should**: some clients (confirmed with Clawdbot) treat a slow *non-streaming* response as indistinguishable from a stalled connection and abort+retry it after a fixed idle window, regardless of any configured request timeout — because from the client's side, zero bytes arrive until the whole completion is done, either way. `stream: true` gets this shim to emit periodic heartbeat SSE frames while generating, which is enough to keep such a client from giving up on a genuinely-still-working, just-slow request.
- `GET /v1/models` — for clients that probe available models before use.
- `GET /health` — `{status, model, loaded}`.

Generation is serialized behind a single lock: MLX/Metal state isn't safe to hit concurrently from multiple threads, and this is meant for one user's own tools, not a multi-tenant service. One consequence worth knowing: if a client aborts its own HTTP request (its own timeout, a retry), this shim has no way to know that and keeps generating anyway — the next request just queues behind it until it finishes, however long that takes.

## A real limitation, not a bug: full-agent-context latency

Feeding this a short, self-contained prompt (a summarization task, a single question) is fast enough for interactive use — tens of seconds. Feeding it a prompt that carries a large fixed system-prompt overhead (a full agent framework's tool/skill definitions — tens of thousands of tokens before the user's actual message even starts) routinely pushes single-request latency past 300–400 seconds on an M-series Mac, because that overhead has to be processed on every single turn regardless of how simple the user's actual question is. That's model-speed math, not something this shim can paper over: a 30B model at ~13–15 tok/s just needs that long to get through that much prompt. If the client you're pointing this at has its own "has this been silent too long" abort logic on top of (not just) an overall request timeout, budget for it — this genuinely can take longer than a fast cloud model would, on every turn, once a large fixed prompt is in the mix.

## Real usage

Built to give two personal projects — [victoria-gateway](https://github.com/GordonWei/victoria-gateway) (an AIOps alert summarizer) and a personal WhatsApp/Telegram assistant gateway — a way to use Muse Glimmer as their local LLM backend without either of them needing to know or care that the model isn't natively supported by the model runner they were originally built against.

## License

MIT — see [LICENSE](LICENSE).
