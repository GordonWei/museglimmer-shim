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

- `POST /v1/chat/completions` — `{model, messages, max_tokens, temperature}` in, standard `{choices: [{message: {role, content}}], usage: {...}}` out.
- `GET /v1/models` — for clients that probe available models before use.
- `GET /health` — `{status, model, loaded}`.

Generation is serialized behind a single lock: MLX/Metal state isn't safe to hit concurrently from multiple threads, and this is meant for one user's own tools, not a multi-tenant service.

## Real usage

Built to give two personal projects — [victoria-gateway](https://github.com/GordonWei/victoria-gateway) (an AIOps alert summarizer) and a personal WhatsApp/Telegram assistant gateway — a way to use Muse Glimmer as their local LLM backend without either of them needing to know or care that the model isn't natively supported by the model runner they were originally built against.

## License

MIT — see [LICENSE](LICENSE).
