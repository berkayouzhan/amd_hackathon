# OptiRoute AI

**AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent**

A token-efficient, category-aware routing agent that intelligently processes batches of diverse AI tasks while minimizing token consumption.

## How It Works

OptiRoute AI routes each task through the most cost-effective path:

```
Task prompt
   │
   ▼
compress_prompt()              — removes redundant whitespace (preserves code blocks)
   │
   ▼
Tier 0: deterministic_solver   — safe arithmetic/unit conversion (0 tokens)
   │  no match? ↓
   ▼
triage.classify()              — classifies into 1 of 8 categories
   │  regex heuristic first (0 tokens), model fallback if uncertain
   ▼
router: model call
   │  "default" categories → try Gemma first (short timeout) → fallback to minimax-m3
   │  "reasoning" categories → minimax-m3
   │  "code" categories → kimi-k2p7-code
   ▼
validator.validate()           — free deterministic checks (0 tokens)
   │  suspicious? ↓
   ▼
Single corrective retry (same model)
   │
   ▼
Answer → /output/results.json
```

## Key Features

- **Zero-Token Solving:** Simple arithmetic is solved deterministically without any API calls
- **Smart Triage:** Two-layer classifier (regex heuristics + lightweight model fallback) categorizes tasks at near-zero cost
- **Optimal Model Selection:** Each category is routed to the best-fit model (MiniMax M3, Kimi K2P7, Gemma 4)
- **Speculative Validation:** Free deterministic checks catch empty/truncated/invalid responses, triggering corrective retries
- **Gemma Circuit-Breaker:** After the first Gemma failure, subsequent tasks skip the timeout — saving runtime in batch processing
- **Robust Batch Processing:** Isolated error handling per task, atomic file writes, deadline-aware processing

## 8 Supported Task Categories

| Category | Model Role | Routing |
|----------|-----------|---------|
| Factual Knowledge | default | Gemma → MiniMax M3 fallback |
| Mathematical Reasoning | reasoning | MiniMax M3 |
| Sentiment Classification | default | Gemma → MiniMax M3 fallback |
| Text Summarization | default | Gemma → MiniMax M3 fallback |
| Named Entity Recognition | default | Gemma → MiniMax M3 fallback |
| Code Debugging | code | Kimi K2P7 Code |
| Logical Reasoning | reasoning | MiniMax M3 |
| Code Generation | code | Kimi K2P7 Code |

## Project Structure

| File | Purpose |
|------|---------|
| `config.py` | Environment variables, model catalog, role assignment |
| `fireworks_client.py` | Fireworks API wrapper — `ALLOWED_MODELS` guard, token tracking, retry |
| `deterministic_solver.py` | Tier 0 — zero-token deterministic solver |
| `triage.py` | 8-category classification (heuristic + model fallback) |
| `prompt_compressor.py` | Whitespace compression (preserves code blocks) |
| `validator.py` | Free, deterministic response validation |
| `router.py` | `OptiRouter` — the main orchestrator |
| `main.py` | Harness entry point — `/input/tasks.json` → `/output/results.json` |
| `fake_fireworks_client.py` | Offline fake client for local testing (not used in scoring) |
| `local_test/tasks.json` | Sample task set covering all 8 categories |
| `tests/` | pytest suite (104 tests, no real API calls) |

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env
# Fill .env with your Fireworks API key:
#   FIREWORKS_API_KEY=fw_...
```

## Running Tests

```bash
pytest tests/ -v
```

All 104 tests run with mocked `FireworksClient` — **none connect to the real Fireworks API**, so they work without an API key and complete in seconds.

## Offline End-to-End Test (no API key needed)

```bash
export TASKS_INPUT_PATH=./local_test/tasks.json
export RESULTS_OUTPUT_PATH=./local_test/results.json
export USE_FAKE_FIREWORKS=1
python main.py
cat local_test/results.json
```

This mode uses `fake_fireworks_client.py` to run the entire pipeline without any real API calls — useful for verifying the pipeline doesn't crash and routes correctly.

## Real API End-to-End Test

```bash
export TASKS_INPUT_PATH=./local_test/tasks.json
export RESULTS_OUTPUT_PATH=./local_test/results.json
python main.py
cat local_test/results.json
```

## Docker Build & Submission

```bash
# Standard build (Intel/AMD machines):
docker build --tag <your-image>:latest .

# Apple Silicon — MUST specify platform:
docker buildx build --platform linux/amd64 --tag <your-image>:latest --push .
```

### Pre-submission Checklist

- [x] Image uses `linux/amd64` manifest
- [x] Image size ≤ 10GB
- [x] `.env` file is NOT copied into the image (`.dockerignore` prevents this)
- [x] 104 automated tests passing
- [x] Gemma circuit-breaker for runtime optimization

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Gemma "try first, fallback silently" | Gemma bonus prize opportunity, but not serverless on Fireworks |
| Circuit-breaker after first Gemma failure | Prevents 6s×N timeout waste across batch |
| Corrective retry on same model | No clear model ladder in Track 1; different model adds complexity |
| `temperature=0.0` default | Consistency > creativity for a deterministic router |
| Tier 0 limited to bare arithmetic | Wrong guess on word problems risks accuracy gate elimination |
| Atomic writes (`tmp` + `os.replace`) | Half-written `results.json` = zero score |
| Running as root in Docker | Mount permission mismatch risk > security for short-lived contest container |

## Technologies

- **Language:** Python 3.11
- **AI Platform:** Fireworks AI
- **Models:** MiniMax M3, Kimi K2P7 Code, Google Gemma 4
- **Containerization:** Docker
- **Testing:** pytest (104 tests)

## License

MIT
