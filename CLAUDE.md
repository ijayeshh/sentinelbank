# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

**SentinelBank** is a fully on-premise AI copilot for Relationship Managers at a US retail bank. A LangGraph agent backed by a locally-served Llama 3.2 3B answers natural-language questions about an RM's customer portfolio: it does text-to-SQL over an on-premise PostgreSQL + pgvector database (customers, accounts, transactions, loans, service requests), retrieves grounding context from internal policy documents via RAG, and can *draft* — never auto-execute — sensitive servicing actions. No data ever leaves the machine, because US bank customer NPI cannot legally go to a third-party API. Every request and response passes through a layered guardrail pipeline: PII protection, prompt-injection detection, entitlement enforcement, SQL safety, human-in-the-loop approval, and a hash-chained audit log. The project exists to demonstrate **indirect prompt injection**: poisoned rows planted in `transactions.description` and `service_requests.notes` enter the model's context through an innocent RM query — rails on, the payload is quarantined, audited, and flagged while the RM still gets their answer; rails off, behaviour visibly degrades. That toggle must stay reproducible.

## Source of truth

`PROJECT_PLAN.md` is the single source of truth. **Re-read it in full at the start of each phase** — not just the phase section, the whole file. Do not add features it does not list.

## Phase discipline

Work strictly phase by phase (1 → 2 → 3 → 4). **Never begin a phase until every acceptance-criteria box of the previous phase is ticked.** Each phase ends with runnable code and passing tests — no half-wired scaffolding carried forward. Section 5 (numeric groundedness stretch goal) does not start before Phase 4 is signed off.

## Hardware constraints

The developer's machine: **Windows 10**, **NVIDIA GTX 1650 with 4 GB VRAM** (~3.4 GB usable after the desktop compositor), **8 GB RAM** — RAM is the harder bottleneck, not the GPU. Violating the rules below causes OOM, not merely slowness.

### The 8 non-negotiable rules

1. **The model must fit entirely in VRAM.** Partial CPU offload drops throughput 3–5×. Use `llama3.2:3b-instruct-q4_K_M` (~2.0 GB weights).
2. **Context window is capped at 4096 tokens.** Defaulting to the model's max context is the single most common cause of OOM on 4 GB cards. Set `num_ctx` explicitly everywhere.
3. **Quantize the KV cache** (`q8_0` for K and V). In Ollama this is *not* a Modelfile parameter — set it via environment variables before starting the server:
   ```powershell
   $env:OLLAMA_FLASH_ATTENTION="1"
   $env:OLLAMA_KV_CACHE_TYPE="q8_0"
   ```
   Flash attention must be enabled for KV quantization to take effect. Document both in `README.md`. If the GTX 1650 (Turing, compute 7.5) does not support flash attention in the installed build, fall back to `num_ctx 2048` and record the deviation.
4. **Load all transformer models once at application startup** (FastAPI `lifespan`), never per-request. A per-request `pipeline(...)` will exhaust RAM. Same for the MiniLM embedder — module-level singleton.
5. **Guardrail models run on CPU** (`device=-1`). Only the main LLM touches the GPU. Install the CPU-only torch build.
6. **Set `torch.set_num_threads(4)`** to prevent thread thrashing against PostgreSQL.
7. **Never run the NeMo comparison stack and the main stack simultaneously.** Separate Docker Compose profile (`docker-compose.nemo.yml`).
8. **Use PowerShell-compatible commands in all documentation.** No `make`, no bash-only syntax in setup instructions.

Docker is memory-capped via `%USERPROFILE%\.wslconfig` (`memory=3GB`, `processors=2`) so it cannot starve Ollama. Python **3.11** specifically (spaCy/Presidio wheels). If RAM gets tight, apply `torch.quantization.quantize_dynamic` (INT8) to the injection classifier.

## Explicitly out of scope — hard prohibitions

Never implement, and never suggest, any of the following:

- **vLLM** (documented as a future swap only)
- **ONNX / ONNX Runtime**
- **LoRA or QLoRA fine-tuning**
- **Observability platforms** (Langfuse, Arize Phoenix, OpenTelemetry collectors)
- **Semantic caching / Redis**
- **Model routing between multiple LLMs**
- **Streaming-aware PII redaction**
- **Llama Guard content-safety rail**
- **Authentication / user management** beyond a hardcoded RM selector
- **Any cloud service of any kind**

## Implementation guidance (PROJECT_PLAN.md §6)

1. **Work strictly phase by phase.** Do not begin Phase 2 until every Phase 1 acceptance box is ticked.
2. **Every phase ends with runnable code and passing tests.** No half-wired scaffolding carried forward.
3. **Determinism is a hard requirement.** `Faker.seed(42)`, `random.seed(42)`, `temperature=0.1`.
4. **Fail closed everywhere.** If a rail cannot make a decision, block rather than allow.
5. **Never let the LLM control security decisions.** Entitlement `rm_id`, tool risk tiers, and SQL validation are all application-layer concerns.
6. **Comment the security-relevant code heavily.** A reviewer reading `sql_guard.py` or `recognizers.py` should understand the reasoning without the README.
7. **Prefer boring, readable code over clever code.** This is a portfolio project; legibility is a feature.
8. **If a dependency fails to install on Windows**, document the workaround in `README.md` rather than silently swapping libraries.
9. **Keep the system prompt short.** Every token costs context in a 4K window.
10. **When the plan conflicts with reality on the machine**, document the deviation in the README rather than quietly diverging.

## Commands

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1   # Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass if blocked
pip install -r requirements.txt
python -m spacy download en_core_web_sm
huggingface-cli login                                 # Prompt-Guard-86M is a gated repo

docker compose up -d                                  # Postgres 16 + pgvector; schema.sql runs on first boot only
python db/seed.py                                     # idempotent: truncate-then-insert

ollama pull llama3.2:3b-instruct-q4_K_M
ollama create sentinelbank-llm -f Modelfile
ollama ps                                             # PROCESSOR must read 100% GPU

uvicorn app.main:app --reload
streamlit run ui/streamlit_app.py

pytest redteam/                                       # all 30 red-team cases
python redteam/report.py                              # writes redteam/RESULTS.md
python -m app.audit.chain verify                      # audit chain integrity CLI
```

Schema changes require `docker compose down -v` — `/docker-entrypoint-initdb.d` only runs on an empty volume.
