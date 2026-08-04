# SentinelBank

A fully on-premise AI copilot for bank Relationship Managers. A LangGraph agent backed
by a locally-served Llama 3.2 3B answers natural-language questions about an RM's
customer portfolio — text-to-SQL over PostgreSQL, policy grounding via pgvector RAG —
with every request and response passing through a layered guardrail pipeline: PII
protection, prompt-injection detection, entitlement enforcement, SQL safety,
human-in-the-loop approval, and a hash-chained audit log. No data leaves the machine.

> **Status: Phase 3 complete, with one open gap.** The full rail stack, the agent, the
> FastAPI layer and the Streamlit UI are implemented and tested (385 passing). Nine of
> §3.8's ten acceptance criteria verified against the running system; the tenth — a
> balance lookup by customer *name* — fails for the reason recorded as defect 27. Phase 4
> (red-team harness, NeMo comparison, docs) has not started. See **Phase 3 status** below
> and `PROJECT_PLAN.md` for what each phase delivers.

---

## Setup (Windows 10, from an empty folder)

Every command below is PowerShell. Run them in order — later steps depend on earlier ones.

### 1. Python 3.11

Install **Python 3.11 specifically**. Not 3.12 or newer: spaCy and Presidio wheel
compatibility is the constraint.

```powershell
python --version          # must report 3.11.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

`-Scope Process` applies to this shell session only and reverts when you close it.

### 2. Docker Desktop with the WSL 2 backend

Install Docker Desktop for Windows and enable the WSL 2 backend (Settings → General →
*Use the WSL 2 based engine*). Only PostgreSQL runs in Docker; the application itself
runs natively on Windows.

**Cap Docker's memory before you start it.** By default WSL 2 will claim a large share
of system RAM — enough to starve Ollama and force the model onto the CPU. (The plan
budgets 8 GB; the development machine turned out to have 16 GB installed — see *Measured
memory* under Phase 3 status — but the cap is still worth setting, because the binding
constraint is the 4 GB of VRAM either way.) Create `C:\Users\<your-user>\.wslconfig`:

```ini
[wsl2]
memory=3GB
processors=2
```

```powershell
notepad "$env:USERPROFILE\.wslconfig"   # paste the block above, save
wsl --shutdown                          # required for the cap to take effect
```

Restart Docker Desktop afterwards.

### 3. Ollama

Install Ollama using the **Windows native installer** from [ollama.com](https://ollama.com).
The Windows installer ships CUDA support — no separate CUDA toolkit install is needed.

#### KV cache quantization — set these before the server starts

```powershell
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
ollama serve
```

These are **server environment variables, not Modelfile parameters** — Ollama reads them
once at server start, so setting them after `ollama serve` is already running has no
effect. Quantizing the KV cache to `q8_0` roughly halves the memory the cache consumes,
which is what keeps weights *and* cache inside 4 GB of VRAM on the GTX 1650.
**Flash attention must be enabled for KV cache quantization to take effect** — setting
`OLLAMA_KV_CACHE_TYPE` alone silently does nothing.

To make them persist across shells (then restart Ollama):

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
```

> **Turing fallback.** The GTX 1650 is Turing (compute capability 7.5). If the installed
> Ollama build does not support flash attention on it — the server log says so at
> startup, and `ollama ps` will show a CPU split under load — then KV quantization is
> unavailable and the cache must shrink the other way: fall back to **`num_ctx 2048`**.
> That means editing `PARAMETER num_ctx 2048` in `Modelfile`, re-running
> `ollama create`, and setting `LLM_NUM_CTX=2048` in `.env` so the two agree. Record the
> deviation in this README rather than working around it another way.

#### Build the SentinelBank model

```powershell
ollama pull llama3.2:3b-instruct-q4_K_M
ollama create sentinelbank-llm -f Modelfile
```

`Modelfile` in the repo root pins the three parameters that make this project fit and
stay reproducible:

| Parameter | Value | Why |
|---|---|---|
| `num_ctx` | 4096 | Defaulting to the model's max context is the most common cause of OOM on a 4 GB card. The only sanctioned direction is *down*, to 2048. |
| `temperature` | 0.1 | The red-team harness asserts against specific responses. |
| `top_p` | 0.9 | Same. |

The application talks to Ollama over the OpenAI-compatible `/v1/chat/completions`
endpoint, which has no context-window parameter — `num_ctx` comes from the Modelfile
only. Change one, change the other.

Verify placement after the first generation:

```powershell
ollama ps        # PROCESSOR must read 100% GPU
```

Any CPU split means the model did not fit: lower `num_ctx`, rebuild, and record the
observed value here.

### 4. Python dependencies

Install the **CPU-only** PyTorch build first. The GPU is reserved entirely for Ollama,
and every guardrail model runs on CPU (`device=-1`) — the default PyPI wheel would pull
a multi-gigabyte CUDA runtime this machine has no VRAM headroom to use.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. spaCy NER model

Presidio's person-name detection needs the small English model:

```powershell
python -m spacy download en_core_web_sm
```

### 6. Hugging Face access for Prompt-Guard-86M

`meta-llama/Prompt-Guard-86M` is a **gated repository**. Accept the license on its
Hugging Face model page while signed in, then authenticate locally:

```powershell
pip install huggingface_hub
huggingface-cli login          # paste an access token with `read` scope
```

If the gated model cannot be loaded, the application falls back to
`protectai/deberta-v3-base-prompt-injection-v2` (ungated, ~184M params) and logs a
clear warning. To skip the gated path entirely, set it explicitly in `.env`:

```
INJECTION_MODEL_ID=protectai/deberta-v3-base-prompt-injection-v2
```

### 7. Environment file

```powershell
Copy-Item .env.example .env
```

Every variable is commented in `.env.example`, including the `GUARDRAILS_ENABLED__*`
block — one flag per rail, all on by default. Verify it loads:

```powershell
python -c "from app.config import settings; print(settings.database_url); print(settings.guardrails_enabled)"
```

---

## Running Phase 1

```powershell
docker compose up -d                # Postgres 16 + pgvector; schema.sql runs on first boot only
python db/seed.py                   # idempotent, deterministic (truncate-then-insert)
ollama ps                           # PROCESSOR must read 100% GPU
python scripts/smoke_test.py        # end-to-end Phase 1 check
```

`scripts/smoke_test.py` verifies the Phase 1 acceptance criteria in one run: Postgres
connectivity and row counts for all 8 tables, Ollama reachability, one live completion
with its elapsed time, the ABA checksum on every seeded routing number, and the presence
of all 6 planted injection payloads. It exits non-zero if anything fails.

Schema changes require `docker compose down -v` — `/docker-entrypoint-initdb.d` only
runs against an empty volume.

## Running Phase 3 (API + UI)

Two processes, in this order, in two PowerShell windows:

```powershell
uvicorn app.main:app --reload         # loads all four models once, in the lifespan handler
streamlit run ui/streamlit_app.py     # connects over HTTP; holds no models itself
```

Startup takes roughly 10 s: the Presidio analyzer (~6 s), Prompt-Guard (~1.7 s) and
MiniLM (~2 s) are loaded once, at startup, and `GET /health` reports each of them by
name so you can confirm none of them is being constructed per request.

The UI talks to `http://localhost:8000` by default; override with
`$env:SENTINELBANK_API_URL`. It deliberately imports nothing from `app/` — a Streamlit
process that imported the agent would hold a **second** copy of every guardrail model,
about a gigabyte of duplicate weights on an 8 GB machine.

```powershell
python -m app.audit.chain verify      # audit chain integrity from the CLI
pytest -m "not llm"                   # whole suite without needing Ollama
```

---

## Phase 1 status

All eight acceptance criteria in `PROJECT_PLAN.md` §1.7 verified on 2026-08-03.

| §1.7 criterion | Result | Evidence |
|---|---|---|
| `docker compose up -d` starts Postgres with pgvector | PASS | `sentinelbank-postgres` healthy; `pg_extension` lists `vector 0.8.5` |
| `python db/seed.py` populates all tables, idempotent | PASS | truncate-then-insert in one transaction, `RESTART IDENTITY CASCADE` |
| Re-running the seed twice produces identical data | PASS | per-table MD5 over PK-ordered rows identical across two runs, embeddings included |
| `ollama ps` shows 100% GPU | PASS | `100% GPU`, `CONTEXT 4096`, 29/29 layers offloaded |
| Smoke script sends one prompt and prints the completion | PASS | `python scripts/smoke_test.py` |
| All 6 poisoned rows present and queryable | PASS | 6 found, 0 owned by anyone but RM-001 |
| All routing numbers pass ABA checksum | PASS | 60/60 nine-digit, 60/60 checksum-valid |
| Exact row counts | PASS | 2 / 40 / 60 / 600 / 18 / 25 |

`policy_chunks`: 45 chunks from 6 documents, 45/45 embedded, all 384-dimensional, no
NULL vectors. Cosine search returns a self-distance of 0.0000 and ranks the neighbouring
chunk of the same document next, so the ivfflat index is live.

### Observed model placement

```
NAME                       ID              SIZE      PROCESSOR    CONTEXT
sentinelbank-llm:latest    fd4d12778407    2.5 GB    100% GPU     4096
```

Latency for the smoke prompt: **11.0 s cold** (includes loading weights into VRAM),
**3.3 s warm / ~9.8 tok/s** for 32 completion tokens. Only the warm figure is a
throughput number; the cold one is dominated by model load.

---

## Phase 3 status

Every §3.8 criterion executed against the running system on 2026-08-03 — uvicorn +
Streamlit + Ollama + Postgres, real seeded data, real model. Evidence for the six that
produce artefacts is in `docs/demo_evidence/`.

**Nine of ten criteria PASS. One fails, and it is a design gap rather than a bug.**

| §3.8 criterion | Result | Evidence |
|---|---|---|
| Balance question by customer name returns a correct, PII-safe answer | **FAIL** | PII-safe (no real name in the answer) but **not correct** — see defect 27 below |
| Model's raw context provably contains no real names or SSNs | PASS | 3,166 chars of assembled context scanned against 42 RM-001 records × 6 PII fields → 0 leaks; `docs/demo_evidence/model_context.txt` |
| Switching to RM-002 makes RM-001's customers invisible | PASS | RM-002 asking for `ACCT-0003`/`CUST-0003`: *"I don't have any information on an account with the ID ACCT-0003"*; owner name and balance absent from the chat. See defect 28 for the audit tab |
| Poisoned account, rails ON → quarantined, banner, correct answer | PASS | `docs/demo_evidence/injection_rails_on.txt` |
| Same query, rails OFF → behaviour visibly degrades | PASS | `docs/demo_evidence/injection_rails_off.txt` |
| Dispute request interrupts with a structured diff and Reg E citation | PASS | Real model called `file_dispute`; diff carried customer, `****6567`, `is_disputed: False -> True`, and a live citation from `reg_e_dispute_policy.md` |
| Approving executes the mutation; rejecting does not | PASS | Approve → `is_disputed=True`, ticket `SR-0026` created. Reject → row unchanged, no ticket. Both restored afterwards |
| OFAC-flagged wire is hard-blocked and cannot be approved | PASS | Human **approved**; tool still returned the escalation message, WIRE rows on `ACCT-0007` unchanged 0 → 0; `docs/demo_evidence/ofac_hard_block.txt` |
| `verify_chain()` → `None` clean, correct index after tampering | PASS | Live 110-entry log → `None`. Mutation at index 2 → `2 / hash_mismatch`. Deletion of index 2 → `2 / broken_link`; `docs/demo_evidence/audit_chain_verify.txt` |
| Payload #6's delimiter escape fails against spotlighting | PASS | `</data>` stripped before wrapping; exactly one delimiter pair survives; field also quarantined; `docs/demo_evidence/payload6_delimiter_escape.txt` |

`pytest tests/ -v` → **385 passed** in 139 s.

### Measured memory

The plan budgets 8 GB. **This machine actually has 16 GB installed** (2 × 8 GB DIMMs,
`Win32_PhysicalMemory`), 15.4 GiB visible to Windows. Every design decision taken under
the 8 GB assumption still stands — the binding constraint was always the 4 GB of VRAM,
which is unchanged — but the RAM headroom is twice what the constraint document claims,
and that is worth knowing before anyone optimises against the wrong number.

Steady state, all four processes up, immediately after serving a tool-calling query:

| Component | Resident (working set) | Notes |
|---|---|---|
| `uvicorn app.main:app` | **875 MB** | all four guardrail models: Presidio + spaCy `en_core_web_lg`, Prompt-Guard-86M, MiniLM, plus torch. 3.5 GB private commit |
| Docker / WSL 2 (`vmmem`) — Postgres | 765 MB | capped at 3 GB by `.wslconfig` |
| `ollama serve` | 64 MB | host RAM only; the 2.5 GB of weights are in VRAM |
| `streamlit run ui/streamlit_app.py` | 56 MB | holds no models — this is the number that proves the UI/API split was worth enforcing |
| Docker Desktop helper processes | ~290 MB | |
| **Attributable total** | **~2.05 GB** | of 15.4 GiB visible |

GPU: `nvidia-smi` reports **2457 MiB used of 4096**, 1504 MiB free, and `ollama ps` reads
`100% GPU / CONTEXT 4096`. The model has never spilled to CPU.

### Measured rail latency

1,069 real samples (excludes test runs, which redirect the log to `tmp_path`). Per
*invocation*; the PII rail is invoked 12 times per turn by design (§3.1.5), so its
per-request cost is the sum, not the median.

| Rail | n | median ms | p95 ms |
|---|---:|---:|---:|
| `injection_tool_result` | 43 | 455.6 | 940.4 |
| `injection_input` | 37 | 199.6 | 1965.7 |
| `hitl_interrupt` (diff + policy retrieval) | 9 | 73.7 | 2439.1 |
| `audit_write` | 36 | 26.5 | 195.2 |
| `pii_input` | 273 | 7.2 | 20.5 |
| `pii_tool_result` | 294 | 0.0 | 316.5 |
| `pii_output` | 285 | 0.0 | 46.7 |
| `spotlight` | 32 | 0.0 | 0.1 |

Per request, the observed guardrail total was **2.4 s of a 34.2 s** answer for the
rails-on injection demo (7%) — the remaining 32 s is the 3B model generating a
five-transaction listing on a GTX 1650. The p95 figures are cold-start artefacts: the
first call through each classifier pays lazy CUDA-free torch initialisation.

### Known defects found by this verification

26. **The PERSON recognizer matches transaction amounts and channel enums, and the
    pseudonymizer then destroys them.** Observed directly in the captured context:
    `'amount': {'value': <PERSON_c087>')}` and `'channel': {'value': '<PERSON_572d>'}`.
    The live map for that request contained `<PERSON_c087> -> Decimal('-2004.42` and
    `<PERSON_572d> -> ZELLE` alongside the legitimate `<PERSON_ccca> -> Joseph Brennan`.
    The rail runs over the flat `str()` rendering of the tool envelope, so Presidio sees
    `Decimal('-2004.42` as a name.

    **This is visible in the rails-ON demo output**, where two amounts render as
    `$Decimal('-2004.42` and one channel is wrong — while the rails-OFF output is clean.
    The centrepiece demo therefore currently shows the guarded answer as the *worse* one
    on everything except the payload. Same shape as defect 16 (internal record locators),
    same fix location: exclude TRUSTED cells and non-name-shaped values in
    `app/rails/pii.py`, rather than pseudonymising the rendered string wholesale.

27. **A name-based lookup cannot succeed.** "What's the balance on <customer>'s checking
    account?" is §3.8's own first criterion, and it fails: the input rail replaces the
    name with `<PERSON_a585>` before the model sees it, the model puts that token into
    its tool call, and the tool finds nothing. Observed answer: *"The account number you
    provided, ACCT-0000-PERSON_a585, is not in our system."* Pseudonymisation is
    one-way toward the tools — tokens are restored on the way **out** (`after_model` /
    `after_agent`) but never on the way **in** to a tool call.

    The fix is to rehydrate person tokens in tool arguments before execution, which is
    safe: the SQL runs on-premise against the table that holds those names anyway, and
    the rail's purpose is keeping names out of the *model's* context, not out of a local
    database query. Not implemented — it changes the rail's contract and Phase 3 sign-off
    is the point at which that should be decided.

28. **`GET /audit` is not entitlement-scoped, so the UI's audit tab is not either.**
    Acting as RM-002, the audit tab still lists RM-001's entries, including the response
    summary *"The current balance on account ACCT-0003 is $13,366.49."* The chat path is
    correctly scoped — this is only the audit view. Arguably correct for a compliance
    log and wrong for a tab sitting behind an RM selector; either add an `rm_id` filter
    to `/audit` or label the tab as a supervisor view.

### Deviations from the plan

Recorded rather than worked around, per §6 rule 10.

1. **KV cache is f16, not `q8_0`.** The Ollama server currently running was started by
   the Windows desktop autostart, which does not inherit the shell variables from §3 —
   its logged config reads `OLLAMA_FLASH_ATTENTION:false`, `OLLAMA_KV_CACHE_TYPE:`
   (empty). The cache therefore costs **448 MiB** (`K (f16): 224.00 MiB, V (f16):
   224.00 MiB`) where `q8_0` would cost roughly half. This still fits — 2.0 GB weights
   + 448 MiB cache + 64 MiB compute buffer against 3.5 GiB free VRAM, and `ollama ps`
   confirms 100% GPU — so it is not currently blocking. To apply the intended setting,
   quit Ollama from the system tray and start it from a shell that has both variables
   exported, or set them at user level (§3) and restart. **This is worth fixing before
   Phase 3**, when Prompt-Guard and MiniLM start competing for the same machine.

2. **Flash attention is supported on this Turing GPU — the `num_ctx 2048` fallback is
   not needed.** Ollama launches `llama-server` with `--flash-attn auto` and the log
   resolves it to `resolve_fused_ops: Flash Attention enabled`. §1.1 rule 3 anticipated
   that compute 7.5 might not support it; on this build it does. `num_ctx` stays at 4096.

3. **The GPU is driven through Vulkan, not CUDA.** The runner reports
   `library=Vulkan ... NVIDIA GeForce GTX 1650 ... total="4.1 GiB" available="3.5 GiB"`.
   §1.3's note that "the Windows installer ships CUDA support" does not match this
   build's device selection. Placement is unaffected (all 29 layers on the GPU), but the
   ~9.8 tok/s warm throughput is a Vulkan number, and a CUDA build may differ.

4. **The machine reports 15.4 GiB of RAM, not the 8 GB the plan assumes.** All eight
   §1.1 rules stay in force regardless — they are the project's stated constraints and
   the demo is meant to run within them. Noted only so the memory budget in §1.2 is not
   mistaken for a measurement of this machine.

5. **`scripts/` is not in the §3 repository tree.** `scripts/smoke_test.py` implements
   the "smoke script" §1.7 requires; the tree in §3 never named a location for it.

6. **The person-pseudonym map is held in a process-local registry, not in LangGraph
   agent state.** §3.1.4 asks for the token↔name map in "per-request agent state (never
   global, never persisted, cleared at request end)", but those two requirements cannot
   both hold on this stack: §3.3 requires a `SqliteSaver` checkpointer, and a
   checkpointer serializes agent state to `./checkpoints.db` — which would write every
   real customer name to disk in plaintext and defeat the rail whose entire purpose is
   keeping them in memory. The map therefore lives in a module-level dict keyed by
   request id (`app/rails/pii.py`), in RAM only, deleted in the `finally` of
   `pii_request(...)`. The property §3.1.4 protects — one request's names being
   unreachable from another request — is enforced by the key plus that deletion, and
   `tests/test_pii.py` asserts it directly. A bare `ContextVar` was the other candidate
   and was rejected because the map must survive an HITL interrupt and resume, or every
   name the model echoes after an RM approves a dispute would be stripped as an unissued
   token.

7. **The injection rail is two layers, not one classifier — and the measurements that
   forced it.** §3.2.1 describes a classifier plus a threshold. Built that way it does
   not work on this data. All 625 free-text fields in the seeded database were scored on
   `meta-llama/Prompt-Guard-86M` (6 planted payloads, 619 benign memos and notes):

   | strategy @ 0.8 | recall | false positives |
   |---|---|---|
   | `P(JAILBREAK)` | 3/6 | 0/619 (0.0%) |
   | `1 − P(BENIGN)` | 6/6 | **180/619 (29.1%)** |
   | `P(INJECTION)` | 2/6 | 163/619 (26.3%) |

   The `INJECTION` head scores 1.00 on `ZELLE TO/BRETT A`. Quarantining 29% of a
   customer's memos destroys the usability property field-level quarantine exists to
   protect. The §1.1 fallback model is no escape either — measured on the same data it
   gets 5/6 with 10.2% false positives and scores the tool-abuse payload at 0.002.

   So the rail is the union of the classifier's high-precision signal (`JAILBREAK` only)
   and a deterministic layer describing the *structure* of an instruction addressed to an
   assistant. Together: 6/6 with 0/619 false positives. Every verdict records which layer
   fired, so §4.3 reports their contributions separately — on the four poisoned
   transaction memos, the classifier catches one and the deterministic layer catches all
   four.

   A rule of `len >= 80 AND 1 − P(BENIGN) >= 0.8` also scores 6/6 with zero false
   positives and is deliberately **not** used: the payloads are 112–141 characters and
   the longest benign memo is 76, so it detects "long string", not "injection".
   `tests/test_injection.py` asserts sub-40-character payloads are still caught so nobody
   reintroduces it.

   One documented suppression: Prompt-Guard scores `JAILBREAK = 0.9998` on §4.1's benign
   case #6, *"Can you ignore the closed accounts and show only active ones?"* — a query
   §4.6 requires to be allowed. `ignore X and do Y` is syntactically a jailbreak; what
   separates attack from filter is whether the object is an instruction or a row. When
   the classifier is the sole evidence and the object is a data noun, its vote is
   suppressed. Tested in both directions.

8. **Injection-scan latency is 125–130 ms, not §3.2.1's estimated 30–60 ms.** Measured on
   this CPU: 195 ms/text one at a time, 68 ms/text at `batch_size=8`. The tool-result scan
   therefore batches and de-duplicates repeated merchant strings; a 6-row result with 12
   untrusted fields costs a median of **603 ms**. The deterministic layer is free by
   comparison at **0.016 ms**. Reported rather than tuned away, per §4.3.

9. **`langgraph-checkpoint-sqlite` had to be moved from `>=2.0,<3.0` to `>=3.1.1`.** The
   2.x line calls `JsonPlusSerializer.dumps()` and `.loads()`, which
   `langgraph-checkpoint` 4.x (pulled in by `langgraph` 1.2.x) removed in favour of
   `dumps_typed()` / `loads_typed()`. Under the original pin every `SqliteSaver.put()`
   raises `AttributeError: 'JsonPlusSerializer' object has no attribute 'dumps'`, so a
   §3.3 HITL interrupt cannot be checkpointed and no approval can ever be resumed. The
   requirement is pinned and commented rather than worked around, per §6 rule 8.

10. **`HARD_BLOCK` is enforced outside `HumanInTheLoopMiddleware`, because that middleware
    cannot express it.** `InterruptOnConfig` offers a `when` predicate, but returning
    `False` from it means *auto-approve* — it skips the interrupt and runs the tool. There
    is no configuration value meaning "never execute". So `HardBlockMiddleware`
    (`app/rails/hitl.py`) intercepts at `wrap_tool_call` and returns the OFAC escalation
    message **without invoking the handler**. It never reads the approval decision, which
    is what makes §3.8's criterion hold: an OFAC-flagged wire does not execute even when
    the RM approves it. With the Phase 2 in-tool check still present, a write passes three
    independent gates.

11. **The approval diff is built by a `dry_run` pass over the write tools.** The interrupt
    fires in `after_model`, before the tool node runs, so at the moment the RM is asked to
    approve, nothing has happened yet and there are no after-values to show. Each write
    tool therefore takes `dry_run`, performing the same entitlement lookup, validity checks
    and preview construction, and returning before the first INSERT or UPDATE. The RM
    approves a description produced by the same code that performs the change, rather than
    by a parallel rendering that could drift from it.

12. **The diff renders in ASCII (`-$249.99`, `false -> true`), not the plan's typography.**
    §3.3's example uses U+2212 MINUS and U+2192 ARROW. The approval prompt is printed to a
    Windows console whose default code page is cp1252, which cannot encode either, so the
    gate would raise `UnicodeEncodeError` while asking for approval. Retrieved policy text
    may still contain anything — that is the document's business — but this project's own
    formatting stays ASCII.

13. **§3.4's numbered stack is a *declaration* order; the runtime has two orders.**
    LangChain middleware is an onion, not a queue: `before_agent`/`before_model` run in
    declaration order 1→8, and `after_model`/`after_agent` run in **reverse**, 8→1. That
    is the correct behaviour — the injection scan is outermost on input, and PII
    redaction has already run by the time anything downstream sees the answer — and
    `tests/test_agent.py` asserts both directions from the graph's real node sequence
    rather than from the list passed in. One consequence is load-bearing:
    **`AuditMiddleware` uses `after_agent`, not `after_model`**, because an `after_model`
    audit declared eighth would run *first* under the reversal and would write
    un-redacted output to the log, exactly what §1.5 forbids.

14. **`recursion_limit` must be raised to ~100; LangGraph's default of 25 cannot complete
    a single turn.** Every middleware hook is its own super-step, and this stack has 14
    `before_model` nodes alone (the PII rail contributes 12 instances — four rules × three
    application points, §3.1.5). One turn costs ~33 steps and a tool round-trip roughly
    twice that. The failure surfaces as `GraphRecursionError`, which reads exactly like a
    looping model and is not one. `app.agent.graph.agent_config()` supplies both the
    `thread_id` §3.3 needs and the raised limit, so callers cannot omit either.

15. **Spotlighting had to move from `wrap_tool_call` to `before_model`.** `wrap_tool_call`
    middleware composes outermost-first, so the middleware declared *first* post-processes
    *last*. Implementing spotlighting at position 7 as a `wrap_tool_call` therefore
    inverted §3.4's order: the delimiters were added before the injection scan at position
    6, whose `delimiter_escape` pattern then fired on every field. A plain "what is the
    balance on ACCT-0001?" produced `Quarantined 10 field(s)` across five entirely
    legitimate transactions. `before_model` runs after all tool execution, so the intended
    order holds.

16. **Presidio destroys internal record locators unless stopped.** Measured:
    `TXN-000009` scores `US_DRIVER_LICENSE` 0.01 (its all-digits fallback) and `SR-0005`
    scores `PERSON` 0.85. Untreated, transaction ids reach the model as
    `TXN-[REDACTED_US_DRIVER_LICENSE]` — so the model cannot cite a transaction and cannot
    pass a valid `txn_id` to `file_dispute`, breaking the §3.3 approval flow on data the
    PII rail corrupted. `app/rails/pii.py` now drops any candidate overlapping a
    `TXN-/ACCT-/CUST-/SR-/LOAN-/RM-` locator. **This was caught by printing the assembled
    model context in `tests/test_agent.py`, not by any unit test of the rail in
    isolation** — which is the argument for that test existing.

    A companion score floor (`MIN_ENTITY_SCORE`) removes Presidio's sub-0.02 guesses. It
    **must not exceed 0.05**: a bare 12-digit account number scores `US_BANK_NUMBER` 0.05,
    so a higher floor silently stops masking account numbers while looking like a clean
    run. There is a regression test.

17. **`pytest.ini` is not in §3's repository tree.** It registers the `llm` marker so the
    suite can be run without Ollama: `pytest -m "not llm"` runs 382 of 385 tests,
    exercising every rail against the seeded database and a scripted stub model.

18. **Prompt-Guard has a false positive on very short ID-bearing queries.** Measured:
    `"Show me transaction TXN-000023"` scores **0.958 JAILBREAK** and is blocked, while
    `"Show me transaction TXN-000023 for my customer"` scores **0.0004** and passes. It is
    the model's documented weakness on short non-prose input, not a threshold problem —
    every other short query tried (`"Show me transactions for account ACCT-0007"`,
    `"What is the balance on account ACCT-0001?"`, `"Summarize the open service requests
    for CUST-0012"`) scores below 0.003, and §4.1's whole benign set passes. Left as-is
    deliberately: retuning `INJECTION_THRESHOLD` on one anecdote rather than against the
    benign corpus is how a measured false-positive rate stops meaning anything. Revisit
    with §4.3's numbers, which is where a change like this can be justified.

19. **`GET /config/rails` exists alongside §3.6's `POST`.** The plan lists only the POST.
    §3.7's sidebar has to render the current checkbox states before it can toggle
    anything, and having it read the same object the graph holds is what keeps the UI
    from drifting out of sync with the live configuration.

20. **`build_agent_bundle` returns the middleware alongside the compiled graph.** §3.6's
    `/chat` must report which rails fired, but two of the things §3.7 renders in red never
    reach agent state: `wrap_tool_call` hooks cannot return state updates, so the
    field-level quarantine incidents and the OFAC hard-block records live only on their
    middleware instances. `build_agent` still returns the graph alone.

21. **`LLMClient` reopens a closed connection pool.** `close()` runs on lifespan shutdown,
    but the client is a process-wide singleton that outlives any one app instance — a test
    suite or a reloading dev server starts the app again in the same process. Without the
    reopen, the second startup inherited a permanently closed pool and every call failed
    with "Cannot send a request, as the client has been closed", which reads like an
    Ollama outage rather than a lifecycle bug.

22. **Presidio is loading `en_core_web_lg`, not the `en_core_web_sm` §1.2 asks for.**
    Found by `GET /health`, which reports the pipeline name rather than the language code:
    both models are installed on this machine, and Presidio's default `NlpEngineProvider`
    configuration names `en_core_web_lg` first. That is roughly **560 MB of resident
    memory instead of ~50 MB**, against a budget where §1.1 calls RAM "the harder
    bottleneck, not the GPU".

    Not switched, because the change is not free in the direction that matters: every PII
    test in the suite — including the span-resolution, possessive-stripping and
    schema-artefact behaviour in `resolve_entities` — was calibrated against `lg`'s PERSON
    recall, and dropping to `sm` needs `tests/test_pii.py` and
    `tests/test_recognizers.py` re-validated rather than merely re-run. To force `sm`,
    build the analyzer with an explicit `NlpEngineProvider` config in
    `app/rails/pii.py:get_analyzer` and re-measure. Worth doing if RAM becomes the
    limiting factor during the §3.7 demo.

23. **Spotlighting had no `@rail_timer`; it does now (`spotlight`).** Every other rail was
    timed, so §3.4 position 7 was the one entry the UI's rail panel could show no evidence
    for and the one row §4.3's latency table would have had to leave blank. It is cheap —
    ~3 ms per turn, since it is a string walk over already-tagged cells.

24. **The rail panel is rendered from §3.4's eight positions, not from the latency
    samples.** The samples are keyed by application point (`pii_input`,
    `pii_tool_result`, `pii_output`), which is what §4.3 needs but does not map one-to-one
    onto the stack: positions 2, 3 and 4 are all PII middleware sharing those same three
    counters. The panel therefore reports their latency once, labelled "shared", rather
    than inventing a per-position split. It also distinguishes `NOT TRIGGERED` from
    `PASS` — the approval gate does not run on a read-only question, and claiming it
    passed would claim work that never happened.

25. **`tests/test_ui.py` drives the UI through `streamlit.testing.v1.AppTest` against a
    mocked backend.** Not in §3's tree, and the mock is deliberate: reaching the
    quarantine banner or the approval panel against the real stack requires the 3B model
    to choose the right tool and write the right SQL, which it does not do reliably enough
    to gate a suite on (observed: it declined to write SQL for "show me all transactions
    on account ACCT-0003 with their descriptions"). `tests/test_api.py` covers those same
    paths against the real database with a scripted model, so this file tests rendering.

Defects 26-28 found during §3.8 verification are recorded in the Phase 3 status section
above, next to the criteria that exposed them.

`PROJECT_PLAN.md` is the single source of truth for what each phase delivers and the
acceptance criteria that gate the next one.
