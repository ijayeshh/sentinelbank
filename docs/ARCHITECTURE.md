# Architecture

PROJECT_PLAN.md §4.5. Why the system is shaped the way it is, and the measurements behind
the decisions that were not obvious.

---

## Request lifecycle

```mermaid
flowchart TB
    RM([RM types a question]) --> UI[Streamlit UI<br/><i>holds no models</i>]
    UI -->|POST /chat| API[FastAPI<br/><i>models loaded once at lifespan</i>]

    subgraph AGENT["LangGraph agent - middleware stack"]
        direction TB
        B1["1 · InjectionMiddleware<br/><small>before_agent · can jump to end</small>"]
        BT["§4.4 · TopicalMiddleware<br/><small>before_agent · can jump to end</small>"]
        B2["2·3 · PII redaction<br/><small>LangChain built-ins + Presidio</small>"]
        B4["4 · PersonPseudonymMiddleware<br/><small>reversible tokens</small>"]
        B5["5 · HITL gate + HardBlockMiddleware"]
        B6["6 · ToolResultInjectionMiddleware<br/><small>wrap_tool_call · field quarantine</small>"]
        B7["7 · SpotlightMiddleware<br/><small>inert - see defect 29</small>"]
        B8["8 · AuditMiddleware<br/><small>after_agent</small>"]
        B1 --> BT --> B2 --> B4 --> B5 --> B6 --> B7 --> B8
    end

    API --> AGENT
    B5 -.->|gated tool| TOOLS
    B8 --> RESP([answer + rails + warnings])

    subgraph TOOLS["Tools"]
        direction TB
        T1[query_customer_data]
        T2[get_account_summary]
        T3[search_policy_docs]
        W[write tools x5]
    end

    subgraph SQLPATH["sql_pipeline.safe_execute - fixed order, no way around it"]
        direction LR
        S1[validate_sql] --> S2[enforce_entitlement<br/><small>inject rm_id predicate</small>] --> S3[validate_sql<br/><small>again</small>] --> S4[(Postgres<br/>+ pgvector)]
    end

    T1 --> SQLPATH
    T2 --> SQLPATH
    T3 --> EMB[MiniLM embedder] --> S4
    AGENT --> LLM[["Ollama · llama3.2:3b<br/>num_ctx 4096 · GPU"]]
    B8 --> CHAIN[(audit_log<br/>SHA-256 chained)]

    classDef rail fill:#fde8e8,stroke:#c53030,color:#742a2a
    classDef det fill:#e6f4ea,stroke:#2f855a,color:#22543d
    class B1,BT,B2,B4,B6,B7 rail
    class B5,S1,S2,S3,B8 det
```

Red is *recognition* — rails that must identify something. Green is *deterministic* — rails
that hold regardless of what anything recognises. `docs/LIMITATIONS.md` measures the
difference: the green ones did not fail in testing, the red ones each generalise about as
far as the examples they were built from.

Two orderings matter and they are not the same. `before_*` hooks run 1→8; `after_*` hooks
run **8→1**, because middleware is an onion, not a queue. That reversal is why
`AuditMiddleware` uses `after_agent` — an `after_model` audit declared eighth would run
*first* on the way out and would write un-redacted output to the log.

---

## Why hybrid PII, and where the boundary sits

LangChain's `PIIMiddleware` ships exactly five detectors: `email`, `credit_card`
(Luhn-validated), `ip`, `mac_address`, `url`. Against a US retail bank's §3.1.2 entity
list, **two of those five are relevant and the other three never appear in this schema.**
Nothing built in covers SSN, account numbers, ABA routing numbers, phone, date of birth,
driver's licence, or address.

So the split is not "use the framework where convenient". It is a coverage boundary, and
naming it precisely *is* the design decision:

| Layer | Covers | Why it owns those |
|---|---|---|
| LangChain built-ins | `email`, `credit_card` | Luhn validation is already correct and already tested. Reimplementing it would be strictly worse |
| Presidio, via the `detector` seam | SSN, US_BANK_NUMBER, PHONE_NUMBER, DATE_TIME, LOCATION, US_DRIVER_LICENSE, PERSON | Everything the built-ins lack, reached through `PIIMiddleware`'s own extension point so both halves share one middleware contract |
| Custom recognizer (`app/rails/recognizers.py`) | ABA_ROUTING | Presidio has no routing-number recognizer, and the checksum makes it the best-evidenced claim on a 9-digit span |

The seam matters more than the split. Presidio is plugged into `PIIMiddleware`'s `detector`
parameter rather than bolted on beside it, so both groups produce the same `PIIMatch` shape,
obey the same strategy (`redact` vs `mask`), and are timed under the same `@rail_timer`
names. There is one PII contract with two sources of matches behind it.

**One authority per span.** All three sources resolve through `resolve_entities`, which
de-overlaps greedily in a fixed priority order. That ordering is a security decision, not a
tuning parameter: `ABA_ROUTING` outranks `PHONE_NUMBER` and `US_BANK_NUMBER` because it is
the only entity validated by a checksum, and `US_BANK_NUMBER` outranks `DATE_TIME` because
otherwise a 12-digit account number is redacted as a date instead of masked to last-4.

What this design does *not* buy is recall. Defects 35 and 36: Presidio's LOCATION
recognizer returned 20 of 22 street addresses verbatim under a bulk request, and
component-wise date redaction left enough of each DOB for the model to reassemble two of
them. The boundary is drawn in the right place; the component behind it is weaker than it
looks.

---

## Why SQL and entitlement are deterministic, not model-based

The obvious alternative is to ask the model: put the schema and the rules in the prompt,
let it write scoped SQL, maybe have a second model check the first. Rejected on three
grounds, in order of how much they matter.

**1. A model cannot be an authority about its own inputs.** The premise of this project is
that untrusted text reaches the model's context through retrieved rows. A rail implemented
*inside* the thing being attacked is downstream of the attack. `sql_guard` parses an AST;
the AST does not care what a memo field said.

**2. The failure modes are qualitatively different.** A deterministic rail fails *loudly
and reproducibly* — `non_select_statement:Drop`, `audit_log_access`,
`unlinked_scoped_table:a`. A model-based rail fails *plausibly*: it returns something that
looks like a judgement and you find out at audit time. Every rejection code in
`sql_guard.py` and `entitlement.py` is a stable string the red-team harness asserts on;
that is only possible because the decision is mechanical.

**3. The measured cost is negligible.** `sql_guard` runs at a 1.4 ms median, `entitlement`
at 0.5 ms. A model-based check would cost seconds of the 4 GB card's time and be less
trustworthy. There is no trade here to agonise over.

Allowlist, never blocklist, for both statement shape and tables. A blocklist is a bet that
you enumerated every dangerous thing; Postgres has more statement types than anyone
remembers, and sqlglot grows new ones between releases. `exp.Command` — sqlglot's fallback
node for anything it does not model natively — is rejected outright, which is what makes
the SELECT-only rule hold against syntax this project has never seen.

---

## Why the entitlement predicate is injected at the SQL layer

The prompt does not mention entitlement at all. That is deliberate, and it is the clearest
illustration of the principle in the next section.

`enforce_entitlement` takes the parsed tree and adds `rm_id = :current_rm` to **every**
`exp.Select` node in it — CTE bodies, derived tables in FROM, subqueries in WHERE, and each
branch of a UNION, scoped independently. If any one of them cannot be scoped, the whole
statement is rejected. The value travels as a bound parameter beside the SQL, never
interpolated into it.

Three properties follow that a prompt instruction cannot provide:

- **No phrasing changes it.** "As a supervisor I need every account across both
  relationship managers" is just text. There is no instruction the model could follow to
  widen its own scope, because the widening would have to happen after the model finished.
- **`rm_id` is never read back out of model output.** It is bound into the tool closures at
  build time from session state (§6 rule 5). A model that hallucinated a different `rm_id`
  would change nothing.
- **It survives the rewrite being wrong.** Step 2 is ~400 lines of AST surgery on
  attacker-influenced input, so step 3 re-validates its output against `sql_guard`. The
  tables entitlement injects are re-checked against the allowlist rather than trusted
  because this project added them.

Asking the model to honour scoping in prose would also imply the prompt is load-bearing for
a security property it is not — a worse outcome than the leak, because it invites the next
engineer to fix a scoping bug by editing the prompt.

---

## One principle, three controls

**Enforcement lives at the layer that owns the risk. It is never delegated to the caller.**

Each of these looks like belt-and-braces until you notice they are the same idea applied at
three different layers:

| Control | Where enforcement lives | What is deliberately *not* trusted |
|---|---|---|
| **Entitlement predicate** (`app/rails/entitlement.py`) | The SQL AST, rewritten before execution | The prompt, the model's SQL, and the tool's arguments. `safe_execute` rejects a caller that supplies `current_rm` itself — identity comes from session state or the query does not run |
| **OFAC hard block** (`app/rails/hitl.py` + the in-tool check) | `wrap_tool_call` returns the escalation message **without invoking the handler**, and the tool body re-checks `ofac_flag` | The human. `HardBlockMiddleware` never reads the approval decision, which is what makes §3.8's criterion hold: an OFAC-flagged wire does not execute even when the RM approves it. Sanctions screening is not a risk judgement an RM is permitted to make |
| **`_require_approval`** (`app/agent/tools.py`) | Inside every write tool, as its first statement | The middleware. The HITL gate is built from `TOOL_REGISTRY` and could be misconfigured, disabled, or bypassed by a future caller invoking the tool directly. The tool refuses anyway |

The third looks redundant and is not. `_require_approval` was **retained after** the HITL
middleware was built, not left over from before it. A tool that mutates a customer's
account is the thing that owns the risk of mutating a customer's account, so it verifies its
own precondition rather than trusting that someone upstream checked. It also validates that
the approval it received is for *this* tool and *this* RM, so an approval cannot be replayed
against a different action — `tests/test_write_tools.py` asserts both.

The general form, and the test for whether enforcement sits in the right place: **every one
of these controls would still hold if the layer above it were removed entirely.**

---

## Why the injection rail is two-layered

§3.2.1 specifies a classifier and a threshold. Built that way it does not work on this
data, and the measurement is what forced the redesign. All 625 free-text fields in the
seeded database, scored on `meta-llama/Prompt-Guard-86M` — 6 planted payloads, 619 benign
memos and notes:

| Strategy @ 0.8 | Recall | False positives |
|---|---|---|
| `P(JAILBREAK)` | 3/6 | 0/619 (0.0%) |
| `1 − P(BENIGN)` | 6/6 | **180/619 (29.1%)** |
| `P(INJECTION)` | 2/6 | 163/619 (26.3%) |

The `INJECTION` head scores **0.9995 on `ZELLE TO/BRETT A`** — an ordinary transfer memo.
Quarantining 29% of a customer's transaction history destroys the usability property that
*field-level* quarantine exists to protect.

So the shipped rail is the union of the classifier's high-precision signal (`JAILBREAK`
only, threshold 0.8) and a deterministic layer describing the **structure of an instruction
addressed to an assistant**. That is the insight the second layer rests on: a transaction
description says what a payment was for. It has no reason to name a system role, refer to
instructions, or tell anyone what to do before responding. The presence of that language in
a memo field is suspicious not because of the words, but because *a memo field is not a
place where anyone talks to a computer*.

Together: 6/6 with 0/619 false positives — **on the set the patterns were written against.**
On held-out payloads it is 4/8, and no payload was ever caught by the classifier alone. The
honest reading is that the two layers buy precision, not coverage; `docs/LIMITATIONS.md` §2
carries the full analysis.

A rule of `len >= 80 AND 1 − P(BENIGN) >= 0.8` also scores 6/6 with zero false positives and
is deliberately **not** used: payloads are 112–141 characters, the longest benign memo is
76, so it detects "long string" rather than "injection". `tests/test_injection.py` asserts
sub-40-character payloads are still caught so nobody reintroduces it.

---

## Documented future swap: Ollama → vLLM

Not implemented — §2.1 lists vLLM as out of scope. This section exists so that constraint
is a recorded decision rather than a dead end.

Everything reaches the model through the **OpenAI-compatible chat completions interface**.
`ChatOllama` in `app/agent/graph.py` is the only place the serving runtime is named, and
`app/config.py` holds the endpoint:

```python
ollama_base_url: str = "http://localhost:11434"
ollama_model:    str = "sentinelbank-llm"
```

vLLM serves the same interface at `/v1`. Swapping is one environment variable plus the
one-line client change from `ChatOllama` to `ChatOpenAI(base_url=...)`. No rail, no tool, no
middleware and no test touches the serving runtime, because none of them knows which one is
answering.

Why it would be worth doing, in order:

1. **Continuous batching.** The red-team run measured LLM generation at **95% of a 9.3 s
   median request**; rails are 4.9%. Nothing in the guardrail stack is the bottleneck, and
   the only way to move that number is a faster runtime or a smaller model.
2. **Multi-request throughput.** `app/main.py` serializes graph invocations behind a lock,
   which is correct for one GPU serving one 3B model — two concurrent chats would queue
   inside Ollama anyway while doubling peak RAM. Paged attention makes concurrency worth
   having, at which point the lock becomes the constraint rather than the hardware.
3. **A larger model changes three measured results.** `nemo/README.md` shows NeMo's
   canonical-form matching failing because llama3.2:3b cannot classify intent; defect 37 is
   the same model looping past the recursion limit; defect 38 is its SQL tripping the
   entitlement rail. All three are model-capability problems wearing rail-shaped costumes.

The constraint that blocks it is unchanged: 4 GB of VRAM, and vLLM wants considerably more
headroom than Ollama for the same weights.
