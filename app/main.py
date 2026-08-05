"""FastAPI entrypoint: /chat, /approve, /audit, /audit/verify, /health, /config/rails.

All transformer models load once in the `lifespan` startup handler, never per-request
(PROJECT_PLAN.md §1.1 rule 4, §3.6).

================================================================================
WHY THE LIFESPAN HANDLER IS THE WHOLE POINT OF THIS MODULE
================================================================================
Four models sit behind this API, and none of them is small on an 8 GB machine:

    Presidio AnalyzerEngine + spaCy en_core_web_sm    ~9.4 s,  ~500 MB
    Prompt-Guard-86M (or the deberta fallback)        ~3-8 s,  ~350 MB
    all-MiniLM-L6-v2                                  ~2 s,    ~90 MB

Constructing any of them inside a request handler does not merely make that request
slow — it allocates a *second* copy while the first is still referenced, and a handful
of queries walks the process into the swap file with Ollama holding 2 GB of VRAM and
Postgres holding its own working set. This is the failure §1.1 rule 4 exists to prevent
and it is the single most likely way for this app to become unusable.

Each model is a `lru_cache`-backed singleton in its own module, so "warming" is just
calling the accessor once. That is deliberate: `/health` then reports residency by
reading `cache_info().currsize` — the real state of the process — rather than a boolean
this module set and hoped stayed true.

================================================================================
ONE REQUEST AT A TIME, ON PURPOSE
================================================================================
`_INVOKE_LOCK` serializes graph invocations. This is not defensive hand-waving; it is
what the hardware already imposes. There is one GTX 1650 serving one 3B model, so two
concurrent chats would queue inside Ollama anyway, while doubling peak RAM and
interleaving the per-request latency samples that `/chat` reports as "which rails fired
and what did they cost". Holding the lock makes the timing window exact.

================================================================================
WHAT /chat REPORTS AND WHERE EACH PART COMES FROM
================================================================================
§3.6 asks for the answer plus the rails that fired, any injection warnings for the UI
banner, and whether the request is now waiting on approval. Those come from three
different places, and the reason is worth stating because it is not arbitrary:

  * **rails fired** — from `timing.get_samples()`, sliced to the window this request
    held the lock. Every rail is already decorated with `@rail_timer` for §4.3's latency
    table, so this is one source of truth rather than a second reporting path that could
    disagree with the audit log.
  * **injection warnings** — the input-scan verdict arrives through agent state
    (`injection_flags`), but the field-level quarantine does not: it happens inside
    `wrap_tool_call`, which cannot return a state update. Those incidents live on the
    `ToolResultInjectionMiddleware` instance, which is why §3.4's `build_agent_bundle`
    hands the middleware back.
  * **interrupted** — `__interrupt__` in the returned state, carrying the structured diff
    §3.3 built. `/approve` resumes it with `Command(resume=...)` on the same `thread_id`.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.agent.graph import AgentBundle, agent_config, build_agent_bundle
from app.audit.chain import count_entries, read_entries, verify_chain_detailed
from app.config import RailsConfig, settings
from app.db import get_session
from app.llm import LLMError, llm
from app.rails.hitl import (
    DEFAULT_CHECKPOINT_PATH,
    RAIL_HARD_BLOCK,
    HardBlockMiddleware,
    build_checkpointer,
)
from app.rails.injection import ToolResultInjectionMiddleware, get_classifier_info
from app.rails.pii import pii_request
from app.rails.sql_pipeline import (
    RAIL_ENTITLEMENT,
    RAIL_SQL_GUARD,
    STAGE_ENTITLEMENT,
    STAGE_EXECUTE,
    get_sql_outcomes,
)
from app.rails.timing import get_samples

logger = logging.getLogger(__name__)

# Where the HITL checkpointer lives. Module-level so tests can redirect it to tmp_path
# before startup rather than writing a real ./checkpoints.db (§3.3).
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH

# See the module docstring. One GPU, one model, one request.
_INVOKE_LOCK = threading.Lock()


# ================================================================================
# Process-wide state, assembled at startup
# ================================================================================


@dataclass
class ModelStatus:
    """Load result for one model, for `/health`.

    `loaded` is not stored — it is a property that asks the singleton itself, so a
    model that was evicted, or was never warmed because startup partially failed,
    reports honestly rather than reporting what startup intended.
    """

    name: str
    probe: Any
    detail: str = ""
    load_ms: float | None = None
    error: str | None = None

    @property
    def loaded(self) -> bool:
        try:
            return bool(self.probe())
        except Exception:  # pragma: no cover - a probe must never break /health
            return False


@dataclass
class AppState:
    """Everything built once at startup."""

    models: dict[str, ModelStatus] = field(default_factory=dict)
    checkpointer: Any = None
    agents: dict[str, AgentBundle] = field(default_factory=dict)
    # thread_id -> the RM who owns it. Needed by /approve, which gets only a thread_id
    # but must resume the graph belonging to the right RM's entitlement scope.
    threads: dict[str, str] = field(default_factory=dict)
    # thread_id -> how many tool calls are awaiting a decision. LangChain requires the
    # resume payload to carry exactly one decision per interrupted call.
    pending: dict[str, int] = field(default_factory=dict)
    started_at: float = 0.0

    def reset(self) -> None:
        self.models.clear()
        self.agents.clear()
        self.threads.clear()
        self.pending.clear()
        self.checkpointer = None


state = AppState()


def _cache_probe(accessor: Any):
    """A probe reporting whether an `lru_cache` singleton is currently resident.

    Reads `cache_info().currsize` rather than calling the accessor — calling it would
    *load* the model, so a `/health` check on a cold process would silently do the very
    per-request loading this module exists to prevent.
    """
    return lambda: accessor.cache_info().currsize > 0


def _warm(name: str, loader: Any, probe: Any, detail: str = "") -> ModelStatus:
    """Load one model, timing it, and never let a failure abort startup.

    A failed guardrail model is a serious condition, but the right response is a `/health`
    that says so plainly — not a process that will not start, which leaves an operator
    with a stack trace and no way to ask the app what is wrong.
    """
    status = ModelStatus(name=name, probe=probe, detail=detail)
    started = time.perf_counter()
    try:
        loaded = loader()
        status.load_ms = round((time.perf_counter() - started) * 1000, 1)
        if detail == "":
            status.detail = type(loaded).__name__
        logger.info("loaded %s in %.0f ms", name, status.load_ms)
    except Exception as exc:
        status.error = f"{type(exc).__name__}: {exc}"
        logger.exception("failed to load %s", name)
    return status


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model once, before the first request (§1.1 rule 4, §3.6)."""
    import torch

    from app.embeddings import get_embedder
    from app.rails.injection import get_classifier
    from app.rails.pii import get_analyzer

    # §1.1 rule 6. `app.embeddings` sets this at import too; repeating it here makes the
    # constraint visible at the one place a reader looks for startup behaviour, and it is
    # idempotent.
    torch.set_num_threads(4)

    state.started_at = time.time()

    # Presidio first: it is the slowest and the largest, so a machine that cannot fit the
    # set fails on the one whose absence is most obvious rather than partway through.
    presidio = _warm("presidio_analyzer", get_analyzer, _cache_probe(get_analyzer))
    state.models["presidio_analyzer"] = presidio
    # spaCy is not warmed separately — Presidio's NlpEngine owns it, and loading a second
    # copy would double ~500 MB for no benefit. It is reported as its own row because
    # "which models are resident" is the question /health answers, and en_core_web_sm is
    # the one whose absence produces a silently worse PERSON recall rather than an error.
    state.models["spacy"] = ModelStatus(
        name="spacy",
        probe=_cache_probe(get_analyzer),
        detail=_spacy_detail() if presidio.error is None else "not loaded",
        error=presidio.error and "not loaded: the Presidio analyzer failed",
    )
    state.models["injection_classifier"] = _warm(
        "injection_classifier", get_classifier, _cache_probe(get_classifier)
    )
    state.models["embedder"] = _warm(
        "embedder", get_embedder, _cache_probe(get_embedder), detail=settings.embedding_model
    )

    # §3.3: file-backed, so an approval that a human takes two minutes to give survives a
    # restart in between. Built once — the connection is owned for the life of the process.
    try:
        state.checkpointer = build_checkpointer(CHECKPOINT_PATH)
    except Exception:  # pragma: no cover
        logger.exception("could not open the HITL checkpointer at %s", CHECKPOINT_PATH)

    try:
        yield
    finally:
        llm.close()
        state.reset()


def _spacy_detail() -> str:
    """The spaCy model actually backing Presidio, read from the live engine."""
    from app.rails.pii import get_analyzer

    try:
        # `nlp` is {language_code: Language}. The key alone is just "en", which does not
        # answer the question /health is being asked — en_core_web_sm and en_core_web_trf
        # have very different PERSON recall, so the pipeline name is the useful part.
        nlp = getattr(get_analyzer().nlp_engine, "nlp", None) or {}
        return (
            ", ".join(
                sorted(f"{lang}_{model.meta.get('name', '?')}" for lang, model in nlp.items())
            )
            or "en_core_web_sm"
        )
    except Exception:  # pragma: no cover
        return "unknown"


app = FastAPI(
    title="SentinelBank",
    description="On-premise RM copilot with a layered guardrail pipeline.",
    version="0.3.0",
    lifespan=lifespan,
)


# ================================================================================
# Agents
# ================================================================================


def get_bundle(rm_id: str) -> AgentBundle:
    """The compiled agent for `rm_id`, built once and reused.

    One graph per RM rather than one per request: every tool closes over `rm_id` so the
    model can never name it (§2.1), which means the binding is fixed at build time. The
    graph is stateless between calls — conversation state lives in the checkpointer,
    keyed by `thread_id` — so reuse is safe as well as necessary.

    `settings.guardrails_enabled` is passed explicitly so that every agent, the
    `/config/rails` endpoint and the Streamlit sidebar all hold the *same* mutable
    RailsConfig instance. That shared identity is what makes a toggle take effect on the
    next message with no rebuild, which is the core demo (§3.4).
    """
    bundle = state.agents.get(rm_id)
    if bundle is None:
        bundle = build_agent_bundle(
            rm_id, settings.guardrails_enabled, checkpointer=state.checkpointer
        )
        state.agents[rm_id] = bundle
    return bundle


# ================================================================================
# Schemas
# ================================================================================


class ChatRequest(BaseModel):
    rm_id: str = Field(default_factory=lambda: settings.default_rm_id)
    thread_id: str
    message: str


class RailReport(BaseModel):
    """One rail's contribution to this request (§3.7's rail status panel).

    `reason` carries the rejecting rail's own code — `audit_log_access`,
    `unlinked_scoped_table:a` — unmodified. Those codes are a contract with §4.2's
    harness, which asserts on *which* rail fired and *why*, and rewording them here
    would break the tests that prove it.

    `group` is what keeps §4.3's latency budget honest. See `RAIL_GROUPS`.
    """

    rail: str
    verdict: str
    score: float | None = None
    reason: str | None = None
    latency_ms: float
    calls: int
    group: Literal["plan", "added", "tool"] = "added"


class SecurityWarning(BaseModel):
    """A banner item for §3.7. `kind` selects the banner; the rest fills it in."""

    kind: Literal["input_injection", "quarantine", "hard_block"]
    message: str
    rail: str
    row_id: str | None = None
    column: str | None = None
    score: float | None = None
    label: str | None = None
    layers: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    preview: str | None = None


class PendingAction(BaseModel):
    """One tool call waiting on the RM, with §3.3's structured diff."""

    tool: str
    args: dict[str, Any]
    diff: str
    allowed_decisions: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    thread_id: str
    rm_id: str
    answer: str
    decision: str
    interrupted: bool
    pending: list[PendingAction] = Field(default_factory=list)
    rails: list[RailReport] = Field(default_factory=list)
    warnings: list[SecurityWarning] = Field(default_factory=list)
    audit_event_id: int | None = None
    elapsed_ms: float


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
    # Only needed to recover a thread this process did not serve — the checkpoint file
    # survives a restart but the in-memory thread->RM map does not.
    rm_id: str | None = None


class AuditEntryOut(BaseModel):
    event_id: int
    timestamp: Any
    rm_id: str
    user_query: str | None = None
    rails_fired: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    decision: str
    response_summary: str | None = None
    prev_hash: str
    entry_hash: str


class AuditPage(BaseModel):
    total: int
    limit: int
    offset: int
    entries: list[AuditEntryOut]


class ChainStatus(BaseModel):
    intact: bool
    entries: int
    broken_at_index: int | None = None
    event_id: int | None = None
    reason: str | None = None
    expected: str | None = None
    found: str | None = None


class RailsUpdate(BaseModel):
    """Partial update: only the flags present in the body are changed.

    Every field is optional and defaults to None rather than True. A body that omits a
    rail must leave it as it was — a model with `True` defaults would silently re-enable
    every rail the caller did not mention, which is a fail-*open* behaviour §6 rule 4
    forbids.
    """

    injection_input: bool | None = None
    pii_builtin: bool | None = None
    pii_presidio: bool | None = None
    person_pseudonym: bool | None = None
    hitl: bool | None = None
    injection_tool_result: bool | None = None
    spotlight: bool | None = None
    audit: bool | None = None
    topical: bool | None = None


# ================================================================================
# Reporting helpers
# ================================================================================

# Rails whose presence in the latency samples means something was flagged rather than
# merely scanned. Everything else reports RAN — the rail did its work and found nothing.
_FLAGGING_RAILS = {"injection_input", "injection_tool_result", "hitl_interrupt"}

# --------------------------------------------------------------------------------
# The rail set, reconciled against §4.3
# --------------------------------------------------------------------------------
# §4.3's latency table names eight rows. The running system emits more names than that,
# and lumping them together makes the table's "total rail overhead" figure wrong in a way
# nobody would notice — so every timed name is classified here, once:
#
#   "plan"  — one of §4.3's eight rows, reported under the plan's own name.
#   "added" — a real guardrail the plan's table did not anticipate. `spotlight` (§3.4
#             position 7, deviation 23), `hitl_interrupt` (§3.3's approval gate) and
#             `hitl_hard_block` (§3.3's sanctions screen) all do guardrail work and all
#             belong in the overhead total; they were simply missing from the table.
#   "tool"  — NOT a guardrail. A `tool_*` span wraps an entire tool body, which *contains*
#             `sql_guard`, `entitlement` and the database round-trip. Its time overlaps
#             the rails nested inside it, so adding it to a rail total counts the same
#             milliseconds twice and buries Postgres inside the guardrail budget.
#
# Anything unrecognised falls to "added" rather than "tool": a new rail nobody classified
# should show up in the overhead figure, not silently vanish from it.
SECTION_4_3_RAILS: frozenset[str] = frozenset(
    {
        "pii_input",
        "injection_input",
        "injection_tool_result",
        RAIL_SQL_GUARD,
        RAIL_ENTITLEMENT,
        "pii_tool_result",
        "pii_output",
        "audit_write",
    }
)

_TOOL_SPAN_PREFIX = "tool_"


def _rail_group(rail: str) -> Literal["plan", "added", "tool"]:
    """Which of the three classes above `rail` belongs to."""
    if rail in SECTION_4_3_RAILS:
        return "plan"
    if rail.startswith(_TOOL_SPAN_PREFIX):
        return "tool"
    return "added"


def _sql_verdicts(
    outcomes: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Verdicts and reason codes for `sql_guard` and `entitlement` (§3.6).

    These two rails run inside a tool rather than inside a middleware, so unlike the
    injection rails nothing about them reaches agent state. `sql_pipeline` records each
    outcome instead, and this turns that record into something the UI can render and
    §4.2's harness can assert on.

    Attribution follows the pipeline's own `stage`, which is what that field is for:
    `validate` and `revalidate` are sql_guard rejecting, `entitlement` is the scoping
    rail rejecting. The first rejection of each kind wins, because it is the one that
    stopped the query — a later tool call succeeding does not un-block it.
    """
    verdicts: dict[str, str] = {}
    reasons: dict[str, str] = {}

    for outcome in outcomes:
        stage, reason = outcome["stage"], outcome["reason"]

        if not outcome["allowed"]:
            # STAGE_EXECUTE means both rails passed and Postgres failed. Blaming a rail
            # for a driver error would be a false positive in the one table that exists
            # to measure false positives.
            if stage == STAGE_EXECUTE:
                continue
            rail = RAIL_ENTITLEMENT if stage == STAGE_ENTITLEMENT else RAIL_SQL_GUARD
            if rail not in verdicts:
                verdicts[rail], reasons[rail] = "BLOCKED", reason
            continue

        # Allowed. sql_guard passed it twice; say so rather than leaving the row blank.
        verdicts.setdefault(RAIL_SQL_GUARD, "PASS")
        # SCOPED is the interesting verdict, and the one §3.8's entitlement criterion
        # turns on: the query that reached the database is not the query the model wrote.
        # A zero-predicate pass is a policy_chunks query with nothing to scope.
        if outcome["predicates"] and verdicts.get(RAIL_ENTITLEMENT) != "BLOCKED":
            verdicts[RAIL_ENTITLEMENT] = "SCOPED"
            reasons[RAIL_ENTITLEMENT] = f"{outcome['predicates']} predicate(s) injected"
        else:
            verdicts.setdefault(RAIL_ENTITLEMENT, "PASS")

    return verdicts, reasons


def _rail_reports(
    samples: list[dict[str, Any]],
    flags: list[dict[str, Any]],
    incidents: list[Any],
    interrupted: bool,
    sql_outcomes: list[dict[str, Any]] | None = None,
    blocked: list[dict[str, Any]] | None = None,
) -> list[RailReport]:
    """Aggregate this request's rails into §3.7's status panel rows.

    Latency is summed per rail rather than averaged: an RM asking "what did the rails
    cost me" wants the total added to their request, and the PII rail is invoked twelve
    times per turn precisely so §4.3 can attribute each application point separately.
    `sql_guard` is summed for the same reason — it genuinely runs twice per query, once
    before the entitlement rewrite and once after it.
    """
    scores: dict[str, float] = {}
    verdicts: dict[str, str] = {}
    reasons: dict[str, str] = {}

    for flag in flags:
        rail = str(flag.get("rail") or "injection_input")
        verdicts[rail] = "BLOCKED"
        if flag.get("reason"):
            reasons[rail] = str(flag["reason"])
        if flag.get("score") is not None:
            scores[rail] = max(scores.get(rail, 0.0), float(flag["score"]))

    if incidents:
        verdicts["injection_tool_result"] = "QUARANTINED"
        scores["injection_tool_result"] = max(float(i.score) for i in incidents)
        reasons["injection_tool_result"] = f"{len(incidents)} field(s) quarantined"

    if interrupted:
        verdicts["hitl_interrupt"] = "AWAITING_APPROVAL"

    # The sanctions screen. It has always been reported in `warnings[]`; this is what puts
    # it in the rail table too, beside the approval gate it is deliberately separate from.
    if blocked:
        verdicts[RAIL_HARD_BLOCK] = "BLOCKED"
        reasons[RAIL_HARD_BLOCK] = str(blocked[0].get("reason") or "ofac_hard_block")

    sql_verdicts, sql_reasons = _sql_verdicts(sql_outcomes or [])
    verdicts.update(sql_verdicts)
    reasons.update(sql_reasons)

    aggregated: dict[str, RailReport] = {}
    for sample in samples:
        rail = sample["rail"]
        report = aggregated.get(rail)
        if report is None:
            report = RailReport(
                rail=rail,
                verdict=verdicts.get(rail, "RAN"),
                score=scores.get(rail),
                reason=reasons.get(rail),
                latency_ms=0.0,
                calls=0,
                group=_rail_group(rail),
            )
            aggregated[rail] = report
        report.latency_ms = round(report.latency_ms + sample["duration_ms"], 3)
        report.calls += 1

    # A rail can flag without appearing in the samples — `hitl_interrupt` is timed around
    # diff construction, which does not run when the tool is auto-approved.
    for rail, verdict in verdicts.items():
        if rail not in aggregated:
            aggregated[rail] = RailReport(
                rail=rail,
                verdict=verdict,
                score=scores.get(rail),
                reason=reasons.get(rail),
                latency_ms=0.0,
                calls=0,
                group=_rail_group(rail),
            )

    return sorted(aggregated.values(), key=lambda r: r.rail)


def _warnings(flags: list[dict[str, Any]], incidents: list[Any], blocked: list[dict]) -> list[SecurityWarning]:
    """The §3.7 banner payload: what was caught, in which row, in which field."""
    warnings: list[SecurityWarning] = []

    for flag in flags:
        warnings.append(
            SecurityWarning(
                kind="input_injection",
                rail=str(flag.get("rail") or "injection_input"),
                message=(
                    "This message was flagged as a prompt-injection attempt and was not "
                    "sent to the model."
                ),
                score=flag.get("score"),
                label=flag.get("label"),
                layers=list(flag.get("layers") or []),
                patterns=list(flag.get("patterns") or []),
            )
        )

    for incident in incidents:
        warnings.append(
            SecurityWarning(
                kind="quarantine",
                rail="injection_tool_result",
                message=(
                    f"Injected content was found in {incident.row_id}.{incident.column} "
                    "and was quarantined. The rest of the record is unchanged and your "
                    "answer is based on it."
                ),
                row_id=incident.row_id,
                column=incident.column,
                score=incident.score,
                label=incident.label,
                layers=list(incident.layers),
                patterns=list(incident.patterns),
                preview=incident.preview,
            )
        )

    for block in blocked:
        warnings.append(
            SecurityWarning(
                kind="hard_block",
                rail=RAIL_HARD_BLOCK,
                message=(
                    f"{block['tool']} was refused: the customer carries a sanctions flag. "
                    "This cannot be approved and has been escalated to compliance."
                ),
                label=block.get("reason"),
            )
        )

    return warnings


def _answer(result: dict[str, Any]) -> str:
    """The last assistant message, which is what the RM sees."""
    from langchain_core.messages import AIMessage

    for message in reversed(result.get("messages") or []):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return ""


def _pending_actions(result: dict[str, Any]) -> list[PendingAction]:
    """Unpack `__interrupt__` into §3.3's structured diffs.

    The description carried on each action request is the diff `build_approval_diff`
    rendered at interrupt time — customer, account, transaction, the before/after change,
    and the Reg E citation retrieved live from the policy RAG. It is passed through
    verbatim: re-rendering it here would mean the RM approves a description produced by
    something other than the code that will perform the change.
    """
    actions: list[PendingAction] = []
    for interrupt in result.get("__interrupt__") or ():
        value = getattr(interrupt, "value", interrupt)
        if not isinstance(value, dict):
            continue
        requests = value.get("action_requests") or []
        configs = value.get("review_configs") or []
        for index, request in enumerate(requests):
            config = configs[index] if index < len(configs) else {}
            actions.append(
                PendingAction(
                    tool=str(request.get("name", "")),
                    args=dict(request.get("args") or {}),
                    diff=str(request.get("description") or ""),
                    allowed_decisions=list(config.get("allowed_decisions") or []),
                )
            )
    return actions


def _decision(result: dict[str, Any], interrupted: bool) -> str:
    """The §1.5 decision value for this turn, matching what the audit rail recorded."""
    if result.get("injection_flags") or result.get("topic_flags"):
        return "BLOCKED"
    if interrupted:
        return "PENDING_APPROVAL"
    return "ALLOWED"


@dataclass
class _Run:
    """One graph invocation and everything observed around it."""

    result: dict[str, Any]
    samples: list[dict[str, Any]]
    incidents: list[Any]
    blocked: list[dict[str, Any]]
    sql_outcomes: list[dict[str, Any]]
    elapsed_ms: float


def _invoke(bundle: AgentBundle, payload: Any, thread_id: str) -> _Run:
    """Run the graph under the lock, capturing this request's rail activity.

    The `pii_request` scope is the important line. §3.1.4 requires the pseudonym map to
    be per-request and destroyed at the end of it — a map that outlived its request and
    was consulted by the next one would resolve `<PERSON_a3f9>` to a customer belonging
    to a different conversation, which is a cross-customer data breach rather than a bug.
    The context manager's `finally` guarantees the teardown even when the graph raises.

    Sampling by index rather than clearing the ring buffer keeps §4.3's aggregate history
    intact — the report over a whole red-team run reads the same file.
    """
    scan = bundle.of_type(ToolResultInjectionMiddleware)
    hard_block = bundle.of_type(HardBlockMiddleware)

    with _INVOKE_LOCK:
        first_sample = len(get_samples())
        first_incident = len(scan.incidents) if scan else 0
        first_block = len(hard_block.blocked) if hard_block else 0
        first_outcome = len(get_sql_outcomes())

        started = time.perf_counter()
        with pii_request(thread_id):
            result = bundle.graph.invoke(payload, agent_config(thread_id))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        return _Run(
            result=result,
            samples=get_samples()[first_sample:],
            incidents=list(scan.incidents[first_incident:]) if scan else [],
            blocked=list(hard_block.blocked[first_block:]) if hard_block else [],
            sql_outcomes=get_sql_outcomes()[first_outcome:],
            elapsed_ms=elapsed_ms,
        )


def _respond(rm_id: str, thread_id: str, run: _Run) -> ChatResponse:
    """Assemble the §3.6 response and record what the next `/approve` will need."""
    pending = _pending_actions(run.result)
    interrupted = bool(pending)

    state.threads[thread_id] = rm_id
    if interrupted:
        state.pending[thread_id] = len(pending)
    else:
        state.pending.pop(thread_id, None)

    flags = list(run.result.get("injection_flags") or [])
    # §4.4's topical rail reports through the same `rails[]` mechanism — its flag dicts
    # already carry `rail` and `reason` — but NOT through `_warnings`, which would label
    # a refusal to give tax advice as a prompt-injection attempt in the red banner.
    rail_flags = flags + list(run.result.get("topic_flags") or [])

    return ChatResponse(
        thread_id=thread_id,
        rm_id=rm_id,
        answer=_answer(run.result),
        decision=_decision(run.result, interrupted),
        interrupted=interrupted,
        pending=pending,
        rails=_rail_reports(
            run.samples,
            rail_flags,
            run.incidents,
            interrupted,
            run.sql_outcomes,
            run.blocked,
        ),
        warnings=_warnings(flags, run.incidents, run.blocked),
        audit_event_id=run.result.get("audit_event_id"),
        elapsed_ms=run.elapsed_ms,
    )


# ================================================================================
# Endpoints
# ================================================================================


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """The main query endpoint (§3.6).

    `rm_id` comes from the request body because §2.1's session model is a hardcoded RM
    selector — authentication is explicitly out of scope. It is nonetheless treated as
    session state, not as data: it is bound into the tool closures at build time and
    reaches `sql_pipeline.safe_execute` from there, so no message content can influence
    it (§6 rule 5).
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message must not be empty")

    bundle = get_bundle(request.rm_id)
    try:
        run = _invoke(
            bundle, {"messages": [("user", request.message)]}, request.thread_id
        )
    except LLMError as exc:
        # Fail closed (§6 rule 4): an unreachable model is an error, never a guessed answer.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _respond(request.rm_id, request.thread_id, run)


@app.post("/approve", response_model=ChatResponse)
def approve(request: ApproveRequest) -> ChatResponse:
    """Resume an interrupted graph with the RM's decision (§3.3, §3.6).

    The `thread_id` is what makes this work: the checkpointer stored the paused state
    under it, so this call rejoins the run exactly where `HumanInTheLoopMiddleware`
    suspended it — mid-`after_model`, before the tool node — rather than starting a new
    one that would have to be talked back into the same tool call.

    Note what approval does *not* buy. `HardBlockMiddleware` never reads the decision, so
    an OFAC-flagged action refuses here just as it would have refused unapproved (§3.8).
    """
    rm_id = request.rm_id or state.threads.get(request.thread_id)
    if rm_id is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown thread_id {request.thread_id!r}. If the server restarted, "
                "resend with rm_id — the checkpoint survives but the thread map does not."
            ),
        )

    count = state.pending.get(request.thread_id)
    if not count:
        raise HTTPException(
            status_code=409,
            detail=f"thread {request.thread_id!r} is not waiting for approval",
        )

    bundle = get_bundle(rm_id)
    # One decision per interrupted tool call — LangChain raises if the counts disagree.
    resume = Command(resume={"decisions": [{"type": request.decision}] * count})

    try:
        run = _invoke(bundle, resume, request.thread_id)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = _respond(rm_id, request.thread_id, run)
    if not response.interrupted:
        response.decision = "APPROVED" if request.decision == "approve" else "REJECTED"
    return response


@app.get("/audit", response_model=AuditPage)
def audit(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    newest_first: bool = Query(default=True),
) -> AuditPage:
    """A page of audit entries (§3.6).

    Entries are already PII-redacted at write time (§1.5) — this endpoint does no
    redaction of its own, and that is deliberate: a reader must see exactly what was
    hashed, or the page could not be used to check the chain by hand.
    """
    entries = read_entries(limit=limit, offset=offset, newest_first=newest_first)
    return AuditPage(
        total=count_entries(),
        limit=limit,
        offset=offset,
        entries=[AuditEntryOut(**entry) for entry in entries],
    )


@app.get("/audit/verify", response_model=ChainStatus)
def audit_verify() -> ChainStatus:
    """Walk the whole chain from genesis and report the first break (§3.5, §3.6).

    Never paginated. Verification is a walk from the genesis sentinel forward, and a page
    of it proves nothing — the linkage check that catches a *deleted* entry needs the
    predecessor that pagination would have left on the previous page.
    """
    total = count_entries()
    verdict = verify_chain_detailed()
    if verdict is None:
        return ChainStatus(intact=True, entries=total)
    return ChainStatus(
        intact=False,
        entries=total,
        broken_at_index=verdict.index,
        event_id=verdict.event_id,
        reason=verdict.reason,
        expected=verdict.expected,
        found=verdict.found,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Ollama reachability, Postgres connectivity, and per-model residency (§3.6).

    "ok" on its own is not a useful answer here. The two conditions this project actually
    degrades under are invisible to a boolean: a guardrail model that failed to load (the
    rails are then running with fewer layers than the README claims), and Prompt-Guard
    falling back to the deberta model (§1.1's documented fallback, which has a measurably
    different false-positive rate — every number in §4.3 is about a different model). Both
    are reported by name.
    """
    models: dict[str, Any] = {}
    for key, status in state.models.items():
        models[key] = {
            "loaded": status.loaded,
            "detail": status.detail,
            "load_ms": status.load_ms,
            "error": status.error,
        }

    classifier = get_classifier_info()
    if classifier is not None:
        models.setdefault("injection_classifier", {}).update(
            {
                "model_id": classifier.model_id,
                "is_fallback": classifier.is_fallback,
                "labels": list(classifier.labels),
                "trigger_labels": sorted(classifier.trigger_labels),
            }
        )

    ollama: dict[str, Any] = {
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_model,
        "num_ctx": settings.llm_num_ctx,
    }
    try:
        served = llm.list_models()
        ollama["reachable"] = True
        ollama["serves_model"] = llm.serves_model()
        ollama["available_models"] = served
    except LLMError as exc:
        ollama["reachable"] = False
        ollama["serves_model"] = False
        ollama["error"] = str(exc)

    postgres: dict[str, Any] = {"database": settings.postgres_db}
    try:
        with get_session() as session:
            postgres["reachable"] = True
            postgres["customers"] = int(
                session.execute(text("SELECT COUNT(*) FROM customers")).scalar() or 0
            )
            postgres["audit_entries"] = int(
                session.execute(text("SELECT COUNT(*) FROM audit_log")).scalar() or 0
            )
            postgres["pgvector"] = bool(
                session.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).scalar()
            )
    except Exception as exc:
        postgres["reachable"] = False
        postgres["error"] = f"{type(exc).__name__}: {exc}"

    healthy = (
        ollama.get("serves_model")
        and postgres.get("reachable")
        and all(entry["loaded"] for entry in models.values())
    )

    return {
        "status": "ok" if healthy else "degraded",
        "uptime_s": round(time.time() - state.started_at, 1) if state.started_at else None,
        "ollama": ollama,
        "postgres": postgres,
        "models": models,
        "checkpointer": {
            "path": CHECKPOINT_PATH,
            "ready": state.checkpointer is not None,
        },
        "agents_built": sorted(state.agents),
        "rails": settings.guardrails_enabled.model_dump(),
    }


@app.get("/config/rails", response_model=RailsConfig)
def read_rails() -> RailsConfig:
    """The live rail configuration, for the §3.7 sidebar to render its checkboxes."""
    return settings.guardrails_enabled


@app.post("/config/rails", response_model=RailsConfig)
def set_rails(update: RailsUpdate) -> RailsConfig:
    """Toggle individual rails at runtime (§3.6, demo support).

    Mutates the single `RailsConfig` instance every compiled agent holds a reference to,
    so the change lands on the next message with no rebuild — that is what makes "ask the
    same question with the rail off" a two-second demo instead of a restart.

    One rail is deliberately not switchable from here, and it is not an oversight:
    `HardBlockMiddleware` ignores the `hitl` flag entirely. §3.3 is explicit that an
    OFAC-flagged wire is refused whether or not the HITL rail is enabled, so there is no
    request body that can disarm sanctions screening.
    """
    changed = update.model_dump(exclude_none=True)
    if not changed:
        raise HTTPException(status_code=422, detail="no rail flags supplied")

    for flag, value in changed.items():
        setattr(settings.guardrails_enabled, flag, value)

    logger.warning(
        "rails changed at runtime: %s",
        ", ".join(f"{flag}={value}" for flag, value in sorted(changed.items())),
    )
    return settings.guardrails_enabled
