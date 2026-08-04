"""Agent tool definitions and their HITL risk tiers (PROJECT_PLAN.md §2.3).

The three AUTO-tier read tools. Write tools (`file_dispute`, `waive_fee`, `block_card`,
`update_contact_info`, `initiate_wire`) are not implemented here — they must not exist
before the HITL gate that guards them (§3.3), because a write tool that is reachable and
merely *documented* as requiring approval is a write tool with no approval.

Everything in this module is a plain Python function, callable and testable with no LLM
and no agent running, as §2.3 requires. Phase 3 wraps them for LangGraph; nothing here
should need to change when it does.

--------------------------------------------------------------------------------
THE SOURCE FLAG — read this before changing any return shape
--------------------------------------------------------------------------------
Every field of every row comes back tagged `TRUSTED` or `UNTRUSTED`. This is not
bookkeeping: Phase 3's tool-result injection rail (§3.2.2) — the signature feature of
this whole project — uses nothing else to decide what to scan and what to spotlight. If
`transactions.description` arrives here tagged TRUSTED, the four planted payloads reach
model context unscanned and the demo silently stops demonstrating anything.

**The flag is per field, not per row.** A transaction row is one poisoned memo attached
to a perfectly ordinary amount, date, and channel. §3.2.2 requires that the offending
*field* be quarantined while "the surrounding legitimate row data" survives, so that the
RM still gets their answer. Carrying the flag next to the value makes that a one-line
local edit for the Phase 3 middleware — `field["value"] = QUARANTINE_TEXT` — and makes it
impossible for the flag to drift away from the data it describes.

**Classification fails closed**, and the reasoning is worth stating because rule 4 below
is a deliberate departure from a literal reading of "UNTRUSTED is description, merchant,
and notes":

  1. A non-string value is TRUSTED. A Decimal, date, bool, int or None cannot carry an
     instruction to a language model. This one rule is what makes every balance,
     aggregate, count, flag, and timestamp TRUSTED without enumerating them.
  2. A string in a known untrusted column is UNTRUSTED.
  3. A string in a known trusted column is TRUSTED.
  4. A string in an *unrecognised* column is UNTRUSTED.

Rule 4 exists because `query_customer_data` executes SQL that, from Phase 3 onward, the
model writes — and the model can be steered by the very payloads this flag exists to
catch. `SELECT t.description AS memo` returns a result key of `memo`. Under name-only
matching that is not in the untrusted set, so it would be tagged TRUSTED and skip the
injection scan entirely: a one-word bypass of the project's primary defence, available to
anything that can influence the generated SQL. Defaulting unknown *strings* to UNTRUSTED
closes it. The cost is scanning some column aliases and expression results that turn out
to be harmless — tens of milliseconds on CPU — which is the right side of the trade
(§6 rule 4: if a rail cannot decide, block rather than allow).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_session
from app.embeddings import DEFAULT_TOP_K, search_policy_chunks
from app.rails.sql_pipeline import safe_execute
from app.rails.timing import rail_timer


class RiskTier(str, Enum):
    """The three HITL tiers from §2.3/§3.3.

    All three are declared even though only AUTO tools exist in this phase: Phase 3's
    `hitl.py` imports this enum to build its `interrupt_on` config, and the tier
    vocabulary should not shift underneath it later.

    AUTO            executes immediately, logged.
    HUMAN_APPROVAL  interrupts the graph; an RM must approve or reject.
    HARD_BLOCK      never executes, regardless of approval (OFAC-flagged wires).
    """

    AUTO = "AUTO"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    HARD_BLOCK = "HARD_BLOCK"


class Source(str, Enum):
    """Whether a value may contain attacker-controlled text.

    TRUSTED    bank-controlled: schema values, numerics, enums, IDs, aggregates, and the
               policy documents we wrote ourselves.
    UNTRUSTED  free text originated outside the bank — or anything this module cannot
               positively prove is trusted. Phase 3 scans and spotlights these.
    """

    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"


# --------------------------------------------------------------------------------
# Column classification
# --------------------------------------------------------------------------------

# The three attacker-controlled free-text columns, matching the >>> UNTRUSTED <<< blocks
# in db/schema.sql. `merchant` is here alongside the two the plan calls out as injection
# vectors because merchant strings are supplied by third-party payees, not by the bank
# (§4.5's threat model names "malicious third-party payee" as an attacker).
UNTRUSTED_COLUMNS = frozenset({"description", "merchant", "notes"})

# Every other column of the tables sql_guard allows the agent to read. Mirrors
# db/schema.sql. A column missing from this set is not a crash — it falls through to
# rule 4 and is treated as untrusted, which is the safe direction for an oversight.
TRUSTED_COLUMNS = frozenset(
    {
        # relationship_managers
        "rm_id", "full_name", "branch",
        # customers
        "customer_id", "ssn", "drivers_license", "date_of_birth", "email", "phone",
        "street_address", "city", "state", "zip_code", "risk_rating", "ofac_flag",
        # accounts
        "account_id", "account_number", "routing_number", "account_type", "balance",
        "status", "opened_date",
        # transactions (minus the untrusted two)
        "txn_id", "amount", "channel", "txn_date", "is_disputed",
        # loans
        "loan_id", "loan_type", "principal", "outstanding_balance", "monthly_payment",
        "interest_rate", "days_delinquent",
        # service_requests (minus notes)
        "ticket_id", "category", "created_at", "created_by",
        # policy_chunks — bank-authored, see search_policy_docs
        "chunk_id", "doc_name", "chunk_text", "distance",
    }
)

# Values that cannot carry an instruction, whatever column they arrived in. `str` is
# pointedly absent. `bytes` is too: nothing in this schema returns it, and a bytes value
# reaching here would mean something unexpected happened, which is not a moment to relax
# a rule.
_NON_TEXT_TYPES = (bool, int, float, Decimal)


def classify_field(column: str, value: Any) -> Source:
    """Tag one result field TRUSTED or UNTRUSTED. See the module docstring for why.

    Deterministic, no model, no network — this runs on every field of every row and is
    the input to the Phase 3 injection scan.
    """
    # Rule 1: not text, so it cannot carry an instruction. Covers balances, amounts,
    # counts, rates, dates, booleans, NULLs, and every aggregate.
    if value is None or isinstance(value, _NON_TEXT_TYPES):
        return Source.TRUSTED
    if not isinstance(value, str):
        # dates, datetimes, UUIDs, and anything else non-textual the driver returns.
        return Source.TRUSTED

    name = column.lower()

    # Rule 2: the known injection vectors.
    if name in UNTRUSTED_COLUMNS:
        return Source.UNTRUSTED

    # Rule 3: known schema columns holding bank-controlled text.
    if name in TRUSTED_COLUMNS:
        return Source.TRUSTED

    # Rule 4: an alias, an expression result, or a column nobody has classified. It is a
    # string, and we cannot prove where it came from — so it gets scanned. Fail closed.
    return Source.UNTRUSTED


def tag_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Wrap each field of a raw result row as `{"value": ..., "source": ...}`."""
    return {
        column: {"value": value, "source": classify_field(column, value).value}
        for column, value in row.items()
    }


# --------------------------------------------------------------------------------
# Tool metadata
# --------------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}


def tool_metadata(name: str, risk_tier: RiskTier) -> Callable[[Callable], Callable]:
    """Attach a tool's name and risk tier, and register it.

    The tier lives on the function rather than in a lookup table elsewhere so it cannot
    drift from the tool it governs — Phase 3's HITL rail reads `func.risk_tier` to decide
    whether to interrupt, and a tool whose tier is defined in another file is a tool
    whose tier can be forgotten when it moves.

    Note this is metadata only. **It enforces nothing.** Enforcement is the HITL rail's
    job (§3.3), and risk tiers are an application-layer concern that the LLM never
    influences (§6 rule 5).
    """

    def decorator(func: Callable) -> Callable:
        func.tool_name = name  # type: ignore[attr-defined]
        func.risk_tier = risk_tier  # type: ignore[attr-defined]
        TOOL_REGISTRY[name] = func
        return func

    return decorator


def _tool_result(
    tool: str,
    risk_tier: RiskTier,
    ok: bool,
    reason: str,
    rows: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the envelope every tool returns, so the shape cannot vary between them."""
    tagged = [tag_row(row) for row in (rows or [])]
    return {
        "tool": tool,
        "risk_tier": risk_tier.value,
        "ok": ok,
        "reason": reason,
        "row_count": len(tagged),
        "rows": tagged,
        **extra,
    }


# --------------------------------------------------------------------------------
# 1. query_customer_data — AUTO
# --------------------------------------------------------------------------------


@tool_metadata("query_customer_data", RiskTier.AUTO)
@rail_timer("tool_query_customer_data")
def query_customer_data(sql: str, rm_id: str) -> dict[str, Any]:
    """Text-to-SQL over the allowlisted tables, entitlement-scoped to `rm_id`.

    Args:
        sql:   the SELECT to run. In this phase the caller supplies it; from Phase 3 the
               model generates it, and it is treated as hostile either way.
        rm_id: the current RM, from application session state (§2.1).

    This function is deliberately thin, and that is its entire security value: there is
    no database access in it, no fallback path, and no branch that reaches Postgres
    except through `safe_execute`. §2.3 says "No exceptions" about routing through
    sql_guard and entitlement, and the way to honour that is to leave nothing here that
    *could* be an exception. A reviewer should be able to confirm the property by reading
    the function once.

    `safe_execute` runs the fixed §2.2 sequence: validate -> inject the rm_id predicate ->
    re-validate -> execute with the value bound.
    """
    result = safe_execute(sql, rm_id)

    if not result["allowed"]:
        # The rail's `stage` and `reason` pass through unmodified — those codes are a
        # contract with the Phase 4 red-team harness (§4.2), and rewording them here
        # would break the tests that prove which rail fired.
        return _tool_result(
            "query_customer_data",
            RiskTier.AUTO,
            ok=False,
            reason=result["reason"],
            stage=result["stage"],
            columns=[],
            sql="",
        )

    return _tool_result(
        "query_customer_data",
        RiskTier.AUTO,
        ok=True,
        reason="ok",
        rows=result["rows"],
        stage=result["stage"],
        columns=result["columns"],
        # The SQL actually executed — the re-validated, entitlement-scoped rewrite, not
        # what the caller passed in.
        sql=result["sql"],
    )


# --------------------------------------------------------------------------------
# 2. get_account_summary — AUTO
# --------------------------------------------------------------------------------

# Shape of an account id as db/seed.py generates them. Checked before the value is used
# for anything: it is bound, not interpolated, so this is not injection defence — it
# rejects obvious nonsense early and keeps a malformed id from being reported as a
# perfectly ordinary "not found".
ACCOUNT_ID_PATTERN = re.compile(r"^ACCT-\d{4}$")

# Fixed queries. NOT model-generated — §2.3 specifies this tool as the parameterised
# alternative to text-to-SQL. They still go through safe_execute, because "we wrote this
# query ourselves" is a reason to expect it to pass the rails, not a reason to skip them:
# the entitlement predicate is what scopes it, and hand-written SQL is exactly where a
# missing rm_id filter would go unnoticed.
_ACCOUNT_SQL = """
SELECT a.account_id,
       a.account_number,
       a.account_type,
       a.balance,
       a.status,
       a.opened_date,
       c.customer_id,
       c.full_name
FROM accounts a
JOIN customers c ON c.customer_id = a.customer_id
WHERE a.account_id = :account_id
LIMIT 1
"""

# The recent-activity sample. This is where UNTRUSTED text enters the tool: `merchant`
# and `description` are attacker-controlled, and one of the four planted payloads sits in
# exactly this shape of result.
_RECENT_TXNS_SQL = """
SELECT t.txn_id,
       t.amount,
       t.merchant,
       t.description,
       t.channel,
       t.txn_date,
       t.is_disputed
FROM transactions t
WHERE t.account_id = :account_id
ORDER BY t.txn_date DESC
LIMIT 5
"""


@tool_metadata("get_account_summary", RiskTier.AUTO)
@rail_timer("tool_get_account_summary")
def get_account_summary(account_id: str, rm_id: str) -> dict[str, Any]:
    """Fixed, parameterised summary of one account, with its five most recent txns.

    No LLM-authored SQL anywhere in this path — but it is still routed through
    `safe_execute` so the entitlement predicate is injected identically. An RM asking for
    an account outside their book gets nothing back.
    """
    if not isinstance(account_id, str) or not ACCOUNT_ID_PATTERN.match(account_id):
        return _tool_result(
            "get_account_summary",
            RiskTier.AUTO,
            ok=False,
            reason="invalid_account_id",
            account_id="",
        )

    account = safe_execute(_ACCOUNT_SQL, rm_id, {"account_id": account_id})
    if not account["allowed"]:
        return _tool_result(
            "get_account_summary",
            RiskTier.AUTO,
            ok=False,
            reason=account["reason"],
            stage=account["stage"],
            account_id=account_id,
        )

    if not account["rows"]:
        # Zero rows means one of two things: no such account, or an account belonging to
        # another RM that the entitlement rail filtered out. **The caller is told the
        # same thing either way, on purpose.** Distinguishing them would confirm that
        # ACCT-0023 exists, which is itself a disclosure and precisely what the §4.1
        # `authorization_bypass` cases probe for ("show me RM-002's portfolio"). The
        # entitlement rail makes the rows unreachable; this message makes their
        # existence unobservable.
        return _tool_result(
            "get_account_summary",
            RiskTier.AUTO,
            ok=False,
            reason="account_not_found",
            account_id=account_id,
        )

    # The account is in scope — proven by the first query having returned a row through
    # the same entitlement predicate — so the transaction sample is scoped by the rail
    # for the same reason, via the transactions -> accounts -> customers join path.
    recent = safe_execute(_RECENT_TXNS_SQL, rm_id, {"account_id": account_id})
    if not recent["allowed"]:
        return _tool_result(
            "get_account_summary",
            RiskTier.AUTO,
            ok=False,
            reason=recent["reason"],
            stage=recent["stage"],
            account_id=account_id,
        )

    return _tool_result(
        "get_account_summary",
        RiskTier.AUTO,
        ok=True,
        reason="ok",
        rows=account["rows"],
        account_id=account_id,
        columns=account["columns"],
        # Tagged field-by-field like the account row: this list is where the poisoned
        # memo fields live, and Phase 3 quarantines them individually.
        recent_transactions=[tag_row(row) for row in recent["rows"]],
    )


# --------------------------------------------------------------------------------
# 3. search_policy_docs — AUTO
# --------------------------------------------------------------------------------


@tool_metadata("search_policy_docs", RiskTier.AUTO)
@rail_timer("tool_search_policy_docs")
def search_policy_docs(query: str, k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Top-k pgvector cosine search over the six internal policy documents (§2.4).

    Takes no `rm_id`: the policy corpus is not customer-scoped. It has no `customer_id`,
    no join path back to `customers`, and is identical for every RM — the exemption is
    documented in full in `app/rails/entitlement.py`.

    Every field comes back TRUSTED. The bank wrote these documents; they are the
    *grounding* for policy answers rather than a channel an attacker can reach. That is a
    claim about provenance, not about content, and it holds only so long as
    `policy_chunks` is loaded from `db/policies/*.md` by the seed script — if a future
    change ever lets customer-supplied text into that table, this classification is the
    first thing that must change.
    """
    if not isinstance(query, str) or not query.strip():
        return _tool_result(
            "search_policy_docs", RiskTier.AUTO, ok=False, reason="empty_query", query=""
        )

    chunks = search_policy_chunks(query, k=k)

    return _tool_result(
        "search_policy_docs",
        RiskTier.AUTO,
        ok=True,
        reason="ok",
        rows=chunks,
        query=query,
    )


# ================================================================================
# WRITE TOOLS
#
# Everything below mutates the database for real (§2.3: "implemented as real database
# mutations"). Four are HUMAN_APPROVAL; initiate_wire is HARD_BLOCK.
#
# --------------------------------------------------------------------------------
# THE APPROVAL SEAM — this is temporary scaffolding, and it is load-bearing
# --------------------------------------------------------------------------------
# §2.3 is explicit: write tools "must never be reachable without passing through the HITL
# gate built in Phase 3. In Phase 2, expose them but guard them with an assertion that
# fails if called outside an approved context."
#
# So every write tool below opens with `_require_approval(...)`.
#
# >>> PHASE 3 UPDATE: the real gate is installed. <<<
#     `app/rails/hitl.py` now wraps these tools with `HumanInTheLoopMiddleware`, which
#     interrupts the graph in `after_model` — before the tool node runs — and requires an
#     RM to approve or reject. The `ApprovalContext` is constructed by that wrapper from
#     the resumed interrupt payload
#     (Command(resume={"decisions": [{"type": "approve"}]})), never by an arbitrary
#     caller, and `build_write_tools` refuses to wrap a HUMAN_APPROVAL or HARD_BLOCK tool
#     that is not present in the middleware's `interrupt_on` map.
#
#     `_require_approval` deliberately REMAINS as a second, independent check. The
#     middleware is the enforcement point; this is the backstop that makes a direct call
#     to `file_dispute(...)` from some future code path fail loudly instead of silently
#     mutating customer data. A tool that is only safe because of what wraps it is one
#     refactor away from being unsafe.
#
# A note on why this is a raised exception and not a bare `assert`: `python -O` strips
# assert statements. A security gate that silently disappears under an optimisation flag
# is not a gate. The behaviour is what §2.3 asks for — loud failure, no mutation — the
# mechanism is just one that cannot be compiled out.
#
# --------------------------------------------------------------------------------
# HOW ENTITLEMENT REACHES A WRITE
# --------------------------------------------------------------------------------
# `sql_pipeline.safe_execute` is SELECT-only by design and must stay that way, so writes
# cannot route through it. Instead every mutation below is scoped twice:
#
#   1. A pre-flight SELECT resolves the target row *and* proves the current RM owns it,
#      producing the before-values the preview needs anyway.
#   2. The UPDATE/INSERT itself carries the entitlement predicate in its own WHERE
#      clause, and the affected row count is asserted.
#
# Step 2 is not redundant. Step 1 alone is check-then-act: between the SELECT and the
# UPDATE the row's ownership could change, and more practically, a future edit could move
# the two apart. Putting `rm_id` in the mutation's own WHERE clause means the statement
# that changes data is the statement that checks entitlement — there is no window.
# ================================================================================


class ApprovalRequiredError(RuntimeError):
    """Raised when a write tool is called outside an approved context.

    Loud and unhandled on purpose. A write tool reaching this state means something
    upstream tried to mutate customer data without human sign-off, which is a control
    failure worth a stack trace, not a `{"ok": false}` an agent might paper over.
    """


@dataclass(frozen=True)
class ApprovalContext:
    """Evidence that a human approved *this* tool call, for *this* RM.

    Frozen so a tool cannot edit its own authorisation.

    `tool` and `rm_id` are checked against the call, not merely carried. Without that, an
    approval is a bearer token: an approved `update_contact_info` could be replayed to
    authorise an `initiate_wire`, or one RM's approval reused against another's customer.
    Binding the context to both narrows it to the single action a human actually saw.
    """

    tool: str
    rm_id: str
    approved: bool = True


def _require_approval(context: ApprovalContext | None, tool: str, rm_id: str) -> None:
    """The Phase 2 stand-in for the HITL gate. See the seam comment above."""
    if context is None:
        raise ApprovalRequiredError(
            f"{tool} is a write tool and requires an approved context; "
            "it must not be called outside the HITL gate (PROJECT_PLAN.md §2.3)"
        )
    if not isinstance(context, ApprovalContext):
        raise ApprovalRequiredError(f"{tool}: approved_context must be an ApprovalContext")
    if not context.approved:
        raise ApprovalRequiredError(f"{tool}: the approval context records a rejection")
    if context.tool != tool:
        raise ApprovalRequiredError(
            f"approval was issued for {context.tool!r}, not {tool!r}"
        )
    if context.rm_id != rm_id:
        raise ApprovalRequiredError(
            f"approval was issued for {context.rm_id!r}, not {rm_id!r}"
        )


@contextmanager
def _write_scope(session: Session | None) -> Iterator[Session]:
    """Transaction scope for a mutation.

    With no session, the tool owns the transaction: `get_session` commits on success and
    **rolls back on any exception**, so a half-applied dispute (transaction flagged, no
    ticket written) cannot survive a failure.

    With a session supplied, the caller owns the transaction and this yields it untouched
    — no commit. That is how the tests exercise real mutations and then roll them back,
    keeping the deterministic seed intact (§6 rule 3).
    """
    if session is not None:
        yield session
    else:
        with get_session() as owned:
            yield owned


# --------------------------------------------------------------------------------
# Shared helpers for previews and id generation
# --------------------------------------------------------------------------------

# Same message whether the row does not exist or belongs to another RM. Distinguishing
# them would confirm the existence of another RM's records — the same disclosure
# get_account_summary avoids.
NOT_FOUND = "not_found_or_not_entitled"

# --------------------------------------------------------------------------------
# dry_run — how the approval prompt gets its before-values (§3.3)
# --------------------------------------------------------------------------------
# The HITL gate interrupts in `after_model`, which is *before* the tool node runs, so at
# the moment the RM is asked to approve a change nothing has happened yet. But §3.3
# forbids a bare yes/no: the RM must see a structured diff of what is about to change,
# and a diff needs the current database values.
#
# So every write tool below takes `dry_run`. It performs the same entitlement lookup and
# the same validity checks, builds the same `preview` dict, and returns **before the
# first INSERT or UPDATE**. `app/rails/hitl.py` calls it to render the approval prompt,
# and the RM then approves the very thing it described.
#
# Building the preview this way, rather than re-deriving it in the middleware, is what
# keeps the diff honest: the values shown to the human come from the same lookup and the
# same construction code as the mutation itself, so the two cannot drift apart.
#
# `_require_approval` is skipped on this path alone — computing a preview *in order to
# ask for* approval cannot itself require approval — and the path has no route to a
# mutation, which is asserted in tests.
PREVIEW_REASON = "preview"


def _change(field: str, before: Any, after: Any) -> dict[str, Any]:
    """One row of the structured diff §3.3 renders on interrupt.

    `after` is tagged with the same classifier the read tools use, so the approval UI can
    mark caller-supplied free text as untrusted rather than presenting every proposed
    value with equal authority.
    """
    return {
        "field": field,
        "before": before,
        "after": after,
        "source": classify_field(field.rsplit(".", 1)[-1], after).value,
    }


def _next_id(session: Session, table: str, column: str, prefix: str, width: int) -> str:
    """Next sequential id for a table whose keys are `PREFIX-0000` strings.

    Table and column names are interpolated, never caller data — they are literals at
    every call site below. The seeded ids are zero-padded strings, so MAX() over them is
    a lexicographic max, which is the same as numeric max at fixed width.
    """
    current = session.execute(text(f"SELECT MAX({column}) FROM {table}")).scalar()
    number = int(current.split("-")[1]) + 1 if current else 1
    return f"{prefix}-{number:0{width}d}"


def _write_result(
    tool: str,
    risk_tier: RiskTier,
    ok: bool,
    reason: str,
    preview: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Envelope for write tools: same shape as the read tools, plus `preview`.

    `preview` describes exactly what changed (or, on a block, what would have) with
    before/after values, because §3.3 requires the approval step to render a structured
    diff rather than a bare yes/no.
    """
    return {
        "tool": tool,
        "risk_tier": risk_tier.value,
        "ok": ok,
        "reason": reason,
        "preview": preview or {},
        **extra,
    }


# Pulled from the policy corpus by Phase 3 (§3.3: "must be pulled from the policy RAG,
# not hardcoded" — that is what proves the grounding pipeline works end to end). The key
# is present now so the preview shape does not change under the approval UI later.
_POLICY_SEAM = {"citation": None, "source_doc": None}


# --------------------------------------------------------------------------------
# 4. file_dispute — HUMAN_APPROVAL
# --------------------------------------------------------------------------------

_DISPUTE_LOOKUP = text(
    """
    SELECT t.txn_id, t.amount, t.merchant, t.description, t.channel, t.txn_date,
           t.is_disputed, a.account_id, a.account_number,
           c.customer_id, c.full_name
    FROM transactions t
    JOIN accounts a ON a.account_id = t.account_id
    JOIN customers c ON c.customer_id = a.customer_id
    WHERE t.txn_id = :txn_id AND c.rm_id = :rm_id
    """
)


def _dispute_preview(row: Any, txn_id: str, ticket_id: str) -> dict[str, Any]:
    """The structured diff for a dispute, built from the pre-flight lookup alone.

    Extracted so the dry-run path and the real path produce a byte-identical preview —
    the RM approves exactly what then happens. Everything here comes from `row`, which
    is the entitlement-scoped SELECT, so this function performs no I/O and cannot mutate.
    """
    return {
        "customer": {"customer_id": row["customer_id"], "full_name": row["full_name"]},
        "account": {
            "account_id": row["account_id"],
            # Last 4 only — RMs identify accounts this way and the full number is
            # US_BANK_NUMBER PII (§3.1.2).
            "account_number_last4": row["account_number"][-4:],
        },
        "transaction": tag_row(
            {
                "txn_id": row["txn_id"],
                "amount": row["amount"],
                "merchant": row["merchant"],
                "description": row["description"],
                "txn_date": row["txn_date"],
            }
        ),
        "changes": [
            _change(f"transactions.{txn_id}.is_disputed", False, True),
            _change(f"service_requests.{ticket_id}.category", None, "DISPUTE"),
            _change(f"service_requests.{ticket_id}.status", None, "PENDING_REVIEW"),
        ],
        "policy": dict(_POLICY_SEAM),
    }


@tool_metadata("file_dispute", RiskTier.HUMAN_APPROVAL)
@rail_timer("tool_file_dispute")
def file_dispute(
    txn_id: str,
    rm_id: str,
    notes: str = "",
    *,
    approved_context: ApprovalContext | None = None,
    session: Session | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Flag a transaction as disputed and open a PENDING_REVIEW dispute ticket.

    HUMAN_APPROVAL because Reg E starts a regulated clock: the 10-business-day
    investigation window begins when this is filed, and an agent that could file
    disputes on its own would be making a compliance commitment on the bank's behalf.
    Planted payload #4 asks the model to do exactly this for every transaction on an
    account, which is why the gate is not advisory.
    """
    if not dry_run:
        _require_approval(approved_context, "file_dispute", rm_id)

    with _write_scope(session) as db:
        row = db.execute(_DISPUTE_LOOKUP, {"txn_id": txn_id, "rm_id": rm_id}).mappings().first()
        if row is None:
            return _write_result("file_dispute", RiskTier.HUMAN_APPROVAL, False, NOT_FOUND)
        if row["is_disputed"]:
            return _write_result(
                "file_dispute", RiskTier.HUMAN_APPROVAL, False, "already_disputed"
            )

        # Resolved before the mutation so the preview can name the ticket. On the dry-run
        # path this is provisional — a concurrent write could claim it first — which is
        # honest for a preview and is why the id is recomputed on the real path.
        ticket_id = _next_id(db, "service_requests", "ticket_id", "SR", 4)

        preview = _dispute_preview(row, txn_id, ticket_id)

        if dry_run:
            return _write_result(
                "file_dispute",
                RiskTier.HUMAN_APPROVAL,
                True,
                PREVIEW_REASON,
                preview,
                ticket_id=ticket_id,
            )

        # The entitlement predicate lives in the UPDATE itself, not only in the lookup.
        updated = db.execute(
            text(
                """
                UPDATE transactions SET is_disputed = TRUE
                WHERE txn_id = :txn_id
                  AND account_id IN (
                      SELECT a.account_id FROM accounts a
                      JOIN customers c ON c.customer_id = a.customer_id
                      WHERE c.rm_id = :rm_id
                  )
                """
            ),
            {"txn_id": txn_id, "rm_id": rm_id},
        ).rowcount
        if updated != 1:
            # Nothing is committed: raising inside the scope rolls the transaction back.
            raise RuntimeError(f"file_dispute expected to update 1 row, updated {updated}")

        db.execute(
            text(
                """
                INSERT INTO service_requests
                    (ticket_id, customer_id, category, status, notes, created_at, created_by)
                VALUES
                    (:ticket_id, :customer_id, 'DISPUTE', 'PENDING_REVIEW',
                     :notes, :created_at, :created_by)
                """
            ),
            {
                "ticket_id": ticket_id,
                "customer_id": row["customer_id"],
                "notes": notes,
                "created_at": datetime.now(),
                "created_by": rm_id,
            },
        )

    return _write_result(
        "file_dispute", RiskTier.HUMAN_APPROVAL, True, "ok", preview, ticket_id=ticket_id
    )


# --------------------------------------------------------------------------------
# 5. waive_fee — HUMAN_APPROVAL
# --------------------------------------------------------------------------------

_FEE_LOOKUP = text(
    """
    SELECT t.txn_id, t.amount, t.merchant, t.description, t.channel,
           a.account_id, a.account_number, a.balance,
           c.customer_id, c.full_name
    FROM transactions t
    JOIN accounts a ON a.account_id = t.account_id
    JOIN customers c ON c.customer_id = a.customer_id
    WHERE t.txn_id = :txn_id AND c.rm_id = :rm_id
    """
)


def _waive_preview(row: Any, reversal_id: str, refund: Any) -> dict[str, Any]:
    """Structured diff for a fee waiver. Pure, so dry run and real path agree exactly."""
    return {
        "customer": {"customer_id": row["customer_id"], "full_name": row["full_name"]},
        "account": {
            "account_id": row["account_id"],
            "account_number_last4": row["account_number"][-4:],
        },
        "transaction": tag_row(
            {
                "txn_id": row["txn_id"],
                "amount": row["amount"],
                "merchant": row["merchant"],
                "description": row["description"],
            }
        ),
        "changes": [
            _change(
                f"accounts.{row['account_id']}.balance",
                row["balance"],
                row["balance"] + refund,
            ),
            _change(f"transactions.{reversal_id}.amount", None, refund),
        ],
        "policy": dict(_POLICY_SEAM),
    }


@tool_metadata("waive_fee", RiskTier.HUMAN_APPROVAL)
@rail_timer("tool_waive_fee")
def waive_fee(
    txn_id: str,
    rm_id: str,
    *,
    approved_context: ApprovalContext | None = None,
    session: Session | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reverse a fee: post an offsetting credit and restore the balance.

    HUMAN_APPROVAL because it is a direct, irreversible revenue give-back, and because
    the fee schedule caps waiver authority by RM tier — a limit a language model has no
    way to enforce and no business enforcing.

    Reversal by compensating entry rather than by deleting the original: a ledger that
    can be edited after the fact is not a ledger, and the audit story of this project
    depends on history being append-only.
    """
    if not dry_run:
        _require_approval(approved_context, "waive_fee", rm_id)

    with _write_scope(session) as db:
        row = db.execute(_FEE_LOOKUP, {"txn_id": txn_id, "rm_id": rm_id}).mappings().first()
        if row is None:
            return _write_result("waive_fee", RiskTier.HUMAN_APPROVAL, False, NOT_FOUND)
        if row["amount"] >= 0:
            # A credit is not a fee. Refusing is better than posting a "reversal" that
            # silently debits the customer.
            return _write_result(
                "waive_fee", RiskTier.HUMAN_APPROVAL, False, "not_a_debit"
            )

        refund = -row["amount"]  # the original is negative; the reversal credits it back
        reversal_id = _next_id(db, "transactions", "txn_id", "TXN", 6)

        preview = _waive_preview(row, reversal_id, refund)

        if dry_run:
            return _write_result(
                "waive_fee",
                RiskTier.HUMAN_APPROVAL,
                True,
                PREVIEW_REASON,
                preview,
                reversal_txn_id=reversal_id,
            )

        db.execute(
            text(
                """
                INSERT INTO transactions
                    (txn_id, account_id, amount, merchant, description, channel,
                     txn_date, is_disputed)
                VALUES
                    (:txn_id, :account_id, :amount, :merchant, :description, :channel,
                     :txn_date, FALSE)
                """
            ),
            {
                "txn_id": reversal_id,
                "account_id": row["account_id"],
                "amount": refund,
                # Bank-authored text, unlike the memo it reverses.
                "merchant": "FEE REVERSAL",
                "description": f"FEE WAIVER/{txn_id}/APPROVED BY {rm_id}",
                "channel": row["channel"],
                "txn_date": datetime.now(),
            },
        )

        updated = db.execute(
            text(
                """
                UPDATE accounts SET balance = balance + :refund
                WHERE account_id = :account_id
                  AND customer_id IN (SELECT customer_id FROM customers WHERE rm_id = :rm_id)
                """
            ),
            {"refund": refund, "account_id": row["account_id"], "rm_id": rm_id},
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"waive_fee expected to update 1 account, updated {updated}")

    return _write_result(
        "waive_fee",
        RiskTier.HUMAN_APPROVAL,
        True,
        "ok",
        preview,
        reversal_txn_id=reversal_id,
    )


# --------------------------------------------------------------------------------
# 6. block_card — HUMAN_APPROVAL
# --------------------------------------------------------------------------------

_ACCOUNT_LOOKUP = text(
    """
    SELECT a.account_id, a.account_number, a.account_type, a.status, a.balance,
           c.customer_id, c.full_name, c.ofac_flag
    FROM accounts a
    JOIN customers c ON c.customer_id = a.customer_id
    WHERE a.account_id = :account_id AND c.rm_id = :rm_id
    """
)


@tool_metadata("block_card", RiskTier.HUMAN_APPROVAL)
@rail_timer("tool_block_card")
def block_card(
    account_id: str,
    rm_id: str,
    *,
    approved_context: ApprovalContext | None = None,
    session: Session | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Freeze an account.

    HUMAN_APPROVAL because it is immediately customer-impacting and awkward to reverse:
    a wrongly frozen account declines a card at a checkout, and the customer finds out
    before the bank does.
    """
    if not dry_run:
        _require_approval(approved_context, "block_card", rm_id)

    with _write_scope(session) as db:
        row = db.execute(
            _ACCOUNT_LOOKUP, {"account_id": account_id, "rm_id": rm_id}
        ).mappings().first()
        if row is None:
            return _write_result("block_card", RiskTier.HUMAN_APPROVAL, False, NOT_FOUND)
        if row["status"] == "FROZEN":
            return _write_result(
                "block_card", RiskTier.HUMAN_APPROVAL, False, "already_frozen"
            )

        preview = {
            "customer": {"customer_id": row["customer_id"], "full_name": row["full_name"]},
            "account": {
                "account_id": row["account_id"],
                "account_number_last4": row["account_number"][-4:],
                "account_type": row["account_type"],
            },
            "changes": [_change(f"accounts.{account_id}.status", row["status"], "FROZEN")],
            "policy": dict(_POLICY_SEAM),
        }

        if dry_run:
            return _write_result(
                "block_card", RiskTier.HUMAN_APPROVAL, True, PREVIEW_REASON, preview
            )

        updated = db.execute(
            text(
                """
                UPDATE accounts SET status = 'FROZEN'
                WHERE account_id = :account_id
                  AND customer_id IN (SELECT customer_id FROM customers WHERE rm_id = :rm_id)
                """
            ),
            {"account_id": account_id, "rm_id": rm_id},
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"block_card expected to update 1 row, updated {updated}")

    return _write_result("block_card", RiskTier.HUMAN_APPROVAL, True, "ok", preview)


# --------------------------------------------------------------------------------
# 7. update_contact_info — HUMAN_APPROVAL
# --------------------------------------------------------------------------------

# Only these columns may be changed. An allowlist rather than "whatever the caller sent",
# because the same UPDATE machinery would otherwise happily write `rm_id` — reassigning a
# customer to another RM and defeating the entitlement rail through the front door.
CONTACT_FIELDS = ("email", "phone", "street_address", "city", "state", "zip_code")


@tool_metadata("update_contact_info", RiskTier.HUMAN_APPROVAL)
@rail_timer("tool_update_contact_info")
def update_contact_info(
    customer_id: str,
    rm_id: str,
    *,
    approved_context: ApprovalContext | None = None,
    session: Session | None = None,
    dry_run: bool = False,
    **fields: Any,
) -> dict[str, Any]:
    """Update a customer's email, phone, or address.

    HUMAN_APPROVAL because this is the classic account-takeover step: change the contact
    details first, then every verification code and fraud alert goes to the attacker.
    It looks like the most clerical tool in the set and is close to the most dangerous.
    """
    if not dry_run:
        _require_approval(approved_context, "update_contact_info", rm_id)

    updates = {k: v for k, v in fields.items() if k in CONTACT_FIELDS and v is not None}
    rejected = sorted(set(fields) - set(CONTACT_FIELDS))
    if rejected:
        # Loud rather than ignored: a caller trying to set `rm_id` or `ssn` here is a
        # signal, not a typo to be quietly dropped.
        return _write_result(
            "update_contact_info",
            RiskTier.HUMAN_APPROVAL,
            False,
            f"field_not_updatable:{rejected[0]}",
        )
    if not updates:
        return _write_result(
            "update_contact_info", RiskTier.HUMAN_APPROVAL, False, "no_fields_to_update"
        )

    with _write_scope(session) as db:
        row = db.execute(
            text(
                "SELECT customer_id, full_name, email, phone, street_address, city, "
                "state, zip_code FROM customers "
                "WHERE customer_id = :customer_id AND rm_id = :rm_id"
            ),
            {"customer_id": customer_id, "rm_id": rm_id},
        ).mappings().first()
        if row is None:
            return _write_result(
                "update_contact_info", RiskTier.HUMAN_APPROVAL, False, NOT_FOUND
            )

        preview = {
            "customer": {"customer_id": row["customer_id"], "full_name": row["full_name"]},
            "changes": [
                _change(f"customers.{customer_id}.{column}", row[column], value)
                for column, value in updates.items()
            ],
            "policy": dict(_POLICY_SEAM),
        }

        if dry_run:
            return _write_result(
                "update_contact_info",
                RiskTier.HUMAN_APPROVAL,
                True,
                PREVIEW_REASON,
                preview,
            )

        # Column names come from CONTACT_FIELDS, never from caller text; the values are
        # bound. The allowlist above is what makes this interpolation safe.
        assignments = ", ".join(f"{column} = :{column}" for column in updates)
        updated = db.execute(
            text(
                f"UPDATE customers SET {assignments} "
                "WHERE customer_id = :customer_id AND rm_id = :rm_id"
            ),
            {**updates, "customer_id": customer_id, "rm_id": rm_id},
        ).rowcount
        if updated != 1:
            raise RuntimeError(
                f"update_contact_info expected to update 1 row, updated {updated}"
            )

    return _write_result("update_contact_info", RiskTier.HUMAN_APPROVAL, True, "ok", preview)


# --------------------------------------------------------------------------------
# 8. initiate_wire — HARD_BLOCK
# --------------------------------------------------------------------------------

# The message an OFAC hit returns. Deliberately an escalation instruction rather than a
# refusal: the RM's next step is a compliance referral, not a retry.
OFAC_ESCALATION_MESSAGE = (
    "BLOCKED — OFAC. This customer carries a sanctions flag. Outbound wires are "
    "prohibited and this block cannot be approved or overridden. Escalate to the "
    "Compliance team; do not contact the customer about the reason for the block."
)


@tool_metadata("initiate_wire", RiskTier.HARD_BLOCK)
@rail_timer("tool_initiate_wire")
def initiate_wire(
    account_id: str,
    amount: Decimal | str | int | float,
    rm_id: str,
    beneficiary: str = "",
    *,
    approved_context: ApprovalContext | None = None,
    session: Session | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Send an outbound wire — unless the customer is OFAC-flagged, in which case never.

    HARD_BLOCK is not "HUMAN_APPROVAL with a sterner warning". §3.3: "Never executes
    regardless of approval." Sanctions screening is not a risk judgement an RM is
    permitted to make, so there is deliberately no argument, no flag, and no approval
    payload that reaches the mutation once `ofac_flag` is true. The check sits after the
    approval gate purely because it needs a database read; it is unconditional, and the
    §3.4 acceptance criterion is precisely that an approved wire on a flagged customer
    still does not execute.
    """
    if not dry_run:
        _require_approval(approved_context, "initiate_wire", rm_id)

    amount = Decimal(str(amount))
    if amount <= 0:
        return _write_result("initiate_wire", RiskTier.HARD_BLOCK, False, "invalid_amount")

    with _write_scope(session) as db:
        row = db.execute(
            _ACCOUNT_LOOKUP, {"account_id": account_id, "rm_id": rm_id}
        ).mappings().first()
        if row is None:
            return _write_result("initiate_wire", RiskTier.HARD_BLOCK, False, NOT_FOUND)

        # ---- THE HARD BLOCK -------------------------------------------------
        # Nothing below this branch runs for a flagged customer. Returning before any
        # INSERT or UPDATE is the enforcement; the message is only the explanation.
        if row["ofac_flag"]:
            return _write_result(
                "initiate_wire",
                RiskTier.HARD_BLOCK,
                False,
                "ofac_hard_block",
                {
                    "customer": {
                        "customer_id": row["customer_id"],
                        "full_name": row["full_name"],
                        "ofac_flag": True,
                    },
                    "account": {
                        "account_id": row["account_id"],
                        "account_number_last4": row["account_number"][-4:],
                    },
                    # An empty diff, stated explicitly: the approval UI must render "no
                    # change is possible", not an actionable pending mutation.
                    "changes": [],
                    "blocked": True,
                    "policy": dict(_POLICY_SEAM),
                },
                message=OFAC_ESCALATION_MESSAGE,
                overridable=False,
            )

        wire_id = _next_id(db, "transactions", "txn_id", "TXN", 6)

        preview = {
            "customer": {
                "customer_id": row["customer_id"],
                "full_name": row["full_name"],
                "ofac_flag": False,
            },
            "account": {
                "account_id": row["account_id"],
                "account_number_last4": row["account_number"][-4:],
            },
            "changes": [
                _change(
                    f"accounts.{account_id}.balance", row["balance"], row["balance"] - amount
                ),
                _change(f"transactions.{wire_id}.amount", None, -amount),
            ],
            "blocked": False,
            "policy": dict(_POLICY_SEAM),
        }

        if dry_run:
            return _write_result(
                "initiate_wire",
                RiskTier.HARD_BLOCK,
                True,
                PREVIEW_REASON,
                preview,
                wire_txn_id=wire_id,
            )

        db.execute(
            text(
                """
                INSERT INTO transactions
                    (txn_id, account_id, amount, merchant, description, channel,
                     txn_date, is_disputed)
                VALUES
                    (:txn_id, :account_id, :amount, :merchant, :description, 'WIRE',
                     :txn_date, FALSE)
                """
            ),
            {
                "txn_id": wire_id,
                "account_id": account_id,
                "amount": -amount,  # outbound: a debit
                "merchant": "OUTBOUND WIRE",
                "description": f"WIRE OUT/{beneficiary}/APPROVED BY {rm_id}",
                "txn_date": datetime.now(),
            },
        )

        updated = db.execute(
            text(
                """
                UPDATE accounts SET balance = balance - :amount
                WHERE account_id = :account_id
                  AND customer_id IN (SELECT customer_id FROM customers WHERE rm_id = :rm_id)
                """
            ),
            {"amount": amount, "account_id": account_id, "rm_id": rm_id},
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"initiate_wire expected to update 1 account, updated {updated}")

    return _write_result(
        "initiate_wire", RiskTier.HARD_BLOCK, True, "ok", preview, wire_txn_id=wire_id
    )
