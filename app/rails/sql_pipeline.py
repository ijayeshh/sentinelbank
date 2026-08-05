"""The single entry point from LLM-authored SQL to the customer database (§2.1, §2.2).

--------------------------------------------------------------------------------
WHY THIS MODULE EXISTS AT ALL
--------------------------------------------------------------------------------
`sql_guard` and `entitlement` are each correct on their own and each useless on their
own. The security property only holds if they run **in this order, every time, with no
path around them**:

    1. validate_sql(raw)              — is this a single read-only SELECT over
                                        allowlisted tables, with a bounded LIMIT?
    2. enforce_entitlement(ast)       — rewrite it so it cannot see another RM's rows
    3. validate_sql(rewritten)        — is what came out of step 2 still all of that?
    4. execute                        — with rm_id bound, never interpolated

§2.2 states the ordering explicitly ("validate -> then inject the entitlement predicate
-> then re-validate the final string"), and this module is that sentence made
executable. Callers — `query_customer_data` in §2.3, and nothing else — get one
function. There is deliberately no exported helper that runs steps 1 and 2 without 3,
because the way this control fails in practice is not that someone disables it; it is
that a later feature needs "just the validated SQL" and quietly grows a second path to
the database.

**Never execute unvalidated SQL under any circumstance.** In code, that reduces to one
invariant a reviewer can check by eye: there is exactly one call to the database in
this file, it is guarded by an assertion, and the string it executes is
`revalidated.safe_sql` — sqlglot's regeneration of the tree that step 3 accepted. Not
the model's original text, not the entitlement rail's output. If step 3 did not run,
that variable does not exist.

--------------------------------------------------------------------------------
WHY STEP 3 IS NOT PARANOIA
--------------------------------------------------------------------------------
Step 2 is a *mutation*: it adds joins, adds WHERE predicates, rewrites `*` into
explicit column lists, and regenerates the SQL text. It is ~400 lines of AST surgery
operating on input an attacker influences. Re-validating its output is what makes the
composition safe rather than merely the parts:

  * the tables entitlement injects (`customers`, `accounts`) are re-checked against
    sql_guard's allowlist rather than trusted because this file added them;
  * the LIMIT clamp is re-asserted after the joins widened the row source;
  * a future change to entitlement — a new join path, a new rewrite — cannot introduce
    a forbidden table or statement shape without step 3 catching it, which means the
    two rails can evolve independently without either one having to know the other's
    internals.

The `stage` field in the returned dict names which step rejected, so an operator
reading the audit log can tell "the model wrote something unsafe" (validate) from "the
model wrote something we could not scope" (entitlement) from "our own rewrite produced
something unsafe" (revalidate). The third is a bug in this project, not an attack, and
it should be impossible to confuse the two.

--------------------------------------------------------------------------------
LATENCY AND ATTRIBUTION
--------------------------------------------------------------------------------
§2.5's `@rail_timer` attaches to the three `_stage_*` helpers below rather than to
`validate_sql` and `enforce_entitlement` themselves. That placement is deliberate:
both of those modules state that they carry no dependency on timing, config or the
database so they can be unit-tested with no services running, and decorating them in
place would break that. Wrapping them here keeps the isolation and still fills the
`sql_guard` and `entitlement` rows of §4.3's latency budget table.

Note that **`sql_guard` is sampled twice per query** — once for step 1 and once for
step 3 — because the rail genuinely runs twice. Summing its samples per request is
therefore the honest per-request cost, not a double count.

The outcome ring buffer below exists for the same reason `injection.py` keeps its
incidents on the middleware instance: §3.6's `/chat` has to report which rails fired,
and these two run inside a tool rather than inside a middleware, so nothing about them
reaches agent state. Without the buffer, a blocked `DROP TABLE` or a scoped-down
portfolio query is attributable only in the audit log — which is exactly the thing
§4.2's `expected_rail` assertion cannot read.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlglot.errors import SqlglotError

from app.db import get_session
from app.rails.entitlement import ENTITLEMENT_PARAM, EntitlementVerdict, enforce_entitlement
from app.rails.sql_guard import DIALECT, SqlVerdict, validate_sql
from app.rails.timing import rail_timer

# §4.3's table names for these two rails. Stable strings: the latency report groups on
# them and the red-team harness asserts `expected_rail` against them.
RAIL_SQL_GUARD = "sql_guard"
RAIL_ENTITLEMENT = "entitlement"

# The stage names that appear in the returned dict, and in the audit log after it.
# Stable strings — the Phase 4 harness asserts on them (§4.2).
STAGE_VALIDATE = "validate"  # step 1: sql_guard on the model's raw SQL
STAGE_ENTITLEMENT = "entitlement"  # step 2: rm_id predicate injection
STAGE_REVALIDATE = "revalidate"  # step 3: sql_guard on the rewritten SQL
STAGE_EXECUTE = "execute"  # step 4: the database


# ================================================================================
# Timed stages (§2.5, §4.3)
# ================================================================================


@rail_timer(RAIL_SQL_GUARD)
def _stage_validate(sql: str) -> SqlVerdict:
    """Step 1: sql_guard on the model's raw SQL."""
    return validate_sql(sql)


@rail_timer(RAIL_ENTITLEMENT)
def _stage_entitlement(tree: Any, rm_id: str) -> EntitlementVerdict:
    """Step 2: rm_id predicate injection."""
    return enforce_entitlement(tree, rm_id)


@rail_timer(RAIL_SQL_GUARD)
def _stage_revalidate(sql: str) -> SqlVerdict:
    """Step 3: sql_guard again, on what the rewrite produced.

    Timed under the same name as step 1 on purpose. It is the same rail doing the same
    work; splitting it into `sql_guard_revalidate` would let a reader of §4.3's table
    conclude the rail costs half what it does.
    """
    return validate_sql(sql)


# ================================================================================
# Outcome buffer — so `/chat` can attribute these two rails (§3.6)
# ================================================================================

# Bounded for the same reason `timing.py`'s sample buffer is: a long-running demo must
# not grow the process without limit on an 8 GB machine. One entry per `safe_execute`,
# and a single turn makes at most a handful of tool calls.
_OUTCOME_BUFFER_SIZE = 256

_OUTCOMES: deque[dict[str, Any]] = deque(maxlen=_OUTCOME_BUFFER_SIZE)


def _record_outcome(
    allowed: bool, stage: str, reason: str, predicates: int = 0
) -> None:
    """Note what the pipeline decided, for the response assembler to read.

    `predicates` is how many entitlement predicates the rewrite injected — one per
    scoped SELECT. Zero is not a failure: a pure `policy_chunks` RAG query has nothing
    customer-owned in it and is correctly left alone. It is recorded so `/chat` can tell
    "the rail ran and had nothing to scope" from "the rail rewrote this query", which is
    the difference between a rail that is working and a rail that is a no-op.
    """
    _OUTCOMES.append(
        {"allowed": allowed, "stage": stage, "reason": reason, "predicates": predicates}
    )


def get_sql_outcomes() -> list[dict[str, Any]]:
    """Pipeline outcomes recorded so far, oldest first."""
    return list(_OUTCOMES)


def clear_sql_outcomes() -> None:
    """Empty the buffer. For tests, and for a clean red-team run."""
    _OUTCOMES.clear()


def _rejected(stage: str, reason: str) -> dict[str, Any]:
    """Build a rejection.

    Every rejection funnels through here so that no partial result — no SQL string, no
    bind parameters, no half-filled row list — can escape alongside `allowed: False`.
    A caller that ignores the flag still gets nothing usable. It is also the one place a
    rejection is recorded, so no future early-return can forget to report itself.
    """
    _record_outcome(False, stage, reason)
    return {
        "allowed": False,
        "stage": stage,
        "reason": reason,
        "sql": "",
        "params": {},
        "columns": [],
        "rows": [],
        "row_count": 0,
    }


def safe_execute(
    sql: str, rm_id: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate, scope, re-validate, and execute LLM-authored SQL.

    Args:
        sql:    SQL as written by the model. Assumed hostile: it may have been steered
                by an injected payload sitting in `transactions.description`.
        rm_id:  the current RM, **from application session state**. Never from model
                output, never from user text (§2.1, §6 rule 5).
        params: additional bind values for placeholders the *caller* put in `sql`. This
                exists for the parameterised, non-LLM-generated tools in §2.3 —
                `get_account_summary` binds `:account_id` — so that they can reuse this
                pipeline verbatim instead of growing a second path to the database.
                Values are bound by the driver and never formatted into the SQL text.

    Returns:
        A dict, always with the same keys:

          allowed    bool  — whether the query reached the database
          stage      str   — which of the four steps produced this verdict
          reason     str   — "ok", or the rejecting rail's reason code
          sql        str   — the SQL actually executed; "" if rejected
          params     dict  — every bound value, caller's plus {"current_rm": rm_id};
                             {} if rejected
          columns    list  — result column names; [] if rejected
          rows       list  — list of dicts, one per row; [] if rejected
          row_count  int   — len(rows)

        On rejection the function returns before touching the database. There is no
        code path that executes anything on a rejection.
    """
    # --- Step 0: the caller may not supply the entitlement value -----------------
    # `current_rm` is the entitlement rail's own placeholder. If a caller could set it,
    # identity would come from the caller rather than from session state, which is the
    # single property §2.1 exists to guarantee — and the merge below would silently let
    # the caller's value win. Reject rather than merge-and-hope. Fail closed.
    caller_params = dict(params or {})
    if ENTITLEMENT_PARAM in caller_params:
        return _rejected(STAGE_VALIDATE, "param_shadows_entitlement")

    # --- Step 1: validate the model's raw SQL -----------------------------------
    # Nothing about the input is trusted yet, including that it is a SELECT, that it is
    # one statement, or that it is SQL at all.
    guard = _stage_validate(sql)
    if not guard.allowed:
        return _rejected(STAGE_VALIDATE, guard.reason)

    # --- Step 2: inject the entitlement predicate -------------------------------
    # Parse sql_guard's *regenerated* output rather than the caller's original string.
    # Re-parsing the original would reintroduce whatever step 1 normalised away and
    # would mean the tree we rewrite is not the tree we validated.
    try:
        tree = sqlglot.parse_one(guard.safe_sql, dialect=DIALECT)
    except SqlglotError:
        # Unreachable in practice: this string was produced by sqlglot's own generator
        # moments ago. If it ever happens, our model of the SQL has broken down and the
        # only safe response is to stop. Fail closed (§6 rule 4).
        return _rejected(STAGE_ENTITLEMENT, "unparseable_validated_sql")
    if tree is None:
        return _rejected(STAGE_ENTITLEMENT, "unparseable_validated_sql")

    entitlement = _stage_entitlement(tree, rm_id)
    if not entitlement.allowed:
        return _rejected(STAGE_ENTITLEMENT, entitlement.reason)

    # One `:current_rm` placeholder per scoped SELECT. Counted off the rewritten SQL
    # rather than tracked through the rail, so it measures what will actually be
    # executed rather than what the rewrite believes it did.
    predicates = entitlement.sql.count(f":{ENTITLEMENT_PARAM}")

    # --- Step 3: re-validate what the rewrite produced --------------------------
    # See the module docstring. Step 2 mutated the tree and regenerated the text; this
    # is the check that the result is still a single read-only SELECT over allowlisted
    # tables with a bounded LIMIT. It runs on `entitlement.sql`, never on `guard.safe_sql`.
    revalidated = _stage_revalidate(entitlement.sql)
    if not revalidated.allowed:
        # Reaching here means the entitlement rail emitted something sql_guard refuses
        # — a defect in this project rather than an attack. It is still handled the
        # same way: refuse, and do not execute.
        return _rejected(STAGE_REVALIDATE, revalidated.reason)

    # --- Step 4: execute --------------------------------------------------------
    # The one and only database call in this module. The assertion is not decoration:
    # it pins the executed string to step 3's output, so a future edit that reorders
    # the steps or reaches for `guard.safe_sql` here fails loudly instead of quietly
    # executing SQL that was never re-validated.
    assert revalidated.allowed and revalidated.safe_sql, "step 3 must precede execution"
    final_sql = revalidated.safe_sql

    # The entitlement value is applied last, so it wins over anything in the caller's
    # dict. Step 0 already rejects that key, making this belt-and-braces rather than the
    # only defence — but the ordering should not be something a reader has to infer.
    bind_values = {**caller_params, **entitlement.params}

    try:
        with get_session() as session:
            # rm_id travels as a bound parameter, beside the SQL and never inside it.
            # `entitlement.params` is {"current_rm": rm_id}; the placeholder may appear
            # several times in the statement (once per scoped SELECT) and one bound
            # value satisfies them all.
            result = session.execute(text(final_sql), bind_values)
            columns = list(result.keys())
            # Rows become plain dicts so callers are not handed live driver objects.
            # Note: a query selecting the same column name from two tables collapses
            # here — harmless for the agent's read paths, and sql_guard has already
            # capped the result at 50 rows either way.
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
    except Exception:
        # Broad on purpose. A driver error, a type conversion failure, a dropped
        # connection — none of them are conditions under which this function should
        # return anything that looks like data.
        #
        # The driver's message is deliberately NOT propagated. Postgres echoes offending
        # values back in its errors, so a message can contain customer PII, and this
        # project keeps PII out of logs by construction (§1.5 audit_log notes). The
        # caller gets the fact of the failure; the values stay in the database.
        return _rejected(STAGE_EXECUTE, "execution_error")

    _record_outcome(True, STAGE_EXECUTE, "ok", predicates)
    return {
        "allowed": True,
        "stage": STAGE_EXECUTE,
        "reason": "ok",
        "sql": final_sql,
        # Everything that was actually bound, caller values included — this is what the
        # audit log should record about the executed statement.
        "params": bind_values,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }
