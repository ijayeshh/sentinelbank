"""End-to-end tests for the SQL pipeline (PROJECT_PLAN.md §2.1, §2.2, acceptance §2.6).

Two things are being proved here, and the second is the harder one.

**The path works.** Model-written SQL goes in, entitlement-scoped rows come out, and an
RM-001 query never returns an RM-002 row — asserted against the real seeded database,
per §2.6.

**The order is real.** §2.2 fixes the sequence as validate -> inject -> *re-validate* ->
execute. Step 3 is the one that is easy to write and easy to accidentally not run,
because in normal operation it never rejects anything: every legitimate query passes it
silently, so a pipeline missing step 3 entirely looks perfectly healthy. The tests in
"Step 3 actually runs" below are built so that they can only pass if it executes —
by making the entitlement rail emit something unsafe and requiring the pipeline to
catch it, and by pinning the executed string to step 3's output.

Requires `docker compose up -d` and `python db/seed.py`. These tests fail rather than
skip when Postgres is unreachable: a security control whose proof reports green when it
did not run is worse than no proof.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.db import get_session
from app.rails import sql_pipeline
from app.rails.entitlement import ENTITLEMENT_PARAM, EntitlementVerdict
from app.rails.sql_pipeline import (
    STAGE_ENTITLEMENT,
    STAGE_EXECUTE,
    STAGE_REVALIDATE,
    STAGE_VALIDATE,
    safe_execute,
)

# Seeded distribution, fixed by Faker.seed(42) in db/seed.py (Phase 1 acceptance).
RM1, RM2 = "RM-001", "RM-002"
CUSTOMERS_RM1, CUSTOMERS_RM2 = 22, 18
POLICY_CHUNKS = 45

# sql_guard clamps at 50, so fixtures ask for 50 to let the entitlement rail — rather
# than the default LIMIT 20 — be what shapes the result.
ALL_CUSTOMERS = "SELECT c.customer_id, c.rm_id FROM customers c LIMIT 50"

REJECTED_SHAPE = {"sql": "", "params": {}, "columns": [], "rows": [], "row_count": 0}


@pytest.fixture(scope="module", autouse=True)
def require_seeded_database():
    """Fail loudly, once, if the seeded database is not there."""
    try:
        with get_session() as session:
            counts = dict(
                session.execute(
                    text("SELECT rm_id, COUNT(*) FROM customers GROUP BY rm_id")
                ).all()
            )
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "pipeline tests require the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )
    assert counts.get(RM1) == CUSTOMERS_RM1 and counts.get(RM2) == CUSTOMERS_RM2, (
        f"seed data has drifted (got {counts}); re-run python db/seed.py"
    )


def customer_ids_of(rm_id: str) -> set[str]:
    """Ground truth, read straight from the table with no rail involved."""
    with get_session() as session:
        return {
            row[0]
            for row in session.execute(
                text("SELECT customer_id FROM customers WHERE rm_id = :rm"), {"rm": rm_id}
            ).all()
        }


# --------------------------------------------------------------------------------
# The happy path, end to end
# --------------------------------------------------------------------------------


def test_full_pipeline_returns_only_the_callers_own_customers():
    result = safe_execute(ALL_CUSTOMERS, RM1)

    assert result["allowed"] is True
    assert result["stage"] == STAGE_EXECUTE
    assert result["reason"] == "ok"
    assert result["row_count"] == CUSTOMERS_RM1 == len(result["rows"])
    assert result["columns"] == ["customer_id", "rm_id"]

    # Every row is positively RM-001's, and none is RM-002's.
    assert {row["rm_id"] for row in result["rows"]} == {RM1}
    assert {row["customer_id"] for row in result["rows"]}.isdisjoint(customer_ids_of(RM2))


def test_the_same_query_partitions_between_the_two_rms():
    seen_by_rm1 = {r["customer_id"] for r in safe_execute(ALL_CUSTOMERS, RM1)["rows"]}
    seen_by_rm2 = {r["customer_id"] for r in safe_execute(ALL_CUSTOMERS, RM2)["rows"]}

    assert len(seen_by_rm2) == CUSTOMERS_RM2
    assert seen_by_rm1.isdisjoint(seen_by_rm2)
    assert len(seen_by_rm1 | seen_by_rm2) == CUSTOMERS_RM1 + CUSTOMERS_RM2


def test_child_table_query_is_scoped_through_the_injected_join():
    """A query naming no customer-owned key column at all still gets scoped."""
    result = safe_execute("SELECT t.txn_id, t.description FROM transactions t LIMIT 50", RM1)

    assert result["allowed"] is True
    # transactions -> accounts -> customers, both hops added by the entitlement rail.
    assert "JOIN accounts" in result["sql"] and "JOIN customers" in result["sql"]

    other = safe_execute("SELECT t.txn_id, t.description FROM transactions t LIMIT 50", RM2)
    assert {r["txn_id"] for r in result["rows"]}.isdisjoint(
        {r["txn_id"] for r in other["rows"]}
    )


def test_policy_chunks_needs_no_scoping_and_still_executes():
    """The RAG corpus is exempt (§2.4). Exempt must still mean *runnable*."""
    result = safe_execute("SELECT p.doc_name FROM policy_chunks p LIMIT 50", RM1)

    assert result["allowed"] is True
    # The whole corpus, identical for every RM (45 chunks from the 6 policy docs).
    assert result["row_count"] == POLICY_CHUNKS
    # No predicate was injected, so the placeholder is absent from the SQL. The bind
    # value is still returned — it describes who ran the query, and an unused bind is
    # harmless — but nothing in the statement consumes it.
    assert ENTITLEMENT_PARAM not in result["sql"]
    assert "JOIN" not in result["sql"].upper()


def test_rm_id_is_bound_not_interpolated():
    """§2.1: a bound parameter. The identity never appears in the SQL text."""
    result = safe_execute(ALL_CUSTOMERS, RM1)

    assert RM1 not in result["sql"]
    assert f":{ENTITLEMENT_PARAM}" in result["sql"]
    assert result["params"] == {ENTITLEMENT_PARAM: RM1}


def test_sql_asking_for_the_other_rms_book_returns_nothing():
    """The §4.1 authorization-bypass shape, run all the way to the database.

    The injected predicate is ANDed, so a caller-supplied `rm_id` filter can only
    narrow the result, never widen it.
    """
    result = safe_execute(
        f"SELECT c.customer_id FROM customers c WHERE c.rm_id = '{RM2}' LIMIT 50", RM1
    )
    assert result["allowed"] is True
    assert result["rows"] == []


def test_limit_is_still_enforced_after_the_rewrite():
    """The clamp has to survive step 2 widening the row source. Step 3 re-checks it."""
    result = safe_execute("SELECT t.txn_id FROM transactions t LIMIT 5000", RM1)

    assert result["allowed"] is True
    assert "LIMIT 50" in result["sql"]
    assert result["row_count"] <= 50


# --------------------------------------------------------------------------------
# Rejections — and never executing on one
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE customers", "non_select_statement:Drop"),
        ("SELECT * FROM audit_log", "audit_log_access"),
        ("SELECT c.customer_id FROM customers c; DROP TABLE customers", "semicolon_separator"),
        ("SELECT c.customer_id FROM customers c -- payload", "sql_comment"),
        ("SELECT table_name FROM information_schema.tables", "system_catalog_access"),
    ],
)
def test_step_1_rejections_are_reported_at_the_validate_stage(sql, reason):
    result = safe_execute(sql, RM1)

    assert result["allowed"] is False
    assert result["stage"] == STAGE_VALIDATE
    assert result["reason"] == reason
    # A rejection carries no SQL, no bind values, and no rows — a caller that ignores
    # the flag still gets nothing usable.
    assert {k: result[k] for k in REJECTED_SHAPE} == REJECTED_SHAPE


def test_unscopeable_query_is_rejected_at_the_entitlement_stage():
    """Passes sql_guard cleanly; the cross join has no provable path to customers."""
    result = safe_execute(
        "SELECT c.customer_id, l.loan_id FROM customers c, loans l LIMIT 50", RM1
    )

    assert result["allowed"] is False
    assert result["stage"] == STAGE_ENTITLEMENT
    assert result["reason"] == "unlinked_scoped_table:l"
    assert {k: result[k] for k in REJECTED_SHAPE} == REJECTED_SHAPE


def test_malformed_rm_id_is_rejected_before_execution():
    """An unset or wrong-shaped session value must not reach the database."""
    result = safe_execute(ALL_CUSTOMERS, "")

    assert result["allowed"] is False
    assert result["stage"] == STAGE_ENTITLEMENT
    assert result["reason"] == "invalid_rm_id"


def test_caller_params_are_bound_alongside_the_entitlement_value():
    """The parameterised tools in §2.3 reuse this pipeline rather than growing a second
    path to the database, so it has to carry their bind values too."""
    result = safe_execute(
        "SELECT a.account_id, a.balance FROM accounts a WHERE a.account_id = :account_id",
        RM1,
        {"account_id": "ACCT-0003"},
    )

    assert result["allowed"] is True
    assert result["row_count"] == 1
    assert result["rows"][0]["account_id"] == "ACCT-0003"
    # Both values were bound; neither was formatted into the SQL text.
    assert result["params"] == {ENTITLEMENT_PARAM: RM1, "account_id": "ACCT-0003"}
    assert "ACCT-0003" not in result["sql"] and RM1 not in result["sql"]


def test_caller_params_cannot_reach_an_out_of_scope_row():
    """A bound account_id is still subject to the entitlement predicate."""
    result = safe_execute(
        "SELECT a.account_id FROM accounts a WHERE a.account_id = :account_id",
        RM1,
        {"account_id": "ACCT-0023"},  # belongs to RM-002
    )
    assert result["allowed"] is True
    assert result["rows"] == []


def test_a_caller_cannot_shadow_the_entitlement_parameter(monkeypatch):
    """The one bind value no caller may supply.

    If `current_rm` could be passed in, identity would come from the caller rather than
    from session state — the single property §2.1 exists to guarantee. Rejected before
    anything else happens, and without touching the database.
    """

    def tripwire():
        raise AssertionError("a shadowing attempt opened a database session")

    monkeypatch.setattr(sql_pipeline, "get_session", tripwire)

    result = safe_execute(ALL_CUSTOMERS, RM1, {ENTITLEMENT_PARAM: RM2})

    assert result["allowed"] is False
    assert result["stage"] == STAGE_VALIDATE
    assert result["reason"] == "param_shadows_entitlement"
    assert {k: result[k] for k in REJECTED_SHAPE} == REJECTED_SHAPE


def test_rejected_query_never_reaches_the_database(monkeypatch):
    """"Return the verdict without executing" — proved, not assumed.

    The session factory is replaced with a tripwire. If any rejection path still opens
    a connection, this fails loudly rather than passing on the strength of the returned
    dict looking right.
    """

    def tripwire():
        raise AssertionError("the pipeline opened a database session on a rejected query")

    monkeypatch.setattr(sql_pipeline, "get_session", tripwire)

    for sql, stage in (
        ("DROP TABLE customers", STAGE_VALIDATE),
        ("SELECT * FROM audit_log", STAGE_VALIDATE),
        ("SELECT c.customer_id, l.loan_id FROM customers c, loans l", STAGE_ENTITLEMENT),
    ):
        result = safe_execute(sql, RM1)
        assert result["allowed"] is False and result["stage"] == stage


def test_execution_failure_is_contained_and_leaks_no_driver_message():
    """A query can be structurally safe and still fail — a hallucinated column name.

    Postgres echoes offending identifiers and values back in its error text, so the
    message is not propagated: the caller learns that it failed, not what was in it.
    """
    result = safe_execute("SELECT c.hallucinated_column FROM customers c", RM1)

    assert result["allowed"] is False
    assert result["stage"] == STAGE_EXECUTE
    assert result["reason"] == "execution_error"
    assert "hallucinated_column" not in result["reason"]
    assert {k: result[k] for k in REJECTED_SHAPE} == REJECTED_SHAPE


# --------------------------------------------------------------------------------
# Step 3 actually runs
#
# These are the tests that would still pass if step 3 were deleted — unless they are
# written specifically so they cannot. Each one fails against a pipeline that skips
# re-validation.
# --------------------------------------------------------------------------------


def test_revalidation_catches_an_unsafe_rewrite(monkeypatch):
    """Make the entitlement rail emit something forbidden. Step 3 must catch it.

    This is the load-bearing test of §2.2's ordering. The stub stands in for the
    realistic failure: not an attacker, but a future edit to entitlement.py — a new
    join path, a changed rewrite — that emits SQL sql_guard would refuse. With step 3
    in place the pipeline rejects at `revalidate`. Without it, this query reads the
    audit log.
    """
    executed: list[str] = []

    def tripwire():
        executed.append("opened")
        raise AssertionError("unsafe rewrite reached the database")

    def emit_forbidden_sql(tree, rm_id):
        return EntitlementVerdict(
            allowed=True,
            reason="ok",
            ast=tree,
            sql="SELECT user_query FROM audit_log LIMIT 20",
            params={ENTITLEMENT_PARAM: rm_id},
        )

    monkeypatch.setattr(sql_pipeline, "enforce_entitlement", emit_forbidden_sql)
    monkeypatch.setattr(sql_pipeline, "get_session", tripwire)

    result = safe_execute(ALL_CUSTOMERS, RM1)

    assert result["allowed"] is False
    assert result["stage"] == STAGE_REVALIDATE
    # sql_guard's own reason code, unmodified — the stage is what says where it fired.
    assert result["reason"] == "audit_log_access"
    assert executed == []


def test_revalidation_catches_a_rewrite_that_dropped_the_limit(monkeypatch):
    """A subtler rewrite defect: still a SELECT, still allowlisted, now unbounded.

    An unbounded result set is a real hazard here, not a style issue — 600 transaction
    rows do not fit a 4096-token context (§1.1 rule 2), and truncation mid-result is
    the condition under which a 3B model starts inventing figures.
    """

    def emit_unbounded_sql(tree, rm_id):
        return EntitlementVerdict(
            allowed=True,
            reason="ok",
            ast=tree,
            sql="SELECT t.txn_id FROM transactions t LIMIT 5000",
            params={ENTITLEMENT_PARAM: rm_id},
        )

    monkeypatch.setattr(sql_pipeline, "enforce_entitlement", emit_unbounded_sql)

    result = safe_execute(ALL_CUSTOMERS, RM1)

    # sql_guard clamps rather than rejects, so this one proceeds — but it proceeds with
    # the clamped string, which is only true if step 3 ran and its output was used.
    assert result["allowed"] is True
    assert "LIMIT 50" in result["sql"]
    assert "5000" not in result["sql"]
    assert result["row_count"] <= 50


def test_validate_sql_is_called_twice_and_the_second_time_on_the_rewrite(monkeypatch):
    """Pin the sequence: step 1 sees the model's SQL, step 3 sees the rewritten SQL."""
    seen: list[str] = []
    real_validate = sql_pipeline.validate_sql

    def spy(sql):
        seen.append(sql)
        return real_validate(sql)

    monkeypatch.setattr(sql_pipeline, "validate_sql", spy)

    result = safe_execute(ALL_CUSTOMERS, RM1)
    assert result["allowed"] is True

    assert len(seen) == 2, "validate_sql must run once before and once after the rewrite"
    # Step 1 saw the raw SQL: no predicate yet.
    assert ENTITLEMENT_PARAM not in seen[0]
    # Step 3 saw the scoped SQL: predicate present. This is the assertion that fails if
    # re-validation is handed `guard.safe_sql` instead of the entitlement output.
    assert f"c.rm_id = :{ENTITLEMENT_PARAM}" in seen[1]


def test_the_executed_string_is_step_3s_output(monkeypatch):
    """The invariant from the module docstring, checked directly.

    What reaches the database must be sqlglot's regeneration from the tree step 3
    accepted — not the model's original text, and not step 2's output either.
    """
    executed: list[str] = []
    validated: list[str] = []
    real_validate = sql_pipeline.validate_sql

    def spy(sql):
        verdict = real_validate(sql)
        validated.append(verdict.safe_sql)
        return verdict

    class FakeResult:
        def keys(self):
            return ["customer_id"]

        def fetchall(self):
            return []

    class FakeSession:
        def execute(self, clause, params=None):
            executed.append(str(clause))
            return FakeResult()

    class FakeScope:
        def __enter__(self):
            return FakeSession()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(sql_pipeline, "validate_sql", spy)
    monkeypatch.setattr(sql_pipeline, "get_session", FakeScope)

    result = safe_execute(ALL_CUSTOMERS, RM1)

    assert result["allowed"] is True
    assert len(executed) == 1
    # validated[1] is step 3's regenerated SQL; validated[0] is step 1's.
    assert executed[0] == validated[1] == result["sql"]
    assert executed[0] != validated[0]
