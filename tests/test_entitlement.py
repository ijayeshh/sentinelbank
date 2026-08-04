"""Tests for the entitlement rail (PROJECT_PLAN.md §2.1, acceptance §2.6).

The §2.6 acceptance criterion is worded deliberately: *"a query issued as RM-001 cannot
return any row belonging to an RM-002 customer — assert this directly against the
seeded data."* So the cross-RM tests in this file run the rewritten SQL against the
real Postgres instance and compare against the real seeded rows. Mocks would only prove
the rail does what this file already assumes it does.

Several tests deliberately run the **unscoped** query first, to show the leak actually
exists before asserting the rail closes it. A negative test that would pass against a
no-op rail is not evidence.

Requires `docker compose up -d` and `python db/seed.py`. These tests fail rather than
skip when Postgres is unreachable: a security control whose proof silently reports
green when it did not run is worse than no proof.

The unit tests at the bottom need no database.
"""

from __future__ import annotations

import sqlglot
import pytest
from sqlalchemy import text
from sqlglot import exp

from app.db import get_session
from app.rails.entitlement import (
    ENTITLEMENT_PARAM,
    EntitlementVerdict,
    enforce_entitlement,
)
from app.rails.sql_guard import validate_sql

DIALECT = "postgres"

# Seeded distribution, fixed by Faker.seed(42) in db/seed.py and asserted in Phase 1's
# acceptance checks. Hard-coded here on purpose: if a reseed changes these, the
# cross-RM tests below have lost their fixture and must be looked at, not auto-adapted.
RM1, RM2 = "RM-001", "RM-002"
CUSTOMERS_RM1, CUSTOMERS_RM2 = 22, 18
LOANS_RM1 = 10
SERVICE_REQUESTS_RM1 = 14

# Every fixture query carries LIMIT 50 (sql_guard's ceiling) so the rail's row filtering
# is what shapes the result, not sql_guard's default LIMIT 20 truncating 22 customers.


# --------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------


def scope(sql: str, rm_id: str) -> EntitlementVerdict:
    """Run the §2.2 pipeline as production will: validate -> enforce -> re-validate.

    Testing `enforce_entitlement` through the same ordering the agent uses is the point
    — a rail that only works on hand-built ASTs has not been tested.
    """
    guard = validate_sql(sql)
    assert guard.allowed, f"fixture rejected by sql_guard: {guard.reason}"

    verdict = enforce_entitlement(sqlglot.parse_one(guard.safe_sql, dialect=DIALECT), rm_id)
    if verdict.allowed:
        # §2.2: the final string is re-validated. A rewrite that produced something
        # sql_guard would refuse must never reach the database.
        revalidated = validate_sql(verdict.sql)
        assert revalidated.allowed, f"rewrite failed re-validation: {revalidated.reason}"
    return verdict


def run_scoped(sql: str, rm_id: str) -> list:
    """Execute the scoped query with rm_id supplied as a bound parameter."""
    verdict = scope(sql, rm_id)
    assert verdict.allowed, f"unexpectedly rejected: {verdict.reason}"
    with get_session() as session:
        return session.execute(text(verdict.sql), verdict.params).all()


def run_raw(sql: str) -> list:
    """Execute a query with NO entitlement rewrite — used to prove a leak exists."""
    with get_session() as session:
        return session.execute(text(sql)).all()


@pytest.fixture(scope="module", autouse=True)
def require_seeded_database():
    """Fail loudly, and only once, if the seeded database is not there."""
    try:
        with get_session() as session:
            counts = dict(
                session.execute(
                    text("SELECT rm_id, COUNT(*) FROM customers GROUP BY rm_id")
                ).all()
            )
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "entitlement tests require the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )
    assert counts.get(RM1) == CUSTOMERS_RM1 and counts.get(RM2) == CUSTOMERS_RM2, (
        f"seed data has drifted (got {counts}); re-run python db/seed.py"
    )


def customer_ids_of(rm_id: str) -> set[str]:
    """The ground truth, read straight from the table with no rail involved."""
    return {
        row[0]
        for row in run_raw(f"SELECT customer_id FROM customers WHERE rm_id = '{rm_id}'")
    }


# --------------------------------------------------------------------------------
# §2.6 — cross-RM isolation, asserted against seeded data
# --------------------------------------------------------------------------------


def test_rm001_query_returns_no_rm002_rows():
    """The core acceptance criterion: RM-001 cannot see an RM-002 customer."""
    rows = run_scoped("SELECT c.customer_id, c.rm_id FROM customers c LIMIT 50", RM1)

    assert len(rows) == CUSTOMERS_RM1
    # Not just "no RM-002 rows" — every returned row is positively RM-001's.
    assert {row.rm_id for row in rows} == {RM1}
    assert {row.customer_id for row in rows}.isdisjoint(customer_ids_of(RM2))


def test_rm002_gets_a_different_non_overlapping_set():
    """The same query as RM-002 returns RM-002's book and nothing of RM-001's."""
    sql = "SELECT c.customer_id, c.rm_id FROM customers c LIMIT 50"
    seen_by_rm1 = {row.customer_id for row in run_scoped(sql, RM1)}
    seen_by_rm2 = {row.customer_id for row in run_scoped(sql, RM2)}

    assert len(seen_by_rm2) == CUSTOMERS_RM2
    assert seen_by_rm1.isdisjoint(seen_by_rm2)
    # Between them the two RMs see all 40 seeded customers: the rail partitions the
    # table, it does not merely shrink it.
    assert len(seen_by_rm1 | seen_by_rm2) == CUSTOMERS_RM1 + CUSTOMERS_RM2


def test_unfiltered_customer_query_is_scoped_down_not_rejected():
    """"Select every customer" is a legitimate RM question — narrow it, don't refuse it.

    Rejecting here would be the easy implementation and the wrong one: the rail's job is
    to make over-fetching impossible, not to make bulk queries impossible. "Which of my
    customers are delinquent?" is exactly this shape (§4.1 benign case 4).
    """
    verdict = scope("SELECT * FROM customers LIMIT 50", RM1)

    assert verdict.allowed is True
    assert verdict.reason == "ok"
    assert f"customers.rm_id = :{ENTITLEMENT_PARAM}" in verdict.sql

    # The unscoped form really does return the whole table — the rail is doing work.
    assert len(run_raw("SELECT customer_id FROM customers LIMIT 50")) == 40
    assert len(run_scoped("SELECT * FROM customers LIMIT 50", RM1)) == CUSTOMERS_RM1


def test_child_tables_are_scoped_through_an_injected_join():
    """§2.1: accounts/transactions/loans/service_requests scope via a join to customers.

    None of these tables carries rm_id, and none of these queries mentions customers.
    """
    for sql, expected in (
        ("SELECT l.loan_id FROM loans l LIMIT 50", LOANS_RM1),
        ("SELECT sr.ticket_id FROM service_requests sr LIMIT 50", SERVICE_REQUESTS_RM1),
    ):
        verdict = scope(sql, RM1)
        assert verdict.allowed, verdict.reason
        assert "JOIN customers" in verdict.sql
        assert len(run_scoped(sql, RM1)) == expected

    # And the two RMs' rows do not overlap.
    sql = "SELECT l.loan_id FROM loans l LIMIT 50"
    assert {r.loan_id for r in run_scoped(sql, RM1)}.isdisjoint(
        {r.loan_id for r in run_scoped(sql, RM2)}
    )


def test_transactions_scope_through_the_two_hop_path():
    """transactions -> accounts -> customers. The only table needing two joins."""
    sql = "SELECT t.txn_id FROM transactions t LIMIT 50"
    verdict = scope(sql, RM1)

    assert "JOIN accounts" in verdict.sql and "JOIN customers" in verdict.sql
    assert {r.txn_id for r in run_scoped(sql, RM1)}.isdisjoint(
        {r.txn_id for r in run_scoped(sql, RM2)}
    )


# --------------------------------------------------------------------------------
# Nested scope units — a scoped outer SELECT is not enough
# --------------------------------------------------------------------------------


def test_subquery_that_would_leak_is_scoped_independently():
    """The vulnerability this rail's per-SELECT design exists to prevent.

    The outer SELECT reads a derived table and names no customer-owned table at all, so
    an implementation that only rewrote the outer query would inject nothing and return
    all 40 customers — including their SSNs.
    """
    sql = "SELECT sub.customer_id, sub.ssn FROM (SELECT customer_id, ssn FROM customers) AS sub LIMIT 50"

    # The leak is real if the rail does nothing.
    assert len(run_raw(sql)) == CUSTOMERS_RM1 + CUSTOMERS_RM2

    verdict = scope(sql, RM1)
    assert verdict.allowed, verdict.reason
    # The predicate sits inside the parentheses, on the inner SELECT.
    inner = verdict.sql[verdict.sql.index("(") : verdict.sql.index(")")]
    assert f"rm_id = :{ENTITLEMENT_PARAM}" in inner

    rows = run_scoped(sql, RM1)
    assert len(rows) == CUSTOMERS_RM1
    assert {r.customer_id for r in rows}.isdisjoint(customer_ids_of(RM2))


def test_where_subquery_is_scoped_independently():
    """A subquery in WHERE is its own scope unit too."""
    sql = (
        "SELECT c.customer_id FROM customers c "
        "WHERE c.customer_id IN (SELECT a.customer_id FROM accounts a) LIMIT 50"
    )
    verdict = scope(sql, RM1)

    assert verdict.allowed, verdict.reason
    # Two predicates: one for the outer customers, one for the subquery's injected join.
    assert verdict.sql.count(f":{ENTITLEMENT_PARAM}") == 2
    assert {r.customer_id for r in run_scoped(sql, RM1)}.isdisjoint(customer_ids_of(RM2))


def test_each_union_branch_is_scoped():
    """A UNION leaks entirely through whichever branch was left alone."""
    sql = (
        "SELECT c.customer_id FROM customers c "
        "UNION SELECT l.customer_id FROM loans l LIMIT 50"
    )
    verdict = scope(sql, RM1)

    assert verdict.allowed, verdict.reason
    assert verdict.sql.count(f":{ENTITLEMENT_PARAM}") == 2

    rows = run_scoped(sql, RM1)
    assert {r.customer_id for r in rows}.isdisjoint(customer_ids_of(RM2))


def test_cte_body_is_scoped():
    """A CTE is a scope unit; referencing its alias later must not re-open it."""
    sql = (
        "WITH book AS (SELECT customer_id, principal FROM loans) "
        "SELECT book.customer_id FROM book LIMIT 50"
    )
    verdict = scope(sql, RM1)

    assert verdict.allowed, verdict.reason
    assert verdict.sql.count(f":{ENTITLEMENT_PARAM}") == 1
    assert len(run_scoped(sql, RM1)) == LOANS_RM1


# --------------------------------------------------------------------------------
# Fail closed — shapes that cannot be scoped
# --------------------------------------------------------------------------------


def test_cross_join_without_a_join_path_is_rejected():
    """The unscopeable shape. `customers` is present but proves nothing about `loans`.

    A rail that merely checked "does customers appear? then add the predicate" would
    accept this and hand the RM every other RM's loans.
    """
    sql = "SELECT c.customer_id, l.loan_id FROM customers c, loans l LIMIT 50"

    # Prove the shape is genuinely dangerous: even with a hand-written rm_id filter on
    # customers, the cross join still returns loans belonging to RM-002.
    leaked = run_raw(
        "SELECT DISTINCT l.customer_id FROM customers c, loans l "
        f"WHERE c.rm_id = '{RM1}'"
    )
    assert not {row[0] for row in leaked}.isdisjoint(customer_ids_of(RM2))

    verdict = scope(sql, RM1)
    assert verdict.allowed is False
    assert verdict.reason == "unlinked_scoped_table:l"
    assert verdict.ast is None and verdict.sql == "" and verdict.params == {}


def test_disconnected_child_tables_are_rejected():
    """Two customer-owned tables, no link between them, no customers to anchor on."""
    verdict = scope(
        "SELECT a.balance, sr.ticket_id FROM accounts a CROSS JOIN service_requests sr",
        RM1,
    )
    assert verdict.allowed is False
    assert verdict.reason == "unlinked_scoped_table:a,sr"


def test_join_on_a_non_key_column_is_not_a_join_path():
    """An equality that is not a foreign key proves nothing about ownership.

    `accounts.customer_id = loans.loan_id` is syntactically a join and semantically
    meaningless. Recognising only the declared FK pairs is what stops an
    attacker-suggested join from being read as proof.
    """
    verdict = scope(
        "SELECT c.customer_id, l.loan_id FROM customers c "
        "JOIN loans l ON l.loan_id = c.customer_id",
        RM1,
    )
    assert verdict.allowed is False
    assert verdict.reason == "unlinked_scoped_table:l"


def test_link_hidden_under_an_or_is_not_a_join_path():
    """`ON ... OR TRUE` holds on some rows only. Only AND conjuncts count."""
    verdict = scope(
        "SELECT c.customer_id, l.loan_id FROM customers c "
        "JOIN loans l ON l.customer_id = c.customer_id OR l.days_delinquent > 0",
        RM1,
    )
    assert verdict.allowed is False
    assert verdict.reason == "unlinked_scoped_table:l"


def test_ambiguous_unqualified_column_is_rejected():
    """An injected join must never silently change what a bare column resolves to."""
    verdict = scope(
        "SELECT customer_id FROM accounts a JOIN transactions t "
        "ON t.account_id = a.account_id",
        RM1,
    )
    assert verdict.allowed is False
    assert verdict.reason == "ambiguous_unqualified_column:customer_id"


def test_cte_shadowing_a_real_table_is_rejected():
    verdict = scope(
        "WITH customers AS (SELECT l.customer_id FROM loans l) "
        "SELECT customers.customer_id FROM customers",
        RM1,
    )
    assert verdict.allowed is False
    assert verdict.reason == "cte_shadows_table:customers"


def test_reserved_alias_prefix_is_rejected():
    """A query that pre-empts the injected alias could shadow the predicate's binding."""
    ast = sqlglot.parse_one(
        "SELECT __ent_customers.loan_id FROM loans AS __ent_customers", dialect=DIALECT
    )
    verdict = enforce_entitlement(ast, RM1)
    assert verdict.allowed is False
    assert verdict.reason == "reserved_alias_conflict:__ent_customers"


def test_non_select_root_is_rejected():
    """sql_guard rejects this first; the rail must not depend on that having happened."""
    ast = sqlglot.parse_one("DROP TABLE customers", dialect=DIALECT)
    verdict = enforce_entitlement(ast, RM1)
    assert verdict.allowed is False
    assert verdict.reason == "unsupported_ast_root:Drop"


# --------------------------------------------------------------------------------
# The rm_id itself — session state, bound, never interpolated
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rm_id",
    [
        "",
        "   ",
        None,
        "RM-1",
        "rm-001",
        # Not because it could inject — the value is bound, so it cannot — but because
        # a value of this shape means it came from somewhere other than session state.
        "RM-001' OR '1'='1",
        "RM-002; DROP TABLE customers",
    ],
)
def test_malformed_rm_id_is_rejected(bad_rm_id):
    ast = sqlglot.parse_one("SELECT c.customer_id FROM customers c", dialect=DIALECT)
    verdict = enforce_entitlement(ast, bad_rm_id)
    assert verdict.allowed is False
    assert verdict.reason == "invalid_rm_id"


def test_rm_id_is_bound_never_interpolated():
    """§2.1: a bound parameter, not string interpolation. The value is not in the SQL."""
    verdict = scope("SELECT c.customer_id FROM customers c", RM1)

    assert verdict.allowed is True
    assert RM1 not in verdict.sql
    assert f":{ENTITLEMENT_PARAM}" in verdict.sql
    assert verdict.params == {ENTITLEMENT_PARAM: RM1}


def test_rm_id_in_the_query_text_does_not_override_the_session():
    """The classic authorization-bypass shape (§4.1): the SQL asks for the other book.

    The injected predicate is ANDed, so the caller's own `rm_id` filter cannot widen
    the result — `rm_id = 'RM-002' AND rm_id = 'RM-001'` matches nothing. Identity comes
    from the session parameter alone.
    """
    rows = run_scoped(
        f"SELECT c.customer_id FROM customers c WHERE c.rm_id = '{RM2}' LIMIT 50", RM1
    )
    assert rows == []


def test_existing_or_predicate_is_parenthesised():
    """`WHERE a OR b` must become `WHERE (a OR b) AND rm_id = ...`, never `a OR b AND ...`.

    Operator precedence would otherwise leave the first disjunct unscoped — a one
    -character-looking bug with a full table's worth of consequences.
    """
    verdict = scope(
        "SELECT c.customer_id FROM customers c "
        "WHERE c.risk_rating = 'HIGH' OR c.ofac_flag = TRUE LIMIT 50",
        RM1,
    )
    assert verdict.allowed is True
    assert "(c.risk_rating = 'HIGH' OR c.ofac_flag = TRUE)" in verdict.sql

    rows = run_scoped(
        "SELECT c.customer_id, c.rm_id FROM customers c "
        "WHERE c.risk_rating = 'HIGH' OR c.ofac_flag = TRUE LIMIT 50",
        RM1,
    )
    assert {row.rm_id for row in rows} <= {RM1}


# --------------------------------------------------------------------------------
# Exempt tables, and non-mutation of the caller's AST
# --------------------------------------------------------------------------------


def test_policy_chunks_is_not_scoped():
    """Bank-authored policy text: no customer_id, no join path, same for every RM."""
    verdict = scope("SELECT p.doc_name, p.chunk_text FROM policy_chunks p", RM1)

    assert verdict.allowed is True
    assert ENTITLEMENT_PARAM not in verdict.sql
    assert "JOIN" not in verdict.sql.upper()
    # Both RMs see the identical corpus — that is the point of the exemption.
    sql = "SELECT p.chunk_id FROM policy_chunks p LIMIT 50"
    assert [r.chunk_id for r in run_scoped(sql, RM1)] == [
        r.chunk_id for r in run_scoped(sql, RM2)
    ]


def test_relationship_managers_is_not_scoped_but_joined_customers_still_are():
    """The exemption must not become a bypass.

    Joining the RM directory is legitimate; it must not widen the customers side.
    """
    sql = (
        "SELECT c.customer_id, rm.branch FROM customers c "
        "JOIN relationship_managers rm ON rm.rm_id = c.rm_id LIMIT 50"
    )
    verdict = scope(sql, RM1)

    assert verdict.allowed is True
    assert f"c.rm_id = :{ENTITLEMENT_PARAM}" in verdict.sql

    rows = run_scoped(sql, RM1)
    assert len(rows) == CUSTOMERS_RM1
    assert {r.customer_id for r in rows}.isdisjoint(customer_ids_of(RM2))


def test_callers_ast_is_never_mutated():
    """Rejection or success, the tree handed in comes back unchanged.

    The caller still needs its original AST to log what was attempted.
    """
    original = "SELECT c.customer_id FROM customers AS c"
    ast = sqlglot.parse_one(original, dialect=DIALECT)

    verdict = enforce_entitlement(ast, RM1)
    assert verdict.allowed is True
    assert ast.sql(dialect=DIALECT) == original
    assert verdict.ast is not ast
    assert isinstance(verdict.ast, exp.Select)
