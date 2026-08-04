"""Unit tests for the SQL safety rail (PROJECT_PLAN.md §2.2, acceptance §2.6).

No LLM, no Postgres, no network — the rail is deliberately dependency-free so it can
be tested in isolation. Every rejection test asserts the *specific* reason code and
not merely `allowed is False`, because Phase 4's red-team harness asserts on which
rail fired and why (§4.2); these codes are a contract.
"""

import pytest

from app.rails.sql_guard import DEFAULT_LIMIT, MAX_LIMIT, validate_sql

# --------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------


def test_rejects_drop_table():
    """The canonical §2.6 case. DROP is not a SELECT, so the root-type check fires."""
    verdict = validate_sql("DROP TABLE customers")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Drop"
    assert verdict.safe_sql == ""


def test_rejects_audit_log_with_its_own_reason():
    """audit_log must never be readable, and must say so distinctly.

    It is also an allowlist miss, so a sloppy implementation would report
    `table_not_allowlisted`. Asserting the specific code proves the dedicated rule
    fired rather than coinciding with the generic one.
    """
    verdict = validate_sql("SELECT * FROM audit_log")
    assert verdict.allowed is False
    assert verdict.reason == "audit_log_access"


def test_rejects_stacked_query():
    verdict = validate_sql("SELECT customer_id FROM customers; DROP TABLE customers")
    assert verdict.allowed is False
    # The `;` is caught before the statement split — both defences cover this shape,
    # and the semicolon check runs first (§2.2 ordering).
    assert verdict.reason == "semicolon_separator"


def test_rejects_trailing_semicolon():
    """Even a lone trailing `;` is rejected: no separator reaches the driver."""
    assert validate_sql("SELECT customer_id FROM customers").allowed is True
    assert validate_sql("SELECT customer_id FROM customers;").reason == "semicolon_separator"


def test_rejects_line_comment():
    verdict = validate_sql("SELECT customer_id FROM customers -- and now the payload")
    assert verdict.allowed is False
    assert verdict.reason == "sql_comment"


def test_rejects_block_comment():
    verdict = validate_sql("SELECT /* smuggled */ customer_id FROM customers")
    assert verdict.allowed is False
    assert verdict.reason == "sql_comment"


def test_rejects_update():
    verdict = validate_sql("UPDATE accounts SET balance = 0 WHERE account_id = 'ACCT-0001'")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Update"


def test_rejects_delete():
    verdict = validate_sql("DELETE FROM transactions WHERE txn_id = 'TXN-00312'")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Delete"


def test_rejects_insert():
    verdict = validate_sql("INSERT INTO customers (customer_id) VALUES ('CUST-9999')")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Insert"


def test_rejects_information_schema():
    verdict = validate_sql("SELECT table_name FROM information_schema.tables")
    assert verdict.allowed is False
    assert verdict.reason == "system_catalog_access"


def test_rejects_pg_catalog():
    verdict = validate_sql("SELECT tablename FROM pg_catalog.pg_tables")
    assert verdict.allowed is False
    assert verdict.reason == "system_catalog_access"


def test_rejects_bare_pg_table():
    """`pg_stat_activity` needs no schema qualifier — the prefix rule covers it."""
    verdict = validate_sql("SELECT query FROM pg_stat_activity")
    assert verdict.allowed is False
    assert verdict.reason == "system_catalog_access"


def test_rejects_cte_reaching_audit_log():
    """Nesting must not launder a forbidden table.

    The CTE alias `leak` is itself a Table node in the AST; the rail excludes alias
    names from the allowlist check but still walks the CTE body, so `audit_log` is
    found where a top-level-only check would miss it.
    """
    verdict = validate_sql(
        "WITH leak AS (SELECT user_query FROM audit_log) SELECT * FROM leak"
    )
    assert verdict.allowed is False
    assert verdict.reason == "audit_log_access"


def test_rejects_subquery_hitting_disallowed_table():
    verdict = validate_sql(
        "SELECT customer_id FROM customers "
        "WHERE customer_id IN (SELECT usename FROM pg_shadow)"
    )
    assert verdict.allowed is False
    assert verdict.reason == "system_catalog_access"


def test_rejects_subquery_hitting_unknown_table():
    verdict = validate_sql(
        "SELECT customer_id FROM customers "
        "WHERE customer_id IN (SELECT customer_id FROM secret_table)"
    )
    assert verdict.allowed is False
    assert verdict.reason == "table_not_allowlisted:secret_table"


def test_rejects_union_reaching_audit_log():
    """A set operation is an accepted root shape, so its branches must be walked."""
    verdict = validate_sql(
        "SELECT customer_id FROM customers UNION SELECT rm_id FROM audit_log"
    )
    assert verdict.allowed is False
    assert verdict.reason == "audit_log_access"


def test_rejects_grant():
    verdict = validate_sql("GRANT ALL ON customers TO attacker")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Grant"


def test_rejects_truncate():
    verdict = validate_sql("TRUNCATE TABLE customers")
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:TruncateTable"


@pytest.mark.parametrize(
    "statement",
    [
        "REVOKE ALL ON customers FROM sentinel",
        "CALL some_procedure()",
        "EXECUTE prepared_stmt",
    ],
)
def test_rejects_unmodelled_statements_via_command_fallback(statement):
    """sqlglot parses these to `exp.Command`, its catch-all for unmodelled syntax.

    Rejecting `Command` is what makes SELECT-only hold against statements this
    sqlglot release has never heard of — the reason why the rule is an allowlist.
    """
    verdict = validate_sql(statement)
    assert verdict.allowed is False
    assert verdict.reason == "non_select_statement:Command"


def test_rejects_pg_function_call():
    verdict = validate_sql("SELECT pg_sleep(10) FROM customers")
    assert verdict.allowed is False
    assert verdict.reason == "system_function_call"


def test_rejects_table_free_select():
    verdict = validate_sql("SELECT 1")
    assert verdict.allowed is False
    assert verdict.reason == "no_table_reference"


def test_rejects_non_literal_limit():
    """A bound decided at execution time cannot be proven here. Fail closed."""
    verdict = validate_sql("SELECT customer_id FROM customers LIMIT $1")
    assert verdict.allowed is False
    assert verdict.reason == "unresolvable_limit"


def test_rejects_garbage():
    verdict = validate_sql("not sql at all ((((")
    assert verdict.allowed is False
    assert verdict.reason == "unparseable_sql"


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_rejects_empty(blank):
    verdict = validate_sql(blank)
    assert verdict.allowed is False
    assert verdict.reason == "empty_sql"


# --------------------------------------------------------------------------------
# Acceptances
# --------------------------------------------------------------------------------


def test_accepts_join_and_injects_default_limit():
    """§2.6: accepts a normal SELECT and adds LIMIT 20."""
    verdict = validate_sql(
        "SELECT c.full_name, a.balance FROM customers c "
        "JOIN accounts a ON a.customer_id = c.customer_id "
        "WHERE a.status = 'ACTIVE'"
    )
    assert verdict.allowed is True
    assert verdict.reason == "ok"
    assert f"LIMIT {DEFAULT_LIMIT}" in verdict.safe_sql
    # The regenerated SQL, not the caller's string, is what flows onward.
    assert "customers" in verdict.safe_sql and "accounts" in verdict.safe_sql


def test_clamps_oversized_limit():
    """§2.2 step 6: a LIMIT above the ceiling is clamped, not rejected."""
    verdict = validate_sql("SELECT txn_id FROM transactions LIMIT 500")
    assert verdict.allowed is True
    assert f"LIMIT {MAX_LIMIT}" in verdict.safe_sql
    assert "500" not in verdict.safe_sql


def test_leaves_small_limit_untouched():
    verdict = validate_sql("SELECT txn_id FROM transactions LIMIT 10")
    assert verdict.allowed is True
    assert "LIMIT 10" in verdict.safe_sql


def test_accepts_limit_exactly_at_ceiling():
    verdict = validate_sql("SELECT txn_id FROM transactions LIMIT 50")
    assert verdict.allowed is True
    assert f"LIMIT {MAX_LIMIT}" in verdict.safe_sql


def test_accepts_union_over_allowlisted_tables():
    verdict = validate_sql(
        "SELECT customer_id FROM customers UNION SELECT customer_id FROM loans"
    )
    assert verdict.allowed is True
    assert f"LIMIT {DEFAULT_LIMIT}" in verdict.safe_sql


def test_accepts_cte_over_allowlisted_tables():
    verdict = validate_sql(
        "WITH recent AS (SELECT account_id, amount FROM transactions) "
        "SELECT account_id FROM recent"
    )
    assert verdict.allowed is True


def test_accepts_public_schema_qualifier():
    verdict = validate_sql("SELECT customer_id FROM public.customers")
    assert verdict.allowed is True


def test_semicolon_inside_string_literal_is_data_not_a_separator():
    """The tokenizer-level check must not false-positive on RM data.

    A merchant memo containing `;` or `--` is ordinary content. This is the concrete
    reason the rail inspects token types instead of scanning the raw string.
    """
    verdict = validate_sql(
        "SELECT txn_id FROM transactions WHERE description = 'PAID; REFUNDED -- OK'"
    )
    assert verdict.allowed is True
    assert f"LIMIT {DEFAULT_LIMIT}" in verdict.safe_sql


def test_aggregate_query_is_accepted():
    verdict = validate_sql(
        "SELECT account_type, SUM(balance) FROM accounts GROUP BY account_type"
    )
    assert verdict.allowed is True
