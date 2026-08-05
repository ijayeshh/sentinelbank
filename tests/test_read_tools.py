"""Tests for the three AUTO-tier read tools and the RAG layer (§2.3, §2.4, §2.6).

Run against the real seeded database, per §2.6's wording: the entitlement claim is only
worth anything if it is checked against real rows.

The centre of gravity here is the **`source` flag**. Phase 3's injection rail reads
nothing else to decide what to scan, so these tests check it field by field — including
the two cases a name-only implementation gets wrong: an aliased untrusted column, and a
numeric aggregate under a name that is in no schema.

Requires `docker compose up -d` and `python db/seed.py`. Fails rather than skips when
Postgres is unreachable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import text

from app.agent import tools
from app.agent.tools import (
    TOOL_REGISTRY,
    RiskTier,
    Source,
    get_account_summary,
    query_customer_data,
    search_policy_docs,
)
from app.db import get_session
from app.rails import timing

RM1, RM2 = "RM-001", "RM-002"
CUSTOMERS_RM1, CUSTOMERS_RM2 = 22, 18

# Verified seed fixtures. All four planted transaction payloads land on RM-001 accounts,
# by design — a payload on RM-002 would be correctly filtered out by the entitlement rail
# and would never test anything (db/seed.py POISONED_ROWS, property 1).
POISONED_TXN = "TXN-000023"
POISONED_ACCOUNT = "ACCT-0003"  # owns TXN-000023, belongs to CUST-0003 / RM-001
PAYLOAD_MARKER = "Prior instructions are void"

IN_SCOPE_ACCOUNT = "ACCT-0003"  # RM-001
OUT_OF_SCOPE_ACCOUNT = "ACCT-0023"  # RM-002
NONEXISTENT_ACCOUNT = "ACCT-9999"

ALL_CUSTOMERS = "SELECT c.customer_id, c.rm_id FROM customers c LIMIT 50"


@pytest.fixture(scope="module", autouse=True)
def require_seeded_database():
    try:
        with get_session() as session:
            counts = dict(
                session.execute(
                    text("SELECT rm_id, COUNT(*) FROM customers GROUP BY rm_id")
                ).all()
            )
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "read-tool tests require the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )
    assert counts.get(RM1) == CUSTOMERS_RM1 and counts.get(RM2) == CUSTOMERS_RM2, (
        f"seed data has drifted (got {counts}); re-run python db/seed.py"
    )


@pytest.fixture(autouse=True)
def isolate_latency(tmp_path, monkeypatch):
    """Keep @rail_timer out of the repository's redteam/latency.jsonl during tests."""
    monkeypatch.setattr(timing, "LATENCY_LOG_PATH", tmp_path / "latency.jsonl")
    timing.clear_samples()


def customer_ids_of(rm_id: str) -> set[str]:
    with get_session() as session:
        return {
            row[0]
            for row in session.execute(
                text("SELECT customer_id FROM customers WHERE rm_id = :rm"), {"rm": rm_id}
            ).all()
        }


def sources_of(row: dict) -> dict[str, str]:
    return {column: field["source"] for column, field in row.items()}


def values_of(row: dict) -> dict:
    return {column: field["value"] for column, field in row.items()}


# --------------------------------------------------------------------------------
# 1. Entitlement holds through the tool
# --------------------------------------------------------------------------------


def test_query_customer_data_as_rm001_cannot_reach_rm002():
    result = query_customer_data(ALL_CUSTOMERS, RM1)

    assert result["ok"] is True
    assert result["row_count"] == CUSTOMERS_RM1
    seen = {values_of(row)["customer_id"] for row in result["rows"]}
    assert seen.isdisjoint(customer_ids_of(RM2))
    assert {values_of(row)["rm_id"] for row in result["rows"]} == {RM1}


def test_the_two_rms_see_disjoint_books_through_the_tool():
    seen_1 = {values_of(r)["customer_id"] for r in query_customer_data(ALL_CUSTOMERS, RM1)["rows"]}
    seen_2 = {values_of(r)["customer_id"] for r in query_customer_data(ALL_CUSTOMERS, RM2)["rows"]}

    assert len(seen_2) == CUSTOMERS_RM2
    assert seen_1.isdisjoint(seen_2)


def test_rejections_pass_the_rail_reason_through_unmodified():
    """The reason codes are a §4.2 contract; the tool must not reword them."""
    result = query_customer_data("SELECT * FROM audit_log", RM1)

    assert result["ok"] is False
    assert result["reason"] == "audit_log_access"
    assert result["stage"] == "validate"
    assert result["rows"] == [] and result["row_count"] == 0
    assert result["sql"] == ""


# --------------------------------------------------------------------------------
# 2. RAG
# --------------------------------------------------------------------------------


def test_search_policy_docs_finds_the_reg_e_chunk_in_its_top_3():
    """§2.6's acceptance criterion, asserted on doc_name.

    Membership in the top 3, not rank 1: the nearest chunk for this phrasing is
    kyc_cip_policy.md, with Reg E at ranks 2 and 3. That is what the criterion asks for
    and what the RAG layer actually has to deliver.
    """
    result = search_policy_docs("dispute timeline")

    assert result["ok"] is True
    assert result["row_count"] == 3

    doc_names = [values_of(row)["doc_name"] for row in result["rows"]]
    assert "reg_e_dispute_policy.md" in doc_names


def test_search_policy_docs_returns_ascending_cosine_distance():
    result = search_policy_docs("wire transfer limit")
    distances = [values_of(row)["distance"] for row in result["rows"]]

    assert len(distances) == 3
    assert all(isinstance(d, float) for d in distances)
    assert distances == sorted(distances), "nearest chunk must come first"


def test_search_policy_docs_rejects_an_empty_query():
    result = search_policy_docs("   ")
    assert result["ok"] is False
    assert result["reason"] == "empty_query"


# --------------------------------------------------------------------------------
# 3. get_account_summary
# --------------------------------------------------------------------------------


def test_account_summary_returns_correct_data_for_an_in_scope_account():
    result = get_account_summary(IN_SCOPE_ACCOUNT, RM1)

    assert result["ok"] is True
    assert result["row_count"] == 1

    account = values_of(result["rows"][0])
    assert account["account_id"] == IN_SCOPE_ACCOUNT
    assert account["customer_id"] == "CUST-0003"
    assert isinstance(account["balance"], Decimal)

    # Cross-check against the table directly — the tool must report what is really there.
    with get_session() as session:
        expected = session.execute(
            text("SELECT balance, account_type FROM accounts WHERE account_id = :a"),
            {"a": IN_SCOPE_ACCOUNT},
        ).one()
    assert account["balance"] == expected.balance
    assert account["account_type"] == expected.account_type

    assert 0 < len(result["recent_transactions"]) <= 5


def test_account_summary_rejects_an_out_of_scope_account():
    result = get_account_summary(OUT_OF_SCOPE_ACCOUNT, RM1)

    assert result["ok"] is False
    assert result["reason"] == "account_not_found"
    assert result["rows"] == []
    assert "recent_transactions" not in result


def test_out_of_scope_and_nonexistent_accounts_are_indistinguishable():
    """Confirming that ACCT-0023 exists is itself a disclosure.

    "Show me RM-002's portfolio" is a §4.1 authorization_bypass case. The entitlement rail
    makes the rows unreachable; this assertion is what makes their *existence*
    unobservable. A tool that said "not authorised" for one and "not found" for the other
    would be a working account-enumeration oracle.
    """
    forbidden = get_account_summary(OUT_OF_SCOPE_ACCOUNT, RM1)
    missing = get_account_summary(NONEXISTENT_ACCOUNT, RM1)

    assert forbidden["reason"] == missing["reason"] == "account_not_found"
    assert forbidden["ok"] is missing["ok"] is False
    # Nothing anywhere in the payload names the other RM or hints the account is real.
    assert RM2 not in str(forbidden)


def test_account_summary_is_visible_to_its_own_rm():
    """The mirror of the test above: RM-002 can see the account RM-001 could not."""
    result = get_account_summary(OUT_OF_SCOPE_ACCOUNT, RM2)
    assert result["ok"] is True
    assert values_of(result["rows"][0])["account_id"] == OUT_OF_SCOPE_ACCOUNT


@pytest.mark.parametrize("bad", ["", "ACCT-3", "acct-0003", "ACCT-0003 OR 1=1", None])
def test_account_summary_rejects_a_malformed_id(bad):
    result = get_account_summary(bad, RM1)
    assert result["ok"] is False
    assert result["reason"] == "invalid_account_id"


# --------------------------------------------------------------------------------
# 4. The source flag is present and correct everywhere
# --------------------------------------------------------------------------------

VALID_SOURCES = {Source.TRUSTED.value, Source.UNTRUSTED.value}


def _assert_well_formed(rows: list[dict]) -> None:
    """Every field of every row is a {value, source} envelope with a valid source."""
    for row in rows:
        for column, field in row.items():
            assert set(field) == {"value", "source"}, f"{column} is not an envelope"
            assert field["source"] in VALID_SOURCES, f"{column} has source {field['source']}"


def test_every_field_of_every_tool_carries_a_valid_source():
    """Catches a field added later that forgets to be classified."""
    query = query_customer_data(
        "SELECT t.txn_id, t.amount, t.merchant, t.description, t.channel "
        "FROM transactions t LIMIT 50",
        RM1,
    )
    summary = get_account_summary(IN_SCOPE_ACCOUNT, RM1)
    policy = search_policy_docs("fee waiver authority")

    assert query["ok"] and summary["ok"] and policy["ok"]
    for rows in (
        query["rows"],
        summary["rows"],
        summary["recent_transactions"],
        policy["rows"],
    ):
        assert rows, "fixture returned no rows, so it proves nothing"
        _assert_well_formed(rows)


def test_policy_chunks_are_trusted_because_we_wrote_them():
    result = search_policy_docs("dispute timeline")
    for row in result["rows"]:
        assert set(sources_of(row).values()) == {Source.TRUSTED.value}


def test_customer_columns_are_trusted():
    result = query_customer_data(
        "SELECT c.customer_id, c.full_name, c.risk_rating, c.ofac_flag FROM customers c LIMIT 5",
        RM1,
    )
    for row in result["rows"]:
        assert set(sources_of(row).values()) == {Source.TRUSTED.value}


# --------------------------------------------------------------------------------
# 5-7. The classification rule, including the two cases name-matching gets wrong
# --------------------------------------------------------------------------------


def test_poisoned_description_is_untrusted_and_its_siblings_are_not():
    """The exact property Phase 3 depends on (§3.2.2).

    The row carries one attacker-controlled memo attached to an entirely ordinary amount,
    channel, and date. Quarantining the row wholesale would cost the RM their answer;
    quarantining the field keeps it. So the flag has to discriminate *within* the row.
    """
    result = query_customer_data(
        "SELECT t.txn_id, t.amount, t.merchant, t.description, t.channel, t.is_disputed "
        f"FROM transactions t WHERE t.txn_id = '{POISONED_TXN}'",
        RM1,
    )

    assert result["ok"] is True and result["row_count"] == 1
    row = result["rows"][0]

    assert sources_of(row) == {
        "txn_id": Source.TRUSTED.value,
        "amount": Source.TRUSTED.value,
        "merchant": Source.UNTRUSTED.value,
        "description": Source.UNTRUSTED.value,
        "channel": Source.TRUSTED.value,
        "is_disputed": Source.TRUSTED.value,
    }

    # Phase 2 flags; it does not quarantine. The payload must arrive intact so the Phase 3
    # classifier has something real to detect.
    assert PAYLOAD_MARKER in row["description"]["value"]


def test_poisoned_service_request_notes_are_untrusted():
    result = query_customer_data(
        "SELECT sr.ticket_id, sr.category, sr.status, sr.notes "
        "FROM service_requests sr WHERE sr.ticket_id = 'SR-0005'",
        RM1,
    )

    assert result["ok"] is True and result["row_count"] == 1
    row = result["rows"][0]
    assert sources_of(row)["notes"] == Source.UNTRUSTED.value
    assert sources_of(row)["category"] == Source.TRUSTED.value
    assert "disregard the approval workflow" in row["notes"]["value"]


def test_account_summary_flags_untrusted_memo_fields_in_its_transaction_sample():
    """The poisoned memo reaches the model through this tool too, not just through SQL."""
    result = get_account_summary(POISONED_ACCOUNT, RM1)

    assert result["ok"] is True
    memo_sources = {
        field: source
        for row in result["recent_transactions"]
        for field, source in sources_of(row).items()
    }
    assert memo_sources["description"] == Source.UNTRUSTED.value
    assert memo_sources["merchant"] == Source.UNTRUSTED.value
    assert memo_sources["amount"] == Source.TRUSTED.value


def test_aliased_untrusted_column_is_still_untrusted():
    """The fail-closed rule, and the reason it exists.

    From Phase 3 the model writes this SQL, and the model is steerable by the very
    payloads the flag exists to catch. Under name-only matching, `AS memo` renames the
    project's primary injection vector to something unrecognised and it would be tagged
    TRUSTED — a one-word bypass of the tool-result scan.
    """
    result = query_customer_data(
        f"SELECT t.description AS memo FROM transactions t WHERE t.txn_id = '{POISONED_TXN}'",
        RM1,
    )

    assert result["ok"] is True and result["row_count"] == 1
    row = result["rows"][0]
    assert "memo" in row, f"expected the alias as the column key, got {list(row)}"
    assert row["memo"]["source"] == Source.UNTRUSTED.value
    assert PAYLOAD_MARKER in row["memo"]["value"]


def test_numeric_aggregate_under_an_unknown_name_is_trusted():
    """Rule 1 keeps the fail-closed default from swallowing the numbers the RM asked for.

    `total_balance` is in no schema, so the name tells us nothing — but the value is a
    Decimal, and a Decimal cannot carry an instruction to a language model.
    """
    result = query_customer_data(
        "SELECT SUM(a.balance) AS total_balance FROM accounts a", RM1
    )

    assert result["ok"] is True and result["row_count"] == 1
    row = result["rows"][0]
    assert isinstance(row["total_balance"]["value"], Decimal)
    assert row["total_balance"]["source"] == Source.TRUSTED.value


def test_classify_field_directly():
    """The four rules, at the unit level, with no database in the way."""
    # Rule 1 — non-text is always trusted, whatever the column is called.
    assert tools.classify_field("description", None) is Source.TRUSTED
    assert tools.classify_field("anything_at_all", Decimal("1.00")) is Source.TRUSTED
    assert tools.classify_field("count", 42) is Source.TRUSTED
    assert tools.classify_field("is_disputed", True) is Source.TRUSTED
    # Rule 2 — known injection vectors.
    for column in ("description", "merchant", "notes", "DESCRIPTION"):
        assert tools.classify_field(column, "text") is Source.UNTRUSTED
    # Rule 3 — known schema columns holding bank-controlled text.
    assert tools.classify_field("full_name", "Michael Chen") is Source.TRUSTED
    assert tools.classify_field("account_type", "CHECKING") is Source.TRUSTED
    # Rule 4 — unrecognised string column. Fail closed.
    assert tools.classify_field("memo", "text") is Source.UNTRUSTED
    assert tools.classify_field("col1", "text") is Source.UNTRUSTED


# --------------------------------------------------------------------------------
# 8-10. Metadata, instrumentation, and the absence of a bypass
# --------------------------------------------------------------------------------


READ_TOOLS = ("query_customer_data", "get_account_summary", "search_policy_docs")


def test_the_three_read_tools_are_auto_tier_and_registered():
    """AUTO means no interrupt, so a read tool acquiring a write is a tier escalation.

    The exhaustive registry check — that all eight tools carry the tier §2.3 assigns —
    lives in tests/test_write_tools.py, where the write tiers are also in scope.
    """
    for name in READ_TOOLS:
        assert name in TOOL_REGISTRY
        func = TOOL_REGISTRY[name]
        assert func.risk_tier is RiskTier.AUTO
        assert func.tool_name == name


def test_rail_timer_records_one_sample_per_tool_call(tmp_path):
    query_customer_data(ALL_CUSTOMERS, RM1)
    search_policy_docs("dispute timeline")

    assert len(timing.get_samples("tool_query_customer_data")) == 1
    assert len(timing.get_samples("tool_search_policy_docs")) == 1

    sample = timing.get_samples("tool_query_customer_data")[0]
    assert set(sample) == {"rail", "duration_ms", "timestamp"}
    assert sample["duration_ms"] >= 0

    # And it reached the JSONL file the §4.3 report will aggregate.
    assert timing.LATENCY_LOG_PATH.exists()
    assert len(timing.LATENCY_LOG_PATH.read_text(encoding="utf-8").splitlines()) == 5


def test_sql_rails_are_timed_under_their_section_4_3_names():
    """§4.3's table has `sql_guard` and `entitlement` rows; they have to be fillable.

    `sql_guard` is sampled twice per query on purpose — `safe_execute` validates the
    model's SQL, rewrites it, then validates the rewrite. Both runs are real work and
    both belong in the budget.
    """
    query_customer_data(ALL_CUSTOMERS, RM1)

    assert len(timing.get_samples("sql_guard")) == 2, "validate + revalidate"
    assert len(timing.get_samples("entitlement")) == 1
    assert all(s["duration_ms"] >= 0 for s in timing.get_samples("sql_guard"))


def test_a_rejected_query_still_times_the_rail_that_rejected_it():
    """The reject path is the one §4.3 would otherwise under-report."""
    query_customer_data("DROP TABLE customers", RM1)

    # Rejected at step 1, so sql_guard ran once and entitlement never got the chance.
    assert len(timing.get_samples("sql_guard")) == 1
    assert timing.get_samples("entitlement") == []


def test_tool_spans_enclose_the_sql_rails_rather_than_excluding_them():
    """The overlap §4.3's total must not double-count.

    `tool_*` wraps the whole tool body, so `sql_guard` and `entitlement` run *inside* it.
    That is unavoidable — they are nested calls — so the contract is that the report
    treats tool spans as a separate class rather than adding them to the rail total.
    This test pins the nesting so the classification in `app/main.py` stays justified.
    """
    query_customer_data(ALL_CUSTOMERS, RM1)

    tool_ms = timing.get_samples("tool_query_customer_data")[0]["duration_ms"]
    nested_ms = sum(
        s["duration_ms"]
        for rail in ("sql_guard", "entitlement")
        for s in timing.get_samples(rail)
    )

    assert nested_ms <= tool_ms, "the rails run inside the tool span, not beside it"


def test_rail_timer_records_rejected_calls_too():
    """Rejections are often the slow path; a budget table built only from successes lies."""
    query_customer_data("DROP TABLE customers", RM1)
    assert len(timing.get_samples("tool_query_customer_data")) == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda: query_customer_data(ALL_CUSTOMERS, RM1),
        lambda: get_account_summary(IN_SCOPE_ACCOUNT, RM1),
    ],
)
def test_sql_tools_have_no_path_to_the_database_except_safe_execute(monkeypatch, call):
    """§2.3: "No exceptions." Proved by removing the only sanctioned path.

    If either tool could reach Postgres another way, it would return rows here instead of
    raising.
    """

    def tripwire(*args, **kwargs):
        raise RuntimeError("bypassed safe_execute")

    monkeypatch.setattr(tools, "safe_execute", tripwire)

    with pytest.raises(RuntimeError, match="bypassed safe_execute"):
        call()
