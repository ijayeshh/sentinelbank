"""Tests for the five write tools (PROJECT_PLAN.md §2.3, tiers per §3.3).

These exercise **real mutations against the seeded database**, which creates a problem
the read-tool tests do not have: the seed is deterministic by requirement (§6 rule 3) and
the red-team harness asserts against specific rows. So every mutating test runs inside a
transaction that is rolled back in the fixture teardown. The tools accept an injected
session precisely so the caller can own that transaction; nothing here commits.

Three properties carry most of the weight:

  * **Unreachable without approval.** §2.3 requires write tools to be inert outside an
    approved context. Tested per tool, not once.
  * **Entitlement applies to writes.** An RM cannot mutate another RM's customer — and
    the target row is checked afterwards to confirm nothing moved.
  * **OFAC is a hard block.** `initiate_wire` against a flagged customer must fail *with*
    a valid approval in hand. That is the §3.8 acceptance criterion and the one assertion
    in this file that distinguishes a hard block from a stern warning.

Requires `docker compose up -d` and `python db/seed.py`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import text

from app.agent.tools import (
    TOOL_REGISTRY,
    ApprovalContext,
    ApprovalRequiredError,
    RiskTier,
    Source,
    block_card,
    file_dispute,
    initiate_wire,
    update_contact_info,
    waive_fee,
)
from app.db import SessionLocal, get_session
from app.rails import timing

RM1, RM2 = "RM-001", "RM-002"

# Verified seed fixtures.
RM1_CUSTOMER = "CUST-0001"
RM1_ACCOUNT = "ACCT-0001"  # ACTIVE, balance 48135.25
RM1_DEBIT_TXN = "TXN-000001"  # -337.07 on ACCT-0001, not disputed

# CUST-0007 is one of the two OFAC-flagged customers AND belongs to RM-001 — so the hard
# block is tested on a customer the RM is otherwise fully entitled to act on. A flagged
# customer belonging to the *other* RM would be refused by entitlement first and would
# prove nothing about OFAC.
OFAC_CUSTOMER = "CUST-0007"
OFAC_ACCOUNT = "ACCT-0007"

RM2_CUSTOMER = "CUST-0023"
RM2_ACCOUNT = "ACCT-0023"
RM2_DEBIT_TXN = "TXN-000221"


@pytest.fixture(scope="module", autouse=True)
def require_seeded_database():
    try:
        with get_session() as session:
            flagged = session.execute(
                text("SELECT customer_id, rm_id FROM customers WHERE ofac_flag")
            ).all()
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "write-tool tests require the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )
    assert (OFAC_CUSTOMER, RM1) in flagged, (
        "the OFAC hard-block test needs CUST-0007 flagged and owned by RM-001; "
        "re-run python db/seed.py"
    )


@pytest.fixture
def tx():
    """A session whose transaction is always rolled back.

    Write tools given a session do not commit — the caller owns the transaction — so the
    mutations below are real, visible to subsequent reads on this session, and gone the
    moment the test ends. The seed stays byte-identical.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(autouse=True)
def isolate_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "LATENCY_LOG_PATH", tmp_path / "latency.jsonl")
    timing.clear_samples()


def approval(tool: str, rm_id: str = RM1) -> ApprovalContext:
    return ApprovalContext(tool=tool, rm_id=rm_id)


def scalar(session, sql: str, **params):
    return session.execute(text(sql), params).scalar()


# --------------------------------------------------------------------------------
# 1. Every write tool is inert without an approved context
# --------------------------------------------------------------------------------

WRITE_CALLS = {
    "file_dispute": lambda ctx, s: file_dispute(
        RM1_DEBIT_TXN, RM1, "test", approved_context=ctx, session=s
    ),
    "waive_fee": lambda ctx, s: waive_fee(RM1_DEBIT_TXN, RM1, approved_context=ctx, session=s),
    "block_card": lambda ctx, s: block_card(RM1_ACCOUNT, RM1, approved_context=ctx, session=s),
    "update_contact_info": lambda ctx, s: update_contact_info(
        RM1_CUSTOMER, RM1, approved_context=ctx, session=s, email="new@example.com"
    ),
    "initiate_wire": lambda ctx, s: initiate_wire(
        RM1_ACCOUNT, Decimal("100.00"), RM1, "ACME", approved_context=ctx, session=s
    ),
}


@pytest.mark.parametrize("tool_name", sorted(WRITE_CALLS))
def test_write_tool_refuses_without_an_approved_context(tool_name, tx):
    """§2.3: write tools must be unreachable outside an approved context."""
    with pytest.raises(ApprovalRequiredError):
        WRITE_CALLS[tool_name](None, tx)


@pytest.mark.parametrize("tool_name", sorted(WRITE_CALLS))
def test_write_tool_refuses_a_rejected_approval(tool_name, tx):
    ctx = ApprovalContext(tool=tool_name, rm_id=RM1, approved=False)
    with pytest.raises(ApprovalRequiredError, match="rejection"):
        WRITE_CALLS[tool_name](ctx, tx)


def test_an_approval_cannot_be_replayed_against_a_different_tool(tx):
    """Otherwise an approval is a bearer token.

    A human who approved a contact-info change did not approve a wire. Binding the
    context to the tool is what keeps one signature from authorising another action.
    """
    stolen = approval("update_contact_info")
    with pytest.raises(ApprovalRequiredError, match="issued for 'update_contact_info'"):
        initiate_wire(RM1_ACCOUNT, Decimal("100"), RM1, approved_context=stolen, session=tx)


def test_an_approval_cannot_be_replayed_by_a_different_rm(tx):
    other_rm = ApprovalContext(tool="block_card", rm_id=RM2)
    with pytest.raises(ApprovalRequiredError, match="issued for 'RM-002'"):
        block_card(RM1_ACCOUNT, RM1, approved_context=other_rm, session=tx)


def test_refusal_happens_before_any_mutation(tx):
    """The guard must be the first thing in the tool, not a check on the way out."""
    before = scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT)
    with pytest.raises(ApprovalRequiredError):
        block_card(RM1_ACCOUNT, RM1, session=tx)
    assert scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT) == before


# --------------------------------------------------------------------------------
# 2. With approval, each tool performs the correct mutation
# --------------------------------------------------------------------------------


def test_file_dispute_flags_the_txn_and_opens_a_pending_ticket(tx):
    result = file_dispute(
        RM1_DEBIT_TXN, RM1, "customer does not recognise this charge",
        approved_context=approval("file_dispute"), session=tx,
    )

    assert result["ok"] is True and result["reason"] == "ok"
    assert result["risk_tier"] == RiskTier.HUMAN_APPROVAL.value

    assert scalar(tx, "SELECT is_disputed FROM transactions WHERE txn_id = :t", t=RM1_DEBIT_TXN) is True

    ticket = tx.execute(
        text("SELECT category, status, created_by, customer_id FROM service_requests "
             "WHERE ticket_id = :tk"),
        {"tk": result["ticket_id"]},
    ).mappings().one()
    assert ticket["category"] == "DISPUTE"
    assert ticket["status"] == "PENDING_REVIEW"
    assert ticket["created_by"] == RM1
    assert ticket["customer_id"] == RM1_CUSTOMER


def test_file_dispute_refuses_a_transaction_already_disputed(tx):
    ctx = approval("file_dispute")
    assert file_dispute(RM1_DEBIT_TXN, RM1, approved_context=ctx, session=tx)["ok"] is True
    second = file_dispute(RM1_DEBIT_TXN, RM1, approved_context=ctx, session=tx)
    assert second["ok"] is False and second["reason"] == "already_disputed"


def test_waive_fee_posts_a_reversal_and_restores_the_balance(tx):
    before = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT)
    fee = scalar(tx, "SELECT amount FROM transactions WHERE txn_id = :t", t=RM1_DEBIT_TXN)

    result = waive_fee(RM1_DEBIT_TXN, RM1, approved_context=approval("waive_fee"), session=tx)
    assert result["ok"] is True

    after = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT)
    assert after == before - fee  # fee is negative, so this is a credit back

    reversal = tx.execute(
        text("SELECT amount, merchant FROM transactions WHERE txn_id = :t"),
        {"t": result["reversal_txn_id"]},
    ).mappings().one()
    assert reversal["amount"] == -fee
    assert reversal["merchant"] == "FEE REVERSAL"

    # The original is preserved: reversal by compensating entry, not by editing history.
    assert scalar(tx, "SELECT amount FROM transactions WHERE txn_id = :t", t=RM1_DEBIT_TXN) == fee


def test_waive_fee_refuses_a_credit(tx):
    credit = tx.execute(
        text("SELECT t.txn_id FROM transactions t JOIN accounts a ON a.account_id = t.account_id "
             "JOIN customers c ON c.customer_id = a.customer_id "
             "WHERE c.rm_id = :rm AND t.amount > 0 ORDER BY t.txn_id LIMIT 1"),
        {"rm": RM1},
    ).scalar()
    result = waive_fee(credit, RM1, approved_context=approval("waive_fee"), session=tx)
    assert result["ok"] is False and result["reason"] == "not_a_debit"


def test_block_card_freezes_the_account(tx):
    assert scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT) == "ACTIVE"

    result = block_card(RM1_ACCOUNT, RM1, approved_context=approval("block_card"), session=tx)

    assert result["ok"] is True
    assert scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT) == "FROZEN"

    again = block_card(RM1_ACCOUNT, RM1, approved_context=approval("block_card"), session=tx)
    assert again["ok"] is False and again["reason"] == "already_frozen"


def test_update_contact_info_updates_only_the_named_fields(tx):
    before = tx.execute(
        text("SELECT email, phone, city FROM customers WHERE customer_id = :c"),
        {"c": RM1_CUSTOMER},
    ).mappings().one()

    result = update_contact_info(
        RM1_CUSTOMER, RM1,
        approved_context=approval("update_contact_info"), session=tx,
        email="updated@example.com", phone="(555) 000-1111",
    )
    assert result["ok"] is True

    after = tx.execute(
        text("SELECT email, phone, city FROM customers WHERE customer_id = :c"),
        {"c": RM1_CUSTOMER},
    ).mappings().one()
    assert after["email"] == "updated@example.com"
    assert after["phone"] == "(555) 000-1111"
    assert after["city"] == before["city"], "untouched fields must not change"


def test_update_contact_info_refuses_to_write_rm_id(tx):
    """The entitlement rail's front door.

    Reassigning a customer to another RM through a "contact update" would move rows out
    of scope without ever defeating the SQL rail. The field allowlist is what stops it.
    """
    result = update_contact_info(
        RM1_CUSTOMER, RM1,
        approved_context=approval("update_contact_info"), session=tx,
        rm_id_new=RM2,
    )
    assert result["ok"] is False
    assert result["reason"].startswith("field_not_updatable")
    assert scalar(tx, "SELECT rm_id FROM customers WHERE customer_id = :c", c=RM1_CUSTOMER) == RM1


def test_initiate_wire_debits_a_clean_customer(tx):
    """The non-flagged path must work, or the OFAC test proves nothing.

    A tool that blocks everything also blocks every attack — the same reason §4.1
    requires a benign set alongside the attacks.
    """
    before = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT)

    result = initiate_wire(
        RM1_ACCOUNT, Decimal("2500.00"), RM1, "ACME SUPPLY CO",
        approved_context=approval("initiate_wire"), session=tx,
    )

    assert result["ok"] is True
    assert scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT) == before - Decimal("2500.00")
    assert scalar(tx, "SELECT amount FROM transactions WHERE txn_id = :t", t=result["wire_txn_id"]) == Decimal("-2500.00")
    assert scalar(tx, "SELECT channel FROM transactions WHERE txn_id = :t", t=result["wire_txn_id"]) == "WIRE"


# --------------------------------------------------------------------------------
# 3. Cross-RM mutations are refused
# --------------------------------------------------------------------------------


def test_cross_rm_file_dispute_is_refused_and_changes_nothing(tx):
    result = file_dispute(
        RM2_DEBIT_TXN, RM1, approved_context=approval("file_dispute"), session=tx
    )
    assert result["ok"] is False and result["reason"] == "not_found_or_not_entitled"
    assert scalar(tx, "SELECT is_disputed FROM transactions WHERE txn_id = :t", t=RM2_DEBIT_TXN) is False


def test_cross_rm_block_card_is_refused_and_changes_nothing(tx):
    # Captured rather than hardcoded: ACCT-0023's seeded status is not the point of this
    # test, and asserting it would couple the entitlement check to unrelated seed detail.
    before = scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM2_ACCOUNT)

    result = block_card(RM2_ACCOUNT, RM1, approved_context=approval("block_card"), session=tx)

    assert result["ok"] is False and result["reason"] == "not_found_or_not_entitled"
    assert scalar(tx, "SELECT status FROM accounts WHERE account_id = :a", a=RM2_ACCOUNT) == before


def test_cross_rm_contact_update_is_refused_and_changes_nothing(tx):
    before = scalar(tx, "SELECT email FROM customers WHERE customer_id = :c", c=RM2_CUSTOMER)
    result = update_contact_info(
        RM2_CUSTOMER, RM1,
        approved_context=approval("update_contact_info"), session=tx,
        email="attacker@example.com",
    )
    assert result["ok"] is False and result["reason"] == "not_found_or_not_entitled"
    assert scalar(tx, "SELECT email FROM customers WHERE customer_id = :c", c=RM2_CUSTOMER) == before


def test_cross_rm_waive_fee_is_refused(tx):
    result = waive_fee(RM2_DEBIT_TXN, RM1, approved_context=approval("waive_fee"), session=tx)
    assert result["ok"] is False and result["reason"] == "not_found_or_not_entitled"


def test_cross_rm_wire_is_refused(tx):
    before = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM2_ACCOUNT)
    result = initiate_wire(
        RM2_ACCOUNT, Decimal("100"), RM1, approved_context=approval("initiate_wire"), session=tx
    )
    assert result["ok"] is False and result["reason"] == "not_found_or_not_entitled"
    assert scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM2_ACCOUNT) == before


# --------------------------------------------------------------------------------
# 4. THE CRITICAL ASSERTION — OFAC is a hard block, not a stern warning
# --------------------------------------------------------------------------------


def test_initiate_wire_is_hard_blocked_on_an_ofac_customer_even_with_approval(tx):
    """§3.3/§3.8: "Never executes regardless of approval."

    The approval here is entirely valid — right tool, right RM, approved=True — and
    RM-001 genuinely owns CUST-0007, so entitlement does not refuse it either. The only
    thing standing between this call and a sanctions violation is the OFAC branch.
    """
    balance_before = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=OFAC_ACCOUNT)
    txn_count_before = scalar(tx, "SELECT COUNT(*) FROM transactions WHERE account_id = :a", a=OFAC_ACCOUNT)

    result = initiate_wire(
        OFAC_ACCOUNT, Decimal("5000.00"), RM1, "ANY BENEFICIARY",
        approved_context=approval("initiate_wire"),  # a genuine, valid approval
        session=tx,
    )

    assert result["ok"] is False
    assert result["reason"] == "ofac_hard_block"
    assert result["risk_tier"] == RiskTier.HARD_BLOCK.value
    assert result["overridable"] is False
    assert "OFAC" in result["message"] and "Compliance" in result["message"]

    # Nothing moved. This is the assertion that matters.
    assert scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=OFAC_ACCOUNT) == balance_before
    assert scalar(tx, "SELECT COUNT(*) FROM transactions WHERE account_id = :a", a=OFAC_ACCOUNT) == txn_count_before

    # The preview says "no change is possible", not "here is a pending change".
    assert result["preview"]["changes"] == []
    assert result["preview"]["blocked"] is True
    assert result["preview"]["customer"]["ofac_flag"] is True


def test_ofac_block_holds_for_every_account_of_the_flagged_customer(tx):
    """CUST-0007 holds two accounts. The flag is on the customer, not the account."""
    for account_id in ("ACCT-0007", "ACCT-0047"):
        result = initiate_wire(
            account_id, Decimal("1.00"), RM1,
            approved_context=approval("initiate_wire"), session=tx,
        )
        assert result["ok"] is False and result["reason"] == "ofac_hard_block"


def test_ofac_block_is_not_a_pii_or_amount_edge_case(tx):
    """No amount, beneficiary, or absent beneficiary makes the block negotiable."""
    for amount, beneficiary in ((Decimal("0.01"), ""), (Decimal("999999"), "X"), (1, "ACME")):
        result = initiate_wire(
            OFAC_ACCOUNT, amount, RM1, beneficiary,
            approved_context=approval("initiate_wire"), session=tx,
        )
        assert result["reason"] == "ofac_hard_block"


# --------------------------------------------------------------------------------
# 5. The preview dict carries usable before/after values
# --------------------------------------------------------------------------------


def test_dispute_preview_renders_a_usable_diff(tx):
    """§3.3 requires a structured diff, not a bare yes/no, at the approval step."""
    result = file_dispute(
        RM1_DEBIT_TXN, RM1, "unrecognised charge",
        approved_context=approval("file_dispute"), session=tx,
    )
    preview = result["preview"]

    assert preview["customer"]["customer_id"] == RM1_CUSTOMER
    # Last 4 only: the full number is US_BANK_NUMBER PII.
    assert len(preview["account"]["account_number_last4"]) == 4

    changes = {c["field"]: c for c in preview["changes"]}
    flag = changes[f"transactions.{RM1_DEBIT_TXN}.is_disputed"]
    assert flag["before"] is False and flag["after"] is True

    ticket_status = next(c for f, c in changes.items() if f.endswith(".status"))
    assert ticket_status["before"] is None and ticket_status["after"] == "PENDING_REVIEW"

    # The transaction context is field-tagged like any other tool result, so the approval
    # UI can mark an attacker-controlled memo rather than render it as fact.
    assert preview["transaction"]["description"]["source"] == Source.UNTRUSTED.value
    assert preview["transaction"]["amount"]["source"] == Source.TRUSTED.value

    # The Reg E citation seam — Phase 3 fills this from the policy RAG (§3.3).
    assert "citation" in preview["policy"]


def test_balance_previews_carry_real_before_and_after_numbers(tx):
    before = scalar(tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT)

    result = waive_fee(RM1_DEBIT_TXN, RM1, approved_context=approval("waive_fee"), session=tx)
    change = next(
        c for c in result["preview"]["changes"] if c["field"].endswith(".balance")
    )

    assert change["before"] == before
    assert change["after"] > change["before"]
    assert change["after"] == scalar(
        tx, "SELECT balance FROM accounts WHERE account_id = :a", a=RM1_ACCOUNT
    ), "the preview's after-value must match what was actually written"


def test_contact_preview_shows_the_old_and_new_values(tx):
    before = scalar(tx, "SELECT email FROM customers WHERE customer_id = :c", c=RM1_CUSTOMER)

    result = update_contact_info(
        RM1_CUSTOMER, RM1,
        approved_context=approval("update_contact_info"), session=tx,
        email="new@example.com",
    )
    change = result["preview"]["changes"][0]

    assert change["before"] == before
    assert change["after"] == "new@example.com"
    assert change["before"] != change["after"]


# --------------------------------------------------------------------------------
# 6. Registry and tiers
# --------------------------------------------------------------------------------


def test_all_eight_tools_are_registered_with_the_tiers_the_plan_assigns():
    assert {name: func.risk_tier for name, func in TOOL_REGISTRY.items()} == {
        "query_customer_data": RiskTier.AUTO,
        "get_account_summary": RiskTier.AUTO,
        "search_policy_docs": RiskTier.AUTO,
        "file_dispute": RiskTier.HUMAN_APPROVAL,
        "waive_fee": RiskTier.HUMAN_APPROVAL,
        "block_card": RiskTier.HUMAN_APPROVAL,
        "update_contact_info": RiskTier.HUMAN_APPROVAL,
        "initiate_wire": RiskTier.HARD_BLOCK,
    }


def test_write_tools_are_timed_like_every_other_rail(tx):
    file_dispute(RM1_DEBIT_TXN, RM1, approved_context=approval("file_dispute"), session=tx)
    assert len(timing.get_samples("tool_file_dispute")) == 1
