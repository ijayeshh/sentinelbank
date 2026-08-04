"""Tests for the human-in-the-loop rail (§3.3).

HITL is the project's compliance control, so the tests are about the boundary between
"drafted" and "executed". Four things must hold, and the third and fourth are the ones
that would matter in an audit:

  * an AUTO tool runs with no ceremony;
  * a HUMAN_APPROVAL tool interrupts, and **nothing has changed in the database at the
    moment the RM is asked** — the approval is genuinely prospective;
  * approve executes and reject does not;
  * a HARD_BLOCK action does not execute **even when an approval is sent**, because
    sanctions screening is not a judgement an RM is permitted to make (§3.8).

The graph is driven by a scripted stub model, so these run with no Ollama and no LLM: the
tool calls are fixed, which is what makes assertions about interrupt/resume deterministic.

Database isolation is by injected session + rollback, as in tests/test_write_tools.py, so
the seeded data stays byte-identical (§6 rule 3). Checkpoints go to tmp_path, never the
repo's ./checkpoints.db.
"""

from __future__ import annotations

import sqlite3

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from sqlalchemy import text

from app.agent.tools import TOOL_REGISTRY, RiskTier, get_account_summary
from app.db import SessionLocal
from app.rails import timing
from app.rails.hitl import (
    GATED_TIERS,
    HardBlockMiddleware,
    build_approval_diff,
    build_hitl_middleware,
    build_write_tools,
    fetch_policy_citation,
    gated_tool_names,
    hitl_config,
)

RM1 = "RM-001"
REG_E_DOC = "reg_e_dispute_policy.md"


# --------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------


class ScriptedModel(FakeMessagesListChatModel):
    """A stub chat model that `create_agent` will accept.

    `FakeMessagesListChatModel` does not implement `bind_tools`, which `create_agent`
    calls unconditionally. Returning `self` is enough — the responses are scripted, so
    the model never needs to know what tools exist. This is what keeps the whole
    interrupt/resume cycle testable without Ollama.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ANN201
        return self


@pytest.fixture
def session():
    """A rolled-back session, so real mutations are exercised without persisting."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(autouse=True)
def isolate_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "LATENCY_LOG_PATH", tmp_path / "latency.jsonl")
    timing.clear_samples()


@pytest.fixture
def saver(tmp_path):
    """A file-backed checkpointer under tmp_path, and its path for reload tests."""
    path = tmp_path / "checkpoints.db"
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    yield saver, path
    connection.close()


def fetch_one(session, sql: str, **params):
    return session.execute(text(sql), params).first()


@pytest.fixture
def disputable_txn(session):
    """An undisputed debit belonging to RM-001."""
    row = fetch_one(
        session,
        """
        SELECT t.txn_id, t.is_disputed
        FROM transactions t
        JOIN accounts a ON a.account_id = t.account_id
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE c.rm_id = :rm AND t.is_disputed = false AND t.amount < 0
        ORDER BY t.txn_id LIMIT 1
        """,
        rm=RM1,
    )
    assert row is not None, "seed data has drifted; re-run python db/seed.py"
    return row[0]


@pytest.fixture
def ofac_account(session):
    """An account belonging to one of the two OFAC-flagged customers (§1.6)."""
    row = fetch_one(
        session,
        """
        SELECT a.account_id, c.rm_id
        FROM accounts a JOIN customers c ON c.customer_id = a.customer_id
        WHERE c.ofac_flag = true
        ORDER BY a.account_id LIMIT 1
        """,
    )
    assert row is not None, "no OFAC-flagged customer; re-run python db/seed.py"
    return row[0], row[1]


def build_test_agent(session, saver, calls: list[dict], rm_id: str = RM1):
    """An agent whose model emits `calls` and then a final answer."""
    hitl = build_hitl_middleware(rm_id, session=session)
    hard_block = HardBlockMiddleware(rm_id, session=session)
    tools = build_write_tools(rm_id, hitl, session=session)

    responses = [AIMessage(content="", tool_calls=calls), AIMessage(content="Done.")]
    agent = create_agent(
        model=ScriptedModel(responses=responses),
        tools=[*tools, _account_summary_tool(rm_id, session)],
        middleware=[hitl, hard_block],
        checkpointer=saver,
    )
    return agent, hard_block


def _account_summary_tool(rm_id: str, session):
    """The AUTO-tier read tool, wrapped the same way the write tools are."""
    from langchain_core.tools import StructuredTool

    def run(account_id: str) -> dict:
        return get_account_summary(account_id, rm_id)

    return StructuredTool.from_function(
        func=run, name="get_account_summary", description="Summarise one account."
    )


def is_disputed(session, txn_id: str) -> bool:
    return fetch_one(
        session, "SELECT is_disputed FROM transactions WHERE txn_id = :t", t=txn_id
    )[0]


# ================================================================================
# 1. Tier routing
# ================================================================================


def test_gated_tiers_come_from_the_phase_2_metadata():
    """The gate is derived from `func.risk_tier`, never a second hand-kept list."""
    gated = set(gated_tool_names())

    assert gated == {
        name
        for name, tool in TOOL_REGISTRY.items()
        if getattr(tool, "risk_tier", RiskTier.AUTO) in GATED_TIERS
    }
    # Every write tool, and no read tool.
    assert gated == {
        "file_dispute",
        "waive_fee",
        "block_card",
        "update_contact_info",
        "initiate_wire",
    }


def test_every_gated_tool_is_in_interrupt_on():
    """No write tool may sit silently outside the gate."""
    hitl = build_hitl_middleware(RM1)

    for name in gated_tool_names():
        assert name in hitl.interrupt_on, f"{name} is a write tool outside the gate"
        assert set(hitl.interrupt_on[name]["allowed_decisions"]) == {"approve", "reject"}


def test_auto_tools_are_not_gated():
    hitl = build_hitl_middleware(RM1)

    for name in ("query_customer_data", "get_account_summary", "search_policy_docs"):
        assert name not in hitl.interrupt_on


def test_auto_tool_executes_without_interrupt(session, saver, disputable_txn):
    """§3.3's AUTO row: executes immediately."""
    account_id = fetch_one(
        session,
        "SELECT a.account_id FROM accounts a JOIN customers c ON c.customer_id = a.customer_id "
        "WHERE c.rm_id = :rm ORDER BY a.account_id LIMIT 1",
        rm=RM1,
    )[0]

    agent, _ = build_test_agent(
        session,
        saver[0],
        [{"name": "get_account_summary", "args": {"account_id": account_id}, "id": "c1"}],
    )
    result = agent.invoke({"messages": [("user", "summarise")]}, hitl_config("t-auto"))

    assert "__interrupt__" not in result


# ================================================================================
# 2. Interrupt, approve, reject
# ================================================================================


def test_file_dispute_interrupts_and_does_not_mutate_before_approval(
    session, saver, disputable_txn
):
    """The property the whole rail exists for: nothing has happened when the RM is asked."""
    assert is_disputed(session, disputable_txn) is False

    agent, _ = build_test_agent(
        session,
        saver[0],
        [{"name": "file_dispute", "args": {"txn_id": disputable_txn}, "id": "c1"}],
    )
    result = agent.invoke({"messages": [("user", "dispute it")]}, hitl_config("t-1"))

    assert "__interrupt__" in result
    # The database is untouched at the moment of the interrupt.
    assert is_disputed(session, disputable_txn) is False
    assert (
        fetch_one(
            session,
            "SELECT COUNT(*) FROM service_requests WHERE created_by = :rm AND status = 'PENDING_REVIEW'",
            rm=RM1,
        )[0]
        == 0
    )


def test_approving_executes_the_mutation(session, saver, disputable_txn):
    agent, _ = build_test_agent(
        session,
        saver[0],
        [{"name": "file_dispute", "args": {"txn_id": disputable_txn}, "id": "c1"}],
    )
    config = hitl_config("t-approve")
    agent.invoke({"messages": [("user", "dispute it")]}, config)

    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert is_disputed(session, disputable_txn) is True
    assert (
        fetch_one(
            session,
            "SELECT COUNT(*) FROM service_requests WHERE created_by = :rm AND category = 'DISPUTE'",
            rm=RM1,
        )[0]
        >= 1
    )


def test_rejecting_leaves_the_database_unchanged(session, saver, disputable_txn):
    agent, _ = build_test_agent(
        session,
        saver[0],
        [{"name": "file_dispute", "args": {"txn_id": disputable_txn}, "id": "c1"}],
    )
    config = hitl_config("t-reject")
    agent.invoke({"messages": [("user", "dispute it")]}, config)

    agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "not authorised"}]}),
        config,
    )

    assert is_disputed(session, disputable_txn) is False


def test_state_survives_the_pause_with_a_fresh_checkpointer(
    session, saver, disputable_txn, tmp_path
):
    """§3.3's persistence requirement, tested the way it actually fails in production.

    The RM's approval can arrive minutes later, after the process has restarted. Resuming
    from a *newly constructed* SqliteSaver over the same file — rather than the in-memory
    object that created the interrupt — is what proves the state is really on disk and
    not merely in the first saver's memory.
    """
    first_saver, path = saver
    config = hitl_config("t-persist")

    agent, _ = build_test_agent(
        session,
        first_saver,
        [{"name": "file_dispute", "args": {"txn_id": disputable_txn}, "id": "c1"}],
    )
    result = agent.invoke({"messages": [("user", "dispute it")]}, config)
    assert "__interrupt__" in result
    assert is_disputed(session, disputable_txn) is False

    # Simulate the process restarting: brand-new connection, saver, and agent.
    connection = sqlite3.connect(path, check_same_thread=False)
    try:
        reloaded = SqliteSaver(connection)
        reloaded.setup()

        resumed_agent, _ = build_test_agent(
            session,
            reloaded,
            [{"name": "file_dispute", "args": {"txn_id": disputable_txn}, "id": "c1"}],
        )
        resumed_agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    finally:
        connection.close()

    assert is_disputed(session, disputable_txn) is True


def test_thread_id_is_required():
    """Without a thread_id there is nowhere to resume from, so it fails loudly."""
    with pytest.raises(ValueError, match="thread_id is required"):
        hitl_config("")


# ================================================================================
# 3. HARD_BLOCK — the §3.8 criterion
# ================================================================================


def test_ofac_wire_is_blocked_even_when_approved(session, saver, ofac_account):
    """Approval is sent and the wire still does not execute.

    This is the criterion §3.8 spells out. The block does not consult the decision at
    all — `HardBlockMiddleware` refuses at `wrap_tool_call` without calling the handler —
    so there is no approval payload that could reach the mutation.
    """
    account_id, rm_id = ofac_account
    before = fetch_one(
        session, "SELECT balance FROM accounts WHERE account_id = :a", a=account_id
    )[0]
    txns_before = fetch_one(
        session, "SELECT COUNT(*) FROM transactions WHERE account_id = :a", a=account_id
    )[0]

    agent, hard_block = build_test_agent(
        session,
        saver[0],
        [
            {
                "name": "initiate_wire",
                "args": {"account_id": account_id, "amount": 500.0, "beneficiary": "ACME"},
                "id": "c1",
            }
        ],
        rm_id=rm_id,
    )
    config = hitl_config("t-ofac")
    agent.invoke({"messages": [("user", "wire 500")]}, config)

    # The RM approves anyway.
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    assert hard_block.blocked, "the hard block did not fire"
    assert hard_block.blocked[0]["reason"] == "ofac_hard_block"

    after = fetch_one(
        session, "SELECT balance FROM accounts WHERE account_id = :a", a=account_id
    )[0]
    txns_after = fetch_one(
        session, "SELECT COUNT(*) FROM transactions WHERE account_id = :a", a=account_id
    )[0]
    assert after == before, "an approved OFAC wire changed the balance"
    assert txns_after == txns_before, "an approved OFAC wire posted a transaction"

    escalation = [
        m for m in result["messages"] if getattr(m, "name", None) == "initiate_wire"
    ]
    assert escalation and "OFAC" in escalation[-1].content


def test_non_ofac_wire_is_not_blocked_by_the_hard_block_middleware(session, ofac_account):
    """The block is targeted: an ordinary customer's wire passes the middleware."""
    clean_account = fetch_one(
        session,
        "SELECT a.account_id FROM accounts a JOIN customers c ON c.customer_id = a.customer_id "
        "WHERE c.rm_id = :rm AND c.ofac_flag = false ORDER BY a.account_id LIMIT 1",
        rm=RM1,
    )[0]

    middleware = HardBlockMiddleware(RM1, session=session)

    assert middleware._ofac_flagged({"account_id": clean_account}) is False
    assert middleware._ofac_flagged({"account_id": ofac_account[0]}) is (
        ofac_account[1] == RM1
    )


def test_hard_block_ignores_tools_of_other_tiers(session):
    """It must not become a second, accidental gate on HUMAN_APPROVAL tools."""
    middleware = HardBlockMiddleware(RM1, session=session)
    called = []

    class Request:
        tool_call = {"name": "file_dispute", "args": {"txn_id": "TXN-000001"}, "id": "c1"}

    middleware.wrap_tool_call(Request(), lambda r: called.append(r) or "ran")

    assert called, "a HUMAN_APPROVAL tool was intercepted by the hard block"


# ================================================================================
# 4. The structured diff (§3.3)
# ================================================================================


def test_diff_is_structured_not_a_yes_no(session, disputable_txn):
    diff = build_approval_diff(
        "file_dispute", {"txn_id": disputable_txn}, RM1, session=session
    )

    for label in ("Tool:", "Customer:", "Account:", "Txn:", "Change:", "Policy:"):
        assert label in diff, f"diff is missing the {label} line:\n{diff}"

    assert "file_dispute" in diff
    assert disputable_txn in diff
    assert "is_disputed: False -> True" in diff
    assert "DISPUTE" in diff and "PENDING_REVIEW" in diff
    # The account is identified by last four only (§3.1.2).
    assert "****" in diff


def test_diff_reg_e_citation_is_retrieved_from_policy_chunks(session, disputable_txn):
    """§3.3: the citation must come from the RAG, not from a constant.

    Asserted against the actual document text rather than merely being non-empty — a
    hardcoded string would pass a "contains Reg E" check, so the test compares the
    retrieved chunk to what is really in db/policies/reg_e_dispute_policy.md.
    """
    citation = fetch_policy_citation("file_dispute")

    assert citation["source_doc"] == REG_E_DOC
    assert citation["citation"]

    # The retrieved text must genuinely appear in the stored policy document.
    stored = fetch_one(
        session,
        "SELECT string_agg(chunk_text, ' ') FROM policy_chunks WHERE doc_name = :d",
        d=REG_E_DOC,
    )[0]
    normalised_stored = " ".join(stored.split())
    assert " ".join(citation["citation"].split()) in normalised_stored

    # And it carries the operative Reg E rule §3.3 quotes.
    assert "10 business days" in citation["citation"]

    diff = build_approval_diff(
        "file_dispute", {"txn_id": disputable_txn}, RM1, session=session
    )
    assert "10 business days" in diff
    assert REG_E_DOC in diff


def test_diff_for_a_refusal_explains_rather_than_asking(session, ofac_account):
    """An action that cannot succeed must not be presented as an approvable change."""
    account_id, rm_id = ofac_account

    diff = build_approval_diff(
        "initiate_wire",
        {"account_id": account_id, "amount": 500.0},
        rm_id,
        session=session,
    )

    assert "ofac_hard_block" in diff
    assert "cannot be approved" in diff
    assert "->" not in diff, "a blocked action must not render an actionable change"


def test_diff_is_ascii_renderable(session, disputable_txn):
    """The prompt is printed to a Windows console; cp1252 must not blow up on it."""
    diff = build_approval_diff(
        "file_dispute", {"txn_id": disputable_txn}, RM1, session=session
    )
    # Only the retrieved policy text may carry non-ASCII; our own formatting must not.
    rendered = diff.split("Policy:")[0]
    rendered.encode("cp1252")


def test_dry_run_mutates_nothing(session, disputable_txn):
    """The diff is built by a dry run, and a dry run must be inert."""
    from app.agent.tools import file_dispute

    before = is_disputed(session, disputable_txn)
    tickets_before = fetch_one(session, "SELECT COUNT(*) FROM service_requests")[0]

    result = file_dispute(disputable_txn, RM1, session=session, dry_run=True)

    assert result["ok"] and result["reason"] == "preview"
    assert result["preview"]["changes"]
    assert is_disputed(session, disputable_txn) == before
    assert fetch_one(session, "SELECT COUNT(*) FROM service_requests")[0] == tickets_before


def test_dry_run_needs_no_approval_but_the_real_call_does(session, disputable_txn):
    """The preview exists in order to ask for approval, so it cannot itself require it."""
    from app.agent.tools import ApprovalRequiredError, file_dispute

    file_dispute(disputable_txn, RM1, session=session, dry_run=True)

    with pytest.raises(ApprovalRequiredError):
        file_dispute(disputable_txn, RM1, session=session)


# ================================================================================
# 5. The wrappers keep rm_id away from the model
# ================================================================================


def test_rm_id_is_absent_from_every_tool_schema():
    """§6 rule 5: the model must be structurally incapable of choosing an identity."""
    hitl = build_hitl_middleware(RM1)

    for tool in build_write_tools(RM1, hitl):
        args = set(tool.args)
        assert "rm_id" not in args
        assert "approved_context" not in args
        assert "session" not in args
        assert "dry_run" not in args
        assert args, f"{tool.name} exposes no arguments at all"


def test_build_write_tools_refuses_when_the_gate_is_missing():
    """The wrapper infers approval from execution, so the gate must provably exist."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    ungated = HumanInTheLoopMiddleware(interrupt_on={})

    with pytest.raises(ValueError, match="not gated"):
        build_write_tools(RM1, ungated)


def test_build_write_tools_refuses_a_partial_gate():
    """Gating four of five write tools is not a gate."""
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    partial = HumanInTheLoopMiddleware(
        interrupt_on={
            name: {"allowed_decisions": ["approve", "reject"]}
            for name in gated_tool_names()
            if name != "initiate_wire"
        }
    )

    with pytest.raises(ValueError, match="initiate_wire"):
        build_write_tools(RM1, partial)
