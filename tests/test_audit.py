"""Tests for the tamper-evident hash-chained audit log (§3.5).

The chain makes the log tamper-*evident*, so these tests are mostly about tampering with
it. Three attacks matter and each is caught by a different half of the verification:

  * changing a field      -> the row's own hash no longer recomputes  (integrity)
  * deleting an entry     -> its successor is orphaned                (linkage)
  * reordering entries    -> the links break on both sides            (linkage)

The fourth claim is not about tampering at all but is the one a bank auditor would ask
about first: the stored text carries no raw PII. §1.5 requires it, and it is checked
against real seeded customer data rather than invented strings.

Every test runs inside a transaction that is rolled back, so the real `audit_log` is
never written to. `write_entry` and friends all accept an explicit `session` for exactly
this reason.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.audit.chain import (
    GENESIS_PREV_HASH,
    RAIL_AUDIT_WRITE,
    canonical_json,
    compute_entry_hash,
    read_entries,
    verify_chain,
    verify_chain_detailed,
    write_entry,
)
from app.db import SessionLocal
from app.rails import timing

RM1 = "RM-001"


# --------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------


@pytest.fixture
def session():
    """A session whose work is always rolled back.

    The audit log is append-only by design and this suite deliberately corrupts it, so
    nothing here may reach the real table. Emptying it inside the transaction gives every
    test a genesis-first chain to reason about.
    """
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM audit_log"))
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(autouse=True)
def isolate_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "LATENCY_LOG_PATH", tmp_path / "latency.jsonl")
    timing.clear_samples()


def write_sequence(session, count: int = 5) -> list:
    """Write `count` entries with varied, realistic content."""
    base = datetime(2026, 8, 3, 9, 0, 0)
    entries = []
    for index in range(count):
        entries.append(
            write_entry(
                RM1 if index % 2 == 0 else "RM-002",
                "ALLOWED" if index % 2 == 0 else "BLOCKED",
                user_query=f"query number {index}",
                response_summary=f"answer number {index}",
                rails_fired=[
                    {
                        "rail": "pii_input",
                        "verdict": "ALLOWED",
                        "score": 0.0 + index / 10,
                        "latency_ms": 7.4,
                    }
                ],
                tool_calls=[{"tool": "query_customer_data", "args": {"n": index}}],
                timestamp=base + timedelta(minutes=index),
                session=session,
            )
        )
    return entries


# ================================================================================
# 1. A sequence of entries forms a valid chain
# ================================================================================


def test_entries_form_a_linked_chain(session):
    entries = write_sequence(session, 5)

    assert entries[0].prev_hash == GENESIS_PREV_HASH
    for previous, current in zip(entries, entries[1:]):
        assert current.prev_hash == previous.entry_hash

    assert len({e.entry_hash for e in entries}) == 5
    assert all(len(e.entry_hash) == 64 for e in entries)


def test_genesis_entry_uses_the_sentinel(session):
    entry = write_entry(RM1, "ALLOWED", user_query="first ever", session=session)

    assert entry.prev_hash == GENESIS_PREV_HASH == "0" * 64


def test_verify_chain_returns_none_on_a_clean_log(session):
    write_sequence(session, 5)

    assert verify_chain(session) is None
    assert verify_chain_detailed(session) is None


def test_verify_chain_is_none_on_an_empty_log(session):
    """An empty log is intact, not broken. Fail closed must not mean fail noisy."""
    assert verify_chain(session) is None


# ================================================================================
# 2. Tampering is detected
# ================================================================================


@pytest.mark.parametrize(
    "column,value",
    [
        ("decision", "APPROVED"),
        ("rm_id", "RM-002"),
        ("user_query", "something else entirely"),
        ("response_summary", "a different answer"),
    ],
)
def test_mutating_a_field_in_a_middle_entry_is_caught_at_that_index(
    session, column, value
):
    """The headline property: a surgical UPDATE cannot pass unnoticed."""
    entries = write_sequence(session, 5)
    target = entries[2]

    session.execute(
        text(f"UPDATE audit_log SET {column} = :value WHERE event_id = :id"),
        {"value": value, "id": target.event_id},
    )

    assert verify_chain(session) == 2

    verdict = verify_chain_detailed(session)
    assert verdict.reason == "hash_mismatch"
    assert verdict.event_id == target.event_id


def test_mutating_the_jsonb_rails_is_caught(session):
    """The JSONB columns are inside the hash too, not just the text ones."""
    entries = write_sequence(session, 4)

    session.execute(
        text("UPDATE audit_log SET rails_fired = CAST(:v AS jsonb) WHERE event_id = :id"),
        {
            "v": json.dumps([{"rail": "pii_input", "verdict": "ALLOWED", "score": 9.9}]),
            "id": entries[1].event_id,
        },
    )

    assert verify_chain(session) == 1


def test_mutating_the_timestamp_is_caught(session):
    entries = write_sequence(session, 3)

    session.execute(
        text("UPDATE audit_log SET timestamp = :t WHERE event_id = :id"),
        {"t": datetime(2020, 1, 1, 0, 0, 0), "id": entries[1].event_id},
    )

    assert verify_chain(session) == 1


def test_deleting_an_entry_is_detected(session):
    """Deletion orphans the successor: its prev_hash points at a hash no longer present.

    Note the surviving rows are individually untouched and their own hashes still
    recompute correctly — only the linkage reveals the gap, which is why verification
    checks both.
    """
    entries = write_sequence(session, 5)

    session.execute(
        text("DELETE FROM audit_log WHERE event_id = :id"), {"id": entries[2].event_id}
    )

    # The entry formerly at index 3 now sits at index 2 and is the orphan.
    assert verify_chain(session) == 2

    verdict = verify_chain_detailed(session)
    assert verdict.reason == "broken_link"
    assert verdict.event_id == entries[3].event_id
    assert verdict.expected == entries[1].entry_hash


def test_deleting_the_genesis_entry_is_detected(session):
    """Truncating history from the front is the tidiest-looking attack and still fails."""
    entries = write_sequence(session, 4)

    session.execute(
        text("DELETE FROM audit_log WHERE event_id = :id"), {"id": entries[0].event_id}
    )

    assert verify_chain(session) == 0
    assert verify_chain_detailed(session).reason == "broken_link"


def test_reordering_entries_is_detected(session):
    """Order is committed to by the chain, not merely recorded by event_id."""
    entries = write_sequence(session, 5)
    first, second = entries[1].event_id, entries[2].event_id

    # Swap two adjacent entries' positions via a temporary id, so ORDER BY event_id
    # returns them the other way round.
    spare = 10_000
    for source, destination in ((first, spare), (second, first), (spare, second)):
        session.execute(
            text("UPDATE audit_log SET event_id = :dst WHERE event_id = :src"),
            {"dst": destination, "src": source},
        )

    broken = verify_chain(session)
    assert broken is not None
    assert broken == 1
    assert verify_chain_detailed(session).reason == "broken_link"


def test_appending_a_forged_entry_is_detected(session):
    """An attacker who adds a row must also produce a hash that chains from the tail."""
    write_sequence(session, 3)

    session.execute(
        text(
            """
            INSERT INTO audit_log
                (timestamp, rm_id, user_query, rails_fired, tool_calls, decision,
                 response_summary, prev_hash, entry_hash)
            VALUES (:t, 'RM-001', 'forged', '[]'::jsonb, '[]'::jsonb, 'ALLOWED',
                    'forged', :p, :e)
            """
        ),
        {"t": datetime(2026, 8, 3, 23, 0, 0), "p": GENESIS_PREV_HASH, "e": "f" * 64},
    )

    assert verify_chain(session) == 3


def test_a_tampered_entry_can_be_repaired_only_by_rewriting_the_tail(session):
    """Why the chain raises the cost of forgery rather than merely detecting one edit.

    Recomputing the tampered entry's own hash is not enough — every later entry stored
    the old value as its prev_hash, so the break simply moves down the chain.
    """
    entries = write_sequence(session, 4)
    target = entries[1]

    session.execute(
        text("UPDATE audit_log SET decision = 'APPROVED' WHERE event_id = :id"),
        {"id": target.event_id},
    )
    assert verify_chain(session) == 1

    # The attacker fixes just that row's hash.
    row = session.execute(
        text(
            "SELECT timestamp, rm_id, user_query, rails_fired, tool_calls, decision, "
            "response_summary, prev_hash FROM audit_log WHERE event_id = :id"
        ),
        {"id": target.event_id},
    ).mappings().first()
    from app.audit.chain import build_payload

    repaired = compute_entry_hash(row["prev_hash"], build_payload(**{
        k: v for k, v in row.items() if k != "prev_hash"
    }))
    session.execute(
        text("UPDATE audit_log SET entry_hash = :h WHERE event_id = :id"),
        {"h": repaired, "id": target.event_id},
    )

    # That entry now verifies, and the break has moved to its successor.
    assert verify_chain(session) == 2


# ================================================================================
# 3. Canonicalization
# ================================================================================


def test_canonical_json_is_key_order_independent():
    """The rule that stops verification producing false failures."""
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}

    assert canonical_json(a) == canonical_json(b)
    assert compute_entry_hash(GENESIS_PREV_HASH, a) == compute_entry_hash(
        GENESIS_PREV_HASH, b
    )


def test_canonical_json_has_no_whitespace_and_sorted_keys():
    assert canonical_json({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


def test_canonical_json_emits_utf8_literally():
    """`ensure_ascii=False` plus a UTF-8 encode — stated so external verifiers agree."""
    assert canonical_json({"m": "café"}) == '{"m":"café"}'.encode("utf-8")


def test_timezone_aware_and_naive_utc_hash_identically():
    """A tz-aware value loses its offset in a TIMESTAMP column; it must not change the hash."""
    from app.audit.chain import build_payload

    aware = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 8, 3, 12, 0, 0)

    common = dict(
        rm_id=RM1,
        user_query=None,
        rails_fired=[],
        tool_calls=[],
        decision="ALLOWED",
        response_summary=None,
    )
    assert build_payload(timestamp=aware, **common) == build_payload(
        timestamp=naive, **common
    )


def test_hash_survives_a_database_round_trip(session):
    """The false-failure test: what is read back must hash to what was written.

    JSONB normalizes and TIMESTAMP drops timezone, so this is the property that a naive
    canonicalization gets wrong — and it fails as a tamper alarm on an untampered log,
    which is worse than not checking at all.
    """
    written = write_sequence(session, 3)
    stored = read_entries(session)

    assert [entry["entry_hash"] for entry in stored] == [e.entry_hash for e in written]
    assert verify_chain(session) is None


def test_non_serializable_payload_is_rejected_not_coerced():
    """A silent str() fallback would hash two different objects identically."""
    with pytest.raises(TypeError):
        canonical_json({"bad": object()})


# ================================================================================
# 4. PII must never reach the table (§1.5)
# ================================================================================


@pytest.fixture(scope="module")
def seeded_customer():
    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT c.full_name, c.ssn, c.phone, a.account_number, a.routing_number "
                "FROM customers c JOIN accounts a ON a.customer_id = c.customer_id "
                "WHERE c.rm_id = 'RM-001' ORDER BY c.customer_id LIMIT 1"
            )
        ).mappings().first()
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.fail(f"audit tests require the seeded database: {exc}")
    finally:
        db.close()
    assert row is not None, "seed data has drifted; re-run python db/seed.py"
    return dict(row)


def test_stored_user_query_contains_no_raw_pii(session, seeded_customer):
    raw = (
        f"Look up {seeded_customer['full_name']}, SSN {seeded_customer['ssn']}, "
        f"account {seeded_customer['account_number']}, "
        f"routing {seeded_customer['routing_number']}"
    )

    entry = write_entry(RM1, "ALLOWED", user_query=raw, session=session)

    stored = session.execute(
        text("SELECT user_query FROM audit_log WHERE event_id = :id"),
        {"id": entry.event_id},
    ).scalar_one()

    for secret in (
        seeded_customer["ssn"],
        seeded_customer["full_name"],
        seeded_customer["routing_number"],
    ):
        assert secret not in stored, f"raw PII reached the audit log: {secret!r}"
        assert secret not in entry.user_query

    # The account number is masked rather than destroyed — last four stay useful to an
    # investigator (§3.1.2).
    assert seeded_customer["account_number"] not in stored
    assert seeded_customer["account_number"][-4:] in stored


def test_stored_response_summary_contains_no_raw_pii(session, seeded_customer):
    raw = (
        f"{seeded_customer['full_name']}'s SSN is {seeded_customer['ssn']} "
        f"and the phone on file is {seeded_customer['phone']}."
    )

    entry = write_entry(
        RM1, "ALLOWED", response_summary=raw, session=session
    )

    stored = session.execute(
        text("SELECT response_summary FROM audit_log WHERE event_id = :id"),
        {"id": entry.event_id},
    ).scalar_one()

    assert seeded_customer["ssn"] not in stored
    assert seeded_customer["full_name"] not in stored
    assert "[REDACTED_US_SSN]" in stored


def test_redaction_happens_before_hashing(session, seeded_customer):
    """The chain commits to the redacted text, so verification never needs the raw values."""
    entry = write_entry(
        RM1,
        "ALLOWED",
        user_query=f"SSN {seeded_customer['ssn']}",
        session=session,
    )

    assert verify_chain(session) is None
    assert seeded_customer["ssn"] not in entry.user_query


# ================================================================================
# 5. Interface details
# ================================================================================


def test_invalid_decision_is_rejected_in_python(session):
    with pytest.raises(ValueError, match="decision must be one of"):
        write_entry(RM1, "MAYBE", session=session)


@pytest.mark.parametrize(
    "decision", ["ALLOWED", "BLOCKED", "PENDING_APPROVAL", "APPROVED", "REJECTED"]
)
def test_every_schema_decision_is_accepted(session, decision):
    entry = write_entry(RM1, decision, session=session)

    assert entry.decision == decision
    assert verify_chain(session) is None


def test_latency_is_recorded_under_audit_write(session):
    timing.clear_samples()

    write_entry(RM1, "ALLOWED", user_query="hello", session=session)

    assert {s["rail"] for s in timing.get_samples()} == {RAIL_AUDIT_WRITE}


def test_cli_verify_reports_intact_and_broken(session, capsys, monkeypatch):
    """`python -m app.audit.chain verify` — exercised through its entry point."""
    from app.audit import chain

    write_sequence(session, 3)

    # The CLI opens its own session; bind it to this rolled-back transaction. `original`
    # is captured before patching so the replacement does not call itself.
    original = chain.read_entries
    monkeypatch.setattr(chain, "read_entries", lambda s=None: original(session))

    assert chain._cli(["chain", "verify"]) == 0
    assert "OK" in capsys.readouterr().out

    session.execute(
        text(
            "UPDATE audit_log SET decision = 'APPROVED' "
            "WHERE event_id = (SELECT MIN(event_id) FROM audit_log)"
        )
    )
    assert chain._cli(["chain", "verify"]) == 1
    assert "BROKEN" in capsys.readouterr().err

    assert chain._cli(["chain", "bogus"]) == 2


def test_cli_show_lists_entries(session, capsys, monkeypatch):
    from app.audit import chain

    write_sequence(session, 3)
    original = chain.read_entries
    monkeypatch.setattr(chain, "read_entries", lambda s=None: original(session))

    assert chain._cli(["chain", "show"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3
