"""Phase 1 smoke test — proves the foundation is real before any agent code exists.

Covers the Phase 1 acceptance boxes that a human would otherwise check by hand
(PROJECT_PLAN.md §1.7):

  1. Postgres is reachable and every one of the 8 tables reports its expected count
  2. The Ollama endpoint is reachable and serving the configured model
  3. One real completion comes back, with wall-clock latency printed
  4. Every routing number in `accounts` passes the ABA checksum the Phase 3 custom
     recognizer applies — undetectable test data would make that rail untestable
  5. All 6 planted injection payloads are present, verbatim, and owned by RM-001

Run it after `docker compose up -d`, `python db/seed.py` and `ollama create`:

    python scripts/smoke_test.py

Exit code 0 means every check passed; 1 means at least one failed. Each check runs
even if an earlier one fails, so one run tells you everything that is wrong rather
than only the first thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as a plain script from anywhere: put the repo root on sys.path so `app` and `db`
# import the same way they do under uvicorn.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import get_session  # noqa: E402
from app.llm import LLMError, llm  # noqa: E402
from app.models import (  # noqa: E402
    Account,
    AuditLog,
    Customer,
    Loan,
    PolicyChunk,
    RelationshipManager,
    ServiceRequest,
    Transaction,
)
from app.rails.recognizers import validate_aba  # noqa: E402
from db.seed import POISONED_ROWS  # noqa: E402

SMOKE_PROMPT = "In one sentence, what is a routing number?"

# All 8 tables, in schema order. `audit_log` is expected to be empty at Phase 1 — it is
# written by the agent, which does not exist yet — so its count is reported, not asserted.
TABLES: list[tuple[str, type, int | None]] = [
    ("relationship_managers", RelationshipManager, 2),
    ("customers", Customer, 40),
    ("accounts", Account, 60),
    ("transactions", Transaction, 600),
    ("loans", Loan, 18),
    ("service_requests", ServiceRequest, 25),
    ("policy_chunks", PolicyChunk, None),  # 40-60, range-checked below
    ("audit_log", AuditLog, None),  # empty until Phase 3
]

_failures: list[str] = []


def _header(title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


def _ok(message: str) -> None:
    print(f"  [PASS] {message}")


def _fail(message: str) -> None:
    _failures.append(message)
    print(f"  [FAIL] {message}")


# =============================================================================
# 1. Postgres connectivity + row counts for all 8 tables
# =============================================================================
def check_database() -> None:
    _header("1. PostgreSQL connectivity and row counts")
    print(f"  {settings.postgres_user}@{settings.postgres_host}:"
          f"{settings.postgres_port}/{settings.postgres_db}\n")
    before = len(_failures)

    with get_session() as session:
        print(f"  {'table':<24} {'rows':>7}   {'expected':>10}")
        print("  " + "-" * 45)
        for name, model, expected in TABLES:
            count = session.scalar(select(func.count()).select_from(model))
            if name == "policy_chunks":
                shown, good = "40-60", 40 <= count <= 60
            elif expected is None:
                shown, good = "-", True
            else:
                shown, good = str(expected), count == expected
            flag = " " if good else "  <-- MISMATCH"
            print(f"  {name:<24} {count:>7}   {shown:>10}{flag}")
            if not good:
                _fail(f"{name}: expected {shown}, found {count}")

    if len(_failures) == before:
        _ok("all 8 tables present with expected volumes")


# =============================================================================
# 2 + 3. Ollama reachability and one real completion
# =============================================================================
def check_llm() -> None:
    _header("2. Ollama endpoint")
    print(f"  {llm.base_url}  (model: {llm.model})\n")

    try:
        served = llm.list_models()
    except LLMError as exc:
        _fail(f"Ollama unreachable: {exc}")
        print("\n  Start it with:  $env:OLLAMA_FLASH_ATTENTION=\"1\"; "
              "$env:OLLAMA_KV_CACHE_TYPE=\"q8_0\"; ollama serve")
        return

    _ok(f"endpoint reachable, serving {len(served)} model(s)")
    if not llm.serves_model():
        _fail(
            f"model '{llm.model}' is not served. Available: {', '.join(served) or 'none'}. "
            "Run: ollama create sentinelbank-llm -f Modelfile"
        )
        return
    _ok(f"model '{llm.model}' is loaded")

    _header("3. One completion")
    print(f"  prompt: {SMOKE_PROMPT}\n")
    try:
        result = llm.complete(SMOKE_PROMPT)
    except LLMError as exc:
        _fail(f"completion failed: {exc}")
        return

    print(f"  completion: {result.text.strip()}\n")
    print(f"  elapsed:    {result.elapsed_s:.2f}s")
    if result.total_tokens:
        tps = result.completion_tokens / result.elapsed_s if result.elapsed_s else 0.0
        print(f"  tokens:     {result.prompt_tokens} prompt + "
              f"{result.completion_tokens} completion  ({tps:.1f} tok/s)")

    if result.text.strip():
        _ok("non-empty completion returned")
    else:
        _fail("model returned an empty completion")

    # Not an assertion — a CPU split is legal, just slow. Phase 1 acceptance requires
    # `ollama ps` to read 100% GPU, which is a human check.
    print("\n  Reminder: run `ollama ps` — PROCESSOR must read 100% GPU.")


# =============================================================================
# 4. ABA checksum over every routing number
# =============================================================================
def check_routing_numbers() -> None:
    _header("4. ABA checksum on every accounts.routing_number")

    with get_session() as session:
        routing_numbers = session.scalars(select(Account.routing_number)).all()

    if not routing_numbers:
        _fail("no routing numbers found — has db/seed.py been run?")
        return

    # The Phase 3 custom recognizer keys on this checksum. If seeded routing numbers
    # did not satisfy it, the PII rail would look like it works while detecting nothing.
    bad = [rn for rn in routing_numbers if not validate_aba(rn)]
    distinct = len(set(routing_numbers))
    print(f"  checked {len(routing_numbers)} routing numbers ({distinct} distinct)")
    if bad:
        _fail(f"{len(bad)} fail the ABA checksum, e.g. {bad[:3]}")
    else:
        _ok("all routing numbers pass the ABA checksum")


# =============================================================================
# 5. The 6 planted injection payloads
# =============================================================================
def check_poisoned_rows() -> None:
    _header("5. Planted injection payloads")
    before = len(_failures)

    with get_session() as session:
        for row in POISONED_ROWS:
            row_id, attack = row["row_id"], row["attack_type"]

            if row["target"] == "transactions.description":
                stored = session.scalar(
                    select(Transaction.description).where(Transaction.txn_id == row_id)
                )
                owner_rm = session.scalar(
                    select(Customer.rm_id)
                    .join(Account, Account.customer_id == Customer.customer_id)
                    .join(Transaction, Transaction.account_id == Account.account_id)
                    .where(Transaction.txn_id == row_id)
                )
            else:
                stored = session.scalar(
                    select(ServiceRequest.notes).where(ServiceRequest.ticket_id == row_id)
                )
                owner_rm = session.scalar(
                    select(Customer.rm_id)
                    .join(ServiceRequest, ServiceRequest.customer_id == Customer.customer_id)
                    .where(ServiceRequest.ticket_id == row_id)
                )

            print(f"\n  {row_id}  {attack}")
            print(f"      column: {row['target']}   owner: {owner_rm}")
            print(f"      text:   {row['payload'][:72]}...")

            if stored is None:
                _fail(f"{row_id} ({attack}) is missing from the database")
                continue
            # Byte-exact: Phase 4's attacks.yaml references these strings verbatim.
            if stored != row["payload"]:
                _fail(f"{row_id} ({attack}) text does not match the specified payload")
                continue
            # A payload on an RM-002 row would be filtered by the entitlement rail and
            # would silently test nothing — the demo would pass while proving nothing.
            if owner_rm != settings.default_rm_id:
                _fail(
                    f"{row_id} ({attack}) belongs to {owner_rm}, not "
                    f"{settings.default_rm_id} — the entitlement rail would hide it"
                )

    if len(_failures) == before:
        print()
        _ok(f"all {len(POISONED_ROWS)} payloads present, verbatim, owned by "
            f"{settings.default_rm_id}")


def main() -> int:
    print("SentinelBank Phase 1 smoke test")

    # Each check is independent so one run reports every problem, not just the first.
    for check in (check_database, check_llm, check_routing_numbers, check_poisoned_rows):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            _fail(f"{check.__name__} raised {type(exc).__name__}: {exc}")

    _header("RESULT")
    if _failures:
        print(f"  {len(_failures)} check(s) FAILED:\n")
        for failure in _failures:
            print(f"    - {failure}")
        print()
        return 1
    print("  All checks passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
