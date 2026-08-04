"""SHA-256 hash-chained audit log (PROJECT_PLAN.md §3.5).

entry_hash = SHA256(prev_hash + canonical_json(payload)); genesis prev_hash = "0"*64.
verify_chain() walks from genesis and returns the first broken index, or None if intact.
CLI: python -m app.audit.chain verify. Entries store PII-redacted text only.

================================================================================
WHAT THIS BUYS, AND WHAT IT DOES NOT
================================================================================
A hash chain makes the log **tamper-evident**, not tamper-proof. Anyone with UPDATE on
`audit_log` can still change a row; what they cannot do is change it *without the change
being detectable*, because every entry commits to the one before it. Editing entry 4
changes its `entry_hash`, which entry 5 stored as its `prev_hash`, and so on to the end
of the table. To forge convincingly an attacker must rewrite every subsequent row, which
is a much louder operation than a single surgical UPDATE — and it is exactly what
`verify_chain()` walks the table to detect.

Three properties follow from the linkage, and all three are asserted in tests:

  * **Mutation** — a changed field means the recomputed hash no longer matches the stored
    one, caught at that entry.
  * **Deletion** — removing an entry orphans its successor, whose `prev_hash` now points
    at a hash no longer present, caught at the successor.
  * **Reordering** — order is not an artefact of `event_id`, it is committed to by the
    chain, so two entries swapped break the links on both sides.

================================================================================
CANONICAL JSON — THE EXACT RULE
================================================================================
Anyone re-verifying this chain from outside this codebase needs the byte-for-byte
serialization rule, because a single differing space or key order changes every hash
downstream and produces a false tamper alarm. The rule is:

    json.dumps(payload,
               sort_keys=True,            # keys ascending by Unicode code point
               separators=(",", ":"),     # no whitespace anywhere
               ensure_ascii=False,        # non-ASCII emitted literally, not \\uXXXX
               default=_json_default)     # datetime -> naive-UTC ISO 8601
      .encode("utf-8")

and then:

    entry_hash = sha256(prev_hash.encode("ascii") + canonical_bytes).hexdigest()

Note `prev_hash` is hashed as its 64-character lowercase hex **string**, exactly as it is
stored, not as decoded bytes. That is the literal reading of §3.5's
`SHA256(prev_hash + canonical_json(payload))` and it keeps the rule reproducible with a
shell one-liner.

Three decisions inside that rule are worth stating because they are the ones that would
otherwise cause false failures:

1. **Timestamps are naive UTC**, serialized by `.isoformat()` with microseconds. The
   column is `TIMESTAMP` without time zone (§1.5), so a timezone-aware value would lose
   its offset on write and re-serialize differently on read — the hash would break on a
   round trip through the database with nobody having tampered with anything. The
   timestamp is also generated here in Python rather than by the column's `DEFAULT NOW()`,
   because a value the database chooses after the hash is computed cannot be inside it.

2. **`event_id` is not in the payload.** It is `SERIAL`, assigned by Postgres during the
   INSERT, so it does not exist when the hash is computed. Ordering is committed to by
   `prev_hash` instead, which is strictly stronger: `event_id` only records the order rows
   were written, while the chain records the order they were *hashed*.

3. **Floats are stored and hashed exactly as Python renders them.** `json.dumps` uses
   `repr`, which is shortest-round-trip, and a JSONB round trip was verified to return
   identical values. Rail scores and latencies are the only floats here.

================================================================================
PII
================================================================================
§1.5 and §3.1.5: `user_query` and `response_summary` are PII-redacted **before** they are
hashed and written, via `app.rails.pii.redact_text`. This is not defence in depth, it is
the primary control for this table — an audit log full of raw SSNs is a liability rather
than a control, and it is durable in a way a request never is. Because redaction happens
before hashing, the chain commits to the redacted text: verification never needs the raw
values, and there is no path by which they can be reconstructed from the log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text
from typing_extensions import NotRequired

from app.db import get_session
from app.rails.pii import redact_text
from app.rails.timing import rail_timer

logger = logging.getLogger(__name__)

# §3.5: the genesis entry has no predecessor, so it commits to a fixed 64-character
# sentinel. Not a real hash of anything — a hash of the empty string would be a value an
# attacker could also produce for a forged "first" entry.
GENESIS_PREV_HASH = "0" * 64

RAIL_AUDIT_WRITE = "audit_write"

# §1.5's CHECK constraint. Mirrored here so a bad decision fails in Python with a clear
# message rather than as an IntegrityError from the driver.
VALID_DECISIONS = frozenset(
    {"ALLOWED", "BLOCKED", "PENDING_APPROVAL", "APPROVED", "REJECTED"}
)

# The payload fields committed to by the hash, in no particular order — `sort_keys`
# imposes the order. Anything not in this tuple is not covered by the chain, so adding a
# column later means deciding, explicitly, whether it is evidence.
HASHED_FIELDS = (
    "timestamp",
    "rm_id",
    "user_query",
    "rails_fired",
    "tool_calls",
    "decision",
    "response_summary",
)


def _json_default(value: Any) -> Any:
    """Serialize types `json` cannot handle, deterministically.

    Only datetimes are expected. Anything else raises rather than being coerced to a
    string: a silent `str()` fallback would hash two different objects to the same bytes
    and quietly weaken the chain.
    """
    if isinstance(value, datetime):
        return _as_naive_utc(value).isoformat()
    raise TypeError(f"audit payload contains a non-serializable {type(value).__name__}")


def _as_naive_utc(value: datetime) -> datetime:
    """Normalize any datetime to naive UTC. See canonicalization note 1."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Serialize `payload` to the canonical bytes the chain hashes.

    The exact rule is documented in the module docstring. Do not change any argument
    here without understanding that it invalidates every hash already in the table.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def compute_entry_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    """entry_hash = SHA256(prev_hash + canonical_json(payload)) — §3.5, literally."""
    digest = hashlib.sha256()
    digest.update(prev_hash.encode("ascii"))
    digest.update(canonical_json(payload))
    return digest.hexdigest()


def build_payload(
    *,
    timestamp: datetime,
    rm_id: str,
    user_query: str | None,
    rails_fired: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    decision: str,
    response_summary: str | None,
) -> dict[str, Any]:
    """Assemble the hashed payload from already-redacted parts.

    Takes redacted text: this function does not redact, so that `write_entry` and
    `verify_chain` provably hash the same thing. Verification reads what is stored — if
    redaction happened here it would run again at verify time, on already-redacted text,
    and any non-idempotent redaction would break the chain.
    """
    return {
        "timestamp": _as_naive_utc(timestamp).isoformat(),
        "rm_id": rm_id,
        "user_query": user_query,
        "rails_fired": rails_fired,
        "tool_calls": tool_calls,
        "decision": decision,
        "response_summary": response_summary,
    }


class AuditState(AgentState[Any]):
    """Agent state plus the id of the entry written for this request."""

    audit_event_id: NotRequired[int]


@dataclass(frozen=True)
class AuditEntry:
    """One written entry, returned so the caller can log or display it."""

    event_id: int
    timestamp: datetime
    rm_id: str
    user_query: str | None
    decision: str
    response_summary: str | None
    prev_hash: str
    entry_hash: str


@rail_timer(RAIL_AUDIT_WRITE)
def write_entry(
    rm_id: str,
    decision: str,
    *,
    user_query: str | None = None,
    response_summary: str | None = None,
    rails_fired: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    timestamp: datetime | None = None,
    session: Any = None,
) -> AuditEntry:
    """Write one audit entry, linked to the current tail of the chain.

    `user_query` and `response_summary` are redacted here, before hashing and before the
    INSERT — the raw strings never reach the database (§1.5).

    The read of the tail and the INSERT share one transaction. Two concurrent writers
    could otherwise read the same tail and fork the chain; the `UNIQUE` constraint on
    `entry_hash` (§1.5) turns the losing fork into an IntegrityError rather than a silent
    branch, which is the fail-closed behaviour §6 rule 4 asks for. This is a single-user
    demo, so the contention is theoretical — the constraint is what makes it safe anyway.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_DECISIONS)}, got {decision!r}"
        )

    # Generated here, not by the column default: a timestamp the database picks during
    # INSERT is chosen after the hash is computed and so cannot be committed to.
    when = _as_naive_utc(timestamp or datetime.now(timezone.utc))

    payload_query = redact_text(user_query) if user_query else user_query
    payload_response = (
        redact_text(response_summary) if response_summary else response_summary
    )

    def _write(db: Any) -> AuditEntry:
        prev_hash = (
            db.execute(
                text("SELECT entry_hash FROM audit_log ORDER BY event_id DESC LIMIT 1")
            ).scalar()
            or GENESIS_PREV_HASH
        )

        payload = build_payload(
            timestamp=when,
            rm_id=rm_id,
            user_query=payload_query,
            rails_fired=rails_fired or [],
            tool_calls=tool_calls or [],
            decision=decision,
            response_summary=payload_response,
        )
        entry_hash = compute_entry_hash(prev_hash, payload)

        event_id = db.execute(
            text(
                """
                INSERT INTO audit_log
                    (timestamp, rm_id, user_query, rails_fired, tool_calls,
                     decision, response_summary, prev_hash, entry_hash)
                VALUES
                    (:timestamp, :rm_id, :user_query, CAST(:rails_fired AS jsonb),
                     CAST(:tool_calls AS jsonb), :decision, :response_summary,
                     :prev_hash, :entry_hash)
                RETURNING event_id
                """
            ),
            {
                "timestamp": when,
                "rm_id": rm_id,
                "user_query": payload_query,
                # Dumped with the same canonical settings so what Postgres parses into
                # JSONB is what was hashed.
                "rails_fired": json.dumps(payload["rails_fired"], sort_keys=True),
                "tool_calls": json.dumps(payload["tool_calls"], sort_keys=True),
                "decision": decision,
                "response_summary": payload_response,
                "prev_hash": prev_hash,
                "entry_hash": entry_hash,
            },
        ).scalar_one()

        return AuditEntry(
            event_id=event_id,
            timestamp=when,
            rm_id=rm_id,
            user_query=payload_query,
            decision=decision,
            response_summary=payload_response,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    if session is not None:
        return _write(session)
    with get_session() as owned:
        return _write(owned)


def read_entries(
    session: Any = None,
    *,
    limit: int | None = None,
    offset: int = 0,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    """Entries in chain order (`event_id` ascending), as plain dicts.

    `limit`/`offset` exist for §3.6's paginated `GET /audit` and the §3.7 audit tab. They
    are deliberately **not** used by `verify_chain`: verification has to walk the chain
    from genesis, and a page of it proves nothing — the first entry of page 3 has a
    `prev_hash` that no row in the page contains, so a paginated verifier would either
    report a false break or have to skip the linkage check that catches deletions.

    `newest_first` reverses the *page* only. The chain order itself is what `event_id`
    ascending means, so pagination happens in that order and the flip is presentational.
    """

    def _read(db: Any) -> list[dict[str, Any]]:
        sql = """
            SELECT event_id, timestamp, rm_id, user_query, rails_fired, tool_calls,
                   decision, response_summary, prev_hash, entry_hash
            FROM audit_log
            ORDER BY event_id {direction}
            """.format(direction="DESC" if newest_first else "ASC")
        params: dict[str, Any] = {}
        if limit is not None:
            sql += " LIMIT :limit"
            params["limit"] = limit
        if offset:
            sql += " OFFSET :offset"
            params["offset"] = offset

        rows = db.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]

    if session is not None:
        return _read(session)
    with get_session() as owned:
        return _read(owned)


def count_entries(session: Any = None) -> int:
    """Total rows in `audit_log`, for the pagination envelope."""

    def _count(db: Any) -> int:
        return int(db.execute(text("SELECT COUNT(*) FROM audit_log")).scalar() or 0)

    if session is not None:
        return _count(session)
    with get_session() as owned:
        return _count(owned)


@dataclass(frozen=True)
class ChainBreak:
    """Where and how the chain failed, for the CLI and the §3.7 audit tab."""

    index: int
    event_id: int
    reason: str
    expected: str
    found: str


def verify_chain_detailed(session: Any = None) -> ChainBreak | None:
    """Walk the chain from genesis; return the first break, or None if intact.

    Two things are checked at every entry, and both are necessary:

      1. **Linkage** — the stored `prev_hash` equals the previous entry's `entry_hash`
         (or the genesis sentinel for the first). This is what catches a deleted or
         reordered entry: the content of the survivors is untouched, so their own hashes
         still recompute correctly, and only the links reveal that something moved.
      2. **Integrity** — recomputing `SHA256(prev_hash + canonical_json(payload))` from
         the row's own stored fields reproduces the stored `entry_hash`. This is what
         catches a mutated field.

    Checking linkage first means a deletion is reported against the orphaned successor
    rather than being masked by an integrity failure further along.
    """
    entries = read_entries(session)

    expected_prev = GENESIS_PREV_HASH
    for index, entry in enumerate(entries):
        if entry["prev_hash"] != expected_prev:
            return ChainBreak(
                index=index,
                event_id=entry["event_id"],
                reason="broken_link",
                expected=expected_prev,
                found=entry["prev_hash"],
            )

        payload = build_payload(
            timestamp=entry["timestamp"],
            rm_id=entry["rm_id"],
            user_query=entry["user_query"],
            rails_fired=entry["rails_fired"],
            tool_calls=entry["tool_calls"],
            decision=entry["decision"],
            response_summary=entry["response_summary"],
        )
        recomputed = compute_entry_hash(entry["prev_hash"], payload)

        if recomputed != entry["entry_hash"]:
            return ChainBreak(
                index=index,
                event_id=entry["event_id"],
                reason="hash_mismatch",
                expected=recomputed,
                found=entry["entry_hash"],
            )

        expected_prev = entry["entry_hash"]

    return None


def verify_chain(session: Any = None) -> int | None:
    """The §3.5 signature: first broken index, or None if the chain is intact.

    Index is the 0-based position in chain order, not the `event_id` — §3.5 asks for "the
    first index at which the chain breaks", and positions stay meaningful even when
    `event_id` values are missing because a row was deleted.
    """
    verdict = verify_chain_detailed(session)
    return None if verdict is None else verdict.index


# ================================================================================
# CLI — python -m app.audit.chain verify
# ================================================================================

class AuditMiddleware(AgentMiddleware):
    """Writes one hash-chained entry per request (§3.4 position 8, §3.5).

    **Uses `after_agent`, not `after_model`, and that is a correctness requirement rather
    than a preference.** `create_agent` runs `after_model` hooks in *reverse* declaration
    order, so this middleware — last in §3.4's stack — would have its `after_model` run
    FIRST, before the PII rails at positions 2 and 3 had redacted the model's output. The
    audit log would then contain exactly the raw customer data §1.5 forbids it to hold.
    `after_agent` runs strictly after every `after_model`, so what is recorded here is the
    final, redacted, name-restored response the RM actually saw.

    `write_entry` redacts `user_query` and `response_summary` again on the way in, so the
    stored text is protected whether or not the PII rail was toggled on.
    """

    state_schema = AuditState

    def __init__(self, rm_id: str, rails: Any = None, session: Any = None) -> None:
        super().__init__()
        from app.config import settings

        self.rm_id = rm_id
        self.rails = rails if rails is not None else settings.guardrails_enabled
        self.session = session
        self.entries: list[AuditEntry] = []

    @property
    def enabled(self) -> bool:
        return bool(self.rails.audit)

    def after_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        messages = state.get("messages") or []
        query = next(
            (
                str(m.content)
                for m in messages
                if isinstance(m, HumanMessage) and m.content
            ),
            None,
        )
        response = next(
            (
                str(m.content)
                for m in reversed(messages)
                if isinstance(m, AIMessage) and m.content
            ),
            None,
        )

        tool_calls = [
            {"tool": call.get("name"), "args": _scrub_args(call.get("args") or {})}
            for m in messages
            if isinstance(m, AIMessage)
            for call in (m.tool_calls or [])
        ]

        try:
            entry = write_entry(
                self.rm_id,
                _decide(state, messages),
                user_query=query,
                response_summary=response,
                rails_fired=_rails_fired(),
                tool_calls=tool_calls,
                session=self.session,
            )
        except Exception:  # pragma: no cover - a failed audit must not lose the answer
            logger.exception("audit write failed; the response is still returned")
            return None

        self.entries.append(entry)
        return {"audit_event_id": entry.event_id}


def _scrub_args(args: dict[str, Any]) -> dict[str, Any]:
    """Redact tool arguments before they are hashed into the log.

    Tool args are model-authored and routinely carry customer data — the SQL string for
    `query_customer_data` can name a customer outright. §1.5 applies to the whole entry,
    not only the two text columns.
    """
    return {key: redact_text(value) if isinstance(value, str) else value for key, value in args.items()}


def _rails_fired() -> list[dict[str, Any]]:
    """The §1.5 `rails_fired` list, aggregated from this request's latency samples.

    Sourced from `timing.get_samples()` rather than from each rail reporting itself,
    because the ring buffer already has exactly one record per rail invocation and a
    parallel reporting path would be a second thing to keep in sync.
    """
    from app.rails.timing import get_samples

    by_rail: dict[str, dict[str, Any]] = {}
    for sample in get_samples():
        entry = by_rail.setdefault(
            sample["rail"],
            {"rail": sample["rail"], "verdict": "RAN", "score": None, "latency_ms": 0.0},
        )
        entry["latency_ms"] = round(entry["latency_ms"] + sample["duration_ms"], 3)
    return sorted(by_rail.values(), key=lambda r: r["rail"])


def _decide(state: Any, messages: list[Any]) -> str:
    """Map what happened to one of §1.5's five decision values."""
    if state.get("injection_flags"):
        return "BLOCKED"
    if state.get("__interrupt__"):
        return "PENDING_APPROVAL"
    return "ALLOWED"


_USAGE = "usage: python -m app.audit.chain [verify|show]"


def _cli(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "verify"

    if command == "verify":
        entries = read_entries()
        verdict = verify_chain_detailed()
        if verdict is None:
            print(f"audit chain OK — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, intact from genesis")
            return 0
        print(
            f"audit chain BROKEN at index {verdict.index} (event_id={verdict.event_id})\n"
            f"  reason:   {verdict.reason}\n"
            f"  expected: {verdict.expected}\n"
            f"  found:    {verdict.found}",
            file=sys.stderr,
        )
        return 1

    if command == "show":
        for index, entry in enumerate(read_entries()):
            print(
                f"[{index:4}] event_id={entry['event_id']:<5} {entry['timestamp']} "
                f"{entry['rm_id']:<8} {entry['decision']:<16} {entry['entry_hash'][:16]}…"
            )
        return 0

    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
