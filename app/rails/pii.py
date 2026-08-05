"""Hybrid PII rail: LangChain PIIMiddleware built-ins + Presidio via its `detector` hook.

Built-ins cover only email/credit_card/ip/mac/url; Presidio covers SSN, bank numbers,
driver's licence, phone, location, DOB, ABA routing, and PERSON (PROJECT_PLAN.md §3.1).
Applies to input, output AND tool results — the SQL tool returns raw customer rows that
must be scanned before entering model context. Also holds the reversible person
pseudonymisation middleware (§3.1.4).

================================================================================
WHY THE DESIGN IS HYBRID — do not collapse it into one approach
================================================================================
§3.1.1 is explicit that this is deliberate. LangChain's `PIIMiddleware` detects exactly
five types: `email`, `credit_card` (Luhn-validated), `ip`, `mac_address`, `url`. Against
this schema that is **two fields**. It cannot detect a person's name, an SSN, a bank
account number, a driver's licence, or a routing number — which is to say it cannot
detect any of the columns that make `customers` regulated data.

So the framework is used two ways rather than replaced:

  * for the two types it genuinely covers, `PIIMiddleware` is used directly, Luhn check
    and all;
  * for everything else, `PIIMiddleware` is *still* the integration surface — its
    `detector` parameter takes a callable, and the callable wraps Presidio's
    `AnalyzerEngine`.

Nothing here reimplements redaction, message traversal, or strategy application. The
value of writing it this way is that it shows where a framework's coverage boundary
actually lies and extends it at the seam the framework provides, instead of building a
parallel system beside it. That reasoning belongs in docs/ARCHITECTURE.md.

The one place the framework genuinely cannot help is names. All four strategies
(`redact`, `mask`, `hash`, `block`) are one-way, and §3.1.4 requires the RM to see real
names in the final answer while the model never sees them at all. That needs a
token<->value map and a restore pass, so `PersonPseudonymMiddleware` below is custom.

================================================================================
WHAT PRESIDIO ACTUALLY RETURNS, AND WHY resolve_entities() EXISTS
================================================================================
Handing Presidio's raw output to `apply_strategy` corrupts data. Measured against one
serialized customer row from this schema, `analyze()` returned **15 matches over 10 real
values**, and the overlaps are not edge cases:

    '100244718831'  (accounts.account_number)  -> DATE_TIME 0.85
                                                  US_BANK_NUMBER 0.40
                                                  US_DRIVER_LICENSE 0.01
    '021000021'     (accounts.routing_number)  -> ABA_ROUTING 0.85
                                                  PHONE_NUMBER 0.75
                                                  US_BANK_NUMBER 0.40
                                                  US_DRIVER_LICENSE 0.01

Two separate problems follow.

**1. Overlapping spans corrupt the string.** `_apply_redact_strategy` walks matches by
descending start offset and splices each one out by index. Two matches covering the same
characters means the second splice operates on offsets the first one already
invalidated, and the result is mangled text — silently, with no exception.

**2. Presidio's confidence score is the wrong thing to rank by.** `DATE_TIME` scores
0.85 on an account number while `US_BANK_NUMBER` scores 0.40. Ranking by score would
redact the account number as a date, when §3.1.2 requires it *masked with the last four
visible* — the RM identifies accounts by those four digits, so this is not a cosmetic
difference, it is the difference between a usable answer and an unusable one.

So `resolve_entities()` below is the single chokepoint: every candidate is filtered and
de-overlapped **once**, in an explicit application-controlled priority order, before any
strategy runs. §6 rule 5 in spirit — which entity wins is a security decision, so it is
decided by a constant in this file, not by a model's confidence output.

================================================================================
THE DATE_TIME FILTER — the most dangerous rule in this file
================================================================================
§3.1.2 asks for `DATE_TIME` redacted "(DOB only)". That parenthesis is load-bearing.
This schema is made of dates: `txn_date`, `opened_date`, `created_at`, `date_of_birth`.
Presidio detects all of them identically, and it detects them *aggressively* — in the
measurement above it also claimed a 12-digit account number as a `DATE_TIME`.

Redacting dates indiscriminately does not produce a more private system, it produces a
broken one: the PII rail runs on tool results before they reach model context (§3.1.5),
so every transaction date would arrive as `[REDACTED_DATE_TIME]` and the model would be
asked to answer "what did this customer spend last month" against data with no months in
it. A 3B model handed that will invent dates rather than refuse.

The rule is therefore **keep the date unless it is provably a date of birth**: a
`DATE_TIME` survives only when a DOB context word sits just before it. Note this is the
opposite default from every other rule in this file, which fail closed. It is the
correct direction here and the trade is stated plainly: a bare date of birth with no
nearby label is not redacted by this rail. In practice the label is always adjacent,
because tool results are serialized rows and the column name `date_of_birth` travels
with the value.

================================================================================
PERFORMANCE
================================================================================
`AnalyzerEngine` construction costs ~9.4 s (it loads spaCy `en_core_web_sm`) and a single
`analyze()` pass costs ~31 ms on this machine's CPU. The engine is therefore a module
singleton built once — §1.1 rule 4 — and §3.6 warms it from the FastAPI `lifespan`
handler. Never construct one per request.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware._redaction import PIIMatch
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import NotRequired

from app.agent.tools import TRUSTED_COLUMNS, UNTRUSTED_COLUMNS
from app.config import RailsConfig, settings
from app.rails.recognizers import ABA_ROUTING_ENTITY, create_aba_recognizer
from app.rails.timing import rail_timer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from presidio_analyzer import AnalyzerEngine

# --------------------------------------------------------------------------------
# Rail names — these are the §4.3 latency-table rows, not free-form labels
# --------------------------------------------------------------------------------
RAIL_PII_INPUT = "pii_input"
RAIL_PII_TOOL_RESULT = "pii_tool_result"
RAIL_PII_OUTPUT = "pii_output"


# ================================================================================
# 1. The Presidio layer
# ================================================================================

# §3.1.2's entity map, minus the two LangChain covers natively. PERSON is included
# because the pseudonym middleware needs the same de-overlapped view of the text — a
# name that overlaps an SSN must lose to the SSN — but PERSON is deliberately absent
# from both strategy groups below, because tokenizing it is not a one-way redaction.
PRESIDIO_ENTITIES: tuple[str, ...] = (
    ABA_ROUTING_ENTITY,
    "US_SSN",
    "US_BANK_NUMBER",
    "PHONE_NUMBER",
    "US_DRIVER_LICENSE",
    "PERSON",
    "LOCATION",
    "DATE_TIME",
)

# Which entity wins when two of them claim the same characters. Ordered most specific
# first, and this ordering is a security decision rather than a tuning parameter:
#
#   ABA_ROUTING first     — the only entity here validated by a checksum
#                           (app/rails/recognizers.py), so when it fires it is the
#                           best-evidenced claim on the span. PHONE_NUMBER and
#                           US_BANK_NUMBER both also match 9 raw digits.
#   US_SSN                — a fixed, unambiguous format; nothing else should take it.
#   US_BANK_NUMBER        — must beat DATE_TIME, or account numbers get redacted as
#                           dates instead of masked to last-4. This is the exact case
#                           measured above and the reason this list is not sorted by
#                           Presidio's score.
#   PHONE_NUMBER
#   US_DRIVER_LICENSE     — Presidio scores this ~0.01 on bare digits; it is a
#                           catch-all and must never outrank a specific entity.
#   PERSON                — spaCy NER, loses to every deterministic format above.
#   LOCATION              — likewise.
#   DATE_TIME last        — the greediest and least specific detector in the set.
ENTITY_PRIORITY: tuple[str, ...] = (
    ABA_ROUTING_ENTITY,
    "US_SSN",
    "US_BANK_NUMBER",
    "PHONE_NUMBER",
    "US_DRIVER_LICENSE",
    "PERSON",
    "LOCATION",
    "DATE_TIME",
)
_PRIORITY_RANK = {entity: index for index, entity in enumerate(ENTITY_PRIORITY)}

# Entities redacted outright (§3.1.2). DATE_TIME only ever reaches here after the DOB
# filter has run, so this set means "DOB", not "every date".
REDACT_ENTITIES: frozenset[str] = frozenset(
    {"US_SSN", "US_DRIVER_LICENSE", "LOCATION", "DATE_TIME", ABA_ROUTING_ENTITY}
)

# Entities masked with the last four characters left visible. LangChain's `mask`
# strategy already does exactly this for any type it has no special case for
# (`_redaction.py`: `f"****{value[-4:]}"`), which is precisely §3.1.2's requirement for
# US_BANK_NUMBER — so there is no custom strategy here and nothing to keep in sync.
MASK_ENTITIES: frozenset[str] = frozenset({"US_BANK_NUMBER", "PHONE_NUMBER"})

# Words that mark a nearby date as a date of birth. Searched in the text *before* the
# match only: English puts the label first ("date_of_birth: 1980-05-03", "born 3 May"),
# and a suffix window would let a sentence mentioning birthdays redact every date in the
# paragraph.
DOB_CONTEXT_WORDS: tuple[str, ...] = (
    "dob",
    "date of birth",
    "date_of_birth",
    "birthdate",
    "birth date",
    "birthday",
    "born",
)

# How far back to look for one of those words. Wide enough to clear a serialized column
# name and its punctuation, narrow enough that one "born" cannot license redacting an
# entire row of transaction dates.
DOB_CONTEXT_WINDOW = 40

# Characters trimmed from the edges of a Presidio span. Measured spans include their
# trailing delimiter — `"Michael Chen'"` and `"1980-05-03'"` both came back with the
# closing quote attached. Replacing the untrimmed span deletes that quote and corrupts
# the serialized row the model then has to read.
#
# Brackets are deliberately NOT in this set. Presidio returns US phone numbers as
# `(854) 314-0409`, leading paren included, and trimming it would leave a stray `(`
# outside the masked span — `(****0409` rather than `****0409`. Only whitespace, quotes
# and sentence punctuation are stripped.
_SPAN_TRIM_CHARS = " \t\r\n'\"`.,:;!?"

# Schema column names, reused from the tool layer so there is one list, not two. spaCy
# tags `date_of_birth` as a PERSON at 0.85 confidence inside a serialized row; without
# this denylist the pseudonym rail would replace the *column name* with a token and hand
# the model a row whose keys it cannot read.
_SCHEMA_COLUMNS = {column.lower() for column in TRUSTED_COLUMNS | UNTRUSTED_COLUMNS}

# --------------------------------------------------------------------------------
# Internal identifiers are never PII, and destroying them breaks the agent
# --------------------------------------------------------------------------------
# The seeded primary keys look enough like weak PII patterns to trip Presidio, and the
# damage is not cosmetic. Measured on real values:
#
#     'TXN-000009'  -> US_DRIVER_LICENSE 0.01   (the all-digits catch-all pattern)
#     'SR-0005'     -> PERSON            0.85   (spaCy reads it as a proper noun)
#
# Left alone, a transaction id reaches the model as `TXN-[REDACTED_US_DRIVER_LICENSE]`.
# The model then cannot cite the transaction, and — worse — cannot pass a valid `txn_id`
# to `file_dispute`, so the HITL approval flow breaks on data the rail corrupted. This
# was caught by printing the assembled context in tests/test_agent.py, not by any
# unit test of the rail in isolation.
#
# These prefixes are bank-internal record locators generated by db/seed.py. They contain
# no customer information — knowing that CUST-0012 exists discloses nothing without the
# row behind it, which entitlement already governs — so any candidate span overlapping
# one is dropped outright.
_INTERNAL_ID_RE = re.compile(r"\b(?:TXN|ACCT|CUST|SR|LOAN|RM)-[0-9A-Za-z]+\b")

# Candidates below this confidence are dropped regardless of entity. The margin here is
# genuinely narrow, so the measured values are recorded rather than left to be rediscovered:
#
#     US_DRIVER_LICENSE  'TXN-000009'    0.01   <- all-digits fallback, a pure guess
#     US_BANK_NUMBER     '272052246567'  0.05   <- MUST survive; §3.1.2 masks it to last-4
#     US_BANK_NUMBER     '100244718831'  0.40   <- same entity, different value
#     US_DRIVER_LICENSE  'J4108603'      0.30   <- a real licence format
#     US_SSN                             0.50
#     PHONE_NUMBER                       0.75
#     PERSON / LOCATION / ABA_ROUTING    0.85
#
# **Do not raise this above 0.05.** Presidio's bank-number recognizer scores a bare
# 12-digit account number at 0.05, so a floor of 0.1 silently stops masking account
# numbers — which looks like the rail working (nothing is flagged) while a primary piece
# of §3.1.2 PII flows straight into model context. That regression was caught by
# tests/test_pii.py::test_account_number_masked_to_last_four.
#
# The floor is a secondary defence: `_INTERNAL_ID_RE` above is what actually fixes the
# record-locator case, and ENTITY_PRIORITY already ranks US_BANK_NUMBER above
# US_DRIVER_LICENSE regardless of score. This only removes the sub-0.02 tier, where
# Presidio is not making a claim so much as declining to rule one out.
MIN_ENTITY_SCORE = 0.02


@dataclass(frozen=True)
class ResolvedMatch:
    """One surviving PII span: which entity, what text, and where."""

    entity: str
    value: str
    start: int
    end: int
    score: float


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    """The process-wide Presidio engine, built on first call.

    `lru_cache` is the singleton. Construction loads spaCy `en_core_web_sm` and costs
    ~9.4 s, which is why §1.1 rule 4 forbids per-request construction and §3.6 warms
    this from the FastAPI `lifespan` handler.

    CPU only. Presidio and spaCy never touch the GPU — the 4 GB GTX 1650 is reserved
    entirely for Ollama (§1.1 rule 5), and `app/embeddings.py` has already capped torch
    at 4 threads (§1.1 rule 6).

    The one custom recognizer (§3.1.3) is registered here. That is the only place
    ABA_ROUTING detection comes from.
    """
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    engine.registry.add_recognizer(create_aba_recognizer())
    return engine


def _has_dob_context(text: str, start: int) -> bool:
    """Whether a DOB label appears just before `start`. See the module docstring."""
    window = text[max(0, start - DOB_CONTEXT_WINDOW) : start].lower()
    return any(word in window for word in DOB_CONTEXT_WORDS)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink a span past surrounding quotes and punctuation."""
    while start < end and text[start] in _SPAN_TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _SPAN_TRIM_CHARS:
        end -= 1
    return start, end


# Possessive endings that spaCy includes in a PERSON span.
_POSSESSIVE_SUFFIXES = ("'s", "’s")


def _strip_possessive(text: str, start: int, end: int) -> int:
    """Drop a trailing `'s` from a PERSON span, returning the new end offset.

    "What's Michael Chen's checking account balance?" is §3.8's acceptance criterion
    verbatim, so the possessive form is the *common* way an RM names a customer, not an
    edge case. spaCy returns the span as `Michael Chen's`, and left alone that produces
    two problems: the map stores a name with a stray `'s` glued on, and the same
    customer mentioned once possessively and once plainly gets two different tokens
    within one request — breaking the consistency §3.1.4 requires.

    Only applied to PERSON. Every other entity in this rail is a formatted identifier
    that never takes a possessive.
    """
    for suffix in _POSSESSIVE_SUFFIXES:
        if text[start:end].endswith(suffix):
            return end - len(suffix)
    return end


# Characters that never appear inside a US retail-bank customer name, and that mark a
# PERSON candidate as a fragment of Python's `repr` rather than a person. See
# `_is_schema_artifact` for the measurements.
_NON_NAME_CHARS = frozenset("0123456789()[]{}=<>|@#$%^&*+")


def _is_schema_artifact(value: str) -> bool:
    """Whether a PERSON candidate is really a column name, an enum, or repr noise.

    spaCy is doing its job — `date_of_birth` genuinely looks like a proper noun out of
    context — but a column name is not a customer, and replacing one with a pseudonym
    token damages the data structure the model reads.

    ----------------------------------------------------------------------------
    THE TWO SHAPE RULES, AND THE MEASUREMENTS BEHIND THEM (README defect 26)
    ----------------------------------------------------------------------------
    The rail scans the flat `str()` rendering of a tool envelope, so spaCy sees Python
    repr fragments and SQL enum values as if they were prose. Both were observed
    destroying real data in the rails-ON demo:

        'amount':  {'value': Decimal('-2004.42')}  -> <PERSON_c087>
        'channel': {'value': 'ZELLE'}              -> <PERSON_572d>

    Rule A — **a name contains no digits and no bracket punctuation.** Catches
    `Decimal('-2004.42` and `8)` (a tail of `datetime.datetime(2025, 6, 3, 22, 28)`).

    Rule B — **a single all-caps token is an enum, not a name.** Measured over every one
    of the 625 free-text fields in the seeded database plus the rendered envelopes of
    four accounts: the *only* single-token all-caps value spaCy ever calls a PERSON is
    `ZELLE`, 65 occurrences, and it is `transactions.channel`. Multi-word all-caps spans
    are left alone because they are usually genuine — `MAUREEN W`, `STACEY H`, the Zelle
    counterparties inside a memo — which is why the rule tests for a single token rather
    than for capitalisation alone.

    Cost of rule B, measured rather than assumed: **7 single-token counterparty first
    names** across the 625 fields (`BRETT`, `STACEY`, `KELLY`, `TINA`, `KATHRYN`, and two
    more) stop being pseudonymized and reach the model in the clear. They are third
    parties named in an untrusted memo, not customers in the RM's book. **Zero of the 40
    seeded customer names are affected by either rule** — every one is Title Case, and
    all 40 are still detected as PERSON inside a serialized row.
    """
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in _SCHEMA_COLUMNS:
        return True
    # Serialized identifiers, never human names as they appear in this schema.
    if "_" in normalized:
        return True

    # Rule A.
    if any(character in _NON_NAME_CHARS for character in value):
        return True

    # Rule B. `split()` rather than `" " in value` so that a span whose whitespace is a
    # tab or a newline is still counted as one token.
    stripped = value.strip()
    return len(stripped.split()) == 1 and not any(c.islower() for c in stripped)


def resolve_entities(text: str) -> list[ResolvedMatch]:
    """Detect PII and return non-overlapping spans, nearest-first by position.

    The single chokepoint. Both strategy detectors and the pseudonym middleware call
    this, so all three see the same de-overlapped view of the text and can never
    disagree about which entity owns a span.

    Steps, in order:
      1. one `analyze()` pass over every §3.1.2 entity;
      2. trim quotes and punctuation off each span;
      3. drop DATE_TIME without DOB context (see the module docstring — the default
         here is deliberately to KEEP the date);
      4. drop PERSON candidates that are schema column names;
      5. greedy non-overlap selection in ENTITY_PRIORITY order.
    """
    if not text or not text.strip():
        return []

    raw = get_analyzer().analyze(text, entities=list(PRESIDIO_ENTITIES), language="en")

    # Spans covering an internal record locator. Computed once per call; see the note on
    # _INTERNAL_ID_RE for why destroying these breaks the agent rather than protecting
    # anyone.
    protected = [(m.start(), m.end()) for m in _INTERNAL_ID_RE.finditer(text)]

    candidates: list[ResolvedMatch] = []
    for result in raw:
        if result.score < MIN_ENTITY_SCORE:
            continue

        start, end = _trim_span(text, result.start, result.end)
        if start >= end:
            continue

        if any(start < id_end and id_start < end for id_start, id_end in protected):
            continue

        # Rule 3: a date survives only if something nearby says it is a birth date.
        if result.entity_type == "DATE_TIME" and not _has_dob_context(text, start):
            continue

        if result.entity_type == "PERSON":
            end = _strip_possessive(text, start, end)
            if start >= end:
                continue

        value = text[start:end]

        # Rule 4: not every proper-noun-shaped token is a person.
        if result.entity_type == "PERSON" and _is_schema_artifact(value):
            continue

        # An entity Presidio returned that we have no priority for cannot be ordered
        # against the others, so it cannot be de-overlapped safely. Drop it rather than
        # guess — it is not in §3.1.2's map and nothing downstream expects it.
        if result.entity_type not in _PRIORITY_RANK:
            continue

        candidates.append(
            ResolvedMatch(
                entity=result.entity_type,
                value=value,
                start=start,
                end=end,
                score=result.score,
            )
        )

    # Rule 5. Sort by priority, then longer span, then position — all three keys are
    # deterministic, so the same text always yields the same redaction. Determinism is a
    # hard requirement (§6 rule 3) and the red-team harness asserts on exact responses.
    candidates.sort(key=lambda m: (_PRIORITY_RANK[m.entity], -(m.end - m.start), m.start))

    chosen: list[ResolvedMatch] = []
    for candidate in candidates:
        # Greedy: a higher-priority span already owns these characters.
        if any(
            candidate.start < taken.end and taken.start < candidate.end for taken in chosen
        ):
            continue
        chosen.append(candidate)

    return sorted(chosen, key=lambda m: m.start)


# Entities redacted when text is written to durable storage. This is the middleware's
# redact set PLUS `PERSON`, and the difference matters.
#
# In a live request a name is pseudonymized and restored, because the RM needs to read
# it (§3.1.4). The audit log is the opposite situation: it is durable, it is read later,
# and the per-request token map that could restore a name is destroyed the moment the
# request ends. A token in a stored row would therefore be permanently unresolvable —
# all of the storage risk of keeping the name, with none of the usefulness. So names are
# destroyed outright on the way to disk.
AUDIT_REDACT_ENTITIES: frozenset[str] = REDACT_ENTITIES | {"PERSON"}


def redact_text(text: str) -> str:
    """One-way redaction for text about to be written to durable storage (§3.5).

    Used by the audit log, where §1.5 requires the stored query and response to be
    PII-redacted: an audit trail full of raw SSNs is a liability, not a control. It
    inverts the usual trade — the middleware keeps text readable for the model and the
    RM, this keeps it *unreadable* for anyone who later reads the table.

    Covers the same ground as the rail as a whole: Presidio's entities through the shared
    `resolve_entities` chokepoint (so the same priority and DOB rules apply, and a
    transaction date in a stored response is not destroyed), plus LangChain's two
    built-ins for email and Luhn-validated credit cards.

    Masked entities keep their last four characters, as §3.1.2 specifies — an audit entry
    reading `****4471` is still useful to an investigator, which a bare `[REDACTED]` is
    not.
    """
    if not text or not text.strip():
        return text

    from langchain.agents.middleware._redaction import detect_credit_card, detect_email

    spans: list[tuple[int, int, str]] = [
        (m.start, m.end, m.entity) for m in resolve_entities(text)
    ]
    # The two LangChain built-ins. Appended after Presidio's spans so that on an overlap
    # the already-de-overlapped Presidio decision wins, keeping one authority per span.
    for match in detect_email(text) + detect_credit_card(text):
        spans.append((match["start"], match["end"], match["type"].upper()))

    chosen: list[tuple[int, int, str]] = []
    for start, end, entity in spans:
        if any(start < t_end and t_start < end for t_start, t_end, _ in chosen):
            continue
        chosen.append((start, end, entity))

    result = text
    # Right to left, so each splice leaves earlier offsets valid.
    for start, end, entity in sorted(chosen, key=lambda s: s[0], reverse=True):
        value = text[start:end]
        if entity in MASK_ENTITIES:
            replacement = f"****{value[-4:]}" if len(value) > 4 else "****"
        else:
            replacement = f"[REDACTED_{entity}]"
        result = result[:start] + replacement + result[end:]

    return result


def _presidio_detector(entities: frozenset[str]):
    """Build a `PIIMiddleware`-compatible detector restricted to `entities`.

    Both detectors resolve against the *full* entity set and then filter, rather than
    asking Presidio only for their own entities. That is what keeps the two strategy
    groups from fighting: the redact detector does not see `100244718831` as a
    DATE_TIME, because US_BANK_NUMBER already won that span globally, so it leaves the
    value untouched for the mask detector to handle.

    Each match carries its **own** entity as `type`. LangChain builds the placeholder
    from `match["type"]`, not from the middleware's `pii_type`, so one instance emits
    accurate per-entity labels — `[REDACTED_US_SSN]` and `[REDACTED_ABA_ROUTING]`, not
    a single lumped `[REDACTED_PRESIDIO_PII]`.
    """

    def detector(content: str) -> list[PIIMatch]:
        return [
            PIIMatch(
                type=match.entity,
                value=match.value,
                start=match.start,
                end=match.end,
            )
            for match in resolve_entities(content)
            if match.entity in entities
        ]

    return detector


# ================================================================================
# 2. Per-request pseudonym state (§3.1.4)
# ================================================================================
#
# WHERE THIS LIVES, AND WHY NOT IN AGENT STATE
# --------------------------------------------------------------------------------
# §3.1.4 says the token<->value map goes in "per-request agent state (never global,
# never persisted, cleared at request end)". Those requirements conflict on this stack:
# §3.3's HITL rail requires a `SqliteSaver` checkpointer, and a checkpointer serializes
# agent state to `./checkpoints.db`. Putting the map in state would write every real
# customer name to disk in plaintext — defeating the rail whose entire purpose is that
# the names stay in process memory.
#
# So the map lives in a module-level registry keyed by request id, in RAM only, deleted
# when the request ends. The security property §3.1.4 protects — one request's names are
# unreachable from another request — is enforced by the key and the deletion.
#
# A bare ContextVar was the other candidate and is rejected for a functional reason: the
# map has to survive an HITL interrupt and resume, or every name the model echoes after
# an RM approves a dispute becomes an unissued token and gets stripped. The registry
# survives the pause because the FastAPI process does.
#
# The deviation from the literal words "agent state" is recorded in README.md (§6 r.10).

# Tokens look like `<PERSON_a3f9>` (§3.1.4).
PERSON_TOKEN_PREFIX = "<PERSON_"
PERSON_TOKEN_RE = re.compile(r"<PERSON_[0-9a-f]{4,8}>")

# What an unissued token becomes. §3.1.4 says strip and flag; a visible marker beats a
# silent deletion, because a hallucinated identifier vanishing without trace is
# indistinguishable to the RM from the model never having mentioned anyone.
UNKNOWN_PERSON_PLACEHOLDER = "[REDACTED_UNKNOWN_PERSON]"

# Flag reason code. A stable string — §4.2's harness asserts on which rail fired.
FLAG_UNISSUED_TOKEN = "unissued_person_token"


@dataclass
class PseudonymMap:
    """One request's name<->token mapping. Never shared, never written to disk.

    `salt` is fresh per request, which is what makes the same name map to a different
    token in a different request (§3.1.4). Without it, tokens would be a stable
    pseudonym across requests and the map would be a re-identification table by itself.
    """

    salt: str
    token_to_name: dict[str, str] = field(default_factory=dict)
    name_to_token: dict[str, str] = field(default_factory=dict)
    flags: list[dict[str, Any]] = field(default_factory=list)

    def token_for(self, name: str) -> str:
        """Return the stable token for `name` within this request, creating it once.

        Keyed case-insensitively and on collapsed whitespace, so "Michael Chen" and
        "michael  chen" in one request collapse to a single token; the first spelling
        seen is what gets restored.
        """
        key = " ".join(name.split()).casefold()
        existing = self.name_to_token.get(key)
        if existing is not None:
            return existing

        # 4 hex chars, as §3.1.4's example shows. Widened on collision rather than
        # allowed to alias — two customers sharing a token would restore one of them as
        # the other, a cross-customer disclosure inside a single answer.
        digest = hashlib.sha256(f"{self.salt}:{key}".encode()).hexdigest()
        token = f"{PERSON_TOKEN_PREFIX}{digest[:8]}>"
        for width in (4, 6, 8):
            candidate = f"{PERSON_TOKEN_PREFIX}{digest[:width]}>"
            if candidate not in self.token_to_name:
                token = candidate
                break

        self.token_to_name[token] = name
        self.name_to_token[key] = token
        return token


# request id -> map. Module-level, but every entry is request-scoped and removed at
# request end; see the block comment above.
_REQUEST_MAPS: dict[str, PseudonymMap] = {}

# Set by `pii_request(...)`. Takes precedence over the graph's thread_id so tests,
# scripts and the FastAPI handler can all scope a request explicitly.
_ACTIVE_REQUEST: ContextVar[str | None] = ContextVar(
    "sentinelbank_pii_request", default=None
)


def current_request_id() -> str | None:
    """The request id in scope, or None.

    Prefers the explicit `pii_request(...)` scope; falls back to LangGraph's
    `thread_id`, which §3.3 already requires for HITL persistence, so a graph run inside
    a FastAPI handler resolves correctly without extra wiring.
    """
    explicit = _ACTIVE_REQUEST.get()
    if explicit is not None:
        return explicit

    try:
        from langgraph.config import get_config

        configurable = get_config().get("configurable") or {}
    except Exception:
        # Not running inside a graph — normal for unit tests and direct calls.
        return None

    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


@contextmanager
def pii_request(request_id: str) -> Iterator[PseudonymMap]:
    """Scope one request's pseudonym map, and guarantee it is destroyed.

        with pii_request(thread_id):
            ...run the agent...

    The `finally` is the whole point: §3.1.4 requires the map cleared at request end,
    and cleanup that only happens on the success path is cleanup that does not happen
    the one time it matters. FastAPI's `/chat` wraps the graph invocation in this.
    """
    existing = _REQUEST_MAPS.get(request_id)
    # An HITL resume re-enters the same request id and must find the same map, not a
    # fresh one — otherwise tokens issued before the interrupt stop resolving.
    request_map = existing or PseudonymMap(salt=secrets.token_hex(16))
    _REQUEST_MAPS[request_id] = request_map

    token = _ACTIVE_REQUEST.set(request_id)
    try:
        yield request_map
    finally:
        _ACTIVE_REQUEST.reset(token)
        _REQUEST_MAPS.pop(request_id, None)


def get_request_map(request_id: str | None = None) -> PseudonymMap | None:
    """The live map for a request, or None if the request is not open."""
    key = request_id if request_id is not None else current_request_id()
    if key is None:
        return None
    return _REQUEST_MAPS.get(key)


def clear_all_requests() -> None:
    """Drop every map. For tests, and for a clean shutdown."""
    _REQUEST_MAPS.clear()


# ================================================================================
# 3. Middleware
# ================================================================================


class PIIAgentState(AgentState[Any]):
    """Agent state plus the rail's flags, for §3.7's security banner."""

    pii_flags: NotRequired[list[dict[str, Any]]]


class TimedPIIMiddleware(PIIMiddleware):
    """`PIIMiddleware` with exactly one application point and a §4.3 latency name.

    §4.3's table has separate `pii_input`, `pii_tool_result` and `pii_output` rows, but
    `PIIMiddleware.before_model` handles input *and* tool results in a single call — so
    one instance covering both points cannot be attributed to one row.

    The fix is to give each instance one job. The constructor rejects any other
    configuration, which means `rail_name` can never drift from what is actually being
    measured. The cost is more instances; the detector work is identical, because each
    instance still scans only the surface it owns.
    """

    def __init__(
        self,
        pii_type: str,
        *,
        rail_name: str,
        flag: str,
        rails: RailsConfig | None = None,
        strategy: str = "redact",
        detector: Any = None,
        apply_to_input: bool = False,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None:
        points = sum([apply_to_input, apply_to_output, apply_to_tool_results])
        if points != 1:
            raise ValueError(
                "TimedPIIMiddleware requires exactly one application point so its "
                f"@rail_timer name is unambiguous; got {points} for {rail_name!r}"
            )

        super().__init__(
            pii_type,
            strategy=strategy,  # type: ignore[arg-type]
            detector=detector,
            apply_to_input=apply_to_input,
            apply_to_output=apply_to_output,
            apply_to_tool_results=apply_to_tool_results,
        )
        self.rail_name = rail_name
        self.flag = flag
        self.rails = rails if rails is not None else settings.guardrails_enabled

    @property
    def name(self) -> str:
        """Unique per (rule, application point).

        Uniqueness is required — `create_agent` rejects duplicate middleware names, and
        this rail contributes twelve instances. The separator is a dot rather than the
        more readable colon because LangGraph turns each middleware name into a graph
        node name, and `:` is reserved there.
        """
        return f"{type(self).__name__}[{self.pii_type}.{self.rail_name}]"

    @property
    def enabled(self) -> bool:
        """Read live on every hook, never captured at construction.

        The instance is baked into a compiled graph, so a build-time `enabled` boolean
        would make the Streamlit toggles inert until the agent was rebuilt — and the core
        demo of this project is flipping a rail off and re-asking the same question.
        """
        return bool(getattr(self.rails, self.flag))

    def before_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None
        return rail_timer(self.rail_name)(super().before_model)(state, runtime)

    def after_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if not self.apply_to_output:
            return None
        return rail_timer(self.rail_name)(super().after_model)(state, runtime)


class PersonPseudonymMiddleware(AgentMiddleware):
    """Reversible person pseudonymization (§3.1.4).

    The model never sees a customer's name; the RM always does. Every other entity in
    this rail is destroyed on the way in, which is fine because the RM does not need an
    SSN echoed back. Names are different — an answer reading "the balance on
    <PERSON_a3f9>'s account" is useless — so this is the one entity that has to survive
    the round trip, and none of LangChain's four strategies can do that.

    Lifecycle, mapped to the hooks:

        before_agent  open the request map; tokenize names in the RM's message
        before_model  tokenize names in tool results before they reach model context
        after_model   restore tokens to names; strip and flag any token we never issued
        after_agent   destroy the map

    The map itself is described in the block comment above `PseudonymMap`.
    """

    state_schema = PIIAgentState

    def __init__(self, rails: RailsConfig | None = None) -> None:
        super().__init__()
        self.rails = rails if rails is not None else settings.guardrails_enabled

    @property
    def enabled(self) -> bool:
        """Read live, so the Streamlit toggle works without rebuilding the graph."""
        return self.rails.person_pseudonym

    # -- helpers ---------------------------------------------------------------

    def _pseudonymize(self, text: str, request_map: PseudonymMap) -> str:
        """Replace every PERSON span with its token, right to left.

        Right to left because replacing left to right invalidates the offsets of every
        span after the one just replaced. `resolve_entities` has already guaranteed the
        spans do not overlap.
        """
        people = [m for m in resolve_entities(text) if m.entity == "PERSON"]
        if not people:
            return text

        result = text
        for match in sorted(people, key=lambda m: m.start, reverse=True):
            token = request_map.token_for(match.value)
            result = result[: match.start] + token + result[match.end :]
        return result

    def _restore(
        self, text: str, request_map: PseudonymMap
    ) -> tuple[str, list[dict[str, Any]]]:
        """Turn tokens back into names; strip and flag anything we did not issue.

        The unissued case is the interesting one. A token in the output that this
        request never created did not come from the data — it came from the model, which
        means it either hallucinated an identifier or was steered into echoing one. Both
        are worth surfacing, and neither should reach the RM looking like a real
        customer.
        """
        flags: list[dict[str, Any]] = []

        def substitute(match: re.Match[str]) -> str:
            token = match.group(0)
            name = request_map.token_to_name.get(token)
            if name is not None:
                return name
            flags.append(
                {
                    "rail": "person_pseudonym",
                    "reason": FLAG_UNISSUED_TOKEN,
                    "token": token,
                }
            )
            return UNKNOWN_PERSON_PLACEHOLDER

        return PERSON_TOKEN_RE.sub(substitute, text), flags

    # -- hooks -----------------------------------------------------------------

    def before_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Tokenize names in the RM's message before anything else runs."""
        if not self.enabled:
            return None
        request_map = get_request_map()
        if request_map is None:
            # No open request means no map to write into. Do nothing rather than mint a
            # global map that would outlive the request — that is the one failure mode
            # §3.1.4 explicitly forbids.
            return None

        return rail_timer(RAIL_PII_INPUT)(self._apply_to_input)(state, request_map)

    def _apply_to_input(
        self, state: Any, request_map: PseudonymMap
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, HumanMessage) or not message.content:
                continue

            content = str(message.content)
            replaced = self._pseudonymize(content, request_map)
            if replaced == content:
                return None

            new_messages = list(messages)
            new_messages[index] = HumanMessage(
                content=replaced, id=message.id, name=message.name
            )
            return {"messages": new_messages}
        return None

    def before_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Tokenize names in tool results — §3.1.5's easily-overlooked application point.

        The SQL tool returns raw `customers` rows. Without this, every name in the
        portfolio reaches model context in the clear and the rail's central claim is
        false.
        """
        if not self.enabled:
            return None
        request_map = get_request_map()
        if request_map is None:
            return None

        return rail_timer(RAIL_PII_TOOL_RESULT)(self._apply_to_tool_results)(
            state, request_map
        )

    def _apply_to_tool_results(
        self, state: Any, request_map: PseudonymMap
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        new_messages = list(messages)
        modified = False

        # Only the tool results from the current step — the ones after the last AI
        # message. Earlier ones were tokenized when they were new, and re-scanning them
        # would waste a ~31 ms Presidio pass per message per turn.
        last_ai = None
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], AIMessage):
                last_ai = index
                break
        start = 0 if last_ai is None else last_ai + 1

        for index in range(start, len(messages)):
            message = messages[index]
            if not isinstance(message, ToolMessage) or not message.content:
                continue

            content = str(message.content)
            replaced = self._pseudonymize(content, request_map)
            if replaced == content:
                continue

            new_messages[index] = ToolMessage(
                content=replaced,
                id=message.id,
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
            modified = True

        return {"messages": new_messages} if modified else None

    def after_model(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Restore names in the answer, and catch identifiers the model invented."""
        if not self.enabled:
            return None
        request_map = get_request_map()
        if request_map is None:
            return None

        return rail_timer(RAIL_PII_OUTPUT)(self._apply_to_output)(state, request_map)

    def _apply_to_output(
        self, state: Any, request_map: PseudonymMap
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage) or not message.content:
                continue

            content = str(message.content)
            restored, flags = self._restore(content, request_map)
            if restored == content and not flags:
                return None

            request_map.flags.extend(flags)

            new_messages = list(messages)
            new_messages[index] = AIMessage(
                content=restored,
                id=message.id,
                name=message.name,
                tool_calls=message.tool_calls,
            )

            update: dict[str, Any] = {"messages": new_messages}
            if flags:
                update["pii_flags"] = list(request_map.flags)
            return update
        return None

    def after_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Destroy the map.

        Belt and braces: `pii_request(...)`'s `finally` is the guarantee, because a
        graph that raises or is interrupted may never reach this hook. Clearing here
        keeps the map out of memory between turns on the normal path.
        """
        if not self.enabled:
            return None
        request_id = current_request_id()
        if request_id is not None:
            _REQUEST_MAPS.pop(request_id, None)
        return None


# ================================================================================
# 4. Assembly
# ================================================================================

# The four rules. Two use LangChain's built-in detectors untouched; two wrap Presidio
# through the `detector` seam. This tuple *is* the §3.1.1 argument in code — the split
# between the first two entries and the last two is exactly the framework's coverage
# boundary.
_RULES: tuple[tuple[str, str, Any, str], ...] = (
    # (pii_type, strategy, detector, RailsConfig flag)
    ("email", "redact", None, "pii_builtin"),
    ("credit_card", "mask", None, "pii_builtin"),
    ("presidio_redact", "redact", _presidio_detector(REDACT_ENTITIES), "pii_presidio"),
    ("presidio_mask", "mask", _presidio_detector(MASK_ENTITIES), "pii_presidio"),
)

# The three application points from §3.1.5, each with its §4.3 row name.
# `apply_to_tool_results` is the one that is easy to overlook and essential here.
_APPLICATION_POINTS: tuple[tuple[str, str], ...] = (
    ("apply_to_input", RAIL_PII_INPUT),
    ("apply_to_tool_results", RAIL_PII_TOOL_RESULT),
    ("apply_to_output", RAIL_PII_OUTPUT),
)


def build_pii_middleware(rails: RailsConfig | None = None) -> list[AgentMiddleware]:
    """Build the PII rail's middleware, honouring the §3.4 toggles.

    Returns them in §3.4's stack order: LangChain built-ins, then Presidio, then the
    person pseudonymizer. Each rule is instantiated once per application point.

    **Every instance is always built**, whatever the flags currently say, and each one
    consults `rails` live at hook time instead. Filtering the list here would bake the
    flags into the compiled graph, so a toggle would do nothing until the agent was
    rebuilt — and the core demo of this project is turning a rail off mid-conversation
    and re-asking the same question.
    """
    config = rails if rails is not None else settings.guardrails_enabled

    middleware: list[AgentMiddleware] = []
    for pii_type, strategy, detector, flag in _RULES:
        for attribute, rail_name in _APPLICATION_POINTS:
            middleware.append(
                TimedPIIMiddleware(
                    pii_type,
                    rail_name=rail_name,
                    flag=flag,
                    rails=config,
                    strategy=strategy,
                    detector=detector,
                    **{attribute: True},
                )
            )

    middleware.append(PersonPseudonymMiddleware(config))
    return middleware
