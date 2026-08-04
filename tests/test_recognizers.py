"""Tests for the custom ABA routing recognizer (§3.1.3).

The claim this file has to establish is narrow and load-bearing: **the checksum, not
the regex, is what makes this detector usable.** So the tests are built in matched
pairs — the same nine-digit shape, once with a valid checksum and once without — and
the false-positive cases are drawn from this schema rather than invented, because the
cost of over-firing here is not a noisy log. The PII rail scans tool results before
they reach model context (§3.1.5), so a false positive deletes a balance out of the
data the model is answering from and the RM gets a confidently wrong number.

Most of the file needs no services. The one test that reads the seeded `accounts`
table requires `docker compose up -d` and `python db/seed.py`, and fails rather than
skips when Postgres is unreachable, matching tests/test_read_tools.py.
"""

from __future__ import annotations

import random

import pytest
from sqlalchemy import text

from app.db import get_session
from app.rails.recognizers import (
    ABA_ROUTING_ENTITY,
    _BARE_SCORE,
    _GROUPED_SCORE,
    create_aba_recognizer,
    generate_valid_aba,
    validate_aba,
)

# --------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------

# Nine-digit strings that pass the checksum, generated the same way db/seed.py
# generates them so the detector is tested against exactly the shape of data it will
# meet in production.
VALID_ROUTING_NUMBERS = [generate_valid_aba(random.Random(seed)) for seed in range(12)]

# Nine-digit strings that do NOT pass the checksum, each one a single digit away from
# a valid number. Single-digit substitution is always caught: the ABA weights are 3,
# 7 and 1, all coprime with 10, so changing one digit shifts the weighted sum by a
# non-zero amount mod 10 every time. These are the cases a `\d{9}` regex cannot tell
# apart from the list above — they differ from a real routing number by one character.
INVALID_ROUTING_NUMBERS = [
    "004332185"[:i] + str((int("004332185"[i]) + 1) % 10) + "004332185"[i + 1 :]
    for i in range(9)
]

# Nine-digit strings this schema actually produces, none of which are routing numbers.
# Each is annotated with where it comes from, because the point of the test is that
# these are realistic, not that they are hard.
DOMAIN_FALSE_POSITIVES = {
    # transactions.amount — a $1,234,567.89 wire, formatting stripped. Every
    # intermediate figure in a balance calculation is a candidate of this shape.
    "transaction amount": "123456789",
    "large balance": "987654321",
    # A Unix timestamp is nine digits for any date between 1973 and 2001, and this
    # schema carries txn_date, opened_date and created_at.
    "unix timestamp": "981173106",
    "unix timestamp (2)": "172959000",
    # Internal references: ticket numbers, confirmation codes, the REF 88213-style
    # strings that appear in the seeded memo fields.
    "customer reference": "400512345",
    "confirmation code": "555000111",
}


@pytest.fixture(scope="module")
def recognizer():
    """One recognizer for the module. Building it imports Presidio (~8 s)."""
    return create_aba_recognizer()


def detect(recognizer, text_to_scan: str):
    """Analyze `text_to_scan` for ABA_ROUTING and return the results."""
    return recognizer.analyze(text_to_scan, entities=[ABA_ROUTING_ENTITY])


def detected_spans(recognizer, text_to_scan: str) -> set[str]:
    """The substrings the recognizer actually flagged."""
    return {
        text_to_scan[result.start : result.end]
        for result in detect(recognizer, text_to_scan)
    }


# --------------------------------------------------------------------------------
# The premise: these two sets are indistinguishable by pattern
# --------------------------------------------------------------------------------


def test_valid_and_invalid_fixtures_are_the_same_shape():
    """Guards the premise of every test below.

    If the invalid fixtures were malformed — eight digits, or letters — the detector
    would reject them on shape and the checksum would never be exercised. Every
    fixture on both sides is exactly nine digits, so the only thing that can separate
    them is the checksum.
    """
    for candidate in VALID_ROUTING_NUMBERS + INVALID_ROUTING_NUMBERS:
        assert len(candidate) == 9 and candidate.isdigit()

    assert all(validate_aba(n) for n in VALID_ROUTING_NUMBERS)
    assert not any(validate_aba(n) for n in INVALID_ROUTING_NUMBERS)


# --------------------------------------------------------------------------------
# Detection of real routing numbers
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("routing_number", VALID_ROUTING_NUMBERS)
def test_valid_routing_numbers_are_detected(recognizer, routing_number):
    results = detect(recognizer, f"Please wire to {routing_number} today.")

    assert len(results) == 1
    assert results[0].entity_type == ABA_ROUTING_ENTITY
    assert results[0].score > 0


def test_every_seeded_routing_number_is_detected(recognizer):
    """§3.1.3's real acceptance test: the detector finds the data the seed planted.

    db/seed.py generates routing numbers with `generate_valid_aba`, which asserts the
    checksum holds. This proves the detector agrees with the generator across all 60
    accounts — the two share `validate_aba`, and this is what would catch them
    drifting apart.
    """
    try:
        with get_session() as session:
            routing_numbers = [
                row[0]
                for row in session.execute(
                    text("SELECT routing_number FROM accounts ORDER BY account_id")
                ).all()
            ]
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "this test requires the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )

    assert len(routing_numbers) == 60, (
        f"seed data has drifted (got {len(routing_numbers)} accounts); re-run db/seed.py"
    )

    undetected = [n for n in routing_numbers if not detect(recognizer, n)]
    assert undetected == [], f"{len(undetected)} seeded routing numbers not detected"


def test_grouped_form_is_detected_and_scores_higher(recognizer):
    """`021-000-021` — how routing numbers are printed on a cheque.

    Scores above the bare form before any context: nine digits are ambiguous, nine
    digits punctuated 3-3-3 are not.
    """
    routing_number = VALID_ROUTING_NUMBERS[0]
    grouped = f"{routing_number[:3]}-{routing_number[3:6]}-{routing_number[6:]}"

    results = detect(recognizer, f"Cheque shows {grouped} at the bottom.")

    assert len(results) == 1
    assert results[0].score == pytest.approx(_GROUPED_SCORE)
    assert results[0].score > _BARE_SCORE


# --------------------------------------------------------------------------------
# The checksum: what a regex alone would get wrong
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("near_miss", INVALID_ROUTING_NUMBERS)
def test_checksum_failures_are_not_detected(recognizer, near_miss):
    """A nine-digit string that fails the checksum produces no match at all.

    Not a low-confidence match — no result. `validate_result` returns False, Presidio
    forces the score to MIN_SCORE, and the result is dropped before it is returned.

    Each of these is one digit away from a genuine routing number, so a detector
    built on `\\d{9}` would flag every one of them with full confidence.
    """
    assert detect(recognizer, f"Please wire to {near_miss} today.") == []


@pytest.mark.parametrize(
    "description,value", sorted(DOMAIN_FALSE_POSITIVES.items())
)
def test_domain_false_positives_are_not_detected(recognizer, description, value):
    """Nine-digit values this schema really produces, none of them routing numbers."""
    assert not validate_aba(value), (
        f"fixture {description!r} ({value}) passes the checksum, so it proves nothing "
        "about the checksum — replace it"
    )
    assert detect(recognizer, value) == [], f"false positive on {description}"


def test_amount_in_context_is_not_detected(recognizer):
    """The realistic form: a currency amount inside a transaction memo.

    Includes the `$` and the decimals, so the regex sees `123456789` with clean word
    boundaries on both sides and does propose it as a candidate. The checksum is the
    only thing standing between that and a redacted balance.
    """
    assert detect(recognizer, "POS DEBIT/ACME CORP/$123456789.00 posted 08/03") == []


def test_account_numbers_are_not_partially_matched(recognizer):
    """A 10-12 digit account number must not yield a 9-digit match inside it.

    `accounts.account_number` is 10-12 digits (§1.5). Without word boundaries on the
    pattern, every account number in the database would offer several overlapping
    nine-digit windows, and ~10% of those would pass the checksum by chance — a
    steady drip of false positives from the most common column in the schema.
    """
    routing_number = VALID_ROUTING_NUMBERS[0]

    # An account number that literally contains a valid routing number.
    for embedded in (f"{routing_number}7", f"7{routing_number}", f"55{routing_number}55"):
        assert detect(recognizer, f"Account {embedded} on file") == [], (
            f"matched inside the longer digit run {embedded}"
        )


def test_other_pii_shapes_do_not_match(recognizer):
    """SSN (3-2-4), phone (3-3-4) and ISO date (4-2-2) are other recognizers' work.

    None of them is 3-3-3, so the grouped pattern must stay off them — a routing
    number recognizer that also claims SSNs would corrupt the §3.1.2 entity map,
    where the two have different anonymization strategies.
    """
    for other in ("123-45-6789", "555-123-4567", "2026-08-03"):
        assert detect(recognizer, f"On file: {other}") == []


# --------------------------------------------------------------------------------
# Context scoring
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Routing number {n} for the transfer",
        "ABA {n} confirmed",
        "Send the wire to {n} this afternoon",
        "ACH origination via {n}",
        "Fedwire {n} on file",
    ],
)
def test_context_words_raise_the_score(recognizer, phrase):
    """A checksum-valid number near banking vocabulary is believed more strongly."""
    routing_number = VALID_ROUTING_NUMBERS[0]

    bare = detect(recognizer, routing_number)[0].score
    with_context = detect(recognizer, phrase.format(n=routing_number))[0].score

    assert bare == pytest.approx(_BARE_SCORE)
    assert with_context > bare


def test_score_without_context_is_not_certainty(recognizer):
    """The base score must stay well below 1.0, and this is the honest part.

    ~10% of arbitrary nine-digit strings pass the ABA checksum by chance, so a
    passing checksum is evidence and not proof. If `validate_result` returned True
    instead of None, Presidio would pin every match at MAX_SCORE — the number would
    claim a certainty the mathematics does not support, and the context boost below
    would have no headroom left to express the difference.
    """
    (result,) = detect(recognizer, VALID_ROUTING_NUMBERS[0])

    assert result.score == pytest.approx(_BARE_SCORE)
    assert result.score < 1.0


def test_unrelated_words_do_not_raise_the_score(recognizer):
    """The boost is specific. Prose about payments is not evidence of a routing number."""
    routing_number = VALID_ROUTING_NUMBERS[0]

    (result,) = detect(recognizer, f"The customer called about {routing_number} yesterday")

    assert result.score == pytest.approx(_BARE_SCORE)


def test_context_word_must_be_a_whole_word(recognizer):
    """The word 'database' contains 'aba' as a substring and must not boost.

    Presidio's default context matching is substring-based, which would fire here.
    This recognizer matches whole tokens instead — a memo reading "database export
    <digits>" is not a wire instruction.
    """
    routing_number = VALID_ROUTING_NUMBERS[0]

    (result,) = detect(recognizer, f"Database export {routing_number} completed")

    assert result.score == pytest.approx(_BARE_SCORE)


def test_context_window_is_bounded(recognizer):
    """A context word far from the match does not boost it.

    Five words, mirroring LemmaContextAwareEnhancer's default prefix window. Without
    a bound, one "wire" anywhere in a long tool result would lift every nine-digit
    figure in it.
    """
    routing_number = VALID_ROUTING_NUMBERS[0]
    far = "wire " + "and then some other unrelated words appeared " + routing_number

    (result,) = detect(recognizer, far)

    assert result.score == pytest.approx(_BARE_SCORE)


# --------------------------------------------------------------------------------
# Integration with Presidio's AnalyzerEngine
# --------------------------------------------------------------------------------


def test_registers_with_analyzer_engine_and_context_still_applies():
    """End-to-end through the engine the Phase 3 PII rail will actually use.

    Two things are being checked. First that the recognizer registers and reports
    ABA_ROUTING through the normal `AnalyzerEngine.analyze` path. Second that the
    engine's own LemmaContextAwareEnhancer applies the context boost — and that the
    recognizer does not *also* apply it, which is why `analyze` only scores context
    itself when `nlp_artifacts` is None. A double boost here would put every
    labelled routing number at the 1.0 ceiling and quietly discard the gradient.

    Loads spaCy, so it is the slow test in this file.
    """
    from presidio_analyzer import AnalyzerEngine

    engine = AnalyzerEngine()
    engine.registry.add_recognizer(create_aba_recognizer())

    routing_number = VALID_ROUTING_NUMBERS[0]
    bare = engine.analyze(routing_number, entities=[ABA_ROUTING_ENTITY], language="en")
    labelled = engine.analyze(
        f"Routing number {routing_number} for the transfer",
        entities=[ABA_ROUTING_ENTITY],
        language="en",
    )

    assert len(bare) == 1 and len(labelled) == 1
    assert bare[0].entity_type == ABA_ROUTING_ENTITY
    assert bare[0].score == pytest.approx(_BARE_SCORE)

    # Boosted, but not pinned to the ceiling — exactly one enhancer ran.
    assert labelled[0].score > bare[0].score
    assert labelled[0].score < 1.0

    # And the checksum still governs inside the engine.
    assert (
        engine.analyze(
            f"Routing number {INVALID_ROUTING_NUMBERS[0]} for the transfer",
            entities=[ABA_ROUTING_ENTITY],
            language="en",
        )
        == []
    )
