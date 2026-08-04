"""Custom Presidio recognizer for ABA routing numbers (PROJECT_PLAN.md §3.1.3).

A bare \\d{9} pattern floods with false positives against amounts and IDs, so the
recognizer only matches when the weighted checksum passes:
    (3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9)) mod 10 == 0
The same formula generates seed routing numbers, so detector and fixtures agree.

--------------------------------------------------------------------------------
WHY A CHECKSUM, AND NOT JUST A PATTERN
--------------------------------------------------------------------------------
This is the file to read if you want to know why this project builds deterministic
rails instead of pattern-matching ones.

A routing number has no distinguishing shape. It is nine digits. That is the entire
specification — no check letter, no prefix, no separator, nothing a regex can anchor
to. So `\\d{9}` is not a *poor* detector of routing numbers; it is a detector of
nine-digit strings, and this schema is full of nine-digit strings that are not
routing numbers:

  * `transactions.amount` — a $1,234,567.89 wire renders as `123456789` once the
    formatting is stripped, and every intermediate figure in a balance calculation
    is a candidate.
  * `accounts.account_number` — 10-12 digits (§1.5), so every one of them *contains*
    several 9-digit substrings.
  * Unix timestamps — 9 digits for any date between 1973 and 2001, and this ledger
    carries `txn_date`, `opened_date`, and `created_at`.
  * Internal references — ticket numbers, confirmation codes, `REF 88213`-style
    strings in the memo fields.

Redacting all of those would be worse than redacting none. The PII rail runs on tool
results before they enter model context (§3.1.5), so a false positive does not merely
produce a noisy log — it deletes a balance out of the data the model is answering
from, and the RM gets a confidently wrong number. **An over-firing PII rail is a
correctness bug in the answer, not a conservative safety margin.**

The checksum is what makes the detector usable. Roughly 90% of arbitrary nine-digit
strings fail it, so it removes about nine of every ten false positives at zero
inference cost, deterministically, with no model and no network. Note the honest
residual: ~10% of random nine-digit strings pass by chance, which is why the base
confidence is 0.5 and not 1.0 — a passing checksum is evidence, not proof. Context
words are what separate the two.

The general principle, which is the reason §3.1.3 asks for exactly one custom
recognizer: a deterministic rail earns its place by *validating*, not by matching. A
rail that only pattern-matches has all the false positives of a model and none of the
recall, and it is the worst of both.

--------------------------------------------------------------------------------
IMPORT WEIGHT
--------------------------------------------------------------------------------
The two checksum helpers are kept free of any `presidio_analyzer` import, and the
Presidio subclass below imports lazily inside a factory. `db/seed.py` imports
`validate_aba`/`generate_valid_aba` from this module long before Presidio is
configured, and an eager import here would drag spaCy and Presidio into every seed
run — ~250 MB of RAM (§1.2) to generate 60 nine-digit strings.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from presidio_analyzer import PatternRecognizer, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpArtifacts

# The weights applied to the nine digits, in order. This is the ABA/routing transit
# number check defined by the American Bankers Association: positions repeat 3-7-1.
_ABA_WEIGHTS = (3, 7, 1, 3, 7, 1, 3, 7, 1)

# Federal Reserve routing symbols occupy 00-12 (and 21-32 for thrift institutions), so
# a real routing number never starts with 4-9. Constraining the leading digit keeps the
# generated fixtures plausible; it is NOT part of the checksum and validate_aba() does
# not enforce it, because a detector that rejected an unusual-but-valid prefix would
# fail open on real customer data.
_LEADING_DIGITS = (0, 1, 2, 3)


def validate_aba(candidate: str) -> bool:
    """Return True if `candidate` is a 9-digit string passing the ABA checksum.

    This is the whole reason the project ships a custom recognizer instead of a regex.
    A 9-digit pattern alone matches transaction amounts, zip+4 concatenations, and
    internal IDs; roughly 90% of arbitrary 9-digit strings fail this checksum, which is
    what makes the detector usable against a transaction ledger.

    Formatting characters are stripped first (real documents write 021-000-021), but the
    result must be exactly 9 digits — a 10-digit account number that happens to contain
    a valid 9-digit substring is not a routing number, and widening this would put the
    PII rail into false-positive territory it cannot recover from.

    Fails closed on anything unparseable: a non-string, an empty string, and a
    wrong-length string all return False rather than raising (§6 rule 4).
    """
    if not isinstance(candidate, str):
        return False

    digits = [int(ch) for ch in candidate if ch.isdigit()]
    if len(digits) != 9:
        return False

    total = sum(weight * digit for weight, digit in zip(_ABA_WEIGHTS, digits))
    return total % 10 == 0


def generate_valid_aba(rng: random.Random) -> str:
    """Generate a 9-digit routing number that passes validate_aba().

    Takes an explicit Random instance rather than touching the module-level `random`,
    so db/seed.py keeps sole ownership of RNG state and a stray call from elsewhere
    cannot shift the seeded data (§6 rule 3).

    The check digit is SOLVED for, not searched for. Rejection sampling — draw nine
    digits, discard if the checksum fails — would consume a variable number of values
    from `rng` per call, so every row generated afterwards would depend on how many
    retries happened to occur. That silently breaks byte-identical reseeding. Here the
    call site always draws exactly 8 values.

    Solving: with d9 weighted 1, the checksum requires
        (weighted_sum_of_first_8 + d9) mod 10 == 0
    so d9 = (-weighted_sum_of_first_8) mod 10, which is always a single digit.
    """
    first_eight = [rng.choice(_LEADING_DIGITS)] + [rng.randint(0, 9) for _ in range(7)]

    partial = sum(weight * digit for weight, digit in zip(_ABA_WEIGHTS, first_eight))
    check_digit = (-partial) % 10

    routing_number = "".join(str(d) for d in first_eight) + str(check_digit)

    # Cheap self-check: the generator and the detector must never diverge, and this
    # assertion is what catches it at seed time rather than in a Phase 3 red-team run.
    assert validate_aba(routing_number), f"generated invalid ABA: {routing_number}"
    return routing_number


# ================================================================================
# THE PRESIDIO RECOGNIZER (§3.1.3)
# ================================================================================

# The entity name Presidio reports and the PII rail's entity map keys on (§3.1.2,
# where ABA_ROUTING is the one row whose detector column reads "Custom recognizer").
# Its anonymization strategy is `redact`.
ABA_ROUTING_ENTITY = "ABA_ROUTING"

# --------------------------------------------------------------------------------
# Patterns: cast a wide net, then let the checksum do the actual deciding
# --------------------------------------------------------------------------------
# These regexes are deliberately permissive. They are the *candidate generator*, not
# the detector — their only job is to find nine-digit sequences cheaply so that
# validate_aba() can rule out roughly nine in ten of them. Tightening the regex here
# to try to reduce false positives would be solving the problem at the wrong layer
# and would cost recall against real routing numbers, which have no distinguishing
# shape to tighten toward.
#
# `\b` on both ends is doing real work: it stops the bare pattern from firing on a
# 9-digit window *inside* a longer digit run. `accounts.account_number` is 10-12
# digits (§1.5), so without the boundaries every account number in the database would
# yield several overlapping candidates, some of which would pass the checksum by
# chance. A routing number is nine digits standing alone, never nine digits carved
# out of twelve.
_BARE_PATTERN = r"\b\d{9}\b"

# The 3-3-3 grouped form, which is how routing numbers are printed on a cheque and
# written in wire instructions (`021-000-021`). validate_aba() strips the separators
# before checksumming, so both forms reach the same validation.
#
# This grouping cannot collide with the other separator-bearing PII in this schema:
# an SSN is 3-2-4 (`123-45-6789`), a US phone number is 3-3-4 (`555-123-4567`), and
# an ISO date is 4-2-2 (`2026-08-03`). None of them match 3-3-3.
_GROUPED_PATTERN = r"\b\d{3}[- ]\d{3}[- ]\d{3}\b"

# Base confidence for a candidate that passed the checksum with no supporting context.
#
# 0.5 rather than 1.0, and this number is the honest part of the file. The checksum
# eliminates ~90% of arbitrary nine-digit strings, but ~10% survive it by chance — a
# transaction amount that happens to satisfy the ABA weights is indistinguishable
# from a routing number by any deterministic test, because it *is* the same nine
# digits. A passing checksum is strong evidence, not proof, and the score says so.
_BARE_SCORE = 0.5

# The grouped form scores higher before any context is considered: nine digits are
# ambiguous, but nine digits a human chose to punctuate as 3-3-3 are almost always a
# routing number. Nothing else in this schema is written that way.
_GROUPED_SCORE = 0.6

# --------------------------------------------------------------------------------
# Context words
# --------------------------------------------------------------------------------
# Presidio's AnalyzerEngine passes these to its LemmaContextAwareEnhancer, which
# raises the score of a match that sits near one of them. This is what closes the gap
# the checksum cannot: "wire to ABA 021000021" and a bare "021000021" in an amount
# column are the same nine digits, and only the surrounding words separate them.
#
# Kept to terms that are genuinely diagnostic of a routing number in banking prose.
# Adding common words ("account", "number", "payment") would boost the score of every
# amount that appears in a sentence about payments, which is precisely the false
# positive this rail exists to avoid.
ABA_CONTEXT_WORDS = [
    "aba",
    "routing",
    "rtn",
    "wire",
    "ach",
    "transit",
    "swift",
    "fedwire",
    "intermediary",
]

# --------------------------------------------------------------------------------
# Context scoring constants
# --------------------------------------------------------------------------------
# These mirror the defaults of Presidio's LemmaContextAwareEnhancer exactly, and they
# are named here because this module applies the same arithmetic itself when it runs
# standalone (see AbaRoutingRecognizer.analyze). Two code paths that produce different
# confidence scores for the same text would make the Phase 4 numbers unreproducible,
# so they produce identical ones.
_CONTEXT_SIMILARITY_FACTOR = 0.35  # added to the score on a supportive word
_MIN_SCORE_WITH_CONTEXT = 0.4  # floor once context is found
_CONTEXT_PREFIX_WORDS = 5  # words BEFORE the match that are inspected
_MAX_SCORE = 1.0

# Alphabetic tokens only. Digits and punctuation are stripped so that "ABA:" and
# "(routing)" tokenize to the bare word, and so the five-word window is five *words*
# rather than five whitespace runs of a numeric table.
_WORD_RE = re.compile(r"[A-Za-z]+")


def _has_supportive_context(text: str, start: int) -> bool:
    """Whether a context word appears in the five words preceding a match.

    Prefix-only, matching LemmaContextAwareEnhancer's defaults (prefix 5, suffix 0).
    English banking prose puts the label before the value — "routing number
    021000021", "ABA: 021000021", "wire via 021000021" — so a suffix window buys
    little and would boost any nine-digit figure that happens to be followed by a
    sentence about wires.

    Matching is whole-token, with a bare plural allowed. Presidio's default
    "substring" mode is looser and mis-fires here: "aba" is a substring of
    "database", so a memo reading "database export 021000021" would score as though
    it were labelled routing data.
    """
    prefix_words = _WORD_RE.findall(text[:start].lower())[-_CONTEXT_PREFIX_WORDS:]
    return any(
        token == word or token == f"{word}s"
        for token in prefix_words
        for word in ABA_CONTEXT_WORDS
    )


def _score_with_context(text: str, start: int, score: float) -> float:
    """Apply the context boost to one match's score. Pure arithmetic, no state."""
    if not _has_supportive_context(text, start):
        return score
    return min(max(score + _CONTEXT_SIMILARITY_FACTOR, _MIN_SCORE_WITH_CONTEXT), _MAX_SCORE)


def _build_recognizer_class() -> type:
    """Define and return the recognizer class, importing Presidio on first use.

    The import is deferred rather than placed at module scope because importing
    `presidio_analyzer` costs **7.9 s and pulls spaCy into the process** (measured on
    this machine). `db/seed.py` imports `validate_aba` and `generate_valid_aba` from
    this module to generate 60 routing numbers, and it must not pay that — §1.2
    budgets Presidio + spaCy at ~250 MB of RAM, which is real money against 8 GB and
    is pure waste in a script that never analyzes any text.

    Defining the class inside a function is the price of that deferral. It is called
    once, from `create_aba_recognizer` below, which is the only supported entry point.
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    class AbaRoutingRecognizer(PatternRecognizer):
        """Presidio recognizer for ABA routing numbers, gated on the checksum.

        Wide regex -> checksum -> context. The regex proposes, `validate_aba`
        disposes, and the context words decide how much to believe what survives.
        """

        def __init__(self, supported_language: str = "en") -> None:
            super().__init__(
                supported_entity=ABA_ROUTING_ENTITY,
                name="AbaRoutingRecognizer",
                supported_language=supported_language,
                patterns=[
                    Pattern("ABA routing (grouped)", _GROUPED_PATTERN, _GROUPED_SCORE),
                    Pattern("ABA routing (bare)", _BARE_PATTERN, _BARE_SCORE),
                ],
                context=ABA_CONTEXT_WORDS,
            )

        def validate_result(self, pattern_text: str) -> bool | None:
            """Drop candidates that fail the ABA checksum.

            Presidio's contract for this hook has three return values, and the choice
            between them is the whole reason this method is four lines and this
            docstring is twenty:

                False -> score is forced to MIN_SCORE (0) and PatternRecognizer drops
                         the result entirely. This is what we want for a failed
                         checksum: not a low-confidence match, no match at all.
                True  -> score is forced to **MAX_SCORE (1.0)**.
                None  -> "no validation was performed"; the pattern's own score stands.

            So a passing checksum returns None, not True — deliberately, and this is
            the one non-obvious line in the file.

            Returning True would pin every checksum-valid nine-digit string at 1.0.
            That is wrong on the facts (~10% of arbitrary nine-digit strings pass the
            checksum, so certainty is not available) and it would also destroy the
            context signal: a score already at the 1.0 ceiling cannot be raised by
            the context enhancer, so `021000021` sitting in an amount column and
            `ABA 021000021` in a wire instruction would be reported identically. The
            gradient between them is the only thing that lets the PII rail treat them
            differently, and it exists only if the passing case leaves headroom.

            `validate_aba` is imported from this same module rather than restated —
            the generator in `db/seed.py` and the detector here must never be able to
            disagree about what a valid routing number is (§1.6).
            """
            if validate_aba(pattern_text):
                return None  # keep the pattern's score; see above
            return False  # checksum failed -> not a routing number, drop it

        def analyze(
            self,
            text: str,
            entities: list[str],
            nlp_artifacts: "NlpArtifacts | None" = None,
            regex_flags: int | None = None,
        ) -> list["RecognizerResult"]:
            """Run the patterns, then apply the context boost when nobody else will.

            Under `AnalyzerEngine` — which is how the Phase 3 PII rail uses this —
            `nlp_artifacts` is always supplied, and the engine's own
            LemmaContextAwareEnhancer applies the boost from `self.context` using
            spaCy lemmas. This method then does nothing extra, and there is no double
            counting.

            Called directly, with no engine and no `nlp_artifacts`, that enhancer never
            runs and every match would be stuck at its base score. So this applies the
            same boost itself, with the same constants — see the block above. That
            keeps the recognizer meaningful as a standalone unit (a rail that can only
            be exercised with a 500 MB spaCy pipeline loaded is a rail nobody tests)
            while production behaviour stays the framework's.
            """
            results = super().analyze(
                text, entities, nlp_artifacts=nlp_artifacts, regex_flags=regex_flags
            )

            if nlp_artifacts is None:
                for result in results:
                    result.score = _score_with_context(text, result.start, result.score)

            return results

    return AbaRoutingRecognizer


def create_aba_recognizer(supported_language: str = "en") -> Any:
    """Build the ABA routing recognizer. The supported entry point for §3.1.

    Phase 3's `app/rails/pii.py` calls this and registers the result on its
    `AnalyzerEngine`'s registry. Returned as a fresh instance rather than a singleton
    because Presidio's registry takes ownership of what it is given, and a recognizer
    shared across two engines with different languages is a bug waiting to happen.
    """
    return _build_recognizer_class()(supported_language=supported_language)
