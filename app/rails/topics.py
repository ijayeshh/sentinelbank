"""Topical restriction — Implementation A of the §4.4 comparison rail.

An RM copilot is not a licensed advisor. Investment advice ("should she buy Nvidia?"),
tax advice ("can he deduct this?") and retirement-account strategy ("is now a good time
to convert to a Roth?") are all regulated activities in a US retail bank, and none of
them is something a 3B model should be improvising to a customer-facing employee.

This is the *only* rail in the project implemented twice. Implementation B is a Colang
flow in `nemo/rails/topics.co`; `nemo/README.md` compares them. Everything about this
file is therefore also an argument about the imperative approach — it is meant to be read
next to 30 lines of Colang.

================================================================================
WHY KEYWORD MATCHING, AND WHERE IT BREAKS
================================================================================
§4.4 allows "classifier or keyword based". This is keyword based, deliberately:

  * a fourth transformer would cost ~200 MB of the RAM budget §1.1 calls the binding
    constraint, to police a topic the bank can enumerate;
  * the injection rail already measured what happens when a classifier meets bank
    vocabulary — 29% false positives on ordinary memos (deviation 7) — and "tax",
    "invest" and "portfolio" are words an RM uses in legitimate sentences all day;
  * a deterministic rule can be read by the compliance officer who has to sign off on
    what the copilot refuses. That is worth more here than marginal recall.

The cost is stated rather than hidden: **this rail matches surface forms.** A paraphrase
that avoids the trigger vocabulary passes it, exactly as the held-out measurement showed
for the injection patterns (`docs/LIMITATIONS.md` §1). It is a topic filter, not a
semantic understanding of advice, and `nemo/README.md` reports the head-to-head against
NeMo's vector-matched canonical forms on the same eight prompts.

================================================================================
THE HARD PART IS THE ADJACENT QUERIES, NOT THE ADVICE
================================================================================
Refusing "which stocks should I buy?" is easy. The rail earns its place by NOT refusing
the four legitimate questions that sit right beside it:

    "What does our fee schedule say about IRAs?"          -> policy lookup, allowed
    "Which of my customers hold money market accounts?"   -> portfolio query, allowed
    "What does the CIP policy require?"                   -> policy lookup, allowed
    "What's this customer's current balance?"             -> account query, allowed

Every one contains vocabulary the naive version of this rail would trip on: `IRA`,
`money market`, `portfolio`, `balance`. So the rule is not "does this mention
investments" — it is **"is this asking the assistant to recommend, or to report?"** A
question about what a document *says*, or what a customer *holds*, is reporting. A
question about what someone *should do* is advice. That distinction is the whole rail,
and it is the same shape as the injection rail's data-noun suppression: the object
decides, not the keyword.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import NotRequired

from app.config import RailsConfig, settings
from app.rails.timing import rail_timer

# §2.5 / §4.3's latency name.
RAIL_TOPICAL = "topical"

# What the RM sees. Names the reason and points somewhere useful, because a refusal with
# no route forward just gets worked around — the RM asks the customer's brother-in-law
# instead, which is worse than the copilot answering.
REFUSAL_MESSAGE = (
    "I can't give investment, tax, or retirement-planning advice — an RM copilot is not "
    "a licensed advisor, and this is a regulated activity. Refer the customer to "
    "Wealth Management (investment and retirement) or to their tax professional.\n\n"
    "I can still help with what our policies say, what a customer holds, and anything "
    "in their account history."
)

# --------------------------------------------------------------------------------
# The two halves of the rule
# --------------------------------------------------------------------------------
# A prompt is refused when it asks for a RECOMMENDATION about a REGULATED SUBJECT. Both
# halves are required, and that conjunction is what keeps the four adjacent queries in
# §4.4's test set out of the net.

# Half one: the ask is for a recommendation, a judgement, or a prediction — not a fact.
# `should` covers most of it. The rest are the ways an RM phrases the same request when
# they are being polite about it.
_ADVICE_RE = re.compile(
    r"\b(?:should|ought to|advise|advice|recommend|recommendation|suggest|"
    r"best (?:option|choice|time|way)|worth (?:it|buying|selling|doing)|"
    r"good (?:idea|time)|what would you do|help (?:me|them|him|her) (?:decide|choose)|"
    r"is (?:now|it) a good time)\b"
    # Permission and eligibility questions are advice too, and they carry no `should`.
    # "Can my customer deduct this expense?" is a request for a tax opinion however it is
    # phrased. Bounded to 40 characters and anchored on the regulated *action* rather than
    # on a bare `can`, so "can you show me the balance" stays out of it.
    r"|\b(?:can|could|may|will|would|is|are)\b[^.?]{0,40}"
    r"\b(?:deduct|deductible|write off|written off|claim|taxable|tax[- ]free|exempt|"
    r"qualify|eligible|penali[sz]ed)\b",
    re.IGNORECASE,
)

# Half two: the subject is one a licensed advisor owns. Deliberately narrow — these are
# the categories a US retail bank's compliance function actually restricts, not every
# money-adjacent noun.
_REGULATED_RE = re.compile(
    r"\b(?:stock|stocks|equity|equities|share price|shares|bond|bonds|etf|etfs|"
    r"mutual fund|mutual funds|index fund|crypto|cryptocurrency|bitcoin|"
    r"invest|investing|investment|investments|portfolio allocation|asset allocation|"
    r"diversif\w*|rebalanc\w*|market timing|"
    r"tax|taxes|taxable|deduct|deduction|deductible|write off|write-off|capital gains|"
    r"ira|roth|401k|401\(k\)|retirement|annuity|annuities|pension|"
    r"estate planning|trust fund)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------------
# The exemption, and why it is not optional
# --------------------------------------------------------------------------------
# Both halves can fire on a question that is plainly a policy lookup — "what does our fee
# schedule say about IRAs" has `ira`, and an RM asking "should I quote the IRA fee?" is
# asking about a document, not about a security. So a request whose object is one of the
# bank's OWN artefacts is exempt: the copilot is reporting what the bank wrote down, which
# is precisely its job.
#
# This is the same construction as the injection rail's `is_data_scoped_request` (§3.2),
# and it is here for the same measured reason: without it the rail refuses legitimate
# work, and a rail that refuses legitimate work gets switched off.
#
# **Kept to the bank's own documents, and nothing else.** The first draft also listed
# `customer`, `account`, `balance` and `transaction`, and that was a bug rather than a
# generous reading: "How should this customer allocate their portfolio?" contains
# `customer`, so the exemption swallowed a textbook allocation request (§4.4's ADV-3).
# Almost every RM query names a customer or an account, so putting those nouns here turns
# the exemption into a universal bypass. A document is a thing the bank wrote down and can
# be quoted; a customer is not.
_BANK_ARTEFACT_RE = re.compile(
    r"\b(?:policy|policies|procedure|fee schedule|schedule of fees|disclosure|"
    r"cip|kyc|reg\s*[a-z]\b|regulation|regulatory|compliance|"
    r"our (?:rules|guidelines|process)|document|documentation|paperwork|"
    r"terms and conditions)\b",
    re.IGNORECASE,
)

# Phrases that make a question unambiguously advice even when a bank artefact is named —
# the exemption must not become a universal bypass. "Should they move their IRA balance
# into equities?" mentions `balance` and is still advice.
_OVERRIDE_RE = re.compile(
    r"\b(?:should (?:i|we|they|he|she|the customer|my customer)\s+(?:buy|sell|move|"
    r"convert|invest|allocate|withdraw|contribute|roll over|rollover)|"
    r"which (?:stock|stocks|fund|funds|etf|etfs)\b|"
    r"how much (?:should|ought)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TopicVerdict:
    """Why the rail did or did not refuse. Frozen — it goes to the audit log."""

    refused: bool
    reason: str
    advice_terms: tuple[str, ...] = ()
    regulated_terms: tuple[str, ...] = ()


def classify_topic(text: str) -> TopicVerdict:
    """Decide whether `text` is asking for regulated advice.

    Returns the matched terms as well as the boolean, so the audit log and §4.4's
    comparison table can show *why* — a refusal an RM cannot interrogate is a refusal
    they will route around.
    """
    if not text or not text.strip():
        return TopicVerdict(refused=False, reason="empty")

    advice = tuple(sorted({m.group(0).lower() for m in _ADVICE_RE.finditer(text)}))
    regulated = tuple(sorted({m.group(0).lower() for m in _REGULATED_RE.finditer(text)}))

    # The unambiguous forms short-circuit both the conjunction and the exemption.
    if _OVERRIDE_RE.search(text):
        return TopicVerdict(True, "explicit_advice_request", advice, regulated)

    if not advice or not regulated:
        # Reporting, not recommending. This is the branch the four adjacent queries take.
        return TopicVerdict(False, "not_advice" if not advice else "not_regulated",
                            advice, regulated)

    if _BANK_ARTEFACT_RE.search(text):
        return TopicVerdict(False, "bank_artefact_lookup", advice, regulated)

    return TopicVerdict(True, "regulated_advice", advice, regulated)


# ================================================================================
# Middleware
# ================================================================================


class TopicState(AgentState[Any]):
    """Agent state plus the rail's flag, so §3.7's banner can render the refusal."""

    topic_flags: NotRequired[list[dict[str, Any]]]


class TopicalMiddleware(AgentMiddleware):
    """Refuse regulated advice before the agent runs (§4.4, Implementation A).

    `before_agent` with `can_jump_to=["end"]`, the same shape as `InjectionMiddleware`:
    there is no partial way to honour a request for investment advice, so the run ends
    with the refusal rather than continuing with a caveat the 3B model may not keep.

    Position in the stack is deliberate — see `app/agent/graph.py`. It runs *after* the
    injection scan, because a prompt that is both an injection and an advice request is
    more usefully reported as an injection.
    """

    state_schema = TopicState

    def __init__(self, rails: RailsConfig | None = None) -> None:
        super().__init__()
        self.rails = rails if rails is not None else settings.guardrails_enabled

    @property
    def enabled(self) -> bool:
        """Read live, so the Streamlit toggle works without rebuilding the graph."""
        return self.rails.topical

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return rail_timer(RAIL_TOPICAL)(self._check)(state)

    def _check(self, state: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, HumanMessage) or not message.content:
                continue

            verdict = classify_topic(str(message.content))
            if not verdict.refused:
                return None

            return {
                "messages": [AIMessage(content=REFUSAL_MESSAGE)],
                "jump_to": "end",
                "topic_flags": [
                    {
                        "rail": RAIL_TOPICAL,
                        "reason": verdict.reason,
                        "advice_terms": list(verdict.advice_terms),
                        "regulated_terms": list(verdict.regulated_terms),
                    }
                ],
            }
        return None
