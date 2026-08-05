"""Tests for the hybrid PII rail (§3.1).

Three claims have to hold here, and they pull against each other:

  1. Regulated identifiers never reach model context — at **all three** application
     points, tool results included (§3.1.5).
  2. Names survive the round trip: the model sees a token, the RM sees the real name,
     and the map is unreachable from any other request (§3.1.4).
  3. **The rail does not destroy the data it is protecting.** A transaction date must
     come through untouched while a date of birth is redacted, and an account number
     must keep its last four digits. Over-redaction here is not a conservative safety
     margin — it is a correctness bug in the RM's answer.

Middleware hooks are driven directly with message-state dicts. No LLM, no graph, no
Ollama. The seeded database is required, because §2.6's reasoning applies just as much
to this rail: an SSN detector that only works on hand-written fixtures proves nothing
about the SSNs actually in `customers`.

A note on fixtures: Presidio's `UsSsnRecognizer` deliberately *rejects* placeholder
SSNs like `123-45-6789` (sequential digits, `invalidate_result`). Writing this file
against invented SSNs would have produced a suite that passes while the rail detects
nothing. Every identifier below comes from the seeded tables for that reason.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import text

from app.config import RailsConfig
from app.db import get_session
from app.rails import timing
from app.rails.pii import (
    FLAG_UNISSUED_TOKEN,
    PERSON_TOKEN_RE,
    RAIL_PII_INPUT,
    RAIL_PII_OUTPUT,
    RAIL_PII_TOOL_RESULT,
    UNKNOWN_PERSON_PLACEHOLDER,
    PersonPseudonymMiddleware,
    build_pii_middleware,
    clear_all_requests,
    get_request_map,
    pii_request,
    resolve_entities,
)

RM1 = "RM-001"


# --------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seeded():
    """One real customer plus one of their accounts, straight from the seed."""
    try:
        with get_session() as session:
            row = session.execute(
                text(
                    """
                    SELECT c.customer_id, c.full_name, c.ssn, c.date_of_birth,
                           c.phone, c.city, a.account_number, a.routing_number
                    FROM customers c
                    JOIN accounts a ON a.customer_id = c.customer_id
                    WHERE c.rm_id = :rm
                    ORDER BY c.customer_id, a.account_id
                    LIMIT 1
                    """
                ),
                {"rm": RM1},
            ).mappings().first()
    except Exception as exc:  # pragma: no cover - environment problem, not a bug
        pytest.fail(
            "PII rail tests require the seeded database.\n"
            "Run: docker compose up -d ; python db/seed.py\n"
            f"Connection error: {exc}"
        )
    assert row is not None, "seed data has drifted; re-run python db/seed.py"
    return dict(row)


@pytest.fixture(scope="module")
def all_routing_numbers():
    with get_session() as session:
        return [
            r[0]
            for r in session.execute(
                text("SELECT routing_number FROM accounts ORDER BY account_id")
            ).all()
        ]


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Keep @rail_timer out of the repo's latency log, and never leak a map."""
    monkeypatch.setattr(timing, "LATENCY_LOG_PATH", tmp_path / "latency.jsonl")
    timing.clear_samples()
    clear_all_requests()
    yield
    clear_all_requests()


# --------------------------------------------------------------------------------
# Helpers: drive the middleware stack over one message list
# --------------------------------------------------------------------------------

# Everything except the pseudonymizer — the redact/mask rules only.
REDACTION_RAILS = RailsConfig(person_pseudonym=False)


def run_before_model(messages: list, rails: RailsConfig = REDACTION_RAILS) -> list:
    """Push `messages` through every before_model hook, in stack order."""
    state = {"messages": list(messages)}
    for middleware in build_pii_middleware(rails):
        update = middleware.before_model(state, None)
        if update:
            state.update(update)
    return state["messages"]


def run_after_model(messages: list, rails: RailsConfig = REDACTION_RAILS) -> dict:
    state = {"messages": list(messages)}
    for middleware in build_pii_middleware(rails):
        update = middleware.after_model(state, None)
        if update:
            state.update(update)
    return state


def serialize(row: dict) -> str:
    """A tool result as the agent will actually see it — a serialized row."""
    return str({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()})


# ================================================================================
# 1. Regulated identifiers are removed at all three application points
# ================================================================================


def test_ssn_redacted_from_input(seeded):
    out = run_before_model([HumanMessage(content=f"Customer SSN is {seeded['ssn']}")])

    assert seeded["ssn"] not in str(out[0].content)
    assert "[REDACTED_US_SSN]" in str(out[0].content)


def test_ssn_redacted_from_tool_results(seeded):
    """§3.1.5's easily-overlooked point. The SQL tool returns raw customer rows."""
    out = run_before_model(
        [
            HumanMessage(content="show me the customer"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(
                content=serialize(seeded), name="query_customer_data", tool_call_id="c1"
            ),
        ]
    )

    content = str(out[-1].content)
    assert seeded["ssn"] not in content
    assert "[REDACTED_US_SSN]" in content


def test_ssn_redacted_from_output(seeded):
    state = run_after_model([AIMessage(content=f"The SSN on file is {seeded['ssn']}.")])

    assert seeded["ssn"] not in str(state["messages"][0].content)


def test_every_seeded_ssn_is_detected():
    """Not just the one fixture — the detector must work across the whole table."""
    with get_session() as session:
        ssns = [r[0] for r in session.execute(text("SELECT ssn FROM customers")).all()]

    assert len(ssns) == 40, "seed data has drifted; re-run python db/seed.py"
    missed = [s for s in ssns if "US_SSN" not in {m.entity for m in resolve_entities(s)}]
    assert missed == [], f"{len(missed)}/40 seeded SSNs undetected: {missed[:3]}"


def test_routing_numbers_are_redacted(all_routing_numbers, seeded):
    """The custom §3.1.3 recognizer, reached through the rail rather than directly."""
    assert len(all_routing_numbers) == 60

    undetected = [
        n
        for n in all_routing_numbers
        if "ABA_ROUTING" not in {m.entity for m in resolve_entities(n)}
    ]
    assert undetected == [], f"{len(undetected)} seeded routing numbers undetected"

    out = run_before_model(
        [HumanMessage(content=f"wire via routing {seeded['routing_number']}")]
    )
    assert seeded["routing_number"] not in str(out[0].content)
    assert "[REDACTED_ABA_ROUTING]" in str(out[0].content)


# ================================================================================
# 2. Masking keeps the data usable
# ================================================================================


def test_account_number_masked_to_last_four(seeded):
    account = seeded["account_number"]
    out = run_before_model([HumanMessage(content=f"balance on {account}?")])
    content = str(out[0].content)

    assert account not in content
    # The RM identifies accounts by the last four; that is the point of `mask`.
    assert f"****{account[-4:]}" in content
    # And no more than four digits survive.
    assert account[:-4] not in content


def test_account_number_masks_rather_than_redacting_as_a_date(seeded):
    """Regression for the overlap finding.

    Presidio claims a 12-digit account number as DATE_TIME at 0.85 and as
    US_BANK_NUMBER at 0.40. Ranking by score redacts it as a date and the RM loses the
    last four digits entirely. ENTITY_PRIORITY is what prevents that.
    """
    entities = {
        m.value: m.entity for m in resolve_entities(f"account {seeded['account_number']}")
    }

    assert entities.get(seeded["account_number"]) == "US_BANK_NUMBER"


def test_routing_number_beats_phone_and_bank_number(seeded):
    """The same overlap problem on 9 digits: ABA_ROUTING is checksum-validated, so it wins."""
    resolved = resolve_entities(f"routing {seeded['routing_number']}")
    matched = [m for m in resolved if m.value == seeded["routing_number"]]

    assert len(matched) == 1, "overlapping spans were not de-duplicated"
    assert matched[0].entity == "ABA_ROUTING"


# ================================================================================
# 3. The DATE_TIME context filter — the test that protects the whole app
# ================================================================================


def test_transaction_date_survives_while_dob_is_redacted(seeded):
    """A serialized row carrying both kinds of date.

    If this test ever fails in the direction of redacting `txn_date`, every
    spending-history answer in the product breaks: the model is handed rows with no
    dates and, being 3B, invents them rather than refusing.
    """
    row = serialize(
        {
            "customer_id": seeded["customer_id"],
            "date_of_birth": seeded["date_of_birth"],
            "txn_date": "2026-07-14",
            "opened_date": "2019-03-02",
            "amount": -249.99,
        }
    )

    out = run_before_model(
        [
            HumanMessage(content="recent activity"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content=row, name="query_customer_data", tool_call_id="c1"),
        ]
    )
    content = str(out[-1].content)

    # The DOB is gone.
    assert seeded["date_of_birth"].isoformat() not in content
    assert "[REDACTED_DATE_TIME]" in content

    # The transaction and account-opening dates are untouched.
    assert "2026-07-14" in content
    assert "2019-03-02" in content
    # And the amount the answer depends on is still there.
    assert "-249.99" in content


@pytest.mark.parametrize(
    "identifier", ["TXN-000009", "ACCT-0001", "CUST-0012", "SR-0005", "LOAN-0003", "RM-001"]
)
def test_internal_record_locators_are_never_redacted(identifier):
    """Regression caught by printing the model's real context in tests/test_agent.py.

    Presidio scores `TXN-000009` as US_DRIVER_LICENSE at 0.01 (its all-digits fallback)
    and spaCy reads `SR-0005` as a PERSON at 0.85. Untreated, a transaction id reaches
    the model as `TXN-[REDACTED_US_DRIVER_LICENSE]` — so the model cannot cite the
    transaction, and cannot pass a valid `txn_id` to `file_dispute`. The HITL approval
    flow then breaks on data this rail corrupted.
    """
    from app.rails.pii import redact_text

    assert redact_text(identifier) == identifier
    assert resolve_entities(identifier) == []


def test_identifier_protection_does_not_weaken_pii_detection(seeded):
    """The guard is narrow: real PII beside an identifier is still destroyed."""
    from app.rails.pii import redact_text

    row = f"{{'txn_id': 'TXN-000009', 'ssn': '{seeded['ssn']}'}}"
    redacted = redact_text(row)

    assert "TXN-000009" in redacted
    assert seeded["ssn"] not in redacted
    assert "[REDACTED_US_SSN]" in redacted


# --------------------------------------------------------------------------------
# Defect 26: the pseudonymizer destroying amounts and channel enums
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "Decimal('-2004.42",  # the span Presidio actually returned, unbalanced quote and all
        "Decimal('2864.63')",
        "8)",  # tail of datetime.datetime(2025, 6, 3, 22, 28)
        "datetime.date(2016, 7, 10)",
        "ZELLE",
        "ACH",
        "CHECKING",
        "MONEY_MARKET",
    ],
)
def test_repr_fragments_and_enums_are_not_treated_as_people(value):
    """Observed in the rails-ON demo: `$Decimal('-2004.42` and a channel reading as a name.

    Both symptoms came from the rail scanning the flat `str()` of a tool envelope, where
    spaCy sees Python repr fragments and SQL enums as prose. Either one makes the guarded
    answer *worse* than the unguarded one, which inverts the whole demo.
    """
    assert [m for m in resolve_entities(value) if m.entity == "PERSON"] == []


def test_a_tool_envelope_survives_pseudonymization_structurally_intact(seeded):
    """The end-to-end shape of defect 26: names go, everything else stays byte-identical."""
    from app.agent.tools import get_account_summary
    from app.rails.pii import PersonPseudonymMiddleware, PseudonymMap

    envelope = get_account_summary("ACCT-0007", "RM-001")
    rendered = str(envelope)
    assert "Decimal(" in rendered and "ZELLE" in rendered, "fixture no longer exercises this"

    replaced = PersonPseudonymMiddleware()._pseudonymize(
        rendered, PseudonymMap(salt="test-salt")
    )

    assert replaced.count("Decimal(") == rendered.count("Decimal(")
    assert replaced.count("datetime.") == rendered.count("datetime.")
    assert replaced.count("ZELLE") == rendered.count("ZELLE")
    assert "<PERSON_" in replaced, "the rail stopped doing its actual job"


def test_every_seeded_customer_name_still_survives_the_shape_rules():
    """The cost of the two shape rules must be zero on the thing they could break.

    Measured when the rules were introduced: 0 of 40. Asserted here so a future tweak to
    `_is_schema_artifact` cannot quietly start dropping real customers.
    """
    from app.rails.pii import _is_schema_artifact

    with get_session() as session:
        names = [r[0] for r in session.execute(text("SELECT full_name FROM customers")).all()]

    assert len(names) == 40, "seed data has drifted; re-run python db/seed.py"

    for name in names:
        assert not _is_schema_artifact(name), name
        row = f"{{'customer_id': 'CUST-0001', 'full_name': '{name}', 'status': 'ACTIVE'}}"
        spans = [m.value for m in resolve_entities(row) if m.entity == "PERSON"]
        assert any(name in span or span in name for span in spans), (
            f"{name!r} is no longer detected inside a serialized row"
        )


def test_multi_word_all_caps_spans_are_still_treated_as_names():
    """Rule B tests for a single token, not for capitalisation.

    `MAUREEN W` is a Zelle counterparty inside a memo — a real third party. Rejecting
    every all-caps span would drop 35 of these; rejecting only single tokens drops 7.
    """
    from app.rails.pii import _is_schema_artifact

    assert not _is_schema_artifact("MAUREEN W")
    assert not _is_schema_artifact("STACEY H")
    assert _is_schema_artifact("ZELLE")


def test_weak_confidence_matches_are_dropped():
    """A 0.01-confidence guess must not destroy data; a real 0.30 licence still must."""
    from app.rails.pii import MIN_ENTITY_SCORE, redact_text

    assert MIN_ENTITY_SCORE > 0.01
    # A genuine driver's licence format scores 0.30 and survives the floor.
    assert redact_text("J4108603") == "[REDACTED_US_DRIVER_LICENSE]"


def test_bare_transaction_dates_are_never_redacted():
    """No DOB context anywhere: nothing date-like should be touched."""
    content = "Transactions on 2026-07-14, 2026-07-15 and 2026-08-01 totalling $412.00"

    out = run_before_model([HumanMessage(content=content)])

    assert str(out[0].content) == content


# ================================================================================
# 4. Reversible person pseudonymization (§3.1.4)
# ================================================================================


def person_middleware() -> PersonPseudonymMiddleware:
    return PersonPseudonymMiddleware()


def test_name_round_trips(seeded):
    """Pseudonymized on the way in, restored on the way out."""
    name = seeded["full_name"]
    mw = person_middleware()

    with pii_request("req-round-trip") as request_map:
        state = {"messages": [HumanMessage(content=f"What is {name}'s balance?")]}
        update = mw.before_agent(state, None)
        assert update is not None

        seen_by_model = str(update["messages"][0].content)
        assert name not in seen_by_model, "the model must never see a real name"
        token = PERSON_TOKEN_RE.search(seen_by_model)
        assert token is not None
        assert request_map.token_to_name[token.group(0)] == name

        # The model answers using the token it was given.
        out_state = {
            "messages": [AIMessage(content=f"{token.group(0)} has a balance of $1,204.10")]
        }
        restored = mw.after_model(out_state, None)
        assert restored is not None
        assert name in str(restored["messages"][0].content)
        assert PERSON_TOKEN_RE.search(str(restored["messages"][0].content)) is None


def test_same_name_gets_one_token_within_a_request(seeded):
    name = seeded["full_name"]
    mw = person_middleware()

    with pii_request("req-consistent"):
        state = {
            "messages": [
                HumanMessage(content=f"{name} called. Does {name} have a joint account?")
            ]
        }
        update = mw.before_agent(state, None)
        content = str(update["messages"][0].content)

    tokens = set(PERSON_TOKEN_RE.findall(content))
    assert len(tokens) == 1, f"one name produced {len(tokens)} tokens: {tokens}"
    assert content.count(next(iter(tokens))) == 2


def test_possessive_and_plain_forms_share_one_token(seeded):
    """Regression: spaCy returns `Michael Chen's` as the PERSON span, possessive included.

    §3.8's acceptance criterion is "What's Michael Chen's checking account balance?", so
    this is the common phrasing rather than an edge case. Left untreated, the map stores
    a name with a stray `'s` and the same customer gets two tokens in one request.
    """
    name = seeded["full_name"]
    mw = person_middleware()

    with pii_request("req-possessive") as request_map:
        state = {
            "messages": [
                HumanMessage(content=f"What is {name}'s balance? Is {name} overdrawn?")
            ]
        }
        update = mw.before_agent(state, None)
        content = str(update["messages"][0].content)

        assert set(request_map.token_to_name.values()) == {name}
        tokens = set(PERSON_TOKEN_RE.findall(content))
        assert len(tokens) == 1, f"possessive form produced a second token: {tokens}"
        # The possessive survives in the text, attached to the token.
        assert f"{next(iter(tokens))}'s balance" in content


def test_different_requests_issue_different_tokens(seeded):
    """The salt is per request, so tokens are not a stable cross-request pseudonym."""
    name = seeded["full_name"]
    mw = person_middleware()

    def token_for_request(request_id: str) -> str:
        with pii_request(request_id):
            update = mw.before_agent(
                {"messages": [HumanMessage(content=f"balance for {name}")]}, None
            )
            return PERSON_TOKEN_RE.search(str(update["messages"][0].content)).group(0)

    assert token_for_request("req-A") != token_for_request("req-B")


def test_token_from_one_request_is_not_resolvable_in_another(seeded):
    """The map is genuinely per-request — this is the cross-customer breach test."""
    name = seeded["full_name"]
    mw = person_middleware()

    with pii_request("req-A"):
        update = mw.before_agent(
            {"messages": [HumanMessage(content=f"balance for {name}")]}, None
        )
        token_a = PERSON_TOKEN_RE.search(str(update["messages"][0].content)).group(0)

    # Request A is closed: its map is gone, not merely out of scope.
    assert get_request_map("req-A") is None

    with pii_request("req-B"):
        state = {"messages": [AIMessage(content=f"{token_a} has a balance of $1.00")]}
        restored = mw.after_model(state, None)

    content = str(restored["messages"][0].content)
    assert name not in content, "a token leaked a name across requests"
    assert UNKNOWN_PERSON_PLACEHOLDER in content


def test_unissued_token_is_stripped_and_flagged():
    """Catches hallucinated identifiers (§3.1.4 step 5)."""
    mw = person_middleware()

    with pii_request("req-hallucination") as request_map:
        state = {
            "messages": [AIMessage(content="Per <PERSON_dead>, the balance is $2,000.")]
        }
        update = mw.after_model(state, None)

    content = str(update["messages"][0].content)
    assert "<PERSON_dead>" not in content
    assert UNKNOWN_PERSON_PLACEHOLDER in content

    flags = update["pii_flags"]
    assert len(flags) == 1
    assert flags[0]["reason"] == FLAG_UNISSUED_TOKEN
    assert flags[0]["token"] == "<PERSON_dead>"
    assert request_map.flags == flags


def test_names_in_tool_results_are_pseudonymized(seeded):
    """The SQL tool returns raw customer rows; the model must not see the names."""
    mw = person_middleware()

    with pii_request("req-tool") as request_map:
        state = {
            "messages": [
                HumanMessage(content="list my customers"),
                AIMessage(content="", tool_calls=[]),
                ToolMessage(
                    content=serialize(seeded),
                    name="query_customer_data",
                    tool_call_id="c1",
                ),
            ]
        }
        update = mw.before_model(state, None)

        content = str(update["messages"][-1].content)
        assert seeded["full_name"] not in content
        assert PERSON_TOKEN_RE.search(content) is not None
        assert seeded["full_name"] in request_map.token_to_name.values()


def test_column_names_are_not_pseudonymized(seeded):
    """Regression: spaCy tags `date_of_birth` as a PERSON at 0.85.

    Tokenizing a column name hands the model a row whose keys it cannot read.
    """
    mw = person_middleware()

    with pii_request("req-columns"):
        row = serialize(
            {"date_of_birth": seeded["date_of_birth"], "full_name": seeded["full_name"]}
        )
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[]),
                ToolMessage(content=row, name="query_customer_data", tool_call_id="c1"),
            ]
        }
        update = mw.before_model(state, None)

    content = str(update["messages"][-1].content)
    assert "date_of_birth" in content, "a schema column name was replaced with a token"
    assert "full_name" in content


def test_map_is_destroyed_when_the_request_ends(seeded):
    with pii_request("req-cleanup") as request_map:
        request_map.token_for(seeded["full_name"])
        assert get_request_map("req-cleanup") is not None

    assert get_request_map("req-cleanup") is None


def test_map_is_destroyed_even_when_the_request_raises(seeded):
    """Cleanup that only runs on the success path is not cleanup."""
    with pytest.raises(RuntimeError):
        with pii_request("req-boom") as request_map:
            request_map.token_for(seeded["full_name"])
            raise RuntimeError("agent exploded")

    assert get_request_map("req-boom") is None


def test_no_map_is_created_outside_a_request(seeded):
    """Fail closed: with no request open, nothing global is minted."""
    mw = person_middleware()
    state = {"messages": [HumanMessage(content=f"balance for {seeded['full_name']}")]}

    assert mw.before_agent(state, None) is None
    assert get_request_map("anything") is None


# ================================================================================
# 5. Assembly, toggles and latency wiring
# ================================================================================


def test_every_rule_covers_all_three_application_points():
    middleware = build_pii_middleware(RailsConfig())
    timed = [m for m in middleware if hasattr(m, "rail_name")]

    # 4 rules x 3 points, plus the pseudonymizer.
    assert len(timed) == 12
    assert len(middleware) == 13
    for rail in (RAIL_PII_INPUT, RAIL_PII_TOOL_RESULT, RAIL_PII_OUTPUT):
        assert sum(1 for m in timed if m.rail_name == rail) == 4

    # Exactly one application point per instance, so its timer name is unambiguous.
    for m in timed:
        assert sum([m.apply_to_input, m.apply_to_output, m.apply_to_tool_results]) == 1


def test_rail_toggles_are_live_rather_than_build_time():
    """Every instance is always built; the flags are consulted at hook time.

    Filtering the list at build time would bake the flags into the compiled graph, so a
    Streamlit checkbox would do nothing until the agent was rebuilt — and turning a rail
    off mid-conversation and re-asking the same question is the demo this project exists
    to deliver (§3.4).
    """
    rails = RailsConfig()
    middleware = build_pii_middleware(rails)

    assert len(middleware) == 13  # 4 rules x 3 application points, plus the pseudonymizer
    assert all(m.enabled for m in middleware)

    rails.pii_builtin = False
    builtin = [m for m in middleware if getattr(m, "flag", None) == "pii_builtin"]
    presidio = [m for m in middleware if getattr(m, "flag", None) == "pii_presidio"]

    # Same objects, no rebuild.
    assert len(middleware) == 13
    assert not any(m.enabled for m in builtin)
    assert all(m.enabled for m in presidio)

    rails.pii_presidio = False
    rails.person_pseudonym = False
    assert not any(m.enabled for m in middleware)


def test_a_disabled_rail_stops_redacting(seeded):
    """The toggle has an observable effect, not just a flag change."""
    rails = RailsConfig()
    raw = f"Customer SSN is {seeded['ssn']}"

    assert seeded["ssn"] not in str(run_before_model([HumanMessage(content=raw)], rails)[0].content)

    rails.pii_presidio = False
    assert seeded["ssn"] in str(run_before_model([HumanMessage(content=raw)], rails)[0].content)


def test_latency_is_recorded_under_the_section_4_3_names(seeded):
    timing.clear_samples()
    run_before_model([HumanMessage(content=f"SSN {seeded['ssn']}")])
    run_after_model([AIMessage(content=f"SSN {seeded['ssn']}")])

    recorded = {s["rail"] for s in timing.get_samples()}
    assert RAIL_PII_INPUT in recorded
    assert RAIL_PII_OUTPUT in recorded
    assert all(s["duration_ms"] >= 0 for s in timing.get_samples())


def test_builtin_email_and_credit_card_still_work():
    """The two types LangChain genuinely covers, reached through the same stack."""
    out = run_before_model(
        [HumanMessage(content="mail rm@example.com card 4111111111111111")]
    )
    content = str(out[0].content)

    assert "rm@example.com" not in content
    assert "[REDACTED_EMAIL]" in content
    # Luhn-valid, so it is masked to the last four.
    assert "4111111111111111" not in content
    assert "1111" in content
