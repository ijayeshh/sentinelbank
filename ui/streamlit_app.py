"""Streamlit UI (PROJECT_PLAN.md §3.7).

Seven required elements: RM selector (RM-001/RM-002), chat, sidebar guardrail toggles
(the core demo control), per-message rail status panel with verdicts/scores/latency,
approval panel rendering the structured diff, red security banner on quarantined
injection, and an audit tab with a chain-verify button.

================================================================================
THIS PROCESS HOLDS NO MODELS
================================================================================
Everything here goes over HTTP to the FastAPI app (§3.6). Nothing in this file imports
`app.agent`, `app.rails` or `app.embeddings`, and that restraint is a hardware
requirement rather than an architectural preference: Streamlit runs in its own Python
process, so a single `from app.agent.graph import build_agent` here would load a **second**
Presidio + spaCy (~560 MB), a second Prompt-Guard (~350 MB) and a second MiniLM (~90 MB)
alongside the copies uvicorn already holds — about a gigabyte of duplicate weights on an
8 GB machine that is also running Postgres in Docker and holding 2 GB of VRAM for Ollama.
The import is the whole failure; it does not announce itself, it just makes the demo swap.

The only shared vocabulary is `RAIL_STACK` below, which mirrors §3.4's eight positions.

================================================================================
WHY THE RAIL PANEL IS BUILT FROM THE STACK, NOT FROM THE TIMINGS
================================================================================
`/chat` reports latency samples keyed by *application point* (`pii_input`,
`pii_tool_result`, ...), which is what §4.3's budget table needs but is not what a viewer
needs. The claim this project makes is "eight separate checks ran on that answer", so the
panel is rendered from §3.4's eight positions and the timings are attributed onto them.

Two of those attributions are many-to-one and the UI says so rather than hiding it:
positions 2, 3 and 4 are all PII middleware sharing the same three `@rail_timer` names, so
their latency is reported once as a shared figure. Splitting it would mean inventing a
number, and a security panel that invents numbers is worse than one that admits a limit.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("SENTINELBANK_API_URL", "http://localhost:8000")

# A 3B model on a GTX 1650 answers a tool-calling question in roughly 10-25 s, and the
# rails add ~1 s of CPU work on top. The read timeout has to clear that comfortably or
# the UI reports a backend failure for a request that was merely thinking.
TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=10.0, pool=5.0)

RM_IDS = ("RM-001", "RM-002")


# ================================================================================
# §3.4's eight positions, as the UI presents them
# ================================================================================
#
# `rails` lists the @rail_timer names whose samples belong to this position. `protects`
# is the one line a viewer reads to understand what the toggle above it turns off — the
# toggles are the core demo control, and a checkbox labelled `pii_presidio` communicates
# nothing to somebody watching over your shoulder.

RAIL_STACK: tuple[dict[str, Any], ...] = (
    {
        "n": 1,
        "flag": "injection_input",
        "label": "Injection scan — your message",
        "protects": "Direct jailbreaks typed into the chat box",
        "rails": ("injection_input",),
        "shared": False,
    },
    {
        "n": 2,
        "flag": "pii_builtin",
        "label": "PII — built-in rules",
        "protects": "Email addresses (redacted) and card numbers (masked to last 4)",
        "rails": ("pii_input", "pii_tool_result", "pii_output"),
        "shared": True,
    },
    {
        "n": 3,
        "flag": "pii_presidio",
        "label": "PII — Presidio + ABA recognizer",
        "protects": "SSNs, account and routing numbers, phones, addresses, dates of birth",
        "rails": ("pii_input", "pii_tool_result", "pii_output"),
        "shared": True,
    },
    {
        "n": 4,
        "flag": "person_pseudonym",
        "label": "Person pseudonymisation",
        "protects": "Customer names — reversible tokens; the model never sees a real one",
        "rails": ("pii_input", "pii_tool_result", "pii_output"),
        "shared": True,
    },
    {
        "n": 5,
        "flag": "hitl",
        "label": "Human approval gate",
        "protects": "Disputes, waivers, card blocks, wires — drafted, never auto-executed",
        "rails": ("hitl_interrupt",),
        "shared": False,
    },
    {
        "n": 6,
        "flag": "injection_tool_result",
        "label": "Injection scan — retrieved data",
        "protects": "Poisoned transaction memos and service-request notes",
        "rails": ("injection_tool_result",),
        "shared": False,
    },
    {
        "n": 7,
        "flag": "spotlight",
        "label": "Spotlighting",
        "protects": "Marks bank data as data, not instructions, and strips fake delimiters",
        "rails": ("spotlight",),
        "shared": False,
    },
    {
        "n": 8,
        "flag": "audit",
        "label": "Hash-chained audit log",
        "protects": "Tamper-evident SHA-256 record of every request",
        "rails": ("audit_write",),
        "shared": False,
    },
)

# Verdicts the backend can report, mapped to how alarming they should look. Anything not
# listed falls through to the neutral style — an unknown verdict must never be silently
# rendered as if it were a pass.
VERDICT_TONE = {
    "BLOCKED": "alert",
    "QUARANTINED": "alert",
    "AWAITING_APPROVAL": "warn",
    "RAN": "ok",
}


# ================================================================================
# HTTP
# ================================================================================


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE, timeout=TIMEOUT)


def api_get(path: str, **params: Any) -> tuple[bool, Any]:
    """GET `path`. Returns (ok, payload-or-error-string)."""
    try:
        with _client() as client:
            response = client.get(path, params=params or None)
            response.raise_for_status()
            return True, response.json()
    except httpx.HTTPStatusError as exc:
        return False, _detail(exc)
    except httpx.HTTPError as exc:
        return False, f"cannot reach the API at {API_BASE}: {exc}"


def api_post(path: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    """POST `payload` to `path`. Returns (ok, payload-or-error-string)."""
    try:
        with _client() as client:
            response = client.post(path, json=payload)
            response.raise_for_status()
            return True, response.json()
    except httpx.HTTPStatusError as exc:
        return False, _detail(exc)
    except httpx.HTTPError as exc:
        return False, f"cannot reach the API at {API_BASE}: {exc}"


def _detail(exc: httpx.HTTPStatusError) -> str:
    """FastAPI puts the useful part in `detail`; the status line rarely says enough."""
    try:
        body = exc.response.json()
    except ValueError:
        return f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
    return f"HTTP {exc.response.status_code}: {body.get('detail', body)}"


@st.cache_data(ttl=10, show_spinner=False)
def fetch_health() -> tuple[bool, Any]:
    """Backend health, cached briefly so every rerun does not re-probe Ollama."""
    return api_get("/health")


# ================================================================================
# Small HTML helpers
# ================================================================================
#
# Inline styles only. This machine is air-gapped by design (§0.1), so a stylesheet or font
# from a CDN would simply fail to load; and rgba backgrounds over the theme's own
# background keep the panel legible in both Streamlit's light and dark themes.

TONES = {
    "ok": ("rgba(22,163,74,0.14)", "#15803d"),
    "warn": ("rgba(217,119,6,0.16)", "#b45309"),
    "alert": ("rgba(220,38,38,0.16)", "#b91c1c"),
    "off": ("rgba(120,113,108,0.18)", "#78716c"),
    "idle": ("rgba(120,113,108,0.10)", "#a8a29e"),
}


def chip(text: str, tone: str = "idle") -> str:
    background, colour = TONES.get(tone, TONES["idle"])
    return (
        f"<span style='background:{background};color:{colour};border:1px solid {colour}44;"
        "border-radius:999px;padding:2px 10px;font-size:0.74rem;font-weight:700;"
        f"letter-spacing:0.03em;white-space:nowrap'>{text}</span>"
    )


def esc(value: Any) -> str:
    """Escape user- and data-controlled text before it goes into an HTML block.

    This matters more here than in an ordinary app: quarantined previews and audit
    entries are literally attacker-authored strings from `transactions.description`. A
    security panel that rendered an injected `<img onerror=...>` while reporting that it
    had contained the attack would be its own counter-example.
    """
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ================================================================================
# Session state
# ================================================================================


def new_thread() -> str:
    return uuid.uuid4().hex[:12]


def init_state() -> None:
    st.session_state.setdefault("rm_id", RM_IDS[0])
    # Per-RM, because switching RM is an entitlement demo, not a topic change: RM-002 must
    # not inherit RM-001's conversation, and its own history should still be there when
    # you switch back.
    st.session_state.setdefault("threads", {rm_id: new_thread() for rm_id in RM_IDS})
    st.session_state.setdefault("history", {rm_id: [] for rm_id in RM_IDS})
    st.session_state.setdefault("pending", {rm_id: None for rm_id in RM_IDS})
    st.session_state.setdefault("rails_loaded", False)


def rm() -> str:
    return st.session_state["rm_id"]


def history() -> list[dict[str, Any]]:
    return st.session_state["history"][rm()]


def pending() -> dict[str, Any] | None:
    return st.session_state["pending"][rm()]


# ================================================================================
# Sidebar — the core demo control
# ================================================================================


def push_rail(flag: str) -> None:
    """POST one toggle to the live backend.

    The API mutates the single `RailsConfig` object every compiled agent holds, so this
    lands on the next message with no rebuild and no restart. That is the whole demo:
    ask a question, flip a rail, ask it again.

    On failure the widget is put back, because a toggle that shows "off" while the rail
    is still running is the one failure mode this panel must never have.
    """
    ok, result = api_post("/config/rails", {flag: bool(st.session_state[f"rail_{flag}"])})
    if not ok:
        st.session_state[f"rail_{flag}"] = not st.session_state[f"rail_{flag}"]
        st.session_state["rail_error"] = result


def set_all_rails(value: bool) -> None:
    ok, result = api_post("/config/rails", {item["flag"]: value for item in RAIL_STACK})
    if ok:
        for flag, current in result.items():
            st.session_state[f"rail_{flag}"] = current
    else:
        st.session_state["rail_error"] = result


def load_rails_once() -> bool:
    """Seed the toggle widgets from the backend's live configuration.

    Once only. After this the widgets own their state and push changes outward — re-reading
    every rerun would let a stale response overwrite a toggle the user just flipped.
    """
    if st.session_state["rails_loaded"]:
        return True
    ok, config = api_get("/config/rails")
    if not ok:
        return False
    for flag, value in config.items():
        st.session_state.setdefault(f"rail_{flag}", bool(value))
    st.session_state["rails_loaded"] = True
    return True


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Relationship Manager")
        st.radio(
            "Acting as",
            RM_IDS,
            key="rm_id",
            horizontal=True,
            label_visibility="collapsed",
        )
        st.caption(
            "Entitlement is enforced in SQL, not in the prompt. Switch RM and ask the "
            "same question — the other portfolio is not merely hidden, it is unreachable."
        )

        st.divider()

        rails_ok = load_rails_once()
        st.markdown("## 🛡️ Guardrails")
        st.caption("Every rail is individually switchable, live. No restart, no rebuild.")

        if not rails_ok:
            st.error("Backend unreachable — rail toggles disabled.")
            render_health()
            return

        left, right = st.columns(2)
        left.button(
            "All on", use_container_width=True, on_click=set_all_rails, args=(True,)
        )
        right.button(
            "All off", use_container_width=True, on_click=set_all_rails, args=(False,)
        )

        off_count = sum(
            1 for item in RAIL_STACK if not st.session_state.get(f"rail_{item['flag']}")
        )
        if off_count:
            background, colour = TONES["alert"]
            st.markdown(
                f"<div style='background:{background};border-left:4px solid {colour};"
                f"padding:8px 12px;border-radius:4px;margin:8px 0;color:{colour};"
                f"font-weight:700'>{off_count} of 8 rails disabled — this configuration "
                "is not safe for real customer data.</div>",
                unsafe_allow_html=True,
            )

        for item in RAIL_STACK:
            st.toggle(
                f"**{item['n']}. {item['label']}**",
                key=f"rail_{item['flag']}",
                on_change=push_rail,
                args=(item["flag"],),
            )
            st.caption(item["protects"])

        if st.session_state.get("rail_error"):
            st.error(st.session_state.pop("rail_error"))

        st.divider()
        st.markdown("### Session")
        if st.button("New conversation", use_container_width=True):
            st.session_state["threads"][rm()] = new_thread()
            st.session_state["history"][rm()] = []
            st.session_state["pending"][rm()] = None
            st.rerun()
        st.caption(f"thread `{st.session_state['threads'][rm()]}`")

        render_health()


def render_health() -> None:
    """Compact backend status. Chiefly here to show the models loaded once, at startup."""
    st.divider()
    st.markdown("### Backend")
    ok, health = fetch_health()
    if not ok:
        st.error(f"{health}\n\nStart it with:\n\n`uvicorn app.main:app --reload`")
        return

    ollama = health["ollama"]
    serves = bool(ollama.get("serves_model"))
    db_ok = bool(health["postgres"].get("reachable"))
    st.markdown(
        chip(health["status"].upper(), "ok" if health["status"] == "ok" else "warn")
        + " "
        + chip("OLLAMA" if serves else "NO MODEL", "ok" if serves else "alert")
        + " "
        + chip("POSTGRES" if db_ok else "NO DB", "ok" if db_ok else "alert"),
        unsafe_allow_html=True,
    )

    with st.expander("Models resident", expanded=False):
        for name, model in health["models"].items():
            detail = model.get("model_id") or model.get("detail") or ""
            st.markdown(
                f"{'✅' if model['loaded'] else '❌'} **{name}** — `{esc(detail)}`"
            )
            if model.get("error"):
                st.caption(f"⚠️ {model['error']}")
        if health["models"].get("injection_classifier", {}).get("is_fallback"):
            st.warning(
                "Prompt-Guard did not load; the fallback classifier is running. Its "
                "false-positive rate is measurably higher — see the README."
            )
        st.caption("Loaded once in the FastAPI lifespan handler, never per request.")


# ================================================================================
# 6. Security banner
# ================================================================================


def render_banners(warnings: list[dict[str, Any]]) -> None:
    """The red banner §3.7 requires, naming the row and the field.

    An RM has to be able to answer "what did you hide, and where did it come from?" — a
    generic "content was filtered" would make the rail indistinguishable from a bug, and
    the whole argument of this project is that the RM still gets a usable answer while
    knowing exactly what was withheld.
    """
    for warning in warnings:
        kind = warning["kind"]

        if kind == "quarantine":
            title = "INJECTION QUARANTINED"
            body = (
                f"Injected content was found in <b>{esc(warning.get('row_id', '?'))}</b> · "
                f"field <code>{esc(warning.get('column'))}</code> and was replaced before "
                "the model saw it. Every other field in that record is intact, and your "
                "answer is based on them."
            )
        elif kind == "input_injection":
            title = "MESSAGE BLOCKED"
            body = (
                "That message was classified as a prompt-injection attempt and was not "
                "sent to the model."
            )
        else:
            title = "HARD BLOCK — SANCTIONS"
            body = esc(warning.get("message"))

        meta = []
        if warning.get("label"):
            meta.append(f"label <b>{esc(warning['label'])}</b>")
        if warning.get("score") is not None:
            meta.append(f"score <b>{float(warning['score']):.4f}</b>")
        if warning.get("layers"):
            meta.append("caught by <b>" + esc(", ".join(warning["layers"])) + "</b>")
        if warning.get("patterns"):
            meta.append("patterns <b>" + esc(", ".join(warning["patterns"])) + "</b>")

        preview = ""
        if warning.get("preview"):
            preview = (
                "<div style='margin-top:8px;font-family:ui-monospace,Consolas,monospace;"
                "font-size:0.76rem;opacity:0.85;word-break:break-word'>"
                f"payload: {esc(warning['preview'])}…</div>"
            )

        background, colour = TONES["alert"]
        st.markdown(
            f"<div style='background:{background};border:1px solid {colour};"
            f"border-left:6px solid {colour};border-radius:6px;padding:12px 16px;"
            "margin:6px 0 12px 0'>"
            f"<div style='color:{colour};font-weight:800;letter-spacing:0.04em;"
            f"font-size:0.82rem'>⚠ {title}</div>"
            f"<div style='margin-top:6px'>{body}</div>"
            f"<div style='margin-top:6px;font-size:0.78rem;opacity:0.9'>"
            + " · ".join(meta)
            + "</div>"
            + preview
            + "</div>",
            unsafe_allow_html=True,
        )


# ================================================================================
# 4. Rail status panel
# ================================================================================


def rail_row(item: dict[str, Any], reports: dict[str, dict[str, Any]]) -> str:
    """One row of the panel: position, name, status, score, latency."""
    enabled = bool(st.session_state.get(f"rail_{item['flag']}", True))
    mine = [reports[name] for name in item["rails"] if name in reports]

    if not enabled:
        status, tone = "OFF", "off"
    elif not mine:
        # Correct and common: the approval gate does not run when no write tool was
        # called. Reporting that as a pass would claim work that never happened.
        status, tone = "NOT TRIGGERED", "idle"
    else:
        flagged = next(
            (report["verdict"] for report in mine if report["verdict"] != "RAN"), None
        )
        status = flagged or "PASS"
        tone = VERDICT_TONE.get(flagged or "RAN", "ok")

    scores = [report["score"] for report in mine if report.get("score") is not None]
    score_text = f"{max(scores):.4f}" if scores else "—"

    if mine:
        latency = sum(report["latency_ms"] for report in mine)
        latency_text = f"{latency:,.1f} ms" + (
            " <i style='opacity:0.6'>shared</i>" if item["shared"] else ""
        )
    else:
        latency_text = "—"

    dim = "" if enabled else "opacity:0.55;"
    return (
        f"<tr style='{dim}border-top:1px solid rgba(120,113,108,0.22)'>"
        f"<td style='padding:6px 8px;opacity:0.55;width:1.6rem'>{item['n']}</td>"
        f"<td style='padding:6px 8px'><b>{esc(item['label'])}</b><br>"
        f"<span style='font-size:0.76rem;opacity:0.7'>{esc(item['protects'])}</span></td>"
        f"<td style='padding:6px 8px;text-align:right'>{chip(status, tone)}</td>"
        f"<td style='padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums'>"
        f"{score_text}</td>"
        f"<td style='padding:6px 8px;text-align:right;font-variant-numeric:tabular-nums;"
        f"white-space:nowrap'>{latency_text}</td>"
        "</tr>"
    )


def render_rail_panel(payload: dict[str, Any]) -> None:
    """§3.7 element 4, rendered from §3.4's eight positions.

    Built from the stack rather than from the timing samples so the panel always has
    eight rows: "eight separate checks ran" is only legible if the ones that found
    nothing are visible too.
    """
    reports = {report["rail"]: report for report in payload.get("rails", [])}
    active = sum(
        1 for item in RAIL_STACK if st.session_state.get(f"rail_{item['flag']}", True)
    )
    total_ms = sum(report["latency_ms"] for report in payload.get("rails", []))

    st.markdown(
        "<div style='display:flex;justify-content:space-between;align-items:baseline;"
        "margin-bottom:4px'>"
        f"<div style='font-weight:800;letter-spacing:0.04em'>RAILS · {active}/8 ACTIVE"
        "</div>"
        f"<div style='font-size:0.78rem;opacity:0.7'>guardrail time {total_ms:,.0f} ms "
        f"of {payload.get('elapsed_ms', 0):,.0f} ms total</div></div>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.86rem'>"
        "<tr style='font-size:0.7rem;opacity:0.6;letter-spacing:0.06em'>"
        "<th></th><th style='text-align:left;padding:0 8px'>RAIL</th>"
        "<th style='text-align:right;padding:0 8px'>VERDICT</th>"
        "<th style='text-align:right;padding:0 8px'>SCORE</th>"
        "<th style='text-align:right;padding:0 8px'>LATENCY</th></tr>"
        + "".join(rail_row(item, reports) for item in RAIL_STACK)
        + "</table>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Positions 2-4 are all PII middleware and share the same three latency counters, "
        "so their figure is the shared total rather than a per-position split. "
        f"Audit entry #{payload.get('audit_event_id') or '—'} · "
        f"decision {payload.get('decision', '—')}"
    )

    with st.expander("Raw latency samples", expanded=False):
        for report in payload.get("rails", []):
            score = report.get("score")
            st.markdown(
                f"`{report['rail']}` — {report['verdict']} · "
                f"{report['latency_ms']:,.1f} ms over {report['calls']} call(s)"
                + (f" · score {score:.4f}" if score is not None else "")
            )


# ================================================================================
# 5. Approval panel
# ================================================================================


def submit_decision(decision: str) -> None:
    payload = pending()
    if payload is None:
        return
    ok, result = api_post(
        "/approve", {"thread_id": payload["thread_id"], "decision": decision}
    )
    if not ok:
        st.session_state["approve_error"] = result
        return
    st.session_state["pending"][rm()] = None
    history().append({"role": "assistant", "payload": result})


def render_approval_panel() -> None:
    """§3.7 element 5: the structured diff plus Approve / Reject.

    The diff is rendered exactly as `app/rails/hitl.py` built it — customer, account,
    transaction, the before -> after change, and the Reg E citation retrieved live from
    the policy index. It is shown verbatim rather than reformatted here, because the RM
    must be approving the description produced by the code that will perform the change;
    a prettier parallel rendering could disagree with it and nobody would notice.
    """
    payload = pending()
    if payload is None:
        return

    background, colour = TONES["warn"]
    st.markdown(
        f"<div style='background:{background};border:1px solid {colour};border-radius:6px;"
        f"padding:10px 16px;margin-top:8px'><span style='color:{colour};font-weight:800;"
        "letter-spacing:0.04em'>⏸ AWAITING YOUR APPROVAL</span>"
        "<div style='margin-top:4px;font-size:0.86rem'>This action changes customer data. "
        "The agent has drafted it and stopped — nothing has been written.</div></div>",
        unsafe_allow_html=True,
    )

    for action in payload.get("pending", []):
        with st.container(border=True):
            st.markdown(f"**{action['tool']}**")
            st.code(action["diff"], language="text")

    approve, reject, _ = st.columns([1, 1, 3])
    approve.button(
        "✅ Approve",
        type="primary",
        use_container_width=True,
        on_click=submit_decision,
        args=("approve",),
    )
    reject.button(
        "✖ Reject",
        use_container_width=True,
        on_click=submit_decision,
        args=("reject",),
    )

    if st.session_state.get("approve_error"):
        st.error(st.session_state.pop("approve_error"))

    st.caption(
        "An OFAC-flagged customer is refused here even if you approve — sanctions "
        "screening never reads the decision."
    )


# ================================================================================
# 2. Chat
# ================================================================================


def send(message: str) -> None:
    history().append({"role": "user", "text": message})

    ok, result = api_post(
        "/chat",
        {
            "rm_id": rm(),
            "thread_id": st.session_state["threads"][rm()],
            "message": message,
        },
    )
    if not ok:
        history().append({"role": "error", "text": result})
        return

    if result.get("interrupted"):
        st.session_state["pending"][rm()] = result
    history().append({"role": "assistant", "payload": result})


def render_turn(turn: dict[str, Any], *, latest: bool) -> None:
    if turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(turn["text"])
        return

    if turn["role"] == "error":
        with st.chat_message("assistant"):
            st.error(turn["text"])
        return

    payload = turn["payload"]
    with st.chat_message("assistant"):
        render_banners(payload.get("warnings", []))

        answer = payload.get("answer") or ""
        if answer:
            st.markdown(answer)
        elif payload.get("interrupted"):
            st.markdown("_Drafted an action and paused for your approval._")

        # The latest turn shows the panel open, because that is what a viewer is looking
        # at; older turns collapse it so a long conversation stays readable.
        if latest:
            render_rail_panel(payload)
        else:
            with st.expander(
                f"Rails · {payload.get('decision', '—')} · "
                f"{payload.get('elapsed_ms', 0):,.0f} ms",
                expanded=False,
            ):
                render_rail_panel(payload)


def render_chat() -> None:
    st.caption(
        f"Portfolio: **{rm()}** — every query is scoped to this RM's customers in SQL, "
        "before the model is involved."
    )

    turns = history()
    last_assistant = max(
        (index for index, turn in enumerate(turns) if turn["role"] == "assistant"),
        default=-1,
    )
    for index, turn in enumerate(turns):
        render_turn(turn, latest=index == last_assistant)

    if pending() is not None:
        render_approval_panel()

    prompt = st.chat_input(
        "Ask about your portfolio…"
        if pending() is None
        else "Approve or reject the pending action first",
        disabled=pending() is not None,
    )
    if prompt:
        with st.spinner("Running the guardrail pipeline…"):
            send(prompt)
        st.rerun()


# ================================================================================
# 7. Audit tab
# ================================================================================


def render_audit() -> None:
    left, _ = st.columns([1, 2])
    with left:
        if st.button("🔗 Verify chain integrity", type="primary", use_container_width=True):
            st.session_state["chain_result"] = api_get("/audit/verify")

    verdict = st.session_state.get("chain_result")
    if verdict is not None:
        ok, result = verdict
        if not ok:
            st.error(result)
        elif result["intact"]:
            st.success(
                f"Chain intact — {result['entries']} entries verified from the genesis "
                "hash. Every entry commits to the one before it."
            )
        else:
            st.error(
                f"**CHAIN BROKEN** at index {result['broken_at_index']} "
                f"(event_id {result['event_id']}) — {result['reason']}\n\n"
                f"expected `{result['expected']}`\n\nfound `{result['found']}`"
            )

    st.divider()

    limit = st.selectbox("Entries per page", (25, 50, 100), index=0)
    ok, page = api_get("/audit", limit=limit, offset=0, newest_first=True)
    if not ok:
        st.error(page)
        return

    st.caption(
        f"{page['total']} entries total · showing the {len(page['entries'])} most recent. "
        "Query and response text is PII-redacted before it is hashed and stored, so what "
        "you see here is exactly what the chain commits to."
    )

    for entry in page["entries"]:
        tone = {"BLOCKED": "alert", "PENDING_APPROVAL": "warn", "REJECTED": "warn"}.get(
            entry["decision"], "ok"
        )
        with st.expander(
            f"#{entry['event_id']} · {entry['timestamp']} · {entry['rm_id']} · "
            f"{entry['decision']}",
            expanded=False,
        ):
            st.markdown(chip(entry["decision"], tone), unsafe_allow_html=True)
            st.markdown(f"**Query** — {esc(entry.get('user_query') or '—')}")
            st.markdown(f"**Response** — {esc(entry.get('response_summary') or '—')}")
            if entry.get("tool_calls"):
                st.markdown("**Tools**")
                st.json(entry["tool_calls"])
            if entry.get("rails_fired"):
                st.markdown("**Rails**")
                st.json(entry["rails_fired"])
            st.markdown(
                "<div style='font-family:ui-monospace,Consolas,monospace;"
                "font-size:0.72rem;opacity:0.75;word-break:break-all'>"
                f"prev {esc(entry['prev_hash'])}<br>hash {esc(entry['entry_hash'])}</div>",
                unsafe_allow_html=True,
            )


# ================================================================================
# Main
# ================================================================================


def main() -> None:
    st.set_page_config(page_title="SentinelBank", page_icon="🛡️", layout="wide")
    init_state()

    st.markdown(
        "<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:4px'>"
        "<h1 style='margin:0'>SentinelBank</h1>"
        "<span style='opacity:0.65'>on-premise RM copilot — no data leaves this machine"
        "</span></div>",
        unsafe_allow_html=True,
    )

    render_sidebar()

    copilot, audit = st.tabs(["💬 Copilot", "📒 Audit log"])
    with copilot:
        render_chat()
    with audit:
        render_audit()


main()
