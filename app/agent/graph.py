"""LangGraph agent assembly and the ordered middleware stack (PROJECT_PLAN.md §3.4).

Order: injection(input) -> PII builtin -> PII Presidio -> person pseudonym -> HITL
-> tool-result injection scan -> spotlight -> audit. Every middleware must be
individually toggleable via RailsConfig — the core demo is turning rails off and on.

================================================================================
WHAT "EXECUTION ORDER" ACTUALLY MEANS HERE
================================================================================
§3.4 numbers the stack 1..8. That is the **declaration** order, and LangChain turns it
into two different runtime orders, because middleware is an onion rather than a queue:

    before_agent / before_model   run in declaration order   1, 2, 3, ... 8
    after_model  / after_agent    run in REVERSE order       8, 7, 6, ... 1

(`langchain/agents/factory.py` wires the after-model chain starting from
`middleware_w_after_model[-1]`.) That is the correct behaviour and it is why the plan's
ordering is what it is:

  * On the way IN, the injection scan is outermost, so a jailbreak in the RM's message
    ends the run before any PII machinery or tool has done work on it.
  * On the way OUT, the same middleware is innermost, so PII redaction and name
    restoration have already happened by the time anything downstream sees the answer.

One consequence is load-bearing enough to state on its own: **`AuditMiddleware` uses
`after_agent`, not `after_model`.** Declared eighth, its `after_model` would run *first*
under the reversal — before the PII rails at positions 2 and 3 had redacted the model's
output — and the audit log would capture exactly the raw customer data §1.5 forbids it to
store. `after_agent` runs strictly after every `after_model`, so it records the final,
redacted, name-restored response.

`MIDDLEWARE_ORDER` below is the declared contract, and `tests/test_agent.py` asserts both
that the assembled stack matches it and that the observed runtime order is 1->8 forward
and 8->1 backward. The plan says to assert the order rather than assume it.

================================================================================
THE MODEL NEVER SEES rm_id
================================================================================
Every tool is built by a factory that closes over `rm_id`, so it appears in no tool
schema. The model cannot name the parameter, so no amount of prompt injection can make
the agent act as another RM. The value reaches `sql_pipeline.safe_execute` from session
state, which rewrites the SQL to carry the entitlement predicate (§2.1). The system
prompt mentions the RM only as context; nothing reads an identity back out of it
(§6 rule 5).

================================================================================
TOOL RESULTS TRAVEL AS ARTIFACTS
================================================================================
The read tools are registered with `response_format="content_and_artifact"`. That is not
cosmetic: a `ToolMessage`'s `.content` is a flat string, and the `source: TRUSTED |
UNTRUSTED` tags that say which fields are attacker-controlled only exist on the structured
envelope. Carrying it on `.artifact` is what lets `ToolResultInjectionMiddleware`
quarantine one poisoned memo while leaving the amount, date and merchant of that same row
intact (§3.2.2), and what lets `SpotlightMiddleware` wrap only the untrusted fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from typing_extensions import NotRequired

from app.agent.prompts import build_system_prompt
from app.agent.tools import (
    DEFAULT_TOP_K,
    get_account_summary,
    query_customer_data,
    search_policy_docs,
)
from app.audit.chain import AuditMiddleware
from app.config import RailsConfig, settings
from app.rails.hitl import HardBlockMiddleware, build_hitl_middleware, build_write_tools
from app.rails.injection import InjectionMiddleware, ToolResultInjectionMiddleware
from app.rails.pii import PersonPseudonymMiddleware, build_pii_middleware
from app.rails.spotlight import SpotlightMiddleware

# §3.4's stack, as a declared contract rather than a comment. The test asserts the
# assembled agent matches this, so reordering the code without reordering this constant
# fails loudly.
MIDDLEWARE_ORDER: tuple[str, ...] = (
    "InjectionMiddleware",
    "PIIMiddleware.builtin",
    "PIIMiddleware.presidio",
    "PersonPseudonymMiddleware",
    "HumanInTheLoopMiddleware",
    "ToolResultInjectionMiddleware",
    "SpotlightMiddleware",
    "AuditMiddleware",
)


# LangGraph counts every middleware hook as its own super-step, and this stack is deep:
# the PII rail alone contributes 12 instances (four rules x three application points,
# §3.1.5), so a single turn costs roughly
#
#     2 before_agent + 14 before_model + 1 model + 14 after_model + 2 after_agent  ~= 33
#
# and a turn that calls a tool costs about twice that. **LangGraph's default
# `recursion_limit` is 25, which does not complete even one turn of this agent** — the
# run dies with GraphRecursionError before the model has answered once. That is not a
# loop and raising the limit is the correct fix, not a workaround.
#
# 100 allows roughly two tool round-trips, which is what a text-to-SQL answer needs
# (query, then summarise), while still catching a genuinely looping 3B model rather than
# letting it spin until the context window fills.
DEFAULT_RECURSION_LIMIT = 100


def agent_config(thread_id: str, recursion_limit: int = DEFAULT_RECURSION_LIMIT) -> dict[str, Any]:
    """The invoke config this agent needs.

    Wraps `hitl_config` (§3.3 requires a `thread_id` to resume an interrupt) and adds the
    raised recursion limit explained above. Callers should use this rather than hand-
    building the dict, because both halves are easy to omit and each produces a confusing
    failure: no `thread_id` loses pending approvals, and the default recursion limit makes
    every request fail.
    """
    from app.rails.hitl import hitl_config

    return {**hitl_config(thread_id), "recursion_limit": recursion_limit}


class SentinelState(AgentState[Any]):
    """The union of every rail's state keys.

    Declared in one place because LangGraph **silently drops** state updates for keys no
    schema knows about — no exception, no warning. A rail returning `injection_flags`
    into an undeclared channel would simply lose it, and §3.7's security banner would
    never fire while every test that only checks messages still passed.
    """

    injection_flags: NotRequired[list[dict[str, Any]]]
    pii_flags: NotRequired[list[dict[str, Any]]]
    quarantine_incidents: NotRequired[list[dict[str, Any]]]
    audit_event_id: NotRequired[int]


def build_llm() -> ChatOllama:
    """The locally served model, over Ollama's OpenAI-compatible endpoint.

    `num_ctx` and `temperature` come from config, which mirrors the Modelfile — §1.1
    rule 2 caps the context at 4096 and §6 rule 3 pins temperature at 0.1 so the
    red-team harness can assert against responses.
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url.removesuffix("/v1"),
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
    )


def build_read_tools(rm_id: str) -> list[StructuredTool]:
    """The three AUTO-tier read tools, bound to `rm_id`.

    `response_format="content_and_artifact"` on all three: the structured envelope with
    its per-field `source` tags has to survive onto the `ToolMessage` or the injection
    and spotlight rails have nothing to work with. See the module docstring.
    """

    def run_query(sql: str) -> tuple[str, dict[str, Any]]:
        """Run a read-only SELECT over the allowlisted tables."""
        result = query_customer_data(sql, rm_id)
        return _summarise(result), result

    def run_summary(account_id: str) -> tuple[str, dict[str, Any]]:
        """Summarise one account and its recent transactions."""
        result = get_account_summary(account_id, rm_id)
        return _summarise(result), result

    def run_policy(query: str, k: int = DEFAULT_TOP_K) -> tuple[str, dict[str, Any]]:
        """Search the bank's internal policy documents."""
        result = search_policy_docs(query, k=k)
        return _summarise(result), result

    return [
        StructuredTool.from_function(
            func=run_query,
            name="query_customer_data",
            description=(
                "Run a read-only SQL SELECT over customers, accounts, transactions, "
                "loans and service_requests. Your portfolio is filtered automatically."
            ),
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=run_summary,
            name="get_account_summary",
            description="Summary of one account (ACCT-0000 format) plus recent transactions.",
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            func=run_policy,
            name="search_policy_docs",
            description="Search internal bank policy documents. Cite the document you use.",
            response_format="content_and_artifact",
        ),
    ]


def _summarise(result: dict[str, Any]) -> str:
    """The text form of a tool result, for the model.

    Deliberately rebuilt from the (possibly quarantined, possibly spotlighted) artifact by
    the middleware rather than trusted from here — this is only the initial rendering.
    """
    if not result.get("ok"):
        return f"tool did not run: {result.get('reason')}"
    return str(result)


def build_middleware(
    rm_id: str,
    rails: RailsConfig,
    hitl: Any = None,
    session: Any = None,
) -> list[AgentMiddleware]:
    """The §3.4 stack, in declaration order. See the module docstring for the semantics.

    Every entry receives the *same* `rails` object and re-reads its flag on each hook, so
    a toggle takes effect on the next message with no rebuild.

    `hitl` is passed in rather than built here so that the instance in the graph is the
    same one `build_write_tools` validated its gate against — two identical-looking
    instances would also trip `create_agent`'s duplicate-name check.
    """
    pii = build_pii_middleware(rails)
    # build_pii_middleware returns 12 timed instances then the pseudonymizer; positions
    # 2-3 of §3.4 are the timed ones and position 4 is the pseudonymizer, so they are
    # split back out here to keep this list readable as the plan's numbered stack.
    pseudonym = [m for m in pii if isinstance(m, PersonPseudonymMiddleware)]
    redactors = [m for m in pii if not isinstance(m, PersonPseudonymMiddleware)]

    gate = hitl if hitl is not None else build_hitl_middleware(rm_id, session=session)

    return [
        InjectionMiddleware(rails),  # 1
        *redactors,  # 2 + 3
        *pseudonym,  # 4
        gate,  # 5
        HardBlockMiddleware(rm_id, rails=rails, session=session),  # 5 (see note)
        ToolResultInjectionMiddleware(rails),  # 6
        SpotlightMiddleware(rails),  # 7
        AuditMiddleware(rm_id, rails=rails, session=session),  # 8
    ]


@dataclass(frozen=True)
class AgentBundle:
    """A compiled agent together with the live middleware instances inside it.

    §3.6's `/chat` response has to report "which rails fired, with their verdicts and
    scores", and not all of that reaches agent state. `wrap_tool_call` hooks cannot
    return state updates, so `ToolResultInjectionMiddleware.incidents` and
    `HardBlockMiddleware.blocked` — the quarantine banner and the OFAC refusal, i.e.
    exactly the two things §3.7 renders in red — exist only on the instances.

    Handing those back is strictly better than the alternative the tests use (patching
    `build_middleware` to intercept the list): the API holds a reference to the same
    objects the graph is running, so what it reports is what actually happened rather
    than a reconstruction.
    """

    rm_id: str
    rails: RailsConfig
    graph: Any
    middleware: list[AgentMiddleware]

    def of_type(self, cls: type) -> Any:
        """The first middleware instance of `cls`, or None."""
        return next((m for m in self.middleware if isinstance(m, cls)), None)


def build_agent_bundle(
    rm_id: str,
    rails: RailsConfig | None = None,
    *,
    checkpointer: Any = None,
    session: Any = None,
    model: Any = None,
) -> AgentBundle:
    """Assemble the full agent for one RM (§3.4), keeping the middleware reachable.

    Args:
        rm_id:        from application session state. Never from the model (§6 rule 5).
        rails:        the shared, mutable RailsConfig. Defaults to the global settings
                      object so the API and the UI toggle the same instance the graph
                      holds.
        checkpointer: required for HITL to survive an interrupt (§3.3).
        session:      an explicit SQLAlchemy session, for tests that roll back.
        model:        override the LLM, for tests that must not need Ollama.
    """
    config = rails if rails is not None else settings.guardrails_enabled
    hitl = build_hitl_middleware(rm_id, session=session)

    tools = [
        *build_read_tools(rm_id),
        *build_write_tools(rm_id, hitl, session=session),
    ]

    # The same `hitl` instance the write tools were validated against goes into the
    # graph, so the "is this tool gated?" check refers to the middleware actually
    # running rather than a second, identical-looking one.
    middleware = build_middleware(rm_id, config, hitl=hitl, session=session)

    graph = create_agent(
        model=model if model is not None else build_llm(),
        tools=tools,
        system_prompt=build_system_prompt(rm_id),
        middleware=middleware,
        state_schema=SentinelState,
        checkpointer=checkpointer,
    )
    return AgentBundle(rm_id=rm_id, rails=config, graph=graph, middleware=middleware)


def build_agent(
    rm_id: str,
    rails: RailsConfig | None = None,
    *,
    checkpointer: Any = None,
    session: Any = None,
    model: Any = None,
):
    """The compiled graph alone, for callers that do not need the middleware."""
    return build_agent_bundle(
        rm_id, rails, checkpointer=checkpointer, session=session, model=model
    ).graph


def middleware_names(middleware: list[AgentMiddleware]) -> list[str]:
    """Collapse an assembled stack to §3.4's eight names, for assertions.

    The PII rail contributes twelve instances (four rules x three application points),
    which is an implementation detail of §3.1.5 rather than a departure from §3.4's
    stack — so they fold back into the two entries the plan names.
    """
    names: list[str] = []
    for item in middleware:
        cls = type(item).__name__
        if cls == "TimedPIIMiddleware":
            label = (
                "PIIMiddleware.presidio"
                if item.pii_type.startswith("presidio")
                else "PIIMiddleware.builtin"
            )
        elif cls == "HardBlockMiddleware":
            # Part of §3.3's HITL rail, not a ninth entry: the plan's position 5 is "the
            # HITL rail", and the hard block is the half of it that cannot live in
            # `interrupt_on`.
            continue
        else:
            label = cls
        if not names or names[-1] != label:
            names.append(label)
    return names
