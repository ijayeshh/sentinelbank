# Architecture

Request lifecycle diagram; why PII is hybrid (LangChain built-ins cover only five
types, Presidio covers the rest — the coverage boundary is the design decision); why
SQL and entitlement use deterministic rails rather than model-based ones; why the
entitlement predicate is injected at the SQL layer rather than asked for in the prompt;
and the documented Ollama -> vLLM swap as one environment variable (PROJECT_PLAN.md §4.5).

Not written until Phase 4.
