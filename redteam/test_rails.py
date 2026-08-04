"""pytest harness parameterised over attacks.yaml (PROJECT_PLAN.md §4.2).

Each case asserts: expected verdict (BLOCKED/ALLOWED), expected rail fired, that no
string from must_not_contain appears in the response, and that latency was recorded.
Not implemented until Phase 4.
"""
