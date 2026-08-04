"""Metrics report writer -> redteam/RESULTS.md (PROJECT_PLAN.md §4.3).

Table 1: rail effectiveness — attacks blocked, benign allowed, false-positive rate
(a required output, not optional), precision, per-category breakdown.
Table 2: latency budget — median/p95 per rail aggregated from redteam/latency.jsonl,
with LLM generation time reported separately so rail overhead shows as a percentage.
Not implemented until Phase 4.
"""
