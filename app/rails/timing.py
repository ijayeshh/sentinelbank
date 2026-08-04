"""@rail_timer latency decorator (PROJECT_PLAN.md §2.5).

Observability platforms are explicitly out of scope (§2.1) — no Langfuse, no Phoenix, no
OpenTelemetry collector. But §4.3's **latency budget table is a required deliverable**,
and the argument that project makes only lands with real numbers: a reviewer wants to
know what the rails cost, as a percentage of a request, measured rather than estimated.

So this is the whole observability story, and it is deliberately about 40 lines: a
decorator, a ring buffer, and an append-only JSONL file that `redteam/report.py`
aggregates into median and p95 per rail in Phase 4.

Two design points worth keeping:

**Latency is recorded even when the wrapped call raises.** A rail that reports only its
successful path understates its own cost — rejections are frequently the *slow* path (a
full AST walk before the reject, a classifier scan that came back positive), and a
budget table built from happy-path timings only would flatter the system. `try/finally`.

**Instrumentation must never be able to break a rail.** A full disk, a locked file, a
read-only checkout — none of those are reasons for a customer query to fail. Every write
error is swallowed. The in-memory buffer is the primary record; the file is a
convenience for the Phase 4 report.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

# Bounded so a long-running demo cannot grow the process's memory without limit. On an
# 8 GB machine an unbounded list is a slow leak, and the report only ever needs a recent
# window: a few thousand samples is far more than the 30-case red-team run produces.
RING_BUFFER_SIZE = 4096

# app/rails/timing.py -> parents[2] is the repository root.
LATENCY_LOG_PATH = Path(__file__).resolve().parents[2] / "redteam" / "latency.jsonl"

_SAMPLES: deque[dict[str, Any]] = deque(maxlen=RING_BUFFER_SIZE)

F = TypeVar("F", bound=Callable[..., Any])


def _record(rail: str, duration_ms: float) -> None:
    """Append one sample to the ring buffer and to the JSONL file.

    The schema — `{rail, duration_ms, timestamp}` — is fixed by §2.5 and consumed by
    §4.3, so it is a contract between the two phases rather than a debug format.
    """
    sample = {
        "rail": rail,
        "duration_ms": round(duration_ms, 3),
        # UTC and ISO-8601: the report sorts and buckets on this, and a local-time
        # string with no offset is ambiguous for exactly one hour every autumn.
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _SAMPLES.append(sample)

    # Read the module global at call time rather than binding it at import, so tests can
    # redirect the path to tmp_path instead of writing into the repository.
    try:
        LATENCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LATENCY_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample) + "\n")
    except OSError:
        # Deliberately silent. See the module docstring: a failed latency write must
        # never surface as a failed customer query.
        pass


def rail_timer(rail_name: str) -> Callable[[F], F]:
    """Decorator recording how long a rail took.

        @rail_timer("sql_guard")
        def validate(...): ...

    `rail_name` is supplied explicitly rather than taken from `func.__name__` because the
    names in the §4.3 table describe *application points*, not functions: the same PII
    scanner is timed separately as `pii_input`, `pii_tool_result`, and `pii_output`.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                # perf_counter is monotonic — immune to NTP steps and DST, unlike
                # time.time(), which can otherwise produce negative durations.
                _record(rail_name, (time.perf_counter() - start) * 1000.0)

        return wrapper  # type: ignore[return-value]

    return decorator


def get_samples(rail: str | None = None) -> list[dict[str, Any]]:
    """Samples recorded so far, oldest first; optionally filtered to one rail."""
    samples = list(_SAMPLES)
    if rail is None:
        return samples
    return [sample for sample in samples if sample["rail"] == rail]


def clear_samples() -> None:
    """Empty the in-memory buffer. For tests, and for a clean red-team run."""
    _SAMPLES.clear()
