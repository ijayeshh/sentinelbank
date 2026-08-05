"""Reproduce the measurements that forced the two-layer injection rail (deviation 7).

    python scripts/measure_classifier_strategies.py

This is the evidence behind the most load-bearing claim in the project: that the plan's
classifier-plus-threshold design (§3.2.1) does not work on this data, and that the
deterministic second layer is therefore not gold-plating. That claim is quoted in
`docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md` and deviation 7, and until this script
existed it was a number recorded from a one-off run — reproducible in principle and by
nobody in practice.

It scores every free-text field in the seeded database (6 planted payloads, 619 benign
memos and notes) under each candidate strategy, and prints the table those documents cite.
Takes about a minute: 625 texts, batched.

Nothing here is used at runtime. It exists so the numbers can be checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text  # noqa: E402

from app.db import get_session  # noqa: E402
from db.seed import POISONED_ROWS  # noqa: E402

THRESHOLD = 0.8


def corpus() -> tuple[list[str], list[str]]:
    """Every free-text field, split into planted payloads and benign fields."""
    with get_session() as session:
        descriptions = [
            r[0] for r in session.execute(text("SELECT description FROM transactions")).all()
        ]
        notes = [
            r[0]
            for r in session.execute(text("SELECT notes FROM service_requests")).all()
            if r[0]
        ]

    planted_text = {row["payload"] for row in POISONED_ROWS}
    fields = descriptions + notes
    planted = [f for f in fields if f in planted_text]
    benign = [f for f in fields if f not in planted_text]
    return planted, benign


def main() -> None:
    from app.rails.injection import (
        _classify_batch,
        get_classifier_info,
        match_patterns,
        scan_text,
    )

    planted, benign = corpus()
    print(f"corpus: {len(planted)} planted payloads, {len(benign)} benign fields, "
          f"{len(planted) + len(benign)} total")
    if len(planted) != len(POISONED_ROWS):
        raise SystemExit(
            f"expected {len(POISONED_ROWS)} planted payloads in the database, found "
            f"{len(planted)} — seed data has drifted, re-run python db/seed.py"
        )

    scores = _classify_batch(planted + benign)
    info = get_classifier_info()
    print(f"classifier: {info.model_id}{'  (FALLBACK)' if info and info.is_fallback else ''}")
    print()

    planted_scores = scores[: len(planted)]
    benign_scores = scores[len(planted) :]

    def strategy(name: str, fn) -> None:
        recall = sum(1 for s in planted_scores if fn(s) >= THRESHOLD)
        fps = [i for i, s in enumerate(benign_scores) if fn(s) >= THRESHOLD]
        pct = 100 * len(fps) / len(benign) if benign else 0.0
        print(f"| `{name}` | {recall}/{len(planted)} | {len(fps)}/{len(benign)} ({pct:.1f}%) |")

    print(f"| Strategy @ {THRESHOLD} | Recall | False positives |")
    print("|---|---|---|")
    strategy("P(JAILBREAK)", lambda s: s.get("JAILBREAK", 0.0))
    strategy("1 - P(BENIGN)", lambda s: 1.0 - s.get("BENIGN", 0.0))
    strategy("P(INJECTION)", lambda s: s.get("INJECTION", 0.0))
    print()

    # The specific memo deviation 7 names, because "an ordinary memo scores 1.00" is the
    # sentence that makes the table land.
    probe = next((f for f in benign if f.startswith("ZELLE TO/BRETT")), None)
    if probe:
        s = _classify_batch([probe])[0]
        print(f"probe: {probe!r}")
        print(f"  INJECTION={s.get('INJECTION', 0.0):.4f}  "
              f"JAILBREAK={s.get('JAILBREAK', 0.0):.4f}  BENIGN={s.get('BENIGN', 0.0):.4f}")
    print()

    # The length gate that was rejected despite scoring perfectly.
    plen = sorted(len(p) for p in planted)
    longest_benign = max(len(b) for b in benign)
    gate_recall = sum(
        1
        for p, s in zip(planted, planted_scores)
        if len(p) >= 80 and (1.0 - s.get("BENIGN", 0.0)) >= THRESHOLD
    )
    gate_fps = sum(
        1
        for b, s in zip(benign, benign_scores)
        if len(b) >= 80 and (1.0 - s.get("BENIGN", 0.0)) >= THRESHOLD
    )
    print(f"rejected length gate (len >= 80 AND 1 - P(BENIGN) >= {THRESHOLD}): "
          f"{gate_recall}/{len(planted)} recall, {gate_fps} false positives")
    print(f"  planted payload lengths {plen[0]}-{plen[-1]} chars; "
          f"longest benign field {longest_benign} chars")
    print("  -> separates the two sets perfectly while detecting 'long string', "
          "not 'injection'. Not used.")
    print()

    # What actually ships.
    shipped_recall = sum(1 for p in planted if scan_text(p).flagged)
    shipped_fps = [b for b in benign if scan_text(b).flagged]
    print(f"SHIPPED RAIL (JAILBREAK union deterministic patterns): "
          f"{shipped_recall}/{len(planted)} recall, {len(shipped_fps)} false positives")
    for b in shipped_fps[:5]:
        print(f"    FP: {b[:90]!r} patterns={match_patterns(b)}")
    print()
    print("Held-out generalisation is the number that matters: "
          "python -m tests.heldout_injections")


if __name__ == "__main__":
    main()
