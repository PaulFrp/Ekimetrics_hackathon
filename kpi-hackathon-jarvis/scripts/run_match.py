"""Run one match locally: your red against your blue on a chosen example.

Usage:
    python scripts/run_match.py examples/finance_short/example_01
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from blue.submission import agent as blue_agent  # noqa: E402
from red.submission import agent as red_agent  # noqa: E402
from shared.scoring import score_match  # noqa: E402
from shared.types import GroundTruth  # noqa: E402


def main(example_dir: str) -> None:
    p = Path(example_dir)
    if not p.is_absolute():
        p = ROOT / p
    document = (p / "document.txt").read_text(encoding="utf-8", errors="replace")
    ground_truth = GroundTruth.model_validate_json(
        (p / "ground_truth.json").read_text(encoding="utf-8")
    )

    # Clear prior debug events buffer if debugging is enabled, then run extraction.
    if __import__("os").environ.get("RED_DEBUG"):
        try:
            from red.base import _RED_DEBUG_EVENTS  # type: ignore
            _RED_DEBUG_EVENTS.clear()
        except Exception:
            pass

    extraction = red_agent.extract(document, ground_truth)
    # If debug enabled, collect and print a human-friendly summary of red's
    # debug events (what changed, which technique).
    if __import__("os").environ.get("RED_DEBUG"):
        try:
            from red.base import _RED_DEBUG_EVENTS  # type: ignore

            # Count techniques
            counts: dict[str, int] = {}
            changes: list[str] = []
            for ev in _RED_DEBUG_EVENTS:
                tech = ev.get("technique", "unknown")
                counts[tech] = counts.get(tech, 0) + 1
                if ev.get("type") == "field_change":
                    kpi = ev.get("kpi")
                    field = ev.get("field")
                    old = ev.get("old")
                    new = ev.get("new")
                    changes.append(f"{kpi}: {field} {old} → {new} ({tech})")
                elif ev.get("type") == "kpi_new":
                    kp = ev.get("kpi_new")
                    name = kp.get("name") if isinstance(kp, dict) else str(kp)
                    changes.append(f"NEW {name}: {kp} ({tech})")

            print("\nRED DEBUG SUMMARY:")
            print("Technique counts:")
            for t, c in counts.items():
                print(f" - {t}: {c}")
            if changes:
                print("\nChanges:")
                for line in changes:
                    print(" - ", line)

        except Exception:
            pass
    judgment = blue_agent.judge(document, extraction.public_view())
    result = score_match(extraction, judgment, ground_truth)

    print(f"red:  {red_agent.name}")
    print(f"blue: {blue_agent.name}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/finance_short/example_01")
