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
    document = (p / "document.txt").read_text()
    ground_truth = GroundTruth.model_validate_json(
        (p / "ground_truth.json").read_text()
    )

    extraction = red_agent.extract(document, ground_truth)
    judgment = blue_agent.judge(document, extraction.public_view())
    result = score_match(extraction, judgment, ground_truth)

    print(f"red:  {red_agent.name}")
    print(f"blue: {blue_agent.name}")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/finance_short/example_01")
