#!/usr/bin/env python3
"""Offline reproduction. No model weights, videos, network, or API key needed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.reproduce import compare, read_jsonl, recompute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/minicpm-o-4.5")
    parser.add_argument("--check", action="store_true", help="Verify metrics and every published event match")
    parser.add_argument("--output", type=Path, help="Write recomputed metrics to a new file")
    args = parser.parse_args()
    metrics, matches = recompute(ROOT, args.results)
    if args.check:
        expected = json.loads((args.results / "metrics.json").read_text())
        if "models" in expected:
            expected = next(r for r in expected["models"] if r["model"] == "minicpm-o-4.5")
        compare(metrics, expected)
        saved = {r["case_id"]: r for r in read_jsonl(args.results / "matches.jsonl")}
        for row in matches:
            compare(row, saved[row["case_id"]], row["case_id"])
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")
    if args.check:
        print("PASS: published metrics and 400 case-level matches reproduced.", file=sys.stderr)


if __name__ == "__main__":
    main()
