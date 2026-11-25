"""Score Streaming ablations and compare paired Full Correct outcomes."""
import argparse
import json
from pathlib import Path

from .score_streaming import evaluate, load_records


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline, condition):
    totals = dict(paired=0, both_correct=0, both_incorrect=0, improved=0, regressed=0, undetermined=0)
    for case, control in baseline.items():
        first, second = control["full_correct"], condition[case]["full_correct"]
        if first is None or second is None:
            totals["undetermined"] += 1
            continue
        totals["paired"] += 1
        key = "both_correct" if first and second else "both_incorrect" if not first and not second else "improved" if second else "regressed"
        totals[key] += 1
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Adjudicated B0 captures keyed by case ID")
    parser.add_argument("--arm", action="append", required=True, metavar="CONDITION=FILE")
    args = parser.parse_args()
    protocol = load(args.protocol)
    control = evaluate(protocol, load_records(args.baseline))
    arms = {}
    for value in args.arm:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in arms or name == "B0":
            parser.error("Each --arm must supply a distinct non-B0 condition and a JSON file")
        result = evaluate(protocol, load_records(Path(path)))
        arms[name] = dict(result, compared_with_B0=compare(control["cases"], result["cases"]))
    print(json.dumps({"B0": control, "ablations": arms}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
