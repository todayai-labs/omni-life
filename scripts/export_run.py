#!/usr/bin/env python3
"""Export new inference runs into the portable scoring and judge schema."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.export import normalize_qa, normalize_streaming, prepare_judge
from evaluation.reproduce import read_jsonl


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streaming", type=Path, required=True, help="Output directory from run_minicpm --track streaming")
    parser.add_argument("--qa", type=Path, required=True, help="Output directory from run_minicpm --track qa")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New portable result directory")
    args = parser.parse_args()
    for path in (args.streaming, args.qa, args.prepared):
        if not path.is_dir():
            parser.error("Input run and prepared directories must exist")
    rows = read_jsonl(ROOT / "dataset/metadata/cases.jsonl")
    annotations = {r["case_id"]: json.loads((ROOT / "dataset" / r["annotation"]).read_text()) for r in rows}
    streams = [normalize_streaming(row, annotations[row["case_id"]], args.streaming, args.prepared) for row in rows]
    with (ROOT / "dataset/qa/qa_zh.csv").open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))
    qa = [normalize_qa(question, args.qa) for question in questions]
    jobs, pairs = prepare_judge(streams, annotations)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "judge").mkdir()
    write_jsonl(args.output / "streaming.jsonl", streams)
    write_jsonl(args.output / "qa.jsonl", qa)
    write_jsonl(args.output / "judge/jobs-input.jsonl", jobs)
    write_jsonl(args.output / "judge/pairs.jsonl", pairs)
    print(json.dumps({"streaming_cases": len(streams), "failed_streaming": sum(r["status"] != "success" for r in streams),
        "qa_questions": len(qa), "failed_qa": sum(r["status"] != "success" for r in qa),
        "judge_pairs": len(pairs), "unique_judge_jobs": len(jobs), "api_calls": 0}))


if __name__ == "__main__":
    main()
