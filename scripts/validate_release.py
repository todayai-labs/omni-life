#!/usr/bin/env python3
"""Validate public dataset identities, denominators, and released results offline."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.reproduce import compare, read_jsonl, recompute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = read_jsonl(ROOT / "dataset/metadata/cases.jsonl")
    if len(rows) != 400 or len({r["case_id"] for r in rows}) != 400:
        raise ValueError("Expected 400 unique cases")
    required = silent = waits = 0
    for row in rows:
        path = ROOT / "dataset" / row["annotation"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["annotation_sha256"]:
            raise ValueError("Annotation identity mismatch: " + row["case_id"])
        value = json.loads(content)
        if value["id"] != row["case_id"]:
            raise ValueError("Annotation ID mismatch")
        active = [a for a in value["annotation"] if a["action"].strip("[]") != "wait"]
        required += len(active)
        silent += not active
        waits += len(value["annotation"]) - len(active)
    if (required, silent, waits) != (555, 110, 685):
        raise ValueError("Unexpected required-event or silence counts")
    with (ROOT / "dataset/qa/qa_zh.csv").open(encoding="utf-8-sig", newline="") as handle:
        qa = list(csv.DictReader(handle))
    if len(qa) != 800 or len({q["qa_id"] for q in qa}) != 800:
        raise ValueError("Expected 800 unique QA questions")
    target = ROOT / "results/minicpm-o-4.5"
    jobs_list = read_jsonl(target / "judge/jobs.jsonl")
    jobs = {row["job_id"]: row for row in jobs_list}
    if len(jobs) != len(jobs_list):
        raise ValueError("Duplicate judge job IDs")
    for job in jobs.values():
        if job["correct"] not in (0, 1) or job["raw_answer"] != str(job["correct"]) or job["finish_reason"] != "stop":
            raise ValueError("Invalid or unfinished raw judge decision")
    inventory = {row["case_id"]: row for row in rows}
    streams = {r["case_id"]: r for r in read_jsonl(target / "streaming.jsonl")}
    pairs = read_jsonl(target / "judge/pairs.jsonl")
    for pair in pairs:
        ident = pair["case_id"]
        labels = json.loads((ROOT / "dataset" / inventory[ident]["annotation"]).read_text())
        annotation = labels["annotation"][pair["annotation_index"]]
        prediction = streams[ident]["predictions"][pair["response_index"]]
        content = jobs[pair["job_id"]]["input"]
        expected_input = {"task": labels.get("query"),
            "context": {k: labels.get(k) for k in ("scene", "assistant_role")},
            "reference_zh": annotation.get("text_zh"), "reference_en": annotation.get("text_en"),
            "candidate": prediction.get("content", prediction.get("text", ""))}
        if content != expected_input:
            raise ValueError("Judge input differs from its referenced case")
        if (pair["gt_start"], pair["gt_end"], pair["reply_onset"]) != (annotation["start"], annotation["end"], prediction["start_media_s"]):
            raise ValueError("Judge pair timestamp mismatch")
    questions = {r["qa_id"]: r for r in qa}
    for record in read_jsonl(target / "qa.jsonl"):
        if record["gold"] != questions[record["qa_id"]]["answer"].split(";"):
            raise ValueError("QA reference differs from the dataset")
    for base in ("scripts", "evaluation", "tests"):
        for path in (ROOT / base).rglob("*.py"):
            ast.parse(path.read_text(), filename=path.name)
    metrics, matches = recompute(ROOT)
    compare(metrics, json.loads((target / "metrics.json").read_text()))
    saved = {r["case_id"]: r for r in read_jsonl(target / "matches.jsonl")}
    for row in matches:
        compare(row, saved[row["case_id"]], row["case_id"])
    print(json.dumps({"status": "passed", "videos": 400, "required_events": required,
        "fully_silent_videos": silent, "wait_intervals": waits, "qa_questions": len(qa),
        "streaming_cases_recomputed": len(matches), "judge_jobs_verified": len(jobs),
        "judge_pairs_verified": len(pairs), "model_calls": 0, "judge_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
