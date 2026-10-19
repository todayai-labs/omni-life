#!/usr/bin/env python3
"""Rejudge saved text pairs with a user-configured OpenAI-compatible endpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.reproduce import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=ROOT / "results/minicpm-o-4.5/judge/jobs.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; the published labels are preserved")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL"), help="Requested API model name; defaults to the historical requested alias in configs/judge.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate requests without network calls or credentials")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/judge.json").read_text())
    args.model = args.model or config["requested_model"]
    jobs = read_jsonl(args.jobs)
    system = (ROOT / "prompts/judge.txt").read_text()
    if len({j["job_id"] for j in jobs}) != len(jobs):
        parser.error("Duplicate job IDs")
    if args.dry_run:
        print(json.dumps({"jobs": len(jobs), "requested_model": args.model, "historical_returned_model": config["observed_returned_model"], "api_calls": 0}))
        return
    if not jobs:
        with args.output.open("x", encoding="utf-8"):
            pass
        print(json.dumps({"jobs_completed": 0, "api_calls": 0}))
        return
    base = os.environ.get("JUDGE_BASE_URL", "").rstrip("/")
    key = os.environ.get("JUDGE_API_KEY", "")
    if not base.startswith("https://") or not key:
        parser.error("Set JUDGE_BASE_URL to an HTTPS API base URL and JUDGE_API_KEY in your environment")
    if args.output.resolve() == args.jobs.resolve():
        parser.error("Output must be separate from published inputs")
    with args.output.open("x", encoding="utf-8") as handle:
        for job in jobs:
            payload = {"model": args.model, "temperature": config["temperature"], "max_tokens": config["max_tokens"], "stream": False,
                       "thinking": {"type": config["thinking"]}, "messages": [
                           {"role": "system", "content": system},
                           {"role": "user", "content": json.dumps(job["input"], ensure_ascii=False)}]}
            request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    value = json.load(response)
                choice = value["choices"][0]
                answer = str(choice["message"].get("content") or "").strip()
                if answer not in ("0", "1") or choice.get("finish_reason") != "stop":
                    raise ValueError("Judge returned an unfinished or nonbinary answer")
                record = {"job_id": job["job_id"], "input": job["input"], "correct": int(answer),
                    "requested_model": args.model, "returned_model": value.get("model"),
                    "raw_answer": answer, "finish_reason": choice["finish_reason"]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            except urllib.error.HTTPError as error:
                raise SystemExit(f"Judge HTTP status {error.code}; completed results are preserved") from None
            except urllib.error.URLError:
                raise SystemExit("Judge connection failed; completed results are preserved") from None
    print(json.dumps({"jobs_completed": len(jobs), "model": args.model}))


if __name__ == "__main__":
    main()
