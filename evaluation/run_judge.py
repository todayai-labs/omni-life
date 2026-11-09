#!/usr/bin/env python3
"""Rejudge saved text pairs with a user-configured OpenAI-compatible endpoint."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.core.reproduce import read_jsonl


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep authenticated requests on the explicitly configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def judge_endpoint(base):
    """Validate an HTTPS API base without echoing a potentially sensitive URL."""
    invalid = ValueError("JUDGE_BASE_URL must be a valid HTTPS base URL without credentials, query, or fragment")
    if not base or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in base) or any(c in base for c in "\\?#"):
        raise invalid
    try:
        parts = urllib.parse.urlsplit(base)
        host, port = parts.hostname, parts.port
        if parts.scheme != "https" or not host or parts.username is not None or parts.password is not None:
            raise invalid
        if parts.netloc.endswith(":") or port is not None and not 1 <= port <= 65535:
            raise invalid
        if ":" in host:
            if not re.fullmatch(r"\[[^\[\]]+\](?::[0-9]+)?", parts.netloc):
                raise invalid
            ipaddress.IPv6Address(host)
            authority = "[" + host + "]"
        elif re.fullmatch(r"[0-9.]+", host) and "." in host:
            ipaddress.IPv4Address(host)
            authority = host
        else:
            authority = host.encode("idna").decode("ascii")
            dns_name = authority.removesuffix(".")
            if len(dns_name) > 253 or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in dns_name.split(".")):
                raise invalid
        if port is not None:
            authority += ":" + str(port)
        return urllib.parse.urlunsplit(("https", authority, parts.path.rstrip("/") + "/chat/completions", "", ""))
    except (ValueError, UnicodeError):
        raise invalid from None


def request_judgment(opener, endpoint, key, payload):
    """Avoid exposing credentials or server response bodies in failures."""
    try:
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        with opener.open(request, timeout=120) as response:
            value = json.load(response)
        choice = value["choices"][0]
        answer = str(choice["message"].get("content") or "").strip()
        if answer not in ("0", "1") or choice.get("finish_reason") != "stop":
            raise ValueError("Invalid judge decision")
        return value, choice, answer
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Judge HTTP status {error.code}; completed results are preserved") from None
    except urllib.error.URLError:
        raise SystemExit("Judge connection failed; completed results are preserved") from None
    except Exception:
        raise SystemExit("Judge request or response validation failed; completed results are preserved") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=ROOT / "results/minicpm-o-4.5/judge/jobs.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file; the published labels are preserved")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL"), help="Requested API model name; defaults to the historical requested alias in evaluation/configs/judge.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate requests without network calls or credentials")
    args = parser.parse_args()
    config = json.loads((ROOT / "evaluation/configs/judge.json").read_text())
    args.model = args.model or config["requested_model"]
    jobs = read_jsonl(args.jobs)
    system = (ROOT / "evaluation/prompts/judge.txt").read_text()
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
    base = os.environ.get("JUDGE_BASE_URL", "")
    key = os.environ.get("JUDGE_API_KEY", "")
    try:
        endpoint = judge_endpoint(base)
    except ValueError as error:
        parser.error(str(error))
    if not key or any(not 33 <= ord(c) <= 126 for c in key):
        parser.error("Set JUDGE_API_KEY to a valid API credential in your environment")
    if args.output.resolve() == args.jobs.resolve():
        parser.error("Output must be separate from published inputs")
    opener = urllib.request.build_opener(NoRedirectHandler())
    with args.output.open("x", encoding="utf-8") as handle:
        for job in jobs:
            payload = {"model": args.model, "temperature": config["temperature"], "max_tokens": config["max_tokens"], "stream": False,
                       "thinking": {"type": config["thinking"]}, "messages": [
                           {"role": "system", "content": system},
                           {"role": "user", "content": json.dumps(job["input"], ensure_ascii=False)}]}
            value, choice, answer = request_judgment(opener, endpoint, key, payload)
            record = {"job_id": job["job_id"], "input": job["input"], "correct": int(answer),
                "requested_model": args.model, "returned_model": value.get("model"),
                "raw_answer": answer, "finish_reason": choice["finish_reason"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    print(json.dumps({"jobs_completed": len(jobs), "model": args.model}))


if __name__ == "__main__":
    main()
