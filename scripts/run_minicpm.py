#!/usr/bin/env python3
"""Run MiniCPM-o 4.5 on prepared media; model dependencies load after dry-run."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.reproduce import read_jsonl
from evaluation.output_parsing import parse_choices


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def prefix_cutoff(question):
    import re
    stem = re.split(r"\n\s*(?:题目：|Question:)", question, flags=re.I)[-1]
    match = re.search(r"第\s*([\d.]+)\s*秒", stem)
    if not match:
        match = re.search(r"\b(?:video|clip)\s+is\s+at\s+([\d.]+)\s*(?:seconds?|s)\b", stem, re.I)
    return float(match[1]) if match else None


def run_qa(args, rows, config):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", init_vision=True, init_audio=True, init_tts=False).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True)
    video_to_case = {r["video"]: r["case_id"] for r in rows}
    with (ROOT / "dataset/qa/qa_zh.csv").open(encoding="utf-8-sig", newline="") as handle:
        questions = [r for r in csv.DictReader(handle) if r["video"] in video_to_case]
    for row in questions:
        ident = video_to_case[row["video"]]
        dest = args.output / row["qa_id"]
        dest.mkdir(exist_ok=False)
        try:
            prepared = args.prepared / ident
            manifest = json.loads((prepared / "manifest.json").read_text())
            cutoff = prefix_cutoff(row["question"])
            frames = [f for f in manifest["frames"] if cutoff is None or f["pts_sec"] <= cutoff]
            if len(frames) > 10:
                frames = [frames[round(i * (len(frames) - 1) / 9)] for i in range(10)]
            content = []
            for frame in frames:
                content += ["视频时间 %.1f 秒：" % frame["pts_sec"], Image.open(prepared / frame["file"]).convert("RGB")]
            if manifest["audio_present"]:
                pcm = (prepared / manifest["audio_file"]).read_bytes()
                if cutoff is not None:
                    pcm = pcm[:int(cutoff * 16000) * 2]
                if pcm:
                    content.append(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)
            options = {k[-1]: v for k, v in row.items() if k.startswith("option_") and v}
            prompt = row["question"] + "\n\n" + "\n".join(k + ". " + v for k, v in options.items())
            prompt += "\n\n" + (ROOT / "prompts/qa.txt").read_text().strip()
            content.append(prompt)
            write_new(dest / "input.json", {"case_id": ident, "qa_id": row["qa_id"], "text": prompt, "cutoff": cutoff})
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            start = time.monotonic()
            raw = model.chat(msgs=[{"role": "user", "content": content}], tokenizer=tokenizer, stream=False,
                generate_audio=False, enable_thinking=False, omni_mode=False, max_new_tokens=64,
                do_sample=False, num_beams=1, length_penalty=1.0)
            if not isinstance(raw, str):
                raise TypeError("Expected native text output")
            write_new(dest / "raw-output.json", {"text": raw})
            picks, valid = parse_choices(raw, options, options)
            write_new(dest / "result.json", {"status": "success", "qa_id": row["qa_id"], "raw": raw,
                "picks": picks, "valid": valid, "exact": valid and set(picks) == set(row["answer"].split(";")),
                "inference_seconds": time.monotonic() - start})
        except Exception as error:
            write_new(dest / "error.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("streaming", "qa"), default="streaming")
    parser.add_argument("--model-path", type=Path, required=True, help="Local copy of the revision in configs/minicpm.json")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing runs are preserved")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--dry-run", action="store_true", help="Validate prepared inputs without loading weights")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/minicpm.json").read_text())
    rows = read_jsonl(ROOT / "dataset/metadata/cases.jsonl")
    if args.case_id:
        rows = [r for r in rows if r["case_id"] in args.case_id]
        if {r["case_id"] for r in rows} != set(args.case_id):
            parser.error("Unknown case ID")
    for row in rows:
        manifest = json.loads((args.prepared / row["case_id"] / "manifest.json").read_text())
        if manifest["source_video_sha256"] != row["video_sha256"]:
            raise ValueError("Prepared video identity mismatch: " + row["case_id"])
        if manifest.get("synthetic_audio_added") is not False:
            raise ValueError("Prepared inputs must contain original audio only")
    if args.dry_run:
        print(json.dumps({"track": args.track, "cases": len(rows), "model_loaded": False, "api_calls": 0}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "config.json", {**config, "track": args.track, "cases": [r["case_id"] for r in rows]})
    if args.track == "qa":
        run_qa(args, rows, config)
        return
    from evaluation.minicpm.adapter import Adapter
    model = Adapter.load({**config, "model_path": str(args.model_path)})
    for row in rows:
        dest = args.output / row["case_id"]
        dest.mkdir()
        try:
            result = model.run_case(args.prepared / row["case_id"], dest)
            coverage = result["coverage"]
            complete = (coverage["frames_received"] == coverage["frames_expected"]
                        and coverage["source_audio_samples_received"] == coverage["source_audio_samples_expected"])
            if not complete:
                result.update(success=False, status="failed", failure_reason="incomplete_source_coverage")
            write_new(dest / "result.json", result)
        except Exception as error:
            write_new(dest / "error.json", {"status": "failed", "type": type(error).__name__, "message": str(error)})


if __name__ == "__main__":
    main()
