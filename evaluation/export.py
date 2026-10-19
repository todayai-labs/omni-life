"""Normalize native run artifacts and construct reference-conditioned judge jobs."""
import json
from pathlib import Path

from .output_parsing import parse_choices
from .scoring import score_case


def prepare_judge(records, annotations):
    """All actual replies at/after each active GT onset; no early-response credit.

    Identical content inputs share a single job. Pair selection does not impose
    an end deadline: the same frozen pool supports every latency tolerance.
    """
    unique, jobs, pairs = {}, [], []
    for record in records:
        ident = record["case_id"]
        gt = annotations[ident]
        for ai, annotation in enumerate(gt["annotation"]):
            if annotation["action"].strip("[]") == "wait":
                continue
            for pi, prediction in enumerate(record["predictions"]):
                onset = prediction.get("start_media_s")
                if onset is None or onset < annotation["start"]:
                    continue
                payload = {"task": gt.get("query"),
                    "context": {k: gt.get(k) for k in ("scene", "assistant_role")},
                    "reference_zh": annotation.get("text_zh"), "reference_en": annotation.get("text_en"),
                    "candidate": prediction.get("content", prediction.get("text", ""))}
                key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                if key not in unique:
                    unique[key] = f"job-{len(unique):06d}"
                    jobs.append({"job_id": unique[key], "input": payload})
                pairs.append({"case_id": ident, "annotation_index": ai,
                    "response_index": prediction.get("source_index", pi),
                    "gt_start": annotation["start"], "gt_end": annotation["end"],
                    "reply_onset": onset, "job_id": unique[key]})
    return jobs, pairs


def read_optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def normalize_streaming(row, annotations, run, prepared):
    ident = row["case_id"]
    directory = Path(run) / ident
    result = read_optional(directory / "result.json")
    manifest = read_optional(Path(prepared) / ident / "manifest.json")
    duration = manifest["duration_sec"] if manifest else row["duration_s"]
    responses = result.get("responses", []) if result else []
    if not result:
        parsed = read_optional(directory / "parsed-output.json")
        if parsed is not None:
            responses = parsed
        elif (directory / "response-snapshots.jsonl").is_file():
            lines = (directory / "response-snapshots.jsonl").read_text().splitlines()
            if lines:
                responses = json.loads(lines[-1])["responses"]
    success = bool(result and result.get("success") and result.get("status") == "success")
    metrics = score_case(annotations["annotation"], responses, duration)
    # Portable pair indices address this public array, after filtering/sorting.
    # Native response indices remain available in the original run artifacts.
    predictions = [{**prediction, "source_index": index}
                   for index, prediction in enumerate(metrics["predictions"])]
    return {"model": "minicpm-o-4.5", "case_id": ident,
        "status": "success" if success else "failed", "duration_s": duration,
        "predictions": predictions}


def normalize_qa(question, run):
    directory = Path(run) / question["qa_id"]
    result = read_optional(directory / "result.json")
    raw = read_optional(directory / "raw-output.json")
    text = raw.get("text", "") if raw else result.get("raw", "") if result else ""
    if not isinstance(text, str):
        text = ""
    options = {k[-1]: v for k, v in question.items() if k.startswith("option_") and v}
    picks, valid = parse_choices(text, options, options)
    success = bool(result and result.get("status") == "success")
    if not success:
        picks, valid = [], False
    gold = question["answer"].split(";")
    return {"model": "minicpm-o-4.5", "case_id": question["case_id"],
        "qa_id": question["qa_id"], "status": "success" if success else "failed",
        "raw": text, "gold": gold, "picks": picks, "valid": valid,
        "exact": valid and set(picks) == set(gold)}
