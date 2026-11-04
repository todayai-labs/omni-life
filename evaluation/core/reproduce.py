"""Recompute published metrics from saved outputs and binary judge decisions."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from . import scoring
from .output_parsing import parse_choices

TOLERANCES = (0, 1, 2, 5, 10)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recompute(root, result_dir=None):
    root = Path(root)
    result_dir = Path(result_dir) if result_dir else root / "results/minicpm-o-4.5"
    inventory = read_jsonl(root / "dataset/metadata/cases.jsonl")
    cases = {row["case_id"]: row for row in inventory}
    records = read_jsonl(result_dir / "streaming.jsonl")
    if len(cases) != 400 or len(records) != 400 or {r["case_id"] for r in records} != set(cases):
        raise ValueError("Streaming evaluation must cover all 400 unique cases")
    jobs = {row["job_id"]: row for row in read_jsonl(result_dir / "judge/jobs.jsonl")}
    if any(row["correct"] not in (0, 1) for row in jobs.values()):
        raise ValueError("Judge decisions must be binary")
    semantic = {}
    for pair in read_jsonl(result_dir / "judge/pairs.jsonl"):
        key = (pair["case_id"], pair["annotation_index"], pair["response_index"])
        if key in semantic:
            raise ValueError("Duplicate semantic pair")
        semantic[key] = jobs[pair["job_id"]]["correct"]
    counts = Counter()
    matched_cases = []
    for record in records:
        ident = record["case_id"]
        gt = json.loads((root / "dataset" / cases[ident]["annotation"]).read_text())
        metrics = scoring.score_case(gt["annotation"], record["predictions"], record["duration_s"])
        active = [a for a in metrics["annotations"] if a["action"] != "wait"]
        predictions = metrics["predictions"]
        failed = record["status"] != "success"
        counts.update(videos=1, failed=int(failed), nodes=len(active), emitted_events=len(predictions))
        counts["false"] += metrics["silence"]["false_interventions"]
        counts["wait_minutes"] += metrics["silence"]["annotated_silence_s"] / 60
        counts["intervention" if active else "silent"] += 1
        if not active:
            counts["silent_correct"] += int(not failed and not predictions)
        result = {"case_id": ident, "failed": failed, "tolerances": {}}
        for tolerance in TOLERANCES:
            def allowed(annotation, prediction):
                onset = prediction.get("start_media_s")
                return (onset is not None and prediction.get("timing_valid", True)
                        and annotation["start"] <= onset < annotation["end"] + tolerance
                        and semantic.get((ident, annotation["source_index"], prediction["source_index"]), 0) == 1)
            pairs = scoring._ordered_matches(active, predictions, allowed)
            correct = len(pairs)
            video_correct = bool(active) and not failed and correct == len(active)
            counts[f"correct_nodes_{tolerance}"] += correct
            counts[f"correct_videos_{tolerance}"] += int(video_correct)
            if tolerance == 0:
                counts["strict_correct_videos"] += int(
                    video_correct and len(predictions) == correct
                    and metrics["silent_window_specificity"]["numerator"] == metrics["counts"]["wait_windows"])
            result["tolerances"][str(tolerance)] = {
                "correct_nodes": correct, "video_correct": video_correct,
                "matches": [{"annotation_index": active[i]["source_index"],
                             "response_index": predictions[j]["source_index"]} for i, j in pairs]}
        matched_cases.append(result)
    if (counts["nodes"], counts["intervention"], counts["silent"]) != (555, 290, 110):
        raise ValueError("Unexpected benchmark denominators")
    with (root / "dataset/qa/qa_zh.csv").open(encoding="utf-8-sig", newline="") as handle:
        questions = {row["qa_id"]: row for row in csv.DictReader(handle)}
    qa = read_jsonl(result_dir / "qa.jsonl")
    if len(qa) != 800 or len({r["qa_id"] for r in qa}) != 800 or {r["qa_id"] for r in qa} != set(questions):
        raise ValueError("QA evaluation must cover all 800 unique questions")
    correct_qa = Counter()
    grouped = defaultdict(dict)
    for record in qa:
        question = questions[record["qa_id"]]
        options = {k[-1]: v for k, v in question.items() if k.startswith("option_") and v}
        picks, valid = parse_choices(record["raw"], options, options)
        if record.get("status", "success") != "success":
            picks, valid = [], False
        gold = question["answer"].split(";")
        exact = valid and set(picks) == set(gold)
        if (picks, valid, exact) != (record["picks"], record["valid"], record["exact"]):
            raise ValueError("Saved QA parsing differs: " + record["qa_id"])
        kind = "content" if record["qa_id"].startswith("qa_c_") else "timing"
        correct_qa[kind] += int(exact)
        grouped[question["video"]][kind] = exact
    both = sum(len(pair) == 2 and all(pair.values()) for pair in grouped.values())
    tp, predicted, nodes = counts["correct_nodes_0"], counts["emitted_events"], counts["nodes"]
    summary = {
        "model": "minicpm-o-4.5", "counts": dict(counts),
        "video_correct": 100 * counts["correct_videos_0"] / 290,
        "action_correct": 100 * tp / 555,
        "silent_correct": 100 * counts["silent_correct"] / 110,
        "false_per_min": counts["false"] / counts["wait_minutes"],
        "qa_content": correct_qa["content"] / 4, "qa_timing": correct_qa["timing"] / 4, "qa_both": both / 4,
        "tolerance_action_correct": {str(t): 100 * counts[f"correct_nodes_{t}"] / 555 for t in TOLERANCES},
        "event_precision": 100 * tp / predicted if predicted else None,
        "event_recall": 100 * tp / nodes, "event_f1": 200 * tp / (predicted + nodes),
        "strict_video_correct": 100 * counts["strict_correct_videos"] / 290,
        "event_confusion": {"tp": tp, "fp": predicted - tp, "fn": nodes - tp}}
    return summary, matched_cases


def compare(actual, expected, path="metrics"):
    """Allow only floating-point roundoff; missing fields are errors."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Missing field: {path}.{key}")
            compare(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Length mismatch: " + path)
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if actual is None or abs(actual - expected) > 1e-10:
            raise ValueError("Numeric mismatch: " + path)
    elif actual != expected:
        raise ValueError("Mismatch: " + path)
