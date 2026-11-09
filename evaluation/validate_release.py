#!/usr/bin/env python3
"""Validate the frozen public dataset and results offline with the standard library."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.core.reproduce import compare, recompute
from evaluation.core.scoring import parse_action

ACTIONS = {"inform", "warn", "feedback", "answer", "wait"}
TOLERANCES = {"0", "1", "2", "5", "10"}
HASH = re.compile(r"[a-f0-9]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, schema, location, optional=()):
    """Require exact JSON types: bool must never pass as an integer."""
    require(type(value) is dict, f"Expected object: {location}")
    require(set(schema) <= value.keys(), f"Missing fields: {location}")
    require(set(value) <= set(schema) | set(optional), f"Unexpected fields: {location}")
    for key, expected in schema.items():
        types = expected if isinstance(expected, tuple) else (expected,)
        require(type(value[key]) in types, f"Invalid type: {location}.{key}")


def unique(rows, key, location):
    require(type(rows) is list and all(type(row) is dict for row in rows),
            f"Expected object array: {location}")
    values = [row.get(key) for row in rows]
    require(all(type(value) is str and value.strip() for value in values),
            f"Missing or invalid {key}: {location}")
    require(len(values) == len(set(values)), f"Duplicate {key}: {location}")
    return {row[key]: row for row in rows}


def finite(value, location):
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{location}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Non-finite number: {location}")


def same_shape(actual, expected, location):
    """Reject type changes and missing/extra fields before numeric comparison."""
    require(type(actual) is type(expected), f"Invalid type: {location}")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"Field coverage mismatch: {location}")
        for key in expected:
            same_shape(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"Length mismatch: {location}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            same_shape(left, right, f"{location}[{index}]")


def load_json(path, lines=False):
    """Reject duplicate keys, NaN/Infinity, and JSONL blank records."""
    path = Path(path)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"Non-finite number {value}: {path}")
    text = path.read_text(encoding="utf-8")
    if lines:
        records = text.splitlines()
        require(all(record.strip() for record in records), f"Blank JSONL record: {path}")
        result = [json.loads(record, object_pairs_hook=pairs, parse_constant=constant)
                  for record in records]
    else:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    finite(result, str(path))
    return result


def relative_path(value, base, location, must_exist=True):
    require(type(value) is str and bool(value), f"Invalid relative path: {location}")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value
            and ":" not in value and "\x00" not in value and str(path) == value,
            f"Unsafe relative path: {location}")
    resolved = (base / value).resolve()
    require(resolved.is_relative_to(base.resolve()), f"Path escapes root: {location}")
    if must_exist:
        require(resolved.is_file(), f"Missing file: {location}")
    return resolved


def validate_dataset(root):
    dataset = root / "dataset"
    metadata = dataset / "metadata"
    rows = load_json(metadata / "cases.jsonl", lines=True)
    cases = unique(rows, "case_id", "cases")
    require(len(cases) == 400, "Expected 400 unique cases")
    for key in ("source_id", "video", "annotation", "video_sha256", "annotation_sha256"):
        unique(rows, key, "cases")
    required = silent = waits = 0
    annotations = {}
    referenced = set()
    for ident, row in cases.items():
        fields(row, {"case_id": str, "source_id": str, "video": str, "annotation": str,
            "source_type": str, "population_id": str, "scenario_id": str,
            "duration_s": (int, float), "requires_intervention": bool,
            "intervention_events": int, "video_sha256": str, "annotation_sha256": str}, ident)
        require(row["source_type"] in {"real", "synthetic"}, f"Invalid source type: {ident}")
        require(row["duration_s"] > 0, f"Invalid duration: {ident}")
        require(all(HASH.fullmatch(row[key]) for key in ("video_sha256", "annotation_sha256")),
                f"Invalid SHA-256: {ident}")
        relative_path(row["video"], dataset, ident + ".video", must_exist=False)
        require(row["video"].endswith(".mp4"), f"Invalid video suffix: {ident}")
        path = relative_path(row["annotation"], dataset, ident + ".annotation")
        require(row["annotation"].startswith("annotations/") and path.name.endswith(".labels.json"),
                f"Invalid annotation path: {ident}")
        referenced.add(path)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["annotation_sha256"],
                f"Annotation identity mismatch: {ident}")
        value = load_json(path)
        fields(value, {"id": str, "form": str, "side": str, "viewpoint": str,
            "target_people": str, "scene": str, "query": list, "annotation": list,
            "target_people_detail": str, "population_id": str, "assistant_role": str,
            "assistant_role_basis": str, "eval_group": str}, ident,
            optional=("capture_context", "pair_id", "task_context", "media_processing"))
        for key in ("capture_context", "pair_id"):
            if key in value:
                require(type(value[key]) is str and bool(value[key]), f"Invalid {key}: {ident}")
        for key in ("task_context", "media_processing"):
            if key in value:
                require(type(value[key]) is dict, f"Invalid {key}: {ident}")
        require(value["id"] == ident and value["population_id"] == row["population_id"],
                f"Annotation identity/population mismatch: {ident}")
        require(bool(value["query"]) and bool(value["annotation"]), f"Empty annotation/query: {ident}")
        for query in value["query"]:
            fields(query, {"start_time_sec": (int, float), "text": str}, ident + ".query", ("text_en",))
            require(query["text"].strip() and 0 <= query["start_time_sec"] < row["duration_s"],
                    f"Invalid standing query: {ident}")
            if "text_en" in query:
                require(type(query["text_en"]) is str and bool(query["text_en"].strip()),
                        f"Invalid English query: {ident}")
        active = 0
        previous = 0
        for index, annotation in enumerate(value["annotation"]):
            location = f"{ident}.annotation[{index}]"
            fields(annotation, {"start": (int, float), "end": (int, float), "action": str,
                                "text_zh": str, "text_en": str}, location)
            start, end = annotation["start"], annotation["end"]
            require(0 <= start < end and start < row["duration_s"], f"Invalid interval: {location}")
            require(start == previous, f"Gap or overlap in annotation intervals: {location}")
            previous = end
            # Frozen endpoints use decimal rounding. Scoring already clips ends in memory.
            require(end <= row["duration_s"] + 0.052000001, f"Annotation exceeds known rounding: {location}")
            action = annotation["action"]
            require(action in ACTIONS or action in {f"[{a}]" for a in ACTIONS},
                    f"Invalid action: {location}")
            if action.strip("[]") != "wait":
                active += 1
                require(annotation["text_zh"].strip() and annotation["text_en"].strip(),
                        f"Missing active reference: {location}")
        require(abs(previous - row["duration_s"]) <= 0.066667001,
                f"Final annotation boundary differs from duration: {ident}")
        require(active == row["intervention_events"] and bool(active) == row["requires_intervention"],
                f"Intervention metadata mismatch: {ident}")
        required += active
        silent += not active
        waits += len(value["annotation"]) - active
        annotations[ident] = value
    actual_paths = {path.resolve() for path in (dataset / "annotations").rglob("*.labels.json")}
    require(actual_paths == referenced, "Annotation file coverage differs from the manifest")
    require((required, silent, waits) == (555, 110, 685), "Unexpected intervention/silence counts")
    source_types = dict(Counter(row["source_type"] for row in rows))
    require(source_types == {"real": 123, "synthetic": 277}, "Unexpected source counts")
    for filename, key, count in (("populations.json", "population_id", 12), ("scenarios.json", "scenario_id", 5)):
        groups = unique(load_json(metadata / filename), key, filename)
        observed = Counter(row[key] for row in rows)
        require(len(groups) == count and set(groups) == set(observed), f"Group coverage mismatch: {filename}")
        for group in groups.values():
            fields(group, {key: str, "name_en": str, "name_zh": str, "case_count": int}, filename)
            require(group["case_count"] == observed[group[key]], f"Group count mismatch: {filename}")
    with (dataset / "qa/qa_zh.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        expected = ["qa_id", "case_id", "question_type", "video", "lang", "question"]
        expected += ["option_" + letter for letter in "ABCDEFGHIJKL"] + ["answer"]
        require(reader.fieldnames == expected, "Unexpected QA CSV header")
        questions = list(reader)
    qa = unique(questions, "qa_id", "QA")
    require(len(qa) == 800, "Expected 800 unique QA questions")
    kinds = defaultdict(list)
    for ident, question in qa.items():
        fields(question, {key: str for key in expected}, ident)
        require(question["case_id"] in cases, f"Unknown QA case: {ident}")
        require(question["video"] == cases[question["case_id"]]["video"], f"QA video mismatch: {ident}")
        kind = question["question_type"]
        require(kind in {"content", "timing"} and ident.startswith("qa_c_" if kind == "content" else "qa_t_"),
                f"Invalid QA type: {ident}")
        require(question["lang"] == "zh" and bool(question["question"].strip()), f"Invalid QA text: {ident}")
        options = [letter for letter in "ABCDEFGHIJKL" if question["option_" + letter].strip()]
        require(options == list("ABCDEFGHIJKL"[:len(options)]) and len(options) >= 2,
                f"Invalid QA options: {ident}")
        gold = question["answer"].split(";")
        require(len(gold) == len(set(gold)) and set(gold) <= set(options), f"Invalid QA answer: {ident}")
        kinds[question["case_id"]].append(kind)
    require(set(kinds) == set(cases) and all(sorted(kinds[c]) == ["content", "timing"] for c in cases),
            "QA must contain one content and one timing question per case")
    statistics = load_json(metadata / "statistics.json")
    expected_stats = {"videos": 400, "source_types": source_types, "assistance_videos": 400 - silent,
        "silent_videos": silent, "required_intervention_events": required, "silence_intervals": waits,
        "populations": 12, "scenarios": 5, "qa_questions": 800,
        "qa_question_types": dict(Counter(q["question_type"] for q in questions))}
    fields(statistics, {key: dict if type(value) is dict else int for key, value in expected_stats.items()}, "statistics")
    for key in ("source_types", "qa_question_types"):
        fields(statistics[key], {name: int for name in expected_stats[key]}, "statistics." + key)
    require(statistics == expected_stats, "Dataset statistics differ from the manifest")
    return cases, annotations, qa


def validate_results(root, cases, annotations, questions):
    target = root / "results/minicpm-o-4.5"
    streams = unique(load_json(target / "streaming.jsonl", lines=True), "case_id", "streaming")
    require(set(streams) == set(cases), "Streaming case coverage mismatch")
    for ident, stream in streams.items():
        fields(stream, {"model": str, "case_id": str, "status": str, "duration_s": (int, float), "predictions": list}, ident)
        require(stream["model"] == "minicpm-o-4.5" and stream["status"] in {"success", "failed"}, f"Invalid streaming status/model: {ident}")
        require(stream["duration_s"] == cases[ident]["duration_s"], f"Streaming duration mismatch: {ident}")
        predictions = stream["predictions"]
        unique(predictions, "response_id", ident + ".predictions")
        for index, prediction in enumerate(predictions):
            location = f"{ident}.predictions[{index}]"
            fields(prediction, {"source_index": int, "response_id": str, "text": str, "action": (str, type(None)),
                "content": str, "valid": bool, "reason": (str, type(None)), "start_media_s": (int, float, type(None)),
                "end_media_s": (int, float, type(None)), "audio_duration_s": (int, float), "onset_source": str,
                "timing_valid": bool}, location)
            require(prediction["source_index"] == index and prediction["audio_duration_s"] >= 0,
                    f"Invalid prediction index/audio duration: {location}")
            start, end = prediction["start_media_s"], prediction["end_media_s"]
            valid_time = start is not None and end is not None and 0 <= start <= end
            require(valid_time == prediction["timing_valid"], f"Prediction timing flag mismatch: {location}")
            parsed = parse_action(prediction["text"], prediction["action"])
            require(all(prediction[key] == value for key, value in parsed.items()),
                    f"Prediction parsing differs from recorded text: {location}")
            # Post-video replies are observations, not corrupt rows; do not drop or clamp them.
    jobs = unique(load_json(target / "judge/jobs.jsonl", lines=True), "job_id", "judge jobs")
    require(len(jobs) == 92, "Expected 92 frozen judge jobs")
    for ident, job in jobs.items():
        fields(job, {"job_id": str, "input": dict, "correct": int, "requested_model": str,
            "returned_model": str, "raw_answer": str, "finish_reason": str}, ident)
        require(job["correct"] in (0, 1) and job["raw_answer"] == str(job["correct"])
                and job["finish_reason"] == "stop", f"Invalid or unfinished judge decision: {ident}")
    pairs = load_json(target / "judge/pairs.jsonl", lines=True)
    require(type(pairs) is list and len(pairs) == 100, "Expected 100 frozen judge pairs")
    seen, referenced_jobs = set(), set()
    for pair in pairs:
        fields(pair, {"case_id": str, "annotation_index": int, "response_index": int,
            "gt_start": (int, float), "gt_end": (int, float), "reply_onset": (int, float), "job_id": str}, "judge pair")
        ident = pair["case_id"]
        key = (ident, pair["annotation_index"], pair["response_index"])
        require(key not in seen, "Duplicate semantic pair")
        seen.add(key)
        require(ident in cases and pair["job_id"] in jobs, "Unknown case/job in judge pair")
        labels = annotations[ident]
        require(0 <= pair["annotation_index"] < len(labels["annotation"])
                and 0 <= pair["response_index"] < len(streams[ident]["predictions"]), "Invalid judge pair index")
        annotation = labels["annotation"][pair["annotation_index"]]
        prediction = streams[ident]["predictions"][pair["response_index"]]
        require(annotation["action"].strip("[]") != "wait", "Judge pair targets silence")
        expected_input = {"task": labels["query"], "context": {k: labels.get(k) for k in ("scene", "assistant_role")},
            "reference_zh": annotation["text_zh"], "reference_en": annotation["text_en"], "candidate": prediction["content"]}
        require(jobs[pair["job_id"]]["input"] == expected_input, "Judge input differs from referenced case")
        require((pair["gt_start"], pair["gt_end"], pair["reply_onset"]) ==
                (annotation["start"], annotation["end"], prediction["start_media_s"]), "Judge pair timestamp mismatch")
        referenced_jobs.add(pair["job_id"])
    require(referenced_jobs == set(jobs), "Unreferenced judge jobs")
    qa = unique(load_json(target / "qa.jsonl", lines=True), "qa_id", "QA results")
    require(set(qa) == set(questions), "QA result coverage mismatch")
    for ident, record in qa.items():
        fields(record, {"model": str, "case_id": str, "qa_id": str, "raw": str,
            "gold": list, "picks": list, "valid": bool, "exact": bool}, ident)
        require(record["model"] == "minicpm-o-4.5" and record["case_id"] == questions[ident]["case_id"], f"QA result identity mismatch: {ident}")
        require(record["gold"] == questions[ident]["answer"].split(";"), f"QA reference mismatch: {ident}")
        require(all(type(p) is str for p in record["picks"]) and len(set(record["picks"])) == len(record["picks"]),
                f"Invalid parsed choices: {ident}")
    saved = unique(load_json(target / "matches.jsonl", lines=True), "case_id", "matches")
    require(set(saved) == set(cases), "Match result coverage mismatch")
    for ident, record in saved.items():
        fields(record, {"case_id": str, "failed": bool, "tolerances": dict}, ident)
        require(set(record["tolerances"]) == TOLERANCES, f"Tolerance coverage mismatch: {ident}")
        for tolerance in record["tolerances"].values():
            fields(tolerance, {"correct_nodes": int, "video_correct": bool, "matches": list}, ident)
            for match in tolerance["matches"]:
                fields(match, {"annotation_index": int, "response_index": int}, ident)
    metrics, matches = recompute(root)
    saved_metrics = load_json(target / "metrics.json")
    # Compare both directions so removed, extra, or mistyped saved fields cannot hide changes.
    same_shape(saved_metrics, metrics, "metrics")
    compare(metrics, saved_metrics)
    compare(saved_metrics, metrics)
    for record in matches:
        same_shape(saved[record["case_id"]], record, record["case_id"])
        compare(record, saved[record["case_id"]], record["case_id"])
        compare(saved[record["case_id"]], record, record["case_id"])
    return {"streaming_cases_recomputed": len(matches), "judge_jobs_verified": len(jobs), "judge_pairs_verified": len(pairs)}


def validate(root=ROOT):
    root = Path(root)
    # Include additional structured metadata in strict syntax/numeric checks.
    for base in ("dataset", "results"):
        for path in (root / base).rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                load_json(path, lines=path.suffix == ".jsonl")
    cases, annotations, qa = validate_dataset(root)
    results = validate_results(root, cases, annotations, qa)
    for base in ("evaluation",):
        for path in (root / base).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {"status": "passed", "videos": len(cases), "required_events": 555,
        "fully_silent_videos": 110, "wait_intervals": 685, "qa_questions": len(qa),
        **results, "model_calls": 0, "judge_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()
