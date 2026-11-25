"""Recompute the main results from the released QA answers and Streaming verdicts."""
import argparse
import csv
import json
import sys
from pathlib import Path

from .choice_format import parse_model_output, parse_selected



def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def records(path, key):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} in {path}")
    return result


def score(directory, questions, media, scenes):
    answers = records(directory / "qa.jsonl", "qa_id")
    runs = records(directory / "streaming.jsonl", "case_id")
    assert answers.keys() == questions.keys() and runs.keys() == scenes
    hits = {"content": set(), "timing": set()}
    for key, answer in answers.items():
        question = questions[key]
        item = media[key]
        assert (answer["case_id"], answer["track"]) == (item["case_id"], item["track"])
        options = {k[-1]: v for k, v in question.items() if k.startswith("option_") and v}
        mode = item["choice_mode"]
        selected = parse_model_output(answer["raw_output"], mode, options)
        gold = parse_selected(question["answer"], mode, options)
        assert gold and (selected == gold) == answer["correct"]
        if selected == gold:
            hits[item["track"]].add(item["case_id"])
    originals, combined = [], []
    for row in runs.values():
        verdict = row["initial_full_correct"]
        supplement = None
        for item in row["additional_records"]:
            if item["route"] == "original":
                assert verdict is None
                verdict = item["full_correct"]
            elif supplement is None:
                supplement = item["full_correct"]
        final = supplement if verdict is None else verdict
        assert verdict is row["original_full_correct"] and final is row["combined_full_correct"]
        originals.append(verdict)
        combined.append(final)
    result = dict(model=directory.name, videos=len(scenes))
    for name, values in (("original", originals), ("combined", combined)):
        result[name+"_correct"] = values.count(True)
        result[name+"_incorrect"] = values.count(False)
        result[name+"_undetermined"] = values.count(None)
        result[name+"_full_correct_pct"] = 100 * values.count(True) / len(scenes)
    result["supplemental_decisions"] = originals.count(None) - combined.count(None)
    for name, cases in (*hits.items(), ("joint", hits["content"] & hits["timing"])):
        result[name+"_correct"] = len(cases)
        result[name+"_pct"] = 100 * len(cases) / len(scenes)
    return result


def check_paper_table(scores, results):
    names = {"qwen3.8-omni-flash-realtime":"Qwen3.8 Flash","qwen3.5-omni-plus-realtime":"Qwen3.5 Plus","qwen3.5-omni-flash-realtime":"Qwen3.5 Flash","minicpm-o-4.5-local":"MiniCPM-o 4.5","gander":"Gander","moss-vl-realtime":"MOSS-VL","realtime-venus-omni":"Venus-Omni","duplexomni":"DuplexOmni (S1)","aura":"AURA","moss-preview-realtime":"MOSS Preview","joyai-vl-interaction":"JoyAI"}
    expected = {row["Respondent"]: list(row.values())[1:] for row in csv_rows(results / "paper-main.csv")}
    assert set(expected) == set(names.values())
    for score_row in scores:
        model = score_row["model"]
        cases = records(results / model / "streaming.jsonl", "case_id").values()
        counts = [r["paper_event_counts"] for r in cases]
        required = sum(r["required"] for r in counts)
        credited = sum(r["credited"] for r in counts)
        extra = sum(r["unmatched_logged_responses"] for r in counts)
        silent = [r for r in cases if not r["paper_event_counts"]["required"]]
        assert required == 500 and len(silent) == 103
        def success(rows):
            return 100 * sum(r["initial_full_correct"] is True for r in rows) / len(rows)
        values = [success(cases), 100 * credited / required, 200 * credited / (required + credited + extra),
                  success(silent), score_row["joint_pct"], score_row["content_pct"], score_row["timing_pct"]]
        assert [round(v, 2) for v in values] == [float(v) for v in expected[names[model]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare every recomputed cell with summary.csv")
    parser.add_argument("--dataset", required=True, type=Path, help="Local Hugging Face dataset snapshot")
    args = parser.parse_args()
    root = args.dataset
    results = root / "results" / "main"
    scenes = {r["case_id"] for r in csv_rows(root / "scenes.csv")}
    questions = {r["qa_id"]: r for track in ("content","timing")
                 for r in csv_rows(root / "qa" / f"qa_{track}_zh.csv")}
    media = {r["qa_id"]: r for r in csv_rows(root / "qa/media-index.csv")}
    assert len(scenes) == 400 and len(questions) == len(media) == 800
    scores = [score(path.parent, questions, media, scenes) for path in sorted(results.glob("*/qa.jsonl"))]
    assert len(scores) == 11 and all(row["combined_undetermined"] == 0 for row in scores)
    if args.check:
        check_paper_table(scores, results)
        expected = csv_rows(results / "summary.csv")
        assert expected == [{k: str(v) for k, v in row.items()} for row in scores]
        print("Verified 8,800 QA answers, 4,400 Streaming case rows and every summary cell.")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)


if __name__ == "__main__":
    main()
