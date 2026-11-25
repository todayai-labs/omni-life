"""Build QA prompts or score a recorded response CSV."""
import argparse
import csv
import json
from pathlib import Path

from .choice_format import parse_model_output, parse_selected


def indexed(path, key):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {row[key]: row for row in rows}
    if len(index) != len(rows) or "" in index:
        raise ValueError(f"Duplicate or blank {key} in {path}")
    return index


def load_questions(root, language):
    content = indexed(root / f"qa/qa_content_{language}.csv", "qa_id")
    timing = indexed(root / f"qa/qa_timing_{language}.csv", "qa_id")
    media = indexed(root / "qa/media-index.csv", "qa_id")
    if len(content) != 400 or len(timing) != 400 or content.keys() & timing.keys():
        raise ValueError("Expected 400 unique questions in each QA track")
    questions = content | timing
    if questions.keys() != media.keys():
        raise ValueError("Questions and the released media index disagree")
    for track, rows in (("content", content), ("timing", timing)):
        if any(media[q]["track"] != track for q in rows) or len({media[q]["case_id"] for q in rows}) != 400:
            raise ValueError(f"Unexpected QA track or scene coverage for {track}")
    return questions, media


def choices(question):
    return {k[-1]: v for k, v in question.items() if k.startswith("option_") and v}


def prompt(question, media, language):
    mode = media["choice_mode"]
    if mode not in {"single", "multiple"}:
        raise ValueError("Unknown QA choice mode")
    instructions = {
        "zh": ("只输出一个选项字母。", "输出所有正确选项的字母，并用分号分隔。"),
        "en": ("Output exactly one option letter.", "Output all correct option letters, separated by semicolons."),
    }
    parts = [question["question"], *(f"{letter}. {text}" for letter, text in choices(question).items()),
             instructions[language][mode == "multiple"]]
    return {"qa_id": question["qa_id"], "text": "\n".join(parts), "media": media}


def evaluate(questions, media, responses):
    if responses.keys() - questions.keys():
        raise ValueError("Response contains an unknown QA ID")
    outcomes = {}
    for key, q in questions.items():
        item = media[key]
        options = choices(q)
        mode = item["choice_mode"]
        expected = parse_selected(q["answer"], mode, options)
        if not expected:
            raise ValueError(f"Invalid published answer for {key}")
        verdict = None
        if key in responses:
            row = responses[key]
            status = (row.get("technical_success") or "true").strip().lower()
            if status not in {"true", "false"}:
                raise ValueError(f"Invalid technical_success for {key}")
            if status == "true":
                if "raw_output" in row:
                    observed = parse_model_output(row["raw_output"], mode, options)
                elif "selected_choices" in row:
                    observed = parse_selected(row["selected_choices"], mode, options)
                else:
                    raise ValueError("Use raw_output or selected_choices in the submission CSV")
                verdict = observed == expected
        outcomes[key] = verdict

    by_track = {track: {media[q]["case_id"]: outcomes[q] for q in questions if media[q]["track"] == track}
                for track in ("content", "timing")}
    if by_track["content"].keys() != by_track["timing"].keys():
        raise ValueError("Content and Timing scenes do not pair")
    paired = {case: None if None in (first, by_track["timing"][case]) else (first and by_track["timing"][case])
              for case, first in by_track["content"].items()}
    groups = {}
    for track, rows in (*by_track.items(), ("joint", paired)):
        values = list(rows.values())
        observed = len(values) - values.count(None)
        groups[track] = dict(total=len(values), observed=observed, correct=values.count(True),
                             accuracy=values.count(True) / len(values) if observed == len(values) else None)
    return {"groups": groups, "questions": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--question", help="Print one prompt and media record with the key omitted")
    action.add_argument("--responses", type=Path, help="CSV with QA IDs and recorded selections or raw outputs")
    args = parser.parse_args()
    questions, media = load_questions(args.dataset, args.lang)
    if args.question:
        if args.question not in questions:
            parser.error("Unknown QA ID")
        result = prompt(questions[args.question], media[args.question], args.lang)
    else:
        result = evaluate(questions, media, indexed(args.responses, "qa_id"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
