"""Release corruption checks using disposable copies of the real frozen artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.validate_release import load_json, validate, validate_dataset


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("dataset", "results"):
            shutil.copytree(ROOT / name, self.root / name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, relative, value):
        (self.root / relative).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_frozen_data_preserves_rounded_endpoints_and_late_responses(self):
        paths = [p for name in ("dataset", "results") for p in (self.root / name).rglob("*") if p.is_file()]
        before = {p: hashlib.sha256(p.read_bytes()).digest() for p in paths}
        summary = validate(self.root)
        self.assertEqual(summary["streaming_cases_recomputed"], 400)
        self.assertEqual(summary["judge_pairs_verified"], 100)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).digest() for p in paths})

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected_before_scoring(self):
        path = self.root / "bad.json"
        for content in ('{"correct": 0, "correct": 1}', '{"time": NaN}', '{"time": Infinity}', '{"time": 1e999}'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Duplicate JSON key|Non-finite"):
                    load_json(path)

    def test_duplicate_match_record_cannot_be_hidden_by_dictionary_conversion(self):
        path = self.root / "results/minicpm-o-4.5/matches.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(path.read_text(encoding="utf-8").splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate case_id: matches"):
            validate(self.root)

    def test_inconsistent_prediction_flags_are_not_hidden_by_rescoring(self):
        path = self.root / "results/minicpm-o-4.5/streaming.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        prediction = next(row["predictions"][0] for row in rows if row["predictions"])
        prediction["valid"] = not prediction["valid"]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Prediction parsing differs"):
            validate(self.root)

    def test_manifest_path_cannot_escape_dataset_root(self):
        path = self.root / "dataset/metadata/cases.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["annotation"] = "../outside.labels.json"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            validate_dataset(self.root)

    def test_unlisted_annotation_is_not_silently_released(self):
        source = next((self.root / "dataset/annotations").rglob("*.labels.json"))
        shutil.copyfile(source, source.parent / "unlisted.labels.json")
        with self.assertRaisesRegex(ValueError, "Annotation file coverage"):
            validate_dataset(self.root)

    def test_stale_statistics_are_rejected(self):
        path = "dataset/metadata/statistics.json"
        stats = json.loads((self.root / path).read_text(encoding="utf-8"))
        stats["source_types"]["real"] = 125
        self.write_json(path, stats)
        with self.assertRaisesRegex(ValueError, "statistics differ"):
            validate_dataset(self.root)

    def test_boolean_cannot_masquerade_as_an_intervention_count(self):
        path = self.root / "dataset/metadata/cases.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["intervention_events"] = False  # False == 0 in Python; it is not a valid count.
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid type: .*intervention_events"):
            validate_dataset(self.root)

    def test_invalid_interval_fails_even_when_manifest_hash_is_refreshed(self):
        path = self.root / "dataset/metadata/cases.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        annotation = self.root / "dataset" / rows[0]["annotation"]
        labels = json.loads(annotation.read_text(encoding="utf-8"))
        labels["annotation"][0]["start"] = -1
        annotation.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
        rows[0]["annotation_sha256"] = hashlib.sha256(annotation.read_bytes()).hexdigest()
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid interval"):
            validate_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
