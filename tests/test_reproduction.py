"""Frozen public artifacts verify the normalization and candidate-pool protocol."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.export import normalize_qa, normalize_streaming, prepare_judge
from evaluation.reproduce import compare, read_jsonl, recompute
from evaluation.scoring import _ordered_matches, score_case


class ReproductionTests(unittest.TestCase):
    def test_filtered_and_reordered_replies_keep_semantic_credit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            directory = base / "run/example"
            directory.mkdir(parents=True)
            annotations = {"annotation": [
                {"start": 0, "end": 5, "action": "inform", "text_zh": "first"},
                {"start": 5, "end": 10, "action": "inform", "text_zh": "second"}]}
            responses = [
                {"text": ""},
                {"text": "", "onset_source": "audio_control_silence", "has_audible_audio": False},
                {"text": "[inform] second", "start_media_s": 7, "end_media_s": 8},
                {"text": "[inform] first", "start_media_s": 2, "end_media_s": 3}]
            (directory / "result.json").write_text(json.dumps(
                {"success": True, "status": "success", "responses": responses}))
            record = normalize_streaming({"case_id": "example", "duration_s": 10},
                annotations, base / "run", base / "prepared")
            self.assertEqual([p["source_index"] for p in record["predictions"]], [0, 1])
            self.assertEqual([p["content"] for p in record["predictions"]], ["first", "second"])
            jobs, pairs = prepare_judge([record], {"example": annotations})
            decisions = {j["job_id"]: int(j["input"]["candidate"] == j["input"]["reference_zh"]) for j in jobs}
            semantic = {(p["annotation_index"], p["response_index"]): decisions[p["job_id"]] for p in pairs}
            # Use the same downstream normalization and pair lookup as recompute.
            scored = score_case(annotations["annotation"], record["predictions"], 10)
            matches = _ordered_matches(scored["annotations"], scored["predictions"],
                lambda a, p: a["start"] <= p["start_media_s"] < a["end"]
                and semantic.get((a["source_index"], p["source_index"]), 0) == 1)
            self.assertEqual(matches, [(0, 0), (1, 1)])
            # Exporting never edits the native response array itself.
            self.assertEqual(responses[2]["start_media_s"], 7)

    def test_published_metrics_and_all_tolerance_matches(self):
        target = ROOT / "results/minicpm-o-4.5"
        summary, matches = recompute(ROOT)
        compare(summary, json.loads((target / "metrics.json").read_text()))
        expected = {r["case_id"]: r for r in read_jsonl(target / "matches.jsonl")}
        for row in matches:
            compare(row, expected[row["case_id"]])

    def test_rebuild_exact_frozen_judge_pool(self):
        target = ROOT / "results/minicpm-o-4.5"
        rows = read_jsonl(ROOT / "dataset/metadata/cases.jsonl")
        annotations = {r["case_id"]: json.loads((ROOT / "dataset" / r["annotation"]).read_text()) for r in rows}
        jobs, pairs = prepare_judge(read_jsonl(target / "streaming.jsonl"), annotations)
        saved_jobs = {r["job_id"]: r for r in read_jsonl(target / "judge/jobs.jsonl")}
        generated_jobs = {r["job_id"]: r for r in jobs}
        canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        self.assertEqual(len(jobs), 92)
        self.assertEqual(len(pairs), 100)
        self.assertEqual({canonical(r["input"]) for r in jobs}, {canonical(r["input"]) for r in saved_jobs.values()})
        def identity(pair, job_map):
            return (pair["case_id"], pair["annotation_index"], pair["response_index"],
                    pair["gt_start"], pair["gt_end"], pair["reply_onset"], canonical(job_map[pair["job_id"]]["input"]))
        self.assertEqual({identity(r, generated_jobs) for r in pairs},
                         {identity(r, saved_jobs) for r in read_jsonl(target / "judge/pairs.jsonl")})

    def test_missing_runs_and_partial_failure_are_not_silent_successes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            row = {"case_id": "example", "duration_s": 10}
            annotations = {"annotation": [{"start": 0, "end": 10, "action": "wait"}]}
            result = normalize_streaming(row, annotations, base / "run", base / "prepared")
            self.assertEqual((result["status"], result["predictions"]), ("failed", []))
            directory = base / "run/example"
            directory.mkdir(parents=True)
            response = {"text": "A real partial reply", "start_media_s": 2, "end_media_s": 3}
            (directory / "response-snapshots.jsonl").write_text(json.dumps({"responses": [response]}) + "\n")
            partial = normalize_streaming(row, annotations, base / "run", base / "prepared")
            self.assertEqual(partial["status"], "failed")
            self.assertEqual(len(partial["predictions"]), 1)
            question = {"qa_id": "qa_c_example", "case_id": "example", "answer": "A", "option_A": "answer"}
            missing = normalize_qa(question, base / "qa")
            self.assertEqual((missing["status"], missing["exact"]), ("failed", False))


if __name__ == "__main__":
    unittest.main()
