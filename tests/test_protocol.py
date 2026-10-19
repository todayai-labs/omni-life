"""Protocol edge cases: timely semantic matching, silence, and strict QA parsing."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.output_parsing import parse_choices
from evaluation.scoring import _ordered_matches, score_case


class ProtocolTests(unittest.TestCase):
    def test_a_response_cannot_complete_two_events(self):
        events = [{"start": 0, "end": 4}, {"start": 4, "end": 8}]
        replies = [{"start_media_s": 5}]
        pairs = _ordered_matches(events, replies, lambda a, p: a["start"] <= p["start_media_s"] < a["end"] + 2)
        self.assertEqual(len(pairs), 1)

    def test_deadline_is_exclusive_and_does_not_credit_early_responses(self):
        event = [{"start": 2, "end": 4}]
        replies = [{"start_media_s": 1}, {"start_media_s": 4}]
        allowed = lambda a, p: a["start"] <= p["start_media_s"] < a["end"]
        self.assertEqual(_ordered_matches(event, replies, allowed), [])

    def test_untagged_utterance_counts_as_unwanted_output(self):
        metrics = score_case([{"start": 0, "end": 10, "action": "wait"}],
            [{"text": "Hello", "start_media_s": 3, "end_media_s": 4}], 10)
        self.assertEqual(len(metrics["predictions"]), 1)
        self.assertEqual(metrics["silence"]["false_interventions"], 1)
        self.assertEqual(metrics["silent_window_specificity"]["numerator"], 0)

    def test_qa_does_not_guess_letters_from_explanatory_prose(self):
        self.assertEqual(parse_choices("<unk>A;C<|round_end|>", "ABCD"), (["A", "C"], True))
        self.assertEqual(parse_choices("I think A or D", "ABCD"), ([], False))

    def test_qa_accepts_option_text_only_when_it_matches(self):
        self.assertEqual(parse_choices("D. Correct text", "ABCD", {"D": "Correct text"}), (["D"], True))
        self.assertEqual(parse_choices("D. Different text", "ABCD", {"D": "Correct text"}), ([], False))


if __name__ == "__main__":
    unittest.main()
