"""Offline release gates must not infer rights from a decision label alone."""
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.check_media_rights import check_media_rights


PENDING = {"rights_status": "not_documented", "privacy_status": "not_documented", "decision": "pending", "evidence_ref": None}
REVIEWED = {"rights_status": "reviewed", "privacy_status": "reviewed", "decision": "cleared", "evidence_ref": "review-offline-test-record"}


class MediaRightsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        metadata = self.root / "dataset/metadata"
        metadata.mkdir(parents=True)
        cases = [{"case_id": "case-%03d" % i, "source_type": "synthetic", "video_sha256": "%064x" % i} for i in range(400)]
        (metadata / "cases.jsonl").write_text("\n".join(json.dumps(row) for row in cases) + "\n")
        asset = self.root / "assets/readme/example.png"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"offline media fixture")
        self.registry = {"schema_version": 1, "videos": [row | PENDING for row in cases],
            "repository_media": [{"path": "assets/readme/example.png", "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(), "contains_source_frames": True} | PENDING]}
        (self.root / "README.md").write_text('![Example](assets/readme/example.png)\n')
        self.save()

    def save(self):
        (self.root / "dataset/metadata/media-rights.json").write_text(json.dumps(self.registry))

    def test_pending_original_videos_do_not_block_structure_check(self):
        result = check_media_rights(self.root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["videos_pending"], 400)
        self.assertEqual(result["repository_media_pending"], 1)
        self.assertEqual(check_media_rights(self.root, require_cleared=True)["status"], "blocked")
        self.registry["repository_media"][0].update(REVIEWED)
        self.save()
        self.assertEqual(check_media_rights(self.root, require_cleared=True)["status"], "passed")

    def test_requested_pending_or_unknown_video_is_refused(self):
        self.assertEqual(check_media_rights(self.root, case_ids=["case-000"])["status"], "blocked")
        self.assertEqual(check_media_rights(self.root, case_ids=["unknown"])["status"], "blocked")
        self.registry["videos"][0].update(REVIEWED)
        self.save()
        self.assertEqual(check_media_rights(self.root, case_ids=["case-000"])["status"], "passed")
        self.assertEqual(check_media_rights(self.root)["videos_pending"], 399)

    def test_cleared_requires_both_reviews_and_nonempty_opaque_evidence(self):
        for target in (self.registry["videos"][0], self.registry["repository_media"][0]):
            for update in ({"decision": "cleared"}, REVIEWED | {"evidence_ref": None},
                           REVIEWED | {"evidence_ref": "https://private.example/evidence"},
                           REVIEWED | {"privacy_status": "not_documented"},
                           REVIEWED | {"rights_status": "not_documented"}):
                with self.subTest(update=update):
                    target.update(PENDING | update)
                    self.save()
                    self.assertEqual(check_media_rights(self.root)["status"], "blocked")
            target.update(PENDING)

    def test_new_unregistered_image_and_readme_reference_are_blocked(self):
        asset = self.root / "docs/new.png"
        asset.parent.mkdir()
        asset.write_bytes(b"unreviewed")
        self.assertIn("Unregistered repository media: docs/new.png", check_media_rights(self.root)["errors"])
        (self.root / "README.md").write_text('<picture><source srcset="assets/readme/example.png 1x, docs/new.png 2x"><img src="assets/readme/example.png"></picture>')
        self.assertIn("README references unregistered media: README.md", check_media_rights(self.root)["errors"])

    def test_force_added_ignored_media_is_still_checked(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True, capture_output=True)
        (self.root / ".gitignore").write_text("private/\n")
        private = self.root / "private"
        private.mkdir()
        (private / "local.png").write_bytes(b"local only")
        (private / "forced.png").write_bytes(b"staged for publication")
        subprocess.run(["git", "-C", str(self.root), "add", "--force", "private/forced.png"], check=True, capture_output=True)
        result = check_media_rights(self.root)
        self.assertIn("Unregistered repository media: private/forced.png", result["errors"])
        self.assertNotIn("Unregistered repository media: private/local.png", result["errors"])

    def test_remote_and_reference_style_images_are_checked(self):
        for text in ('![Remote](https://example.com/figure.png)',
                     '![Figure][figure]\n\n[figure]: https://example.com/figure.png',
                     '![Figure][]\n\n[Figure]: https://example.com/figure.png',
                     '![Figure]\n\n[Figure]: https://example.com/figure.png',
                     '<img src="https://example.com/figure.png">'):
            with self.subTest(text=text):
                (self.root / "README.md").write_text(text)
                self.assertEqual(check_media_rights(self.root)["status"], "blocked")

    def test_case_coverage_and_hash_changes_are_blocked(self):
        self.registry["videos"].pop()
        self.save()
        self.assertIn("Video registry must cover all 400 cases exactly once", check_media_rights(self.root)["errors"])
        self.registry["videos"][0]["video_sha256"] = "f" * 64
        self.save()
        self.assertIn("Video identity mismatch: case-000", check_media_rights(self.root)["errors"])
        (self.root / "assets/readme/example.png").write_bytes(b"replacement content")
        self.assertIn("Repository media hash mismatch: assets/readme/example.png", check_media_rights(self.root)["errors"])

    def test_brand_exception_cannot_be_applied_to_video_frames(self):
        row = self.registry["repository_media"][0]
        row.update(REVIEWED | {"decision": "repository_presentation_only"})
        self.save()
        self.assertEqual(check_media_rights(self.root, require_cleared=True)["status"], "blocked")
        row["contains_source_frames"] = False
        self.save()
        self.assertEqual(check_media_rights(self.root, require_cleared=True)["status"], "blocked")

    def test_registered_brand_presentation_scope_is_accepted(self):
        row = self.registry["repository_media"][0]
        row.update(REVIEWED)
        hero = self.root / "assets/readme/hero.svg"
        hero.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        self.registry["repository_media"].append({"path": "assets/readme/hero.svg",
            "sha256": hashlib.sha256(hero.read_bytes()).hexdigest(), "contains_source_frames": False} |
            REVIEWED | {"decision": "repository_presentation_only"})
        self.save()
        result = check_media_rights(self.root, require_cleared=True)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["presentation_only_media"], 1)


if __name__ == "__main__":
    unittest.main()
