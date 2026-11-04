"""Release regressions that a permissive Git ignore file or score check could miss."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.check_release import MANIFEST, check


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        for name in [*self.manifest["files"], MANIFEST]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (self.root / "README.md").write_text("[Questions](dataset/qa/qa_zh.csv)\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_exported_archive_is_checked_without_git(self):
        (self.root / ".env.example").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
        result = check(self.root)
        self.assertEqual((result["frozen_files"], result["file_inventory"]), (419, "archive"))
        self.assertGreater(result["local_links"], 0)

    def test_qa_byte_change_is_rejected_even_if_the_csv_remains_readable(self):
        with (self.root / "dataset/qa/qa_zh.csv").open("ab") as handle:
            handle.write(b"\n")
        with self.assertRaisesRegex(ValueError, "Frozen input changed: dataset/qa/qa_zh.csv"):
            check(self.root)

    def test_manifest_cannot_read_outside_the_release(self):
        name = next(iter(self.manifest["files"]))
        self.manifest["files"]["../private.txt"] = self.manifest["files"].pop(name)
        (self.root / MANIFEST).write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
            check(self.root)

    def test_markdown_and_html_links_must_resolve(self):
        for text in ("[Missing](does-not-exist.md)", '<img src="missing.png">',
                     '<source srcset="missing.png 1x, also-missing.png 2x">', '[guide]: missing.md'):
            with self.subTest(text=text):
                (self.root / "README.md").write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Broken local documentation link"):
                    check(self.root)

    def test_git_tracked_env_is_blocked_even_when_gitignore_excludes_it(self):
        if shutil.which("git") is None:
            self.skipTest("Git CLI is required for this inventory test")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / ".gitignore").write_text(".env\n", encoding="utf-8")
        (self.root / ".env").write_text("EXAMPLE_SECRET=fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.root), "add", "-f", ".env"], check=True)
        with self.assertRaisesRegex(ValueError, "Environment credentials file"):
            check(self.root)

    def test_archive_does_not_hide_env_files_using_gitignore(self):
        (self.root / ".gitignore").write_text(".env\n", encoding="utf-8")
        (self.root / ".env").write_text("EXAMPLE_SECRET=fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Environment credentials file"):
            check(self.root)

    def test_private_key_header_is_rejected_under_an_innocent_filename(self):
        (self.root / "notes.txt").write_text("-----BEGIN " + "PRIVATE KEY-----\nfixture\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Private key material"):
            check(self.root)

    def test_raw_media_is_not_part_of_the_public_release(self):
        (self.root / "source.mp4").write_bytes(b"fixture, not an actual video")
        with self.assertRaisesRegex(ValueError, "Excluded media"):
            check(self.root)

    def test_symbolic_link_is_not_a_release_artifact(self):
        (self.root / "linked-questions").symlink_to(self.root / "dataset/qa/qa_zh.csv")
        with self.assertRaisesRegex(ValueError, "Symbolic link"):
            check(self.root)


if __name__ == "__main__":
    unittest.main()
