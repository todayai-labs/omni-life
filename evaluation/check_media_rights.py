#!/usr/bin/env python3
"""Check media review records; this gate records decisions, never grants rights."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".svg", ".bmp", ".tif", ".tiff", ".ico", ".pdf",
                  ".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}
LOCAL_ONLY_DIRS = {".git", ".venv", "venv", "__pycache__", ".cache", ".pytest_cache"}
BRAND_PATHS = {"assets/readme/hero.svg", "assets/readme/hero-mobile.svg"}
REVIEW_FIELDS = {"rights_status", "privacy_status", "decision", "evidence_ref"}


def read_json(path):
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def review_valid(record, allow_brand=False):
    if any(record.get(key) not in {"not_documented", "reviewed"} for key in ("rights_status", "privacy_status")):
        return False
    decision = record.get("decision")
    if decision not in ({"pending", "cleared", "repository_presentation_only"} if allow_brand else {"pending", "cleared"}):
        return False
    evidence = record.get("evidence_ref")
    if evidence is not None and (not isinstance(evidence, str) or not re.fullmatch(r"review-[A-Za-z0-9_-]{8,64}", evidence)):
        return False
    return decision == "pending" or (record["rights_status"] == record["privacy_status"] == "reviewed" and evidence is not None)


class ImageReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"img", "source", "video", "audio"}:
            self.references.extend(attrs[key] for key in ("src", "poster") if attrs.get(key))
            self.references.extend(item.strip().split()[0] for item in attrs.get("srcset", "").split(",") if item.strip())


def readme_references(path):
    text = path.read_text(encoding="utf-8")
    parser = ImageReferences()
    parser.feed(text)
    references = parser.references
    references.extend(a or b for a, b in re.findall(r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))", text))
    definitions = {name.strip().lower(): a or b for name, a, b in re.findall(r"(?m)^\s*\[([^\]]+)\]:\s*(?:<([^>]+)>|(\S+))", text)}
    for label, name in re.findall(r"!\[([^\]]*)\]\[([^\]]*)\]", text):
        references.append(definitions.get((name or label).strip().lower(), "unresolved-reference:"))
    for label in re.findall(r"!\[([^\]]+)\](?![\[(])", text):
        if label.strip().lower() in definitions:
            references.append(definitions[label.strip().lower()])
    return references


def repository_files(root):
    if (root / ".git").exists():
        # Include force-added ignored files while excluding private local runs.
        result = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                check=True, capture_output=True)
        return [root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    return [path for path in root.rglob("*") if path.is_file() and not LOCAL_ONLY_DIRS.intersection(path.relative_to(root).parts)]


def check_media_rights(root, require_cleared=False, case_ids=()):
    root = Path(root).resolve()
    errors = []
    registry = read_json(root / "dataset/metadata/media-rights.json")
    if not isinstance(registry, dict) or set(registry) != {"schema_version", "videos", "repository_media"} or registry["schema_version"] != 1:
        raise ValueError("Invalid media-rights registry structure")
    cases = [json.loads(line) for line in (root / "dataset/metadata/cases.jsonl").read_text().splitlines() if line.strip()]
    case_map = {row["case_id"]: row for row in cases}
    if len(cases) != 400 or len(case_map) != 400:
        raise ValueError("Expected exactly 400 unique metadata cases")
    videos, media = registry["videos"], registry["repository_media"]
    if not isinstance(videos, list) or not isinstance(media, list):
        raise ValueError("Media registry collections must be lists")
    video_map = {}
    for row in videos:
        if not isinstance(row, dict) or set(row) != REVIEW_FIELDS | {"case_id", "source_type", "video_sha256"}:
            raise ValueError("Invalid video review fields")
        ident = row["case_id"]
        if ident in video_map or ident not in case_map:
            raise ValueError("Duplicate or unknown video case ID")
        video_map[ident] = row
        if any(row[key] != case_map[ident][key] for key in ("source_type", "video_sha256")):
            errors.append("Video identity mismatch: " + ident)
        if not review_valid(row):
            errors.append("Invalid or unsupported review decision: " + ident)
    if set(video_map) != set(case_map):
        errors.append("Video registry must cover all 400 cases exactly once")
    registered = set()
    for row in media:
        if not isinstance(row, dict) or set(row) != REVIEW_FIELDS | {"path", "sha256", "contains_source_frames"}:
            raise ValueError("Invalid repository media review fields")
        rel = row["path"]
        if not isinstance(rel, str) or "\\" in rel or Path(rel).is_absolute() or ".." in Path(rel).parts or Path(rel).as_posix() != rel or rel in registered:
            raise ValueError("Invalid or duplicate repository media path")
        registered.add(rel)
        path = root / rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            errors.append("Missing or unsafe registered media: " + rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("Repository media hash mismatch: " + rel)
        brand = rel in BRAND_PATHS and row["contains_source_frames"] is False
        if not isinstance(row["contains_source_frames"], bool) or not review_valid(row, allow_brand=brand):
            errors.append("Invalid or unsupported review decision: " + rel)
        if require_cleared and row["decision"] not in {"cleared", "repository_presentation_only"}:
            errors.append("Repository media review pending: " + rel)
    for path in repository_files(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in MEDIA_SUFFIXES and rel not in registered:
            errors.append("Unregistered repository media: " + rel)
        if path.name.lower().startswith("readme") and path.suffix.lower() == ".md":
            for reference in readme_references(path):
                parsed = urlsplit(reference)
                if parsed.scheme or parsed.netloc:
                    errors.append("README uses unregistered remote or inline media: " + rel)
                    continue
                target = (path.parent / unquote(parsed.path)).resolve()
                if not target.is_relative_to(root) or target.relative_to(root).as_posix() not in registered:
                    errors.append("README references unregistered media: " + rel)
    for ident in case_ids:
        record = video_map.get(ident)
        if record is None or record["decision"] != "cleared" or not review_valid(record):
            errors.append("Requested video is not cleared: " + ident)
    return {"status": "passed" if not errors else "blocked", "registered_videos": len(videos),
            "videos_pending": sum(row["decision"] == "pending" for row in videos),
            "registered_repository_media": len(media), "repository_media_pending": sum(row["decision"] == "pending" for row in media),
            "presentation_only_media": sum(row["decision"] == "repository_presentation_only" for row in media),
            "require_cleared": require_cleared, "requested_cases": list(case_ids), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cleared", action="store_true", help="Require review clearance for media included in this repository")
    parser.add_argument("--case-id", action="append", default=[], help="Require clearance for a requested source video; does not grant access")
    args = parser.parse_args()
    try:
        result = check_media_rights(ROOT, args.require_cleared, args.case_id)
    except (ValueError, TypeError, KeyError, OSError, subprocess.SubprocessError):
        result = {"status": "blocked", "errors": ["Invalid or unreadable media registry, metadata, or README reference"]}
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
