#!/usr/bin/env python3
"""Check frozen artifact hashes, release-file hygiene, and local documentation links."""
import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "dataset/metadata/release-manifest.json"
BLOCKED = set(".pem .key .p12 .pfx .keystore .mp4 .mov .mkv .webm .avi .m4v .wav .mp3 .flac .pcm .s16le .aac .ogg .m4a .safetensors .bin .pt .pth .ckpt .gguf .onnx .h5 .hdf5 .tflite .pkl .pickle .zip .tar .gz .tgz .bz2 .xz .7z .rar .db .sqlite .sqlite3".split())
CACHE_DIRS = {".git", "__pycache__", ".venv", "venv", ".cache", ".pytest_cache"}
PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name
            and ":" not in name and "\x00" not in name and str(path) == name, "Unsafe manifest path")
    target = root / name
    require(target.resolve().is_relative_to(root.resolve()), "Manifest path escapes release root")
    return target


def manifest_files(root):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate release-manifest key")
            result[key] = value
        return result
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    require(type(manifest) is dict and manifest.get("schema_version") == 1, "Invalid release manifest")
    files = manifest.get("files")
    require(type(files) is dict and len(files) == 419 and manifest.get("file_count") == 419,
            "Expected 419 frozen release inputs")
    require(manifest.get("base_commit") == "b099f9d763f333505d623259dc09d4e3d0cc3f58", "Unexpected release baseline")
    for name, digest in files.items():
        require(type(name) is str and type(digest) is str and re.fullmatch(r"[a-f0-9]{64}", digest), "Invalid manifest identity")
        target = safe_path(root, name)
        require(target.is_file() and not target.is_symlink(), f"Missing or linked frozen input: {name}")
        require(hashlib.sha256(target.read_bytes()).hexdigest() == digest, f"Frozen input changed: {name}")
    return files


def public_files(root):
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                capture_output=True, check=True)
        return sorted({root / name for name in result.stdout.decode("utf-8").split("\0") if name})
    # An exported archive has no Git index; inspect its contents except tool caches.
    return sorted(path for path in root.rglob("*") if (path.is_file() or path.is_symlink())
                  and not set(path.relative_to(root).parts) & CACHE_DIRS)


def local_links(text):
    text = re.sub(r"(?ms)^\s*(`{3}|~{3})[^\n]*\n.*?^\s*\1[^\n]*$", "", text)
    text = re.sub(r"`+[^`]*`+", "", text)
    yield from (m.group(1).strip("<>") for m in re.finditer(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)", text))
    yield from (m.group(1).strip("<>") for m in re.finditer(r"(?m)^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", text))
    for match in re.finditer(r"\b(href|src|srcset)\s*=\s*(['\"])(.*?)\2", text, re.I | re.S):
        if match.group(1).lower() == "srcset":
            yield from (part.strip().split()[0] for part in match.group(3).split(",") if part.strip())
        else:
            yield match.group(3)


def check(root=ROOT):
    root = Path(root).resolve()
    frozen = manifest_files(root)
    files = public_files(root)
    public = {path.relative_to(root).as_posix() for path in files}
    require(set(frozen) <= public, "Frozen inputs missing from the release file set")
    links = 0
    for path in files:
        name = path.relative_to(root).as_posix()
        require(not path.is_symlink(), f"Symbolic link in release: {name}")
        require(path.is_file(), f"Missing tracked file: {name}")
        require(not (path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example"),
                f"Environment credentials file in release: {name}")
        require(path.suffix.lower() not in BLOCKED, f"Excluded media, model, key, archive, or database: {name}")
        require(not PRIVATE_KEY.search(path.read_bytes()), f"Private key material in release: {name}")
        if path.suffix.lower() not in {".md", ".markdown", ".html"}:
            continue
        for value in local_links(path.read_text(encoding="utf-8")):
            url = urlsplit(html.unescape(value))
            if url.scheme or url.netloc or not url.path:
                continue
            require(not url.path.startswith("/"), f"Absolute local documentation link: {name}")
            target = (path.parent / unquote(url.path)).resolve()
            require(target.is_relative_to(root), f"Documentation link escapes release root: {name}")
            require(target.exists(), f"Broken local documentation link: {name} -> {value}")
            relative = target.relative_to(root).as_posix()
            require(relative in public or target.is_dir() and any(p.startswith(relative + "/") for p in public),
                    f"Documentation target is not included in release: {name} -> {value}")
            links += 1
    return {"status": "passed", "frozen_files": len(frozen), "public_files": len(files), "local_links": links,
            "file_inventory": "git" if (root / ".git").exists() else "archive"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(check(), indent=2))


if __name__ == "__main__":
    main()
