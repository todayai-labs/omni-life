#!/usr/bin/env python3
"""Prepare original video/audio for causal replay, without synthesized audio."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.reproduce import read_jsonl


def digest(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def prepare(row, videos, output, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    source = (Path(videos) / row["video"]).resolve()
    if not source.is_relative_to(Path(videos).resolve()):
        raise ValueError("Video path escapes video root")
    if digest(source) != row["video_sha256"]:
        raise ValueError("Video identity mismatch: " + row["case_id"])
    labels = json.loads((ROOT / "dataset" / row["annotation"]).read_text())
    query = labels["query"]
    if len(query) != 1 or query[0]["start_time_sec"] != 0:
        raise ValueError("Expected exactly one task instruction at time zero")
    dest = Path(output) / row["case_id"]
    dest.mkdir(parents=True, exist_ok=False)
    system = (ROOT / "prompts/streaming.txt").read_text().replace("{{USER_REQUEST}}", query[0]["text"])
    (dest / "SYSTEM.txt").write_text(system, encoding="utf-8")
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)], text=True))
    audio_present = any(s.get("codec_type") == "audio" for s in probe["streams"])
    audio_path = dest / "audio_16k_mono.s16le"
    if audio_present:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source), "-vn", "-ac", "1", "-af", "aresample=16000:async=1:first_pts=0", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", str(audio_path)], check=True)
    proc = subprocess.run([ffmpeg, "-hide_banner", "-i", str(source), "-an", "-vf", "select='isnan(prev_selected_t)+gte(t-prev_selected_t,1)',scale=640:-2,showinfo", "-vsync", "vfr", "-q:v", "5", str(dest / "frame_%06d.jpg")], capture_output=True, text=True, check=True)
    times = [float(t) for t in re.findall(r"\bn:\s*\d+.*?\bpts_time:([\d.eE+-]+)", proc.stderr)]
    frames = sorted(dest.glob("frame_*.jpg"))
    if not frames or len(frames) != len(times):
        raise ValueError("Frame timestamp/file mismatch")
    manifest = {
        "id": row["case_id"], "source_video_sha256": row["video_sha256"],
        "duration_sec": float(probe["format"]["duration"]), "system_file": "SYSTEM.txt",
        "system_sha256": digest(dest / "SYSTEM.txt"), "audio_present": audio_present,
        "audio_file": audio_path.name if audio_present else None,
        "audio_samples": audio_path.stat().st_size // 2 if audio_present else 0,
        "audio_sha256": digest(audio_path) if audio_present else None,
        "synthetic_audio_added": False, "sample_rate_hz": 16000, "channels": 1,
        "frames": [{"pts_sec": t, "file": p.name, "sha256": digest(p)} for t, p in zip(times, frames)]}
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Root containing the relative video paths in cases.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", help="Select one or more cases; default: all 400")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without extracting media")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "dataset/metadata/cases.jsonl")
    if args.case_id:
        rows = [r for r in rows if r["case_id"] in args.case_id]
        if {r["case_id"] for r in rows} != set(args.case_id):
            parser.error("Unknown case ID")
    for row in rows:
        if args.dry_run:
            source = args.videos / row["video"]
            if digest(source) != row["video_sha256"]:
                raise ValueError("Video identity mismatch: " + row["case_id"])
            print(json.dumps({"case_id": row["case_id"], "input_verified": True}))
        else:
            manifest = prepare(row, args.videos, args.output, args.ffmpeg, args.ffprobe)
            print(json.dumps({"case_id": row["case_id"], "frames": len(manifest["frames"]), "audio_samples": manifest["audio_samples"]}))


if __name__ == "__main__":
    main()
