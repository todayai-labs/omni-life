"""Small source-only replay and append-only recording helpers."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


class Recorder:
    def __init__(self, output, start):
        self.output = Path(output)
        self.start = start

    def log(self, filename, event, **fields):
        now = time.monotonic()
        record = {"utc": utc(), "monotonic_s": now, "media_wall_s": now - self.start,
                  "event": event, **fields}
        with (self.output / filename).open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record


def packets(prepared, chunk_s=1.0):
    """Audio released at segment end; frames only when their PTS is observable.

    Missing audio is represented by None, not by fabricated zero samples.
    The real decoded partial final packet is retained without padding.
    """
    prepared = Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    audio = (prepared / manifest["audio_file"]).read_bytes() if manifest["audio_present"] else None
    frames = manifest["frames"]
    if audio is None:
        for index, frame in enumerate(frames):
            yield {"index": index, "send_at_s": frame["pts_sec"], "start_s": frame["pts_sec"],
                   "end_s": frame["pts_sec"], "audio": None, "frames": [frame]}
        return
    sample_step = round(chunk_s * 16000)
    count = len(audio) // 2
    cursor = 0
    for index, start in enumerate(range(0, count, sample_step)):
        end = min(count, start + sample_step)
        chosen = []
        while cursor < len(frames) and frames[cursor]["pts_sec"] <= end / 16000:
            chosen.append(frames[cursor])
            cursor += 1
        yield {"index": index, "send_at_s": end / 16000, "start_s": start / 16000,
               "end_s": end / 16000, "audio": audio[start * 2:end * 2], "frames": chosen}
    while cursor < len(frames):
        frame = frames[cursor]
        cursor += 1
        yield {"index": index + cursor, "send_at_s": frame["pts_sec"], "start_s": frame["pts_sec"],
               "end_s": frame["pts_sec"], "audio": None, "frames": [frame]}


def wait_release(start, at):
    remaining = start + at - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
