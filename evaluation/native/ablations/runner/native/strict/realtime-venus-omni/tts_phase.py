"""Input-only task-speech probe primitives. No model imports or model controls."""
from __future__ import annotations

import hashlib
import io
from itertools import chain
import json
import math
import queue
import re
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE = 16000
CONTROL = re.compile(r"(<\|[^<>|]+\|>|\[DONE\]|\[ERROR\])")
TURN_END = {"<|round_start|>", "<|round_end|>", "<|silence|>", "<|eot_id|>",
            "<|im_end|>", "[DONE]", "[ERROR]"}


def save_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def original_query(manifest):
    query = manifest.get("query")
    if (not isinstance(query, list) or len(query) != 1
            or query[0].get("start_time_sec", 0) != 0
            or not isinstance(query[0].get("text"), str) or not query[0]["text"].strip()):
        raise ValueError("Probe requires the unchanged, single initial source query")
    return query[0]["text"]


def build_tts_only_system(original_rendered, query):
    """Verify the frozen task block, then return an empty system body."""
    if not isinstance(original_rendered, str) or not isinstance(query, str) or not query:
        raise ValueError("System and original query must be nonempty strings")
    marker = "当前用户交给你的任务：\n"
    position = original_rendered.rfind(marker)
    if position < 0 or (position > 0 and original_rendered[position - 1] != "\n"):
        raise ValueError("Final explicit user-task block not found")
    suffix = original_rendered[position:]
    if suffix not in {marker + query, marker + query + "\n"}:
        raise ValueError("Final task block differs from original query; refusing broad replacement")
    generic = original_rendered[:position]
    if not generic.strip():
        raise ValueError("Generic system prompt would be empty")
    return ""  # This condition sends no natural-language initialization at all.


def save_generic_system(output, original_rendered, query):
    generic = build_tts_only_system(original_rendered, query)
    with (Path(output) / "generic-SYSTEM.txt").open("x") as handle:
        handle.write(generic)
    return generic


def failure_kind(error):
    """Classify genuine evidence, without hiding failed input as silence."""
    text = str(error or "")
    if any(fragment in text for fragment in ("Connection", "Timeout", "timed out", "CUDA out of memory",
            "CUDA error", "FileNotFoundError", "PermissionError", "OSError", "ConnectionResetError",
            "session is already active", "source_start_late", "transport")):
        return "infrastructure"
    if any(fragment in text for fragment in ("audio not enough", "maximum context",
            "max_model_len", "context length", "media transaction failed")):
        return "native_input_limit"
    if any(fragment in text for fragment in ("Final task block", "user-task block", "query differ", "coverage incomplete")):
        return "protocol_setup_or_coverage"
    return "native_unknown" if error else None


def load_task_audio(index_path, case_id, manifest):
    path = Path(index_path)
    index = json.loads(path.read_text())
    entry = dict(index["cases"][case_id])
    if entry.get("text") != original_query(manifest):
        raise ValueError("TTS text differs from the original query; no transcript fallback")
    wav_path = Path(entry["wav_file"])
    if not wav_path.is_absolute():
        wav_path = path.parent / wav_path
    wav_bytes = wav_path.read_bytes()
    if hashlib.sha256(wav_bytes).hexdigest() != entry["wav_sha256"]:
        raise ValueError(f"{case_id}: task WAV differs from the frozen audio index")
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("Task WAV must be uncompressed PCM16, 16000 Hz, mono")
        expected = handle.getnframes()
        pcm = handle.readframes(expected)
    if not pcm or len(pcm) != expected * 2:
        raise ValueError("Task WAV is empty or truncated")
    entry.update(wav_file=str(wav_path.resolve()), audio_samples=expected,
                 duration_sec=expected / SAMPLE_RATE, sample_rate=SAMPLE_RATE,
                 channels=1, sample_width_bytes=2, added_silence_samples=0,
                 text_also_present_in_system_prompt=False,
                 task_input_modality="TTS-audio-only-v2")
    return pcm, entry


class PhaseLedger:
    """Append-only wall clocks; phase ownership follows actual playback start."""
    def __init__(self, output, task_duration, clock=time.monotonic):
        self.output, self.clock = Path(output), clock
        self.duration = float(task_duration)
        if not math.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Task duration must be positive")
        self.task_start = self.source_start = None
        self.lock = threading.RLock()
        self.records = []
        for phase in ("TaskPhase", "SourcePhase"):
            (self.output / phase).mkdir(exist_ok=False)
        self.handles = {(phase, name): (self.output / phase / name).open("x")
                        for phase in ("TaskPhase", "SourcePhase")
                        for name in ("input.jsonl", "raw-output.jsonl", "runlog.jsonl")}
        self.root = {name: (self.output / name).open("x")
                     for name in ("input.jsonl", "raw-output.jsonl", "runlog.jsonl")}

    def begin_task(self):
        with self.lock:
            if self.task_start is not None:
                raise RuntimeError("Task already started")
            self.task_start = self.clock()
            self.event("runlog.jsonl", "task_playback_start", force_phase="TaskPhase")
        return self.task_start

    def begin_source(self):
        with self.lock:
            if self.task_start is None or self.source_start is not None:
                raise RuntimeError("Invalid source start")
            now = self.clock()
            deadline = self.task_start + self.duration
            if now + 1e-7 < deadline:
                raise RuntimeError("Source playback before task speech EOF")
            self.source_start = now
            self.event("runlog.jsonl", "source_playback_start", force_phase="SourcePhase",
                       planned_video_t0_monotonic_s=deadline,
                       actual_video_t0_monotonic_s=now, producer_start_lag_s=now - deadline,
                       waited_for_model_ack=False, source_clock_rebased_after_inference=False)
        return now

    def phase_at(self, now=None):
        now = self.clock() if now is None else now
        return "SourcePhase" if self.source_start is not None and now >= self.source_start else "TaskPhase"

    def event(self, filename, event, *, force_phase=None, at=None, **fields):
        with self.lock:
            now = self.clock() if at is None else at
            phase = force_phase or self.phase_at(now)
            item = {"event": event, "utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_s": now, "phase": phase,
                    "session_s": now - self.task_start if self.task_start is not None else None,
                    "source_media_s": now - self.source_start if self.source_start is not None else None,
                    **fields}
            encoded = json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
            self.root[filename].write(encoded)
            self.root[filename].flush()
            self.handles[(phase, filename)].write(encoded)
            self.handles[(phase, filename)].flush()
            self.records.append(item)
            return item

    def clock_record(self):
        return {"task_start_monotonic_s": self.task_start,
                "task_audio_duration_s": self.duration,
                "planned_video_t0_monotonic_s": (self.task_start + self.duration
                                                  if self.task_start is not None else None),
                "actual_video_t0_monotonic_s": self.source_start,
                "source_timing_policy": "independent_producer_starts_after_real_task_WAV_EOF",
                "output_phase_policy": "actual_receive_time_not_input_packet_kind"}

    def close(self):
        for handle in [*self.handles.values(), *self.root.values()]:
            handle.close()


class Responses:
    """Phase-separated views of unmodified received text, not a model turn reset."""
    def __init__(self, ledger, parse_action=None):
        self.ledger, self.rows, self.current, self.turn = ledger, [], None, 0
        self.previous_phase = None
        self.parse_action = parse_action

    def flush(self, **flags):
        if self.current is not None and self.current["text"].strip():
            self.current.update(flags)
            self.rows.append(self.current)
        self.current = None

    def add(self, text, at=None, *, ended=False, controls=False):
        now = self.ledger.clock() if at is None else at
        phase = self.ledger.phase_at(now)
        parts = CONTROL.split(str(text)) if controls else [str(text)]
        for part in parts:
            if not part:
                continue
            if controls and CONTROL.fullmatch(part):
                if part in TURN_END:
                    self.flush(native_control=part)
                    self.turn += 1
                    self.previous_phase = None
                continue
            crossed = self.current is not None and self.current["phase"] != phase
            if crossed:
                self.flush(cross_phase_turn=True)
            if self.current is None and part.strip():
                self.current = {"response_id": f"r{len(self.rows):04d}", "text": "",
                    "phase": phase, "native_turn_id": self.turn,
                    "start_monotonic_s": now, "end_monotonic_s": now,
                    "onset_source": "actual_native_text_receive",
                    "continued_from_other_phase": bool(self.previous_phase and self.previous_phase != phase)}
            if self.current is not None:
                self.current["text"] += part
                self.current["end_monotonic_s"] = now
                self.previous_phase = phase
        if ended:
            self.flush()
            self.turn += 1
            self.previous_phase = None

    def finish(self):
        self.flush(incomplete_turn_at_source_end=True)
        turns = {}
        for row in self.rows:
            turn = turns.setdefault(row["native_turn_id"], {"native_turn_id": row["native_turn_id"],
                "text": "", "projection_response_ids": [], "phases": []})
            turn["text"] += row["text"]
            turn["projection_response_ids"].append(row["response_id"])
            turn["phases"].append(row["phase"])
        for row in self.rows:
            zero = self.ledger.source_start
            row["start_media_s"] = row["start_monotonic_s"] - zero if zero is not None else None
            row["end_media_s"] = row["end_monotonic_s"] - zero if zero is not None else None
            turn = turns[row["native_turn_id"]]
            parsed = self.parse_action(turn["text"]) if self.parse_action is not None else None
            # Only explicit labels actually emitted by this same native turn are
            # carried as a parsed FIELD. Never prepend an invented tag to text.
            # The fixed scorer checks any conflict against a source delta's tag.
            if (row["phase"] == "SourcePhase" and row["continued_from_other_phase"]
                    and parsed and parsed.get("valid") and parsed.get("action") is not None):
                match = re.match(r"^\s*\[[A-Za-z_-]+\]", turn["text"])
                row["action"] = parsed["action"]
                row["action_provenance"] = {"native_turn_id": row["native_turn_id"],
                    "origin": "frozen_parse_action_valid_full_native_turn",
                    "raw_tag": match[0] if match else None, "text_projection_unmodified": True,
                    "source_onset_not_backdated": True}
            elif row["phase"] == "SourcePhase" and row["continued_from_other_phase"]:
                row["action_inheritance_rejected"] = {"native_turn_id": row["native_turn_id"],
                    "reason": parsed.get("reason") if parsed else "frozen_parser_not_supplied"}
        task = [row for row in self.rows if row["phase"] == "TaskPhase"]
        source = [row for row in self.rows if row["phase"] == "SourcePhase"]
        save_new(self.ledger.output / "TaskPhase" / "parsed-output.json", task)
        save_new(self.ledger.output / "SourcePhase" / "parsed-output.json", source)
        save_new(self.ledger.output / "parsed-output.json", source)
        save_new(self.ledger.output / "all-parsed-output.json", self.rows)
        save_new(self.ledger.output / "full-native-turns.json", list(turns.values()))
        return task, source


class PlaybackProducer(threading.Thread):
    """Unbounded release queue, independent of native generation throughput."""
    def __init__(self, ledger, pcm, source_packets, duration, *, task_packet_samples=16000,
                 first_task_packet_samples=None):
        super().__init__(name="tts-independent-causal-playback", daemon=True)
        self.ledger, self.pcm, self.source_packets = ledger, pcm, source_packets
        self.duration, self.task_step = duration, task_packet_samples * 2
        self.first_task_step = self.task_step if first_task_packet_samples is None else 2 * first_task_packet_samples
        if self.first_task_step <= 0:
            raise ValueError("FirstTask packet must be nonempty")
        self.queue, self.stop_event, self.error = queue.Queue(), threading.Event(), None
        self.source_done = threading.Event()

    def wait(self, deadline):
        while not self.stop_event.is_set():
            remaining = deadline - self.ledger.clock()
            if remaining <= 0:
                return True
            self.stop_event.wait(min(remaining, 0.05))
        return False

    def run(self):
        try:
            start = self.ledger.begin_task()
            for offset in chain((0,), range(self.first_task_step, len(self.pcm), self.task_step)):
                step = self.first_task_step if offset == 0 else self.task_step
                data = self.pcm[offset:offset + step]
                end = (offset + len(data)) // 2
                deadline = start + end / SAMPLE_RATE
                if not self.wait(deadline):
                    return
                self.ledger.event("input.jsonl", "task_audio_release", force_phase="TaskPhase",
                    start_sample=offset // 2, end_sample=end, audio_samples=len(data) // 2,
                    **({"real_task_pcm16_sha256": hashlib.sha256(data).hexdigest()}
                       if self.first_task_step != self.task_step else {}),
                    release_lag_s=max(0, self.ledger.clock() - deadline))
                self.queue.put({"kind": "task", "audio": data, "release_at": self.ledger.clock(),
                                "start_sample": offset // 2, "end_sample": end})
            # This boundary is reached by playback, never by model completion.
            zero = self.ledger.begin_source()
            self.queue.put({"kind": "source_start", "release_at": zero})
            for packet in self.source_packets():
                # Video playback releases frames at their actual source PTS,
                # even when the official native AV transaction waits for audio.
                # These are availability events, NOT extra native prefill/generate.
                for frame in packet["frames"]:
                    frame_deadline = zero + frame["pts_sec"]
                    if not self.wait(frame_deadline):
                        return
                    frame_now = self.ledger.clock()
                    self.ledger.event("input.jsonl", "source_frame_release", force_phase="SourcePhase",
                        frame=frame, source_pts_s=frame["pts_sec"],
                        release_lag_s=max(0, frame_now - frame_deadline),
                        official_AV_packet_index=packet["index"], extra_native_generation=False)
                    self.queue.put({"kind": "source_frame_available", "frame": frame,
                                    "release_at": frame_now})
                deadline = zero + packet["send_at_s"]
                if not self.wait(deadline):
                    return
                now = self.ledger.clock()
                item = {**packet, "kind": "source", "release_at": now}
                self.ledger.event("input.jsonl", "source_packet_release", force_phase="SourcePhase",
                    packet_index=packet["index"], source_start_s=packet["start_s"],
                    source_end_s=packet["end_s"], frames=packet["frames"],
                    source_audio_samples=len(packet["audio"] or b"") // 2,
                    release_lag_s=max(0, now - deadline), expected_release_s=packet["send_at_s"])
                self.queue.put(item)
            if self.wait(zero + self.duration):
                self.ledger.event("runlog.jsonl", "source_playback_eof", force_phase="SourcePhase")
                self.source_done.set()
        except BaseException as error:
            self.error = error
            self.ledger.event("runlog.jsonl", "producer_failure", error=repr(error))
        finally:
            self.queue.put(None)


class TaskTail:
    def __init__(self):
        self.pending = b""
        self.received = 0

    def hold(self, pcm):
        self.pending += pcm

    def source_audio(self, pcm, source_audio_present):
        if self.pending and not source_audio_present:
            raise RuntimeError("task_audio_tail_not_consumable_without_source_audio; no padding, discard or text fallback")
        if self.pending and pcm:
            count = len(self.pending) // 2
            combined = self.pending + pcm
            self.pending = b""
            return combined, count
        return pcm, 0
