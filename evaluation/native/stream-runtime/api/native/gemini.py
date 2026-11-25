"""Native Gemini Live evaluation of frozen Omni-Life audio/video inputs.

Credentials exist only in process memory. Each phase has independent probe and
formal runs; successful identities are never repeated. Main time/content scores
are computed offline, not inferred from the optional tag diagnostic scorer.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import fcntl
import getpass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import queue
import re
import signal
import sys
import tempfile
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import websockets

ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
CONTROL = re.compile(r'<(?:unk|\|(?:eot_id|round_end|im_end|endoftext|end_of_text|begin_of_text|im_start|start_header_id|end_header_id)\|)>')
STOP = None
CONNECTION_GATE = None


class StopRequested(RuntimeError):
    pass


class ConnectionGate:
    """One shared, signal-aware connection-rate and quota cooldown gate."""
    def __init__(self, minimum_interval=1.0, max_quota_events=0):
        self.minimum_interval = minimum_interval
        self.next_allowed = 0.0
        self.lock = asyncio.Lock()
        self.quota_events = 0
        self.max_quota_events = max_quota_events

    async def penalize(self, retry=0, *, count_event=True):
        async with self.lock:
            if count_event:
                self.quota_events += 1
                if self.max_quota_events and self.quota_events >= self.max_quota_events:
                    STOP.set()
                    print(json.dumps({'utc': utc(), 'event': 'quota_guard_stopped', 'quota_events': self.quota_events, 'reason': 'limited single-session quota validation exhausted; no further model calls'}), flush=True)
            delay = min(90.0, 45.0 * (retry + 1))
            self.next_allowed = max(self.next_allowed, time.monotonic() + delay)
            return delay

    async def wait(self, stop):
        while not stop.is_set():
            async with self.lock:
                delay = self.next_allowed - time.monotonic()
                if delay <= 0:
                    self.next_allowed = time.monotonic() + self.minimum_interval
                    return
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
        raise StopRequested("Graceful stop before opening the next native session")


class DurableLogs:
    """A dedicated disk thread preserves ordered, flushed raw/input records.

    Only the thread accesses the file handles. Slow shared-filesystem writes do
    not stall any source playback clock. A failed writer retains undelivered
    records in a local recovery journal and raises, never silently drops them.
    """
    def __init__(self, attempt):
        self.attempt = attempt
        self.pending = queue.Queue(maxsize=8192)
        self.error = None
        self.recovery = None
        self.overflow = []
        self.files = {name: (attempt / name).open("x", encoding="utf-8")
                      for name in ("input-events.jsonl", "raw-output.jsonl")}
        self.thread = threading.Thread(target=self._write, name="gemini-durable-logs", daemon=True)
        self.thread.start()

    def _write(self):
        fallback = None
        try:
            while True:
                item = self.pending.get()
                try:
                    if item is None:
                        break
                    channel, line = item
                    if self.error is None:
                        try:
                            self.files[channel].write(line)
                            self.files[channel].flush()
                            continue
                        except Exception as error:
                            self.error = error
                    if fallback is None:
                        directory = Path(tempfile.mkdtemp(prefix="omnilife-gemini-io-recovery-"))
                        self.recovery = str(directory / "undelivered.jsonl")
                        fallback = Path(self.recovery).open("x", encoding="utf-8")
                    fallback.write(json.dumps({"attempt": str(self.attempt), "channel": channel,
                                               "record": json.loads(line)}, ensure_ascii=False) + "\n")
                    fallback.flush()
                finally:
                    self.pending.task_done()
        finally:
            if fallback is not None:
                fallback.close()
            for handle in self.files.values():
                handle.close()

    def emit(self, channel, value):
        line = json.dumps(value, ensure_ascii=False) + "\n"
        try:
            self.pending.put_nowait((channel, line))
        except queue.Full:
            self.overflow.append((channel, line))
            raise RuntimeError("Durable log queue exhausted; stop transport rather than lose records")

    def check(self):
        if self.error is not None:
            raise RuntimeError("Durable log write failed; recovery=%s: %s" % (self.recovery, self.error))

    def finish(self):
        self.pending.put(None)
        self.pending.join()
        self.thread.join()
        if self.overflow:
            directory = Path(tempfile.mkdtemp(prefix="omnilife-gemini-io-overflow-"))
            self.recovery = str(directory / "undelivered.jsonl")
            with Path(self.recovery).open("x", encoding="utf-8") as handle:
                for channel, line in self.overflow:
                    handle.write(json.dumps({"attempt": str(self.attempt), "channel": channel,
                                             "record": json.loads(line)}, ensure_ascii=False) + "\n")
                handle.flush()
            self.error = RuntimeError("Log queue overflow preserved in local recovery journal")
        self.check()


async def save_async(path, value):
    await asyncio.to_thread(save, path, value)


def read_preloaded_media(prepared, manifest):
    frames = {frame["file"]: (prepared / frame["file"]).read_bytes() for frame in manifest["frames"]}
    pcm = (prepared / manifest["audio_file"]).read_bytes() if manifest["audio_present"] else b""
    return frames, pcm, digest(prepared / "manifest.json")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def clean_native_output(raw):
    text = unicodedata.normalize("NFKC", raw if isinstance(raw, str) else "")
    text = CONTROL.sub("", text).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*\n?", "", text)
        text = text[:-3].strip()
    return text


def parse_choices(raw, allowed, options=None):
    """Same explicit-choice parser as benchmark evaluation/output_parsing.py."""
    text = re.sub(r"<think>.*?</think>", "", clean_native_output(raw), flags=re.S).strip().upper()
    text = re.sub(r"^(?:答案|ANSWER)\s*[:：]\s*", "", text)
    if not re.fullmatch(r"[A-L](?:\s*[;,，；、\s]\s*[A-L])*[。.]?", text):
        match = re.fullmatch(r"([A-L])[.．、：:]\s*(.+)", text, flags=re.S)
        norm = lambda s: re.sub(r"\s+", "", clean_native_output(s)).rstrip("。.").upper()
        if match and options and match[1] in options and norm(match[2]) == norm(options[match[1]]):
            return [match[1]], True
        return [], False
    picks = sorted(set(re.findall("[A-L]", text)))
    return picks, bool(picks and set(picks) <= set(allowed))


def prefix_cutoff(question):
    stem = re.split(r"\n\s*(?:题目：|Question:)", question, flags=re.I)[-1]
    match = re.search(r"第\s*([\d.]+)\s*秒", stem)
    if not match:
        match = re.search(r"\b(?:video|clip)\s+is\s+at\s+([\d.]+)\s*(?:seconds?|s)\b", stem, re.I)
    return float(match[1]) if match else None


def render_system(template, query):
    if len(query) != 1 or query[0].get("start_time_sec") != 0:
        raise ValueError("Exactly one t=0 user task is required")
    if template.count("{{USER_REQUEST}}") != 1:
        raise ValueError("Final SP must contain exactly one USER_REQUEST placeholder")
    return template.replace("{{USER_REQUEST}}", query[0]["text"])


def select_probe(cases, labels):
    """Deterministic coverage of warning, feedback, answer, and silent cases."""
    selected = []
    for desired in ("warn", "feedback", "answer", "wait"):
        for ident in sorted(cases):
            actions = {str(x.get("action", "")).strip("[]") for x in labels[ident]["annotation"]}
            matches = actions <= {"wait"} if desired == "wait" else desired in actions
            if matches and ident not in selected:
                selected.append(ident)
                break
    for ident in sorted(cases):
        if len(selected) >= 3:
            break
        if ident not in selected:
            selected.append(ident)
    return selected


def load_source(args):
    labels = {}
    source = {}
    for path in sorted(args.data.rglob("*.labels.json")):
        ident = path.stem.removesuffix(".labels")
        if ident in labels:
            raise ValueError("Duplicate source identity: " + ident)
        gt = json.loads(path.read_text(encoding="utf-8"))
        if gt["id"] != ident:
            raise ValueError("Source filename/id mismatch: " + ident)
        labels[ident] = gt
        source[ident] = {"id": ident, "labels": str(path), "labels_sha256": digest(path),
                         "manifest": str(args.prepared / ident / "manifest.json")}
    if len(source) != 400:
        raise ValueError("Expected exactly 400 final source identities")
    rows = []
    if args.phase == "qa":
        if not args.qa_csv:
            raise ValueError("--qa-csv is required for QA")
        with args.qa_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("lang", "zh") == "zh"]
        if len(rows) != 800 or len({row["qa_id"] for row in rows}) != 800:
            raise ValueError("Expected exactly 800 unique Chinese QA questions")
        groups = {}
        for row in rows:
            ident = Path(row["video"]).stem
            if ident not in source:
                raise ValueError("QA video not in final source: " + ident)
            kind = "content" if row["qa_id"].startswith("qa_c_") else "timing"
            if kind in groups.setdefault(ident, set()):
                raise ValueError("Duplicate QA type for video: " + ident)
            groups[ident].add(kind)
        if len(groups) != 400 or any(kinds != {"content", "timing"} for kinds in groups.values()):
            raise ValueError("Expected one content/timing question pair per video")
    return source, labels, rows


def validate_prepared(args, item, gt, template):
    path = Path(item["manifest"]).parent
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["id"] != item["id"] or manifest["query"] != gt["query"]:
        raise ValueError("Prepared query/source mismatch: " + item["id"])
    system = (path / manifest["system_file"]).read_text(encoding="utf-8")
    if system.strip() != render_system(template, gt["query"]).strip():
        raise ValueError("Prepared SP is not final SP + exact source query: " + item["id"])
    if digest(path / manifest["system_file"]) != manifest["system_sha256"]:
        raise ValueError("Prepared SP record mismatch")
    if not manifest["frames"] or manifest["sample_rate_hz"] != 16000 or manifest["channels"] != 1:
        raise ValueError("Unexpected prepared AV format")
    if manifest.get("synthetic_audio_added"):
        raise ValueError("Synthetic source audio is forbidden")
    previous = -1.0
    for frame in manifest["frames"]:
        if frame["pts_sec"] < previous or frame["pts_sec"] < 0:
            raise ValueError("Invalid frame PTS sequence")
        previous = frame["pts_sec"]
        if digest(path / frame["file"]) != frame["sha256"]:
            raise ValueError("Prepared frame mismatch")
    if manifest["audio_present"]:
        audio = path / manifest["audio_file"]
        if audio.stat().st_size != manifest["audio_samples"] * 2 or digest(audio) != manifest["audio_sha256"]:
            raise ValueError("Prepared source audio mismatch")
    else:
        if manifest["audio_samples"] != 0:
            raise ValueError("Missing source audio is not zero samples")
    return path, manifest, system


def make_schedule(manifest, pcm):
    schedule = [(frame["pts_sec"], "video", frame) for frame in manifest["frames"]]
    for offset in range(0, len(pcm), 6400):
        chunk = pcm[offset:offset + 6400]
        # Release only after the last source sample in this packet occurred.
        schedule.append(((offset + len(chunk)) / 32000, "audio", (offset, chunk)))
    return sorted(schedule, key=lambda event: (event[0], event[1]))


async def session(key, setup, attempt, manifest, prepared, qa_row=None):
    """Concurrent send/receive; original-speed causal AV for streaming."""
    frame_bytes, pcm, manifest_digest = await asyncio.to_thread(read_preloaded_media, prepared, manifest)
    logs = None
    responses, usage, current = [], [], None
    t0 = None
    receiver = None
    sent_audio = sent_frames = 0
    first_audio = first_text = None
    max_drift = 0.0
    status = "failed"
    error = None
    latency = {}
    turn_finished = asyncio.Event()
    qa_completion = None

    def record(kind, **fields):
        logs.check()
        logs.emit("input-events.jsonl", {"utc": utc(), "elapsed_s": time.monotonic() - t0 if t0 is not None else None,
                                        "kind": kind, **fields})

    def finish(reason):
        nonlocal current
        if current is not None:
            current["response_id"] = "r%d" % len(responses)
            current["finish_reason"] = reason
            current["interrupted"] = reason == "interrupted"
            current["complete"] = reason == "turnComplete" or (qa_row is not None and "turn_complete_media_s" in current)
            current["generation_complete"] = "generation_complete_media_s" in current
            current["turn_complete"] = current["complete"]
            responses.append(current)
            current = None

    if qa_row is None:
        expected_audio, expected_frames = len(pcm) // 2, len(manifest["frames"])
        specification = {"setup": setup, "manifest": manifest, "prepared": str(prepared),
                         "manifest_sha256": manifest_digest, "playback_speed": 1,
                         "audio_chunk_s": 0.2, "frame_rule": "frozen source PTS approximately 1 FPS",
                         "tail_wait_s": 2, "no_forced_turns": True, "no_synthetic_audio": True}
        await save_async(attempt / "input-spec.json", specification)
    else:
        cutoff = prefix_cutoff(qa_row["question"])
        frames = [frame for frame in manifest["frames"] if cutoff is None or frame["pts_sec"] <= cutoff + 1e-8]
        if not frames:
            raise ValueError("No causal frame before requested QA cutoff")
        if len(frames) > 10:
            frames = [frames[round(i * (len(frames) - 1) / 9)] for i in range(10)]
        if cutoff is not None:
            pcm = pcm[:max(0, int(cutoff * 16000) * 2)]
        expected_audio, expected_frames = len(pcm) // 2, len(frames)
        options = {name[-1]: value for name, value in qa_row.items() if name.startswith("option_") and value}
        prompt = qa_row["question"] + "\n\n" + "\n".join(name + ". " + value for name, value in options.items())
        prompt += "\n\n请仅依据上面的视频画面与音轨作答。请输出所有你选择的选项字母，用分号分隔，例如 A;C。不要解释，不要输出其他文字。"
        parts, ledger = [], []
        for frame in frames:
            jpeg = frame_bytes[frame["file"]]
            parts.extend([{"text": "视频时间 %.1f 秒：" % frame["pts_sec"]},
                          {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}])
            ledger.append({"kind": "image", "path": str(prepared / frame["file"]),
                           "sha256": frame["sha256"], "time": frame["pts_sec"]})
        parts.append({"text": prompt})
        pacing = 0.2 if len(pcm) / 32000 > 60 else 0.0
        await save_async(attempt / "input.json", {"setup": setup, "content": ledger, "text": prompt,
                                     "cutoff": cutoff, "prepared": str(prepared),
                                     "manifest_sha256": manifest_digest,
                                     "audio_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                                     "audio_samples": expected_audio,
                                     "audio_source": str(prepared / manifest["audio_file"]) if pcm else None,
                                     "transport_audio_chunk_interval_s": pacing})
    logs = await asyncio.to_thread(DurableLogs, attempt)
    try:
        async with websockets.connect(ENDPOINT, additional_headers={"x-goog-api-key": key},
                                      open_timeout=30, close_timeout=3, max_size=None,
                                      ping_interval=20, ping_timeout=30) as ws:
            record("setup", setup=setup)
            await ws.send(json.dumps({"setup": setup}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), 30))
            await save_async(attempt / "setup-response.json", ack)
            if "setupComplete" not in ack:
                raise RuntimeError("Provider rejected setup: " + json.dumps(ack))
            t0 = time.monotonic()

            async def receive():
                nonlocal current, first_audio, first_text, qa_completion
                async for payload in ws:
                    elapsed = time.monotonic() - t0
                    event = json.loads(payload)
                    logs.check()
                    logs.emit("raw-output.jsonl", {"utc": utc(), "receive_media_s": elapsed, "event": event})
                    if event.get("error") or (qa_row is not None and ("error" in event or "goAway" in event)):
                        raise RuntimeError("Provider error/goAway: " + json.dumps(event))
                    # The raw event records the remaining connection lifetime.
                    # Streaming continues on the same socket and clock until
                    # frame/PCM coverage and the full observation tail complete.
                    if "usageMetadata" in event:
                        usage.append(event["usageMetadata"])
                    content = event.get("serverContent", {})
                    turn_parts = content.get("modelTurn", {}).get("parts", [])
                    audio_parts = [part["inlineData"] for part in turn_parts
                                   if part.get("inlineData", {}).get("mimeType", "").startswith("audio/")]
                    text = content.get("outputTranscription", {}).get("text", "")
                    if not text:
                        text = "".join(part.get("text", "") for part in turn_parts if not part.get("thought"))
                    if audio_parts or text:
                        if current is None:
                            current = {"text": "", "start_media_s": elapsed, "end_media_s": elapsed,
                                       "timing_valid": True, "output_audio_bytes": 0}
                        if qa_row is not None and qa_completion is not None:
                            current["late_output_after_qa_completion"] = True
                        current["text"] += text
                        current["end_media_s"] = elapsed
                        if text:
                            if first_text is None:
                                first_text = elapsed
                            current.setdefault("first_text_media_s", elapsed)
                        if audio_parts:
                            if first_audio is None:
                                first_audio = elapsed
                            current.setdefault("first_audio_media_s", elapsed)
                            current["start_media_s"] = current["first_audio_media_s"]
                            current["output_audio_bytes"] += sum(len(base64.b64decode(part["data"])) for part in audio_parts)
                    if content.get("generationComplete"):
                        if current is not None:
                            current["generation_complete_media_s"] = elapsed
                        if qa_row is not None:
                            # Official QA generation completion includes the
                            # final output transcription. Do not wait for audio
                            # playback/turnComplete or manufacture a new turn.
                            if qa_completion is None:
                                reason = "turnComplete" if content.get("turnComplete") else "generationComplete"
                                qa_completion = {"event": reason, "media_s": elapsed,
                                                 "rule": "generationComplete_or_turnComplete"}
                            turn_finished.set()
                            # Continue recording already-arriving raw/usage
                            # until normal close/final cancellation. This does
                            # not add playback, a wait, or another model turn.
                        # Streaming grouping remains unchanged: keep the
                        # active utterance until its native turn boundary.
                    if content.get("interrupted"):
                        finish("interrupted")
                    if content.get("turnComplete"):
                        if qa_row is not None:
                            if current is not None:
                                current.setdefault("turn_complete_media_s", elapsed)
                            if qa_completion is None:
                                qa_completion = {"event": "turnComplete", "media_s": elapsed,
                                                 "rule": "generationComplete_or_turnComplete"}
                            turn_finished.set()
                        else:
                            finish("turnComplete")
                            turn_finished.set()

            receiver = asyncio.create_task(receive())

            async def check():
                logs.check()
                if receiver.done():
                    if receiver.cancelled():
                        raise RuntimeError("Receiver cancelled before source delivery")
                    if receiver.exception():
                        raise receiver.exception()
                    if qa_row is not None and turn_finished.is_set():
                        # GC may arrive while the remaining original input is
                        # still uploading. Finish delivery before acceptance.
                        return
                    raise RuntimeError("Receiver closed before session completed")

            if qa_row is None:
                for timestamp, kind, item in make_schedule(manifest, pcm):
                    await asyncio.sleep(max(0, t0 + timestamp - time.monotonic()))
                    await check()
                    drift = time.monotonic() - t0 - timestamp
                    max_drift = max(max_drift, drift)
                    if drift > 2:
                        raise RuntimeError("Playback drift exceeds 2 seconds; transport failure")
                    if kind == "video":
                        jpeg = frame_bytes[item["file"]]
                        await ws.send(json.dumps({"realtimeInput": {"video": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}}))
                        sent_frames += 1
                        record("video", pts_s=timestamp, path=str(prepared / item["file"]), sha256=item["sha256"], drift_s=drift)
                    else:
                        offset, chunk = item
                        await ws.send(json.dumps({"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(chunk).decode()}}}))
                        sent_audio += len(chunk) // 2
                        record("audio", start_s=offset / 32000, end_s=timestamp, samples=len(chunk) // 2,
                               sha256=hashlib.sha256(chunk).hexdigest(), drift_s=drift)
                await asyncio.sleep(max(0, t0 + manifest["duration_sec"] - time.monotonic()))
                if pcm:
                    await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                    record("audio_stream_end")
                await asyncio.sleep(2)
                await check()
                latency = {"first_audio_media_s": first_audio, "first_text_media_s": first_text,
                           "max_playback_drift_s": max_drift, "total_session_seconds": time.monotonic() - t0}
            else:
                await ws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": parts}], "turnComplete": not bool(pcm)}}))
                sent_frames = len(frames)
                record("qa_content", text=prompt, frames=ledger)
                if pcm:
                    await ws.send(json.dumps({"realtimeInput": {"activityStart": {}}}))
                    record("activity_start")
                    for offset in range(0, len(pcm), 32000):
                        if pacing:
                            await asyncio.sleep(pacing)
                        await check()
                        chunk = pcm[offset:offset + 32000]
                        await ws.send(json.dumps({"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000", "data": base64.b64encode(chunk).decode()}}}))
                        sent_audio += len(chunk) // 2
                        record("audio", start_s=offset / 32000, samples=len(chunk) // 2,
                               sha256=hashlib.sha256(chunk).hexdigest())
                    await ws.send(json.dumps({"realtimeInput": {"activityEnd": {}}}))
                    record("activity_end")
                request_done = time.monotonic() - t0
                waiter = asyncio.create_task(turn_finished.wait())
                try:
                    await asyncio.wait([waiter, receiver], timeout=90, return_when=asyncio.FIRST_COMPLETED)
                    if not turn_finished.is_set():
                        await check()
                        raise TimeoutError("QA response did not complete within 90 seconds")
                    await check()
                finally:
                    waiter.cancel()
                    await asyncio.gather(waiter, return_exceptions=True)
                onset = first_audio if first_audio is not None else first_text
                latency = {"first_audio_s": first_audio, "first_text_s": first_text,
                           "first_output_after_upload_s": onset - request_done if onset is not None else None,
                           "upload_seconds": request_done, "total_session_seconds": time.monotonic() - t0}
            if sent_audio != expected_audio or sent_frames != expected_frames:
                raise RuntimeError("Incomplete source coverage")
            status = "success"
    except asyncio.CancelledError:
        error = {"error_type": "CancelledError", "error": "Session cancelled or exceeded its wall-clock timeout"}
        raise
    except Exception as problem:
        error = {"error_type": type(problem).__name__, "error": str(problem).replace(key, "[REDACTED]")}
        if "quota" in error["error"].lower() and CONNECTION_GATE is not None:
            error["shared_cooldown_s"] = await CONNECTION_GATE.penalize()
            error["quota_event_counted"] = True
    finally:
        if qa_row is not None and error is not None:
            status = "failed"
        if receiver is not None:
            if qa_row is not None and status == "success" and receiver.done() and not receiver.cancelled():
                # Normal socket close may deliver a final provider failure
                # after the QA completion check. Do not hide an already
                # observed receiver exception in cancellation/gather below.
                late_error = receiver.exception()
                if late_error is not None:
                    status = "failed"
                    error = {"error_type": type(late_error).__name__,
                             "error": str(late_error).replace(key, "[REDACTED]")}
                    if "quota" in error["error"].lower() and CONNECTION_GATE is not None:
                        error["shared_cooldown_s"] = await CONNECTION_GATE.penalize()
                        error["quota_event_counted"] = True
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        reason = "session_end" if status == "success" else "failed_session_end"
        if qa_row is not None and status == "success" and qa_completion is not None:
            reason = qa_completion["event"]
        finish(reason)
        transport = {"status": status, "responses": responses, "latency": latency, "usage": usage,
                     "billing": {"api_cost": None, "reason": "Provider monetary charge unavailable; complete usage events retained"},
                     "coverage": {"source_audio_present": manifest["audio_present"], "audio_samples_sent": sent_audio,
                                  "video_frames_sent": sent_frames, "audio_samples_expected": expected_audio,
                                  "video_frames_expected": expected_frames, "complete": sent_audio == expected_audio and sent_frames == expected_frames},
                     "error": error, "finished_utc": utc()}
        if qa_row is not None:
            transport["protocol_completion"] = qa_completion
        try:
            await asyncio.to_thread(logs.finish)
        except Exception as log_error:
            transport.update(status="failed", error={"error_type": type(log_error).__name__,
                             "error": str(log_error).replace(key, "[REDACTED]"),
                             "local_recovery_journal": logs.recovery})
        await save_async(attempt / "parsed-output.json", responses)
        await save_async(attempt / "transport-result.json", transport)
    return transport
