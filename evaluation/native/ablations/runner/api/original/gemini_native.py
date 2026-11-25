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
                    if event.get("error") or event.get("goAway") or (qa_row is not None and ("error" in event or "goAway" in event)):
                        raise RuntimeError("Provider error/goAway: " + json.dumps(event))
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


def diagnostic_scorer(path):
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("frozen_label_diagnostic_scoring", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_case


def recover_qa_generation_complete(attempt, value, row, manifest, prepared, setup):
    """Recover only the old complete-input/GC-but-no-TC timeout, never answers.

    Candidate correctness and option validity are intentionally irrelevant.
    All original files and native response grouping remain untouched.
    """
    try:
        if value.get("status") != "failed" or not isinstance(value.get("error"), dict) or value["error"].get("error_type") != "TimeoutError":
            return None
        if value["error"].get("error") != "QA response did not complete within 90 seconds":
            return None
        original_result = json.loads((attempt / "result.json").read_text())
        original_source = json.loads((attempt / "source.json").read_text())
        if attempt.parent.name != row["qa_id"] or original_result.get("status") != "failed" or original_result.get("id") != row["qa_id"] or original_result.get("error") != "QA response did not complete within 90 seconds":
            return None
        video_id = Path(row["video"]).stem
        if manifest.get("id") != video_id or "setupComplete" not in json.loads((attempt / "setup-response.json").read_text()):
            return None
        if original_result.get("video_id") != video_id or original_source.get("phase") != "qa" or original_source.get("qa_id") != row["qa_id"] or original_source.get("source", {}).get("id") != video_id:
            return None
        recorded = json.loads((attempt / "input.json").read_text())
        options = {name[-1]: text for name, text in row.items() if name.startswith("option_") and text}
        prompt = row["question"] + "\n\n" + "\n".join(name + ". " + text for name, text in options.items())
        prompt += "\n\n请仅依据上面的视频画面与音轨作答。请输出所有你选择的选项字母，用分号分隔，例如 A;C。不要解释，不要输出其他文字。"
        cutoff = prefix_cutoff(row["question"])
        frames = [f for f in manifest["frames"] if cutoff is None or f["pts_sec"] <= cutoff + 1e-8]
        if len(frames) > 10:
            frames = [frames[round(i * (len(frames) - 1) / 9)] for i in range(10)]
        ledger = [{"kind": "image", "path": str(prepared / f["file"]), "sha256": f["sha256"], "time": f["pts_sec"]} for f in frames]
        pcm = (prepared / manifest["audio_file"]).read_bytes() if manifest["audio_present"] else b""
        if cutoff is not None:
            pcm = pcm[:max(0, int(cutoff * 16000) * 2)]
        if recorded.get("setup") != setup or recorded.get("text") != prompt or recorded.get("content") != ledger or recorded.get("audio_samples") != len(pcm) // 2 or recorded.get("audio_pcm_sha256") != hashlib.sha256(pcm).hexdigest():
            return None
        coverage = value.get("coverage", {})
        if coverage != {"source_audio_present": manifest["audio_present"], "audio_samples_sent": len(pcm) // 2,
                        "video_frames_sent": len(frames), "audio_samples_expected": len(pcm) // 2,
                        "video_frames_expected": len(frames), "complete": True}:
            return None
        events = [json.loads(line) for line in (attempt / "input-events.jsonl").read_text().splitlines() if line.strip()]
        contents = [event for event in events if event.get("kind") == "qa_content"]
        if len(contents) != 1 or contents[0].get("text") != prompt or contents[0].get("frames") != ledger:
            return None
        if not events or events[0].get("kind") != "setup" or events[0].get("setup") != setup:
            return None
        last_clock, samples = -1.0, 0
        for event in events:
            if datetime.fromisoformat(event["utc"].replace("Z", "+00:00")).utcoffset() is None:
                return None
            elapsed = event.get("elapsed_s")
            if event.get("kind") != "setup" and elapsed is None:
                return None
            if elapsed is not None:
                if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < last_clock or elapsed < 0:
                    return None
                last_clock = elapsed
            if event.get("kind") == "audio":
                count = event.get("samples")
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or abs(event.get("start_s", -1) - samples / 16000) > 1e-9:
                    return None
                chunk = pcm[samples * 2:(samples + count) * 2]
                if len(chunk) != count * 2 or event.get("sha256") != hashlib.sha256(chunk).hexdigest():
                    return None
                samples += count
        if samples != len(pcm) // 2:
            return None
        starts = [i for i, e in enumerate(events) if e.get("kind") == "activity_start"]
        ends = [i for i, e in enumerate(events) if e.get("kind") == "activity_end"]
        audio_indices = [i for i, e in enumerate(events) if e.get("kind") == "audio"]
        if pcm:
            if len(starts) != 1 or len(ends) != 1 or not (starts[0] < min(audio_indices) <= max(audio_indices) < ends[0]):
                return None
            upload_end = events[ends[0]].get("elapsed_s")
            if events[-1] != events[ends[0]]:
                return None
        else:
            if starts or ends or audio_indices:
                return None
            upload_end = contents[0].get("elapsed_s")
            if events[-1] != contents[0]:
                return None
        raw = [json.loads(line) for line in (attempt / "raw-output.jsonl").read_text().splitlines() if line.strip()]
        replayed, current, usage, gc, first_audio, first_text = [], None, [], [], None, None
        prior = -1.0
        for native in raw:
            if datetime.fromisoformat(native["utc"].replace("Z", "+00:00")).utcoffset() is None:
                return None
            elapsed = native.get("receive_media_s")
            if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < prior or elapsed < 0:
                return None
            prior = elapsed
            event = native["event"]
            content = event.get("serverContent", {})
            if "error" in event or "goAway" in event or content.get("interrupted") or content.get("turnComplete"):
                return None
            if "usageMetadata" in event:
                usage.append(event["usageMetadata"])
            parts = content.get("modelTurn", {}).get("parts", [])
            audios = [part["inlineData"] for part in parts if part.get("inlineData", {}).get("mimeType", "").startswith("audio/")]
            text = content.get("outputTranscription", {}).get("text", "") or "".join(part.get("text", "") for part in parts if not part.get("thought"))
            if audios or text:
                if gc:
                    return None
                if current is None:
                    current = {"text": "", "start_media_s": elapsed, "end_media_s": elapsed, "timing_valid": True, "output_audio_bytes": 0}
                current["text"] += text
                current["end_media_s"] = elapsed
                if text:
                    first_text = elapsed if first_text is None else first_text
                    current.setdefault("first_text_media_s", elapsed)
                if audios:
                    first_audio = elapsed if first_audio is None else first_audio
                    current.setdefault("first_audio_media_s", elapsed)
                    current["start_media_s"] = current["first_audio_media_s"]
                    current["output_audio_bytes"] += sum(len(base64.b64decode(a["data"], validate=True)) for a in audios)
            if content.get("generationComplete"):
                gc.append(elapsed)
                if current is not None:
                    current["generation_complete_media_s"] = elapsed
        if not gc:
            return None
        core = ("text", "start_media_s", "end_media_s", "timing_valid", "output_audio_bytes", "first_text_media_s", "first_audio_media_s", "generation_complete_media_s")
        originals = value.get("responses", [])
        expected = [current] if current is not None else []
        if len(originals) != len(expected) or any(old.get(k) != observed.get(k) for old, observed in zip(originals, expected) for k in core) or usage != value.get("usage") or json.loads((attempt / "parsed-output.json").read_text()) != originals:
            return None
        if not isinstance(upload_end, (int, float)) or isinstance(upload_end, bool) or not math.isfinite(upload_end):
            return None
        if gc[-1] < upload_end:
            return None
        unknown = ["first_output_after_upload_s", "upload_seconds", "total_session_seconds"]
        onset = first_audio if first_audio is not None else first_text
        latency = {"first_audio_s": first_audio, "first_text_s": first_text, "generation_complete_s": gc[-1],
                   "ledger_upload_end_s": upload_end, "first_output_after_ledger_end_s": onset - upload_end if onset is not None else None,
                   "first_output_after_upload_s": None, "upload_seconds": None, "total_session_seconds": None,
                   "unknown_fields": unknown, "timing_source": "retained raw receive_media_s and input elapsed_s"}
        metadata = {"recovery_reason": "qa_generation_complete_without_turn_complete", "original_status": "failed",
                    "protocol_completion": "generationComplete", "original_elapsed_s": original_result.get("elapsed_s"),
                    "original_finished_utc": value.get("finished_utc"), "original_error": value.get("error"),
                    "original_latency": value.get("latency"), "latency_reconstruction": {
                        "source": "raw-output receive_media_s + input-events elapsed_s", "unknown_fields": unknown,
                        "original_elapsed_preserved": True}}
        return {**value, "status": "success", "error": None, "latency": latency, "_qa_protocol_recovery": metadata}
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def reusable_transport(folder, qa_row=None, manifest=None, prepared=None, setup=None):
    """Recover completed native calls if only offline finalization was interrupted."""
    for attempt in sorted(folder.glob("attempt-*")):
        path = attempt / "transport-result.json"
        if path.is_file():
            try:
                value = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                note = folder / ("unreadable-transport-%s.json" % attempt.name)
                if not note.exists():
                    save(note, {"source": str(path), "reason": "incomplete JSON from an interrupted attempt; original retained", "not_model_silence": True})
                continue
            if value.get("status") == "success" and value.get("coverage", {}).get("complete"):
                return attempt, value
            if qa_row is not None and folder.parent.parent.name == "qa" and manifest is not None and prepared is not None and setup is not None:
                recovered = recover_qa_generation_complete(attempt, value, qa_row, manifest, prepared, setup)
                if recovered is not None:
                    return attempt, recovered
    return None


async def main(args, key):
    global ENDPOINT
    ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.%s.GenerativeService.BidiGenerateContent" % args.api_version
    source, labels, rows = await asyncio.to_thread(load_source, args)
    template = await asyncio.to_thread(args.sp.read_text, encoding="utf-8")
    model = args.model.removeprefix("models/")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("Invalid model identifier")
    run = args.root / model / ("probes" if args.probe else "runs") / args.phase
    await asyncio.to_thread(run.mkdir, parents=True, exist_ok=True)
    lock = await asyncio.to_thread((run / "run.lock").open, "a+")
    try:
        await asyncio.to_thread(fcntl.flock, lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another runner already owns this phase")
    selected = sorted(source)
    if args.caseids:
        if not args.probe:
            raise ValueError("--caseids is only allowed for isolated probes")
        selected = [item.strip() for item in args.caseids.split(",") if item.strip()]
        if len(set(selected)) != len(selected) or any(ident not in source for ident in selected):
            raise ValueError("Duplicate/unknown probe identities")
    elif args.probe:
        selected = select_probe(source, labels)
    tasks = [(ident, source[ident], None) for ident in selected] if args.phase == "streaming" else [
        (row["qa_id"], source[Path(row["video"]).stem], row) for row in rows if Path(row["video"]).stem in selected]
    expected = 400 if args.phase == "streaming" else 800
    if not args.probe and len(tasks) != expected:
        raise ValueError("Formal phase has incomplete input population")
    scorer_path = args.scorer or args.root / "code" / "scoring.py"
    scorer = await asyncio.to_thread(diagnostic_scorer, scorer_path)
    sp_digest = await asyncio.to_thread(digest, args.sp)
    qa_digest = await asyncio.to_thread(digest, args.qa_csv) if args.qa_csv else None
    scorer_digest = await asyncio.to_thread(digest, scorer_path) if scorer else None
    config = {"model": model, "phase": args.phase, "probe": args.probe, "planned_cases": len(tasks),
              "selected_video_ids": selected, "api_version": args.api_version, "endpoint": ENDPOINT,
              "data": str(args.data), "prepared": str(args.prepared), "final_sp": str(args.sp),
              "system_sha256": sp_digest, "source": list(source.values()),
              "qa_csv": str(args.qa_csv) if args.qa_csv else None,
              "qa_csv_sha256": qa_digest,
              "streaming": {"native_continuous_av": True, "proactive_audio": True, "playback_speed": 1,
                            "audio_chunk_s": 0.2, "tail_wait_s": 2, "temperature": 0,
                            "context_compression": "official slidingWindow", "task_route": "embedded final SP only"},
              "qa": {"max_frames": 10, "prefix_rule": "same reference prefix_cutoff",
                     "max_output_tokens": 1024, "temperature": 0, "parser": "robust explicit parse_choices"},
              "main_score": "offline time + binary content judge; tag not required; extra output not deducted",
              "tag_diagnostic_scorer": str(scorer_path) if scorer else None,
              "tag_diagnostic_scorer_sha256": scorer_digest}
    config_path = run / "config.json"
    if await asyncio.to_thread(config_path.exists):
        previous = json.loads(await asyncio.to_thread(config_path.read_text))
        # Runtime/IO implementation may change; frozen model, data, media and
        # native inference configuration must remain exactly the same.
        immutable = ("model", "phase", "probe", "planned_cases", "selected_video_ids", "api_version",
                     "endpoint", "data", "prepared", "final_sp", "system_sha256", "source", "qa_csv",
                     "qa_csv_sha256", "streaming", "qa", "main_score")
        if any(previous.get(name) != config.get(name) for name in immutable):
            raise ValueError("Frozen run configuration changed; do not mix experiments")
    else:
        await save_async(config_path, config)
    runner_digest = await asyncio.to_thread(digest, __file__)
    await save_async(run / ("environment-%d.json" % time.time_ns()), {"utc": utc(), "python": sys.version,
         "platform": platform.platform(), "host": platform.node(), "pid": os.getpid(),
         "websockets": importlib.metadata.version("websockets"), "concurrency": args.concurrency,
         "runner_sha256": runner_digest, "runtime_revision": "v2 nonblocking IO + shared quota cooldown",
         "minimum_connect_interval_s": args.min_connect_interval})
    snapshot = run / ("runner-%s.py" % runner_digest)
    if not await asyncio.to_thread(snapshot.exists):
        await asyncio.to_thread(snapshot.write_bytes, await asyncio.to_thread(Path(__file__).read_bytes))
    successes, failures = {}, {}
    artifact_errors = []
    def previous_successes():
        completed = {}
        for ident, _, _ in tasks:
            for path in sorted((run / "cases" / ident).glob("attempt-*/result.json")):
                result = json.loads(path.read_text())
                if result.get("status") == "success":
                    completed.setdefault(ident, result)
        return completed
    successes.update(await asyncio.to_thread(previous_successes))
    sem = asyncio.Semaphore(args.concurrency)
    progress_lock = asyncio.Lock()
    log_lock = asyncio.Lock()

    async def progress():
        async with progress_lock:
            await asyncio.to_thread(atomic, run / "progress.json", {"utc": utc(), "model": model, "phase": args.phase,
                 "probe": args.probe, "success": len(successes), "failed": len(failures),
                 "pending": len(tasks) - len(successes) - len(failures), "planned": len(tasks),
                 "concurrency": args.concurrency, "pid": os.getpid(), "stop_requested": STOP.is_set(),
                 "shared_quota_cooldown_remaining_s": max(0, CONNECTION_GATE.next_allowed - time.monotonic())})

    async def log(event, **fields):
        entry = {"utc": utc(), "event": event, **fields}
        def write():
            with (run / "runlog.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                handle.flush()
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        async with log_lock:
            await asyncio.to_thread(write)

    async def worker(ident, item, row):
        if ident in successes:
            return
        async with sem:
            if STOP.is_set():
                return
            folder = run / "cases" / ident
            await asyncio.to_thread(folder.mkdir, parents=True, exist_ok=True)
            gt = labels[item["id"]]
            for retry in range(args.max_attempts):
                if STOP.is_set():
                    return
                numbers = await asyncio.to_thread(lambda: [int(path.name.split("-")[-1]) for path in folder.glob("attempt-*") if path.is_dir()])
                attempt = folder / ("attempt-%03d" % (max(numbers, default=0) + 1))
                await asyncio.to_thread(attempt.mkdir)
                started = time.monotonic()
                await log("case_start", id=ident, video_id=item["id"], attempt=attempt.name)
                await save_async(attempt / "source.json", {"source": item, "phase": args.phase, "qa_id": ident if row else None})
                try:
                    path, manifest, system = await asyncio.to_thread(validate_prepared, args, item, gt, template)
                    setup = {"model": "models/" + model, "generationConfig": {"responseModalities": ["AUDIO"], "temperature": 0},
                             "outputAudioTranscription": {}, "contextWindowCompression": {"slidingWindow": {}}}
                    if row is None:
                        setup.update(systemInstruction={"parts": [{"text": system}]}, proactivity={"proactiveAudio": True})
                    else:
                        setup["generationConfig"]["maxOutputTokens"] = 1024
                        setup["realtimeInputConfig"] = {"automaticActivityDetection": {"disabled": True}}
                    timeout = manifest["duration_sec"] + 120 if row is None else max(150, manifest["duration_sec"] * 0.2 + 120)
                    cached = await asyncio.to_thread(reusable_transport, folder, row, manifest, path, setup)
                    if cached is not None:
                        cached_attempt, transport = cached
                        recovery = {"native_attempt": str(cached_attempt), "no_new_model_call": True,
                                    "input_and_raw_retained_at": str(cached_attempt)}
                        recovery.update(transport.pop("_qa_protocol_recovery", {}))
                        await save_async(attempt / "recovery-from.json", recovery)
                        await save_async(attempt / "parsed-output.json", transport["responses"])
                        await save_async(attempt / "transport-result.json", {**transport, "recovery": recovery})
                    else:
                        await CONNECTION_GATE.wait(STOP)
                        transport = await asyncio.wait_for(session(key, setup, attempt, manifest, path, row), timeout)
                    if transport["status"] != "success":
                        raise RuntimeError((transport.get("error") or {}).get("error", "Native session failed"))
                    outcome = {"id": ident, "video_id": item["id"], "status": "success",
                               "responses": len(transport["responses"]), "latency": transport["latency"],
                               "usage": transport["usage"], "billing": transport["billing"],
                               "coverage": transport["coverage"], "source_audio_present": manifest["audio_present"]}
                    if cached is not None:
                        outcome["reused_native_transport"] = recovery
                        outcome["elapsed_scope"] = "recovery_finalization_only; original elapsed retained in recovery"
                    if row is None:
                        await save_async(attempt / "ground-truth.json", gt)
                        outcome["main_content_judge_pending"] = True
                        if scorer:
                            try:
                                outcome["metrics"] = await asyncio.to_thread(scorer, gt["annotation"], transport["responses"], manifest["duration_sec"])
                            except Exception as score_error:
                                outcome["diagnostic_scoring_error"] = str(score_error)
                            outcome["metrics_scope"] = "tag/time diagnostic only; NOT final main correct score"
                    else:
                        text = "".join(response["text"] for response in transport["responses"])
                        options = {name[-1]: value for name, value in row.items() if name.startswith("option_") and value}
                        picks, valid = parse_choices(text, options, options)
                        gold = row["answer"].split(";")
                        exact = valid and set(picks) == set(gold)
                        score = 0.0 if not valid else float(exact) if len(set(gold)) == 1 else len(set(picks) & set(gold)) / len(set(gold))
                        outcome.update(qa_id=ident, video=row["video"],
                                       qa_type="content" if ident.startswith("qa_c_") else "timing",
                                       text=text, picks=picks, valid=valid, exact=exact, score=score,
                                       wrong_picks=sorted(set(picks) - set(gold)))
                        await save_async(attempt / "ground-truth.json", {"answer": row["answer"]})
                    outcome.update(elapsed_s=time.monotonic() - started, finished_utc=utc(), attempt=str(attempt))
                    await save_async(attempt / "result.json", outcome)
                    if row is None:
                        await save_async(attempt / "scored-result.json", outcome)
                    successes[ident] = outcome
                    failures.pop(ident, None)
                    await log("case_complete", id=ident, responses=outcome["responses"], elapsed_s=outcome["elapsed_s"])
                    await progress()
                    return
                except StopRequested:
                    await save_async(attempt / "result.json", {"id": ident, "status": "stopped_before_connect",
                                     "attempt": str(attempt), "finished_utc": utc(), "not_model_failure": True})
                    return
                except Exception as problem:
                    if ident in successes:
                        # Final progress/log failure must never turn a completed
                        # native case into another paid model call.
                        artifact_errors.append({"id": ident, "error": str(problem).replace(key, "[REDACTED]")})
                        STOP.set()
                        return
                    transport_path = attempt / "transport-result.json"
                    partial = json.loads(await asyncio.to_thread(transport_path.read_text)) if await asyncio.to_thread(transport_path.exists) else None
                    outcome = {"id": ident, "video_id": item["id"], "status": "failed",
                               "error_type": type(problem).__name__, "error": str(problem).replace(key, "[REDACTED]"),
                               "elapsed_s": time.monotonic() - started, "finished_utc": utc(),
                               "attempt": str(attempt), "coverage": partial.get("coverage") if partial else None,
                               "partial_response_count": len(partial["responses"]) if partial else 0,
                               "not_model_silence": True}
                    await save_async(attempt / "result.json", outcome)
                    await log("case_failed", id=ident, error_type=outcome["error_type"], error=outcome["error"])
                    if "quota" in outcome["error"].lower():
                        await CONNECTION_GATE.penalize(
                            retry, count_event=not bool(partial and
                                (partial.get("error") or {}).get("quota_event_counted")))
                    if STOP.is_set():
                        return
                    if retry + 1 < args.max_attempts:
                        if "quota" not in outcome["error"].lower():
                            try:
                                await asyncio.wait_for(STOP.wait(), timeout=5 * (retry + 1))
                            except asyncio.TimeoutError:
                                pass
                    else:
                        failures[ident] = outcome
                        await progress()
    await progress()
    await asyncio.gather(*(worker(*task) for task in tasks))
    ledger = {"model": model, "phase": args.phase, "probe": args.probe, "utc": utc(),
              "planned": len(tasks), "results": [successes[ident] for ident in sorted(successes)],
              "failures": [failures[ident] for ident in sorted(failures)],
              "main_content_judge_pending": args.phase == "streaming", "stop_requested": STOP.is_set(),
              "pending": len(tasks) - len(successes) - len(failures), "artifact_errors": artifact_errors}
    await asyncio.to_thread(atomic, run / "results.json", ledger)
    await log("run_stopped" if STOP.is_set() else "run_complete", success=len(successes), failed=len(failures), planned=len(tasks))
    await progress()
    await asyncio.to_thread(lock.close)
    return ledger


async def pipeline(args, key):
    global STOP, CONNECTION_GATE
    STOP = asyncio.Event()
    CONNECTION_GATE = ConnectionGate(args.min_connect_interval, args.max_quota_events)
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, STOP.set)
    requested_phase = args.phase
    if args.probe_then_full:
        args.probe = True
        args.phase = "streaming"
        args.caseids = args.caseids or "adult_candle_pos,adult_candle_neg,ask_ball_rule"
        stream_probe = await main(args, key)
        if STOP.is_set():
            return
        passed = not stream_probe["failures"] and len(stream_probe["results"]) == stream_probe["planned"]
        qa_probe = None
        if passed and args.qa_csv:
            args.phase = "qa"
            args.caseids = "ask_ball_rule"
            qa_probe = await main(args, key)
            if STOP.is_set():
                return
            passed = not qa_probe["failures"] and len(qa_probe["results"]) == qa_probe["planned"]
            passed = passed and bool(qa_probe["results"]) and all(
                result.get("text", "").strip() and result["coverage"]["complete"] for result in qa_probe["results"])
        report = {"utc": utc(), "passed": bool(passed), "streaming_probe": {
                  "planned": stream_probe["planned"], "success": len(stream_probe["results"]),
                  "failed": len(stream_probe["failures"])},
                  "qa_probe": None if qa_probe is None else {
                  "planned": qa_probe["planned"], "success": len(qa_probe["results"]),
                  "failed": len(qa_probe["failures"]),
                  "nonempty_answers": sum(bool(result.get("text", "").strip()) for result in qa_probe["results"])},
                  "rule": "Full source coverage, no transport failures, nonempty native QA transcription; streaming silence is not a probe failure"}
        await save_async(args.root / ("readiness-%d.json" % time.time_ns()), report)
        if not passed:
            raise RuntimeError("Probe readiness failed; formal evaluation was NOT started")
        args.probe = False
        args.caseids = None
    phases = ("streaming", "qa") if requested_phase == "all" else (requested_phase,)
    for phase in phases:
        if STOP.is_set():
            break
        args.phase = phase
        await main(args, key)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    for name in ("root", "prepared", "data", "sp"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--qa-csv", type=Path)
    parser.add_argument("--scorer", type=Path)
    parser.add_argument("--phase", choices=("streaming", "qa", "all"), required=True)
    parser.add_argument("--api-version", choices=("v1alpha", "v1beta"), default="v1alpha")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--caseids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--min-connect-interval", type=float, default=1.0)
    parser.add_argument("--max-quota-events", type=int, default=0, help="Stop opening sessions after this many quota failures; zero disables the guard")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--credential-stdin", action="store_true", help="Read one secret line from stdin without echo or persistence")
    parser.add_argument("--probe-then-full", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_attempts < 1 or args.min_connect_interval < 0:
        parser.error("Concurrency and max attempts must be positive")
    return args


if __name__ == "__main__":
    arguments_value = arguments()
    credential = sys.stdin.readline().rstrip("\r\n") if arguments_value.credential_stdin else getpass.getpass("Google credential: ")
    if not credential:
        raise SystemExit("Empty credential")
    if arguments_value.detach:
        pid = os.fork()
        if pid:
            print(json.dumps({"detached_pid": pid}), flush=True)
            sys.exit(0)
        os.setsid()
        console = arguments_value.root / "logs" / ("%s-%s-%s-%d.log" % (
            arguments_value.model.removeprefix("models/"), arguments_value.phase,
            "probe" if arguments_value.probe else "formal", time.time_ns()))
        console.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(console), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        fd = os.open("/dev/null", os.O_RDONLY)
        os.dup2(fd, 0)
        os.close(fd)
    asyncio.run(pipeline(arguments_value, credential))
