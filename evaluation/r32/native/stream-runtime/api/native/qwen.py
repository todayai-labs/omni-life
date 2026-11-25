#!/usr/bin/env python3
"""Causal, source-audio Qwen Realtime evaluation with auditable native events."""
from __future__ import annotations

import argparse
import asyncio
import base64
import getpass
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
import uuid
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import websockets

ROOT = Path(__file__).resolve().parents[1]
PREPARED = Path("prepared")
ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
MODELS = ("qwen3.5-omni-plus-realtime", "qwen3.5-omni-flash-realtime", "qwen3.8-omni-flash-realtime")
PROBE_CASES = ("elder_stand_f_pos", "deaf_sign_home", "orbit_leash_11")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any, credential: str) -> Any:
    if isinstance(value, str):
        return value.replace(credential, "[REDACTED]") if credential else value
    if isinstance(value, dict):
        return {str(k): redact(v, credential) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(v, credential) for v in value]
    return value


def save(path: Path, value: Any, credential: str = "") -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(redact(value, credential), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def classify_failure(result):
    detail = str(result.get("error", "")).lower()
    if result.get("fatal_credential_error"):
        return "credential_or_permission", False
    if any(word in detail for word in ("connect call failed", "connection refused")) and any(
            host in detail for host in ("127.0.0.1", "localhost", "::1")):
        return "local_transport_unavailable", True
    if result.get("error_type") in ("ConnectionRefusedError", "ConnectionResetError", "BrokenPipeError") or any(
            word in detail for word in ("no close frame", "connection closed", "receive_error")):
        return "temporary_transport_or_server", True
    if result.get("error_type") == "gaierror" or any(word in detail for word in (
            "no route to host", "network is unreachable", "host unreachable", "nodename nor servname",
            "nodename or servname", "name or service not known", "temporary failure in name resolution")):
        return "temporary_transport_or_server", True
    if "append image before append audio" in detail:
        # The same native error can occur after a VAD commit despite source
        # audio having been delivered. Do not call that a video-only case.
        audio_present = result.get("source_audio_present")
        if audio_present is None and result.get("attempt"):
            try:
                recorded = json.loads((Path(result["attempt"]) / "input-spec.json").read_text())
                manifest = json.loads(Path(recorded["source_manifest"]).read_text())
                audio_present = bool(manifest.get("audio_present"))
            except (OSError, ValueError, KeyError):
                pass
        if audio_present is False:
            return "native_video_only_input_rejected", False
        return "native_audio_buffer_state_error", True
    if "configuration" in detail or "session.updated" in detail and "mismatch" in detail:
        return "configuration_rejected", False
    if any(word in detail for word in ("inappropriate content", "data_inspection", "content_policy", "safety", "content_filter")):
        return "provider_refusal", False
    if any(word in detail for word in ("429", "too many", "ratelimit", "rate_limit", "throttl", "limit_exceeded")):
        return "rate_limit", True
    if any(word in detail for word in ("timeout", "timed out", "timing drift", "input drift", "send drift", "clock drift", "disconnect", "connection", "502", "503", "504", "internal_server", "500")):
        return "temporary_transport_or_server", True
    return "provider_or_input_error", False


def reuse_probe(case_root, output_root, model, case, prepared):
    """Reference a verified identical successful probe, without another API call."""
    for status_file in sorted((output_root / "models" / model / "probe" / "cases" / case).glob("attempt-*/status.json")):
        status = json.loads(status_file.read_text())
        if status.get("status") != "completed":
            continue
        probe = status_file.parent
        recorded = json.loads((probe / "input-spec.json").read_text())
        source = prepared / case
        manifest = json.loads((source / "manifest.json").read_text())
        echoes = json.loads((probe / "session_config.json").read_text())
        updates = [e for e in echoes.get("provider_echoes", []) if e.get("type") == "session.updated"]
        if not updates or not verify_session(echoes["requested"]["session"], updates[-1]["session"])["passed"]:
            continue
        if recorded["instructions"] != (source / manifest["system_file"]).read_text() or recorded["query"] != manifest["query"]:
            continue
        if recorded.get("task_tts") or recorded.get("padding") or recorded.get("manual_response_create") or recorded.get("manual_commit") or recorded.get("empty_audio_control_probe"):
            continue
        selected = case_root / "attempt-reused-probe"
        selected.mkdir(exist_ok=True)
        for item in probe.iterdir():
            if item.is_file() and not (selected / item.name).exists():
                (selected / item.name).symlink_to(os.path.relpath(item, selected))
        save(selected / "reused-probe.json", {"source_attempt": str(probe), "no_new_model_call": True,
                                              "selection_reason": "verified same prepared task, SP and media protocol"})
        return selected
    return None


def schedule(manifest: dict[str, Any], chunk_seconds: float = 0.2) -> list[dict[str, Any]]:
    """Send audio only when its last sample is available; keep every frame's PTS."""
    rate = int(manifest.get("sample_rate_hz", 16000))
    if rate != 16000 or int(manifest.get("channels", 1)) != 1:
        raise ValueError("Prepared audio must be original-source 16 kHz mono PCM.")
    audio_present = bool(manifest.get("audio_present"))
    samples = int(manifest.get("audio_samples", 0)) if audio_present else 0
    if audio_present and samples <= 0:
        raise ValueError("Audio-present manifest has no original audio samples.")
    chunk_samples = round(rate * chunk_seconds)
    if chunk_samples <= 0:
        raise ValueError("Audio chunk size must be positive.")
    events: list[dict[str, Any]] = []
    for start in range(0, samples, chunk_samples):
        end = min(samples, start + chunk_samples)
        events.append({"kind": "audio", "send_due_s": end / rate, "source_start_s": start / rate,
                       "source_end_s": end / rate, "sample_start": start, "sample_end": end,
                       "file": manifest["audio_file"]})
    first_audio_due = min(samples, chunk_samples) / rate if samples else 0.0
    frames = manifest.get("frames", [])
    for index, frame in enumerate(frames):
        pts = float(frame["pts_sec"])
        if not math.isfinite(pts) or pts < 0:
            raise ValueError("Invalid source frame PTS.")
        due = max(pts, first_audio_due) if index == 0 and samples else pts
        events.append({"kind": "video", "send_due_s": due, "source_pts_s": pts,
                       "frame_index": index, "file": frame["file"],
                       "native_first_frame_delay_s": due - pts if index == 0 else 0.0})
    return sorted(events, key=lambda event: (event["send_due_s"], 0 if event["kind"] == "audio" else 1))


def session_config(model: str, instructions: str) -> dict[str, Any]:
    return {"type": "session.update", "session": {
        "model": model, "instructions": instructions, "modalities": ["text"],
        "audio": {"input": {"format": {"type": "pcm", "sample_rate": 16000}}},
        "turn_detection": {"type": "semantic_vad"},
        "temperature": 0, "seed": 0, "enable_search": False, "tools": [],
    }}


def verify_session(requested: dict[str, Any], returned: dict[str, Any]) -> dict[str, Any]:
    """Refuse to silently measure different provider inference settings."""
    checks = {}
    for field in ("model", "instructions", "modalities", "temperature", "turn_detection", "audio"):
        actual = returned.get(field)
        expected = requested[field]
        if field == "model":
            same = isinstance(actual, str) and (actual == expected or actual.startswith(expected + "-"))
        elif field == "audio":
            actual_format = actual.get("input", {}).get("format") if isinstance(actual, dict) else None
            expected_format = expected["input"]["format"]
            same = isinstance(actual_format, dict) and all(actual_format.get(key) == value for key, value in expected_format.items())
        elif field == "turn_detection":
            same = isinstance(actual, dict) and actual.get("type") == expected["type"]
        else:
            same = actual == expected
        checks[field] = {"requested": expected, "returned": actual, "matches": same}
    for field in ("seed", "enable_search", "tools"):
        checks[field] = {"requested": requested[field], "returned": returned.get(field),
                         "matches": returned.get(field) == requested[field] if field in returned else None}
    mismatches = [field for field, check in checks.items() if check["matches"] is False]
    return {"passed": not mismatches, "mismatches": mismatches, "checks": checks}


def fatal_credential_error(value: Any) -> bool:
    serialized = json.dumps(value, ensure_ascii=False).lower()
    words = ("unauthorized", "invalid_api_key", "invalidapikey", "authentication", "authorizationfailed",
             "insufficient_quota", "insufficientquota", "insufficient balance", "arrears", "accessdenied",
             "permission denied", "not authorized", "notauthor", "forbidden", "invalid api-key")
    return any(word in serialized for word in words) or any(
        marker in serialized for marker in ('"status_code": 401', '"status_code": 403', '"status": 401', '"status": 403'))


class ResponseParser:
    """Keep text and transcript channels separate and preserve native response IDs."""

    def __init__(self) -> None:
        self.responses: dict[str, dict[str, Any]] = {}
        self.item_response: dict[str, str] = {}
        self.current_id: str | None = None
        self.provider_models: list[str] = []
        self.usage: list[dict[str, Any]] = []

    def feed(self, event: dict[str, Any], media_s: float | None, received_at: str) -> None:
        kind = event.get("type", "")
        session = event.get("session", {})
        for candidate in (session.get("model"), event.get("model"), event.get("response", {}).get("model")):
            if candidate and candidate not in self.provider_models:
                self.provider_models.append(candidate)
        native_response = event.get("response", {})
        rid = event.get("response_id") or native_response.get("id")
        if not rid and event.get("item_id"):
            rid = self.item_response.get(event["item_id"])
        if not rid and kind.startswith("response."):
            rid = self.current_id
        if kind == "response.created" and rid:
            self.current_id = rid
        if rid is None and kind.startswith("response."):
            rid = "unidentified-response"
        if not rid:
            return
        response = self.responses.setdefault(str(rid), {
            "response_id": str(rid), "created_media_s": media_s, "first_text_media_s": None,
            "first_audio_media_s": None, "last_output_media_s": None, "text": "", "transcript": "",
            "status": None, "status_details": None, "usage": None, "first_output_utc": None,
        })
        item = event.get("item", {})
        if item.get("id"):
            self.item_response[item["id"]] = str(rid)
        channel = None
        if kind in ("response.text.delta", "response.output_text.delta"):
            channel = "text"
        elif kind in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            channel = "transcript"
        if channel:
            delta = event.get("delta", "")
            if isinstance(delta, str):
                response[channel] += delta
                if delta and response["first_text_media_s"] is None:
                    response["first_text_media_s"] = media_s
                if delta:
                    self._onset(response, media_s, received_at)
        elif kind in ("response.audio.delta", "response.output_audio.delta") and event.get("delta"):
            if response["first_audio_media_s"] is None:
                response["first_audio_media_s"] = media_s
            self._onset(response, media_s, received_at)
        elif kind in ("response.text.done", "response.output_text.done"):
            self._final_text(response, "text", event.get("text", ""), media_s, received_at)
        elif kind in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
            self._final_text(response, "transcript", event.get("transcript", ""), media_s, received_at)
        elif kind == "response.done":
            response["status"] = native_response.get("status")
            response["status_details"] = native_response.get("status_details")
            response["usage"] = native_response.get("usage")
            if response["usage"] is not None:
                self.usage.append({"response_id": str(rid), "usage": response["usage"]})
            # A server may deliver only a final response, without any text delta.
            contents = [part for output in native_response.get("output", []) for part in output.get("content", [])]
            final_text = "".join(part.get("text", "") for part in contents if part.get("type") in ("text", "output_text"))
            final_transcript = "".join(part.get("transcript", "") for part in contents if "transcript" in part)
            self._final_text(response, "text", final_text, media_s, received_at)
            self._final_text(response, "transcript", final_transcript, media_s, received_at)

    @staticmethod
    def _onset(response: dict[str, Any], media_s: float | None, received_at: str) -> None:
        if response["first_output_utc"] is None:
            response["first_output_utc"] = received_at
        response["last_output_media_s"] = media_s

    def _final_text(self, response: dict[str, Any], channel: str, value: Any,
                    media_s: float | None, received_at: str) -> None:
        if isinstance(value, str) and value:
            had_output = bool(response[channel])
            response[channel] = value
            if not had_output:
                if response["first_text_media_s"] is None:
                    response["first_text_media_s"] = media_s
                self._onset(response, media_s, received_at)

    def export(self) -> dict[str, Any]:
        rows = []
        for response in self.responses.values():
            row = dict(response)
            row["content"] = row["text"] or row["transcript"]
            starts = [value for value in (row["first_text_media_s"], row["first_audio_media_s"]) if value is not None]
            row["start_media_s"] = min(starts) if starts else None
            row["end_media_s"] = row["last_output_media_s"]
            rows.append(row)
        return {"responses": rows, "provider_models": self.provider_models, "usage": self.usage}


async def run_case(model: str, case: str, credential: str, prepared: Path,
                   output_root: Path, scope: str, empty_audio_control=False) -> dict[str, Any]:
    case_root = output_root / "models" / model / scope / "cases" / case
    case_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for status_file in case_root.glob("attempt-*/status.json"):
        try:
            previous = json.loads(status_file.read_text())
            if previous.get("status") == "completed":
                completed.append(str(status_file.parent))
        except (OSError, ValueError):
            continue
    if completed:
        return {"model": model, "case": case, "status": "skipped_completed", "attempt": sorted(completed)[-1]}
    if scope == "run":
        reused = reuse_probe(case_root, output_root, model, case, prepared)
        if reused:
            return {"model": model, "case": case, "status": "skipped_completed", "attempt": str(reused), "reused_probe": True}
    attempt = case_root / ("attempt-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8])
    attempt.mkdir(exist_ok=False)
    started = time.monotonic()
    origin: float | None = None
    wall_origin: float | None = None
    parser = ResponseParser()
    result: dict[str, Any] = {"model": model, "case": case, "status": "running", "started_utc": utc(), "attempt": str(attempt)}
    save(attempt / "status.json", result, credential)
    save(attempt / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "websockets": websockets.__version__, "argv": sys.argv, "script": str(Path(__file__).resolve()),
        "prepared": str(prepared), "concurrency_scope": scope,
        "connection_route": "direct official WebSocket; proxy disabled for this run only"}, credential)
    save(attempt / "runtime-source.json", {"source": Path(__file__).read_text(),
                                         "source_file": str(Path(__file__).resolve())}, credential)
    log_file = (attempt / "runlog.jsonl").open("x", encoding="utf-8")
    input_file = (attempt / "input-events.jsonl").open("x", encoding="utf-8")
    raw_file = (attempt / "raw-output.jsonl").open("x", encoding="utf-8")

    def write_event(handle: Any, record: Any) -> None:
        handle.write(json.dumps(redact(record, credential), ensure_ascii=False) + "\n")
        handle.flush()

    def log(kind: str, **details: Any) -> None:
        write_event(log_file, {"utc": utc(), "event": kind, **details})

    def clock_drift():
        if origin is None or wall_origin is None:
            return 0.0
        return (time.time() - wall_origin) - (time.monotonic() - origin)

    receiver_task: asyncio.Task[Any] | None = None
    try:
        source = prepared / case
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        result.update(source_audio_present=bool(manifest.get("audio_present")),
                      source_audio_samples=int(manifest.get("audio_samples", 0)))
        if manifest.get("synthetic_audio_added"):
            raise ValueError("This protocol forbids synthesized task audio or silence.")
        instructions = (source / manifest.get("system_file", "SYSTEM.txt")).read_text(encoding="utf-8")
        events = schedule(manifest)
        duration = float(manifest["duration_sec"])
        audio = (source / manifest["audio_file"]).read_bytes() if manifest.get("audio_present") else b""
        if audio and len(audio) != 2 * int(manifest["audio_samples"]):
            raise ValueError("Prepared PCM length differs from its declared original samples.")
        for frame in manifest.get("frames", []):
            if not (source / frame["file"]).is_file():
                raise FileNotFoundError(source / frame["file"])
        config = session_config(model, instructions)
        endpoint = ENDPOINT + "?model=" + quote(model, safe="")
        save(attempt / "input-spec.json", {"case": case, "source_manifest": str(source / "manifest.json"),
            "source_video": manifest.get("source_video"), "prepared": str(source), "duration_sec": duration,
            "query": manifest.get("query"), "instructions": instructions, "audio_present": bool(audio),
            "audio_samples": manifest.get("audio_samples", 0), "audio_chunk_seconds": 0.2,
            "frame_pts_sec": [frame["pts_sec"] for frame in manifest.get("frames", [])],
            "playback_rate": 1, "observation_seconds": 2, "max_input_drift_seconds": 2,
            "query_delivery": "session.instructions (unchanged SYSTEM.txt)",
            "empty_audio_control_probe": empty_audio_control and not bool(audio),
            "generation_trigger": "provider-native semantic VAD only", "task_tts": False,
            "padding": False, "manual_commit": False, "manual_response_create": False,
            "first_frame_rule": "With original audio, send first frame after first causal audio packet; preserve its source PTS.",
            "schedule": events}, credential)
        save(attempt / "session_config.json", {"endpoint": endpoint, "requested": config}, credential)
        created = asyncio.Event()
        updated = asyncio.Event()
        provider_error = asyncio.Event()
        disconnected = asyncio.Event()
        errors: list[dict[str, Any]] = []
        echoes: list[dict[str, Any]] = []
        sent_audio_samples = 0
        sent_frames = 0
        last_input_media_s: float | None = None
        max_drift = 0.0

        async with websockets.connect(endpoint, proxy=None, additional_headers={"Authorization": "Bearer " + credential},
                open_timeout=30, close_timeout=10, max_size=None, ping_interval=20, ping_timeout=30) as ws:
            log("connected", endpoint=endpoint)

            async def receive() -> None:
                try:
                    async for message in ws:
                        received_mono = time.monotonic()
                        media_s = received_mono - origin if origin is not None else None
                        try:
                            event = json.loads(message)
                        except (TypeError, ValueError):
                            write_event(raw_file, {"received_utc": utc(), "media_s": media_s, "non_json_message": str(message)})
                            errors.append({"type": "non_json_provider_message"})
                            provider_error.set()
                            continue
                        received_at = utc()
                        write_event(raw_file, {"received_utc": received_at, "media_s": media_s, "event": event})
                        parser.feed(event, media_s, received_at)
                        kind = event.get("type")
                        if kind in ("session.created", "session.updated"):
                            echoes.append(event)
                            save(attempt / "session_config.json", {"endpoint": endpoint, "requested": config, "provider_echoes": echoes}, credential)
                            (created if kind == "session.created" else updated).set()
                        elif kind == "error":
                            errors.append(event)
                            provider_error.set()
                            log("provider_error", details=event)
                        elif kind == "response.done" and event.get("response", {}).get("status") == "failed":
                            errors.append(event)
                            provider_error.set()
                            log("provider_generation_failed", details=event.get("response", {}).get("status_details"))
                        if abs(clock_drift()) > 1:
                            errors.append({"type": "clock_drift", "wall_minus_monotonic_s": clock_drift()})
                            provider_error.set()
                except Exception as exc:
                    errors.append({"type": "receive_error", "message": str(exc)})
                    log("receive_error", message=str(exc))
                    provider_error.set()
                finally:
                    disconnected.set()

            receiver_task = asyncio.create_task(receive())

            async def wait_signal(signal: asyncio.Event, timeout: float, stage: str) -> None:
                tasks = [asyncio.create_task(item.wait()) for item in (signal, provider_error, disconnected)]
                try:
                    done, _ = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                    if provider_error.is_set():
                        raise RuntimeError(f"Provider rejected {stage}: {json.dumps(errors[-1], ensure_ascii=False)}")
                    if not signal.is_set():
                        if disconnected.is_set():
                            raise RuntimeError(f"Provider disconnected during {stage}.")
                        raise TimeoutError(f"No {stage} acknowledgement within {timeout:g} seconds.")
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            await wait_signal(created, 30, "session.created")
            await ws.send(json.dumps(config, ensure_ascii=False))
            write_event(input_file, {"sent_utc": utc(), "media_s": None, "event": config})
            await wait_signal(updated, 30, "session.updated")
            verification = verify_session(config["session"], echoes[-1].get("session", {}))
            save(attempt / "session-verification.json", verification, credential)
            if not verification["passed"]:
                raise RuntimeError("Provider session settings differ or are not echoed: " + ", ".join(verification["mismatches"]))
            origin = time.monotonic()
            wall_origin = time.time()
            log("playback_started", duration_sec=duration)
            if empty_audio_control and not audio:
                # A single zero-byte API control message, not a PCM sample or
                # silence. Isolated interface probe; never covertly pad media.
                control = {"type": "input_audio_buffer.append", "audio": "",
                           "event_id": "evt-" + uuid.uuid4().hex}
                await ws.send(json.dumps(control))
                write_event(input_file, {"sent_utc": utc(), "media_s": time.monotonic() - origin,
                                        "event": control, "kind": "empty_audio_buffer_control",
                                        "audio_bytes": 0, "source_samples": 0})

            async def wait_until(deadline: float) -> None:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    blockers = [asyncio.create_task(item.wait()) for item in (provider_error, disconnected)]
                    try:
                        await asyncio.wait(blockers, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for blocker in blockers:
                            blocker.cancel()
                        await asyncio.gather(*blockers, return_exceptions=True)
                if provider_error.is_set():
                    raise RuntimeError(f"Native provider error: {json.dumps(errors[-1], ensure_ascii=False)}")
                if disconnected.is_set():
                    raise RuntimeError("Provider disconnected before playback and observation completed.")
                if abs(clock_drift()) > 1:
                    raise RuntimeError(f"Clock drift exceeds 1 second ({clock_drift():.6f}).")

            for item in events:
                await wait_until(origin + item["send_due_s"])
                actual = time.monotonic() - origin
                drift = actual - item["send_due_s"]
                max_drift = max(max_drift, drift)
                if drift > 2:
                    raise RuntimeError(f"Causal input drift exceeded 2 seconds ({drift:.6f}).")
                if item["kind"] == "audio":
                    chunk = audio[2 * item["sample_start"]:2 * item["sample_end"]]
                    outgoing = {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}
                else:
                    outgoing = {"type": "input_image_buffer.append", "image": base64.b64encode((source / item["file"]).read_bytes()).decode("ascii")}
                outgoing["event_id"] = "evt-" + uuid.uuid4().hex
                before_send = time.monotonic() - origin
                await ws.send(json.dumps(outgoing))
                after_send = time.monotonic() - origin
                max_drift = max(max_drift, after_send - item["send_due_s"])
                write_event(input_file, {"sent_utc": utc(), "event_id": outgoing["event_id"],
                    "event_type": outgoing["type"], "media_s": before_send, "send_completed_media_s": after_send,
                    "send_duration_s": after_send - before_send, "input_drift_s": before_send - item["send_due_s"],
                    "path": str(source / item["file"]), **item})
                if item["kind"] == "audio":
                    sent_audio_samples += item["sample_end"] - item["sample_start"]
                else:
                    sent_frames += 1
                last_input_media_s = after_send
                if after_send - item["send_due_s"] > 2:
                    raise RuntimeError(f"Causal input send drift exceeded 2 seconds ({after_send - item['send_due_s']:.6f}).")
            await wait_until(origin + duration + 2)
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
            if sent_audio_samples != int(manifest.get("audio_samples", 0)) or sent_frames != len(manifest.get("frames", [])):
                raise RuntimeError("Incomplete prepared-media delivery.")
            result.update(status="completed", observation_completed_media_s=time.monotonic() - origin)
            save(attempt / "transport-result.json", {"status": "completed", "sent_audio_samples": sent_audio_samples,
                "sent_frames": sent_frames, "duration_sec": duration, "last_input_media_s": last_input_media_s,
                "max_input_drift_s": max_drift, "wall_minus_monotonic_s": clock_drift(),
                "provider_errors": errors, "session_acknowledged": updated.is_set()}, credential)
    except asyncio.CancelledError:
        result.update(status="failed", error_type="CancelledError", error="Evaluation queue interrupted; partial attempt retained",
                      failure_category="environment_interruption", retryable=True)
        log("interrupted", reason=result["error"])
        save(attempt / "transport-result.json", {**result, "sent_audio_samples": locals().get("sent_audio_samples", 0),
            "sent_frames": locals().get("sent_frames", 0), "max_input_drift_s": locals().get("max_drift"),
            "provider_errors": locals().get("errors", [])}, credential)
        raise
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        response_status = getattr(getattr(exc, "response", None), "status_code", None)
        result["fatal_credential_error"] = response_status in (401, 403) or fatal_credential_error(
            {"error": str(exc), "provider_errors": locals().get("errors", [])})
        log("failed", error_type=type(exc).__name__, error=str(exc))
        save(attempt / "transport-result.json", {**result, "sent_audio_samples": locals().get("sent_audio_samples", 0),
            "sent_frames": locals().get("sent_frames", 0), "max_input_drift_s": locals().get("max_drift"),
            "provider_errors": locals().get("errors", [])}, credential)
    finally:
        if receiver_task is not None:
            if not receiver_task.done():
                receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)
        parsed = parser.export()
        save(attempt / "parsed-output.json", parsed, credential)
        save(attempt / "latency.json", {"clock": "monotonic seconds from acknowledged session.update playback origin",
            "setup_s": origin - started if origin is not None else None, "wall_s": time.monotonic() - started,
            "responses": [{"response_id": row["response_id"], "first_text_media_s": row["first_text_media_s"],
                "first_audio_media_s": row["first_audio_media_s"], "start_media_s": row["start_media_s"]} for row in parsed["responses"]],
            "usage": parsed["usage"], "api_cost": None,
            "api_cost_status": "Record native provider usage; no invented monetary cost."}, credential)
        result.update(finished_utc=utc(), wall_s=time.monotonic() - started,
            provider_models=parsed["provider_models"], response_count=len(parsed["responses"]))
        save(attempt / "status.json", result, credential)
        log("finished", status=result["status"], response_count=result["response_count"])
        for handle in (log_file, input_file, raw_file):
            handle.close()
    return redact(result, credential)
