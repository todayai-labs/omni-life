"""Integer-sample v3 source-aligned AV transport for official-ASR models.

This is an application-pipeline adapter, NOT a native audio encoder baseline.
Video is released by source PTS independently of ASR/model compute. Audio reaches
the official streaming ASR only after each real segment's end. All transcripts,
including revisions, are recorded and become visible only after ASR completes.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import traceback
import uuid


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SourceAlignedASRAdapter:
    minimum_frames = 1

    def run_case(self, prepared_dir, output_dir):
        import requests
        prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((prepared_dir / "manifest.json").read_text())
        self.manifest = manifest
        self.case_usage = {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "asr_calls": 0}
        system = (prepared_dir / manifest["system_file"]).read_text()
        audio = (prepared_dir / manifest["audio_file"]).read_bytes() if manifest["audio_present"] else b""
        if audio and digest(audio) != manifest["audio_sha256"]:
            raise ValueError("Source audio hash mismatch")
        if digest(system.encode()) != manifest["system_sha256"]:
            raise ValueError("Source system hash mismatch")
        streams = {k: (output_dir / k).open("x", encoding="utf8") for k in
                   ("input.jsonl", "raw-output.jsonl", "runlog.jsonl")}
        origin, lock = None, threading.Lock()
        def log(kind, event):
            now = time.monotonic()
            value = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": now,
                     "media_wall_s": now - origin if origin is not None else None, **event}
            with lock:
                streams[kind].write(json.dumps(value, ensure_ascii=False) + "\n")
                streams[kind].flush()
            return value
        session_id = manifest["id"] + "-" + uuid.uuid4().hex
        endpoint = self.config.get("asr_url", "http://127.0.0.1:8034")
        health = requests.get(endpoint + "/health", timeout=20).json() if audio else None
        if audio:
            log("raw-output.jsonl", {"kind": "asr_health", "raw": health})
        expected_asr = self.config.get("asr_expected", {})
        if audio and expected_asr:
            actual = health.get("config", {})
            if any(actual.get(k) != v for k, v in expected_asr.items()):
                raise ValueError("ASR runtime differs from frozen expected configuration")
            if not health.get("runtime_identity", {}).get("bridge_sha256"):
                raise ValueError("ASR runtime identity missing")
        if audio:
            requests.post(endpoint + "/start?session=" + session_id, timeout=30).raise_for_status()
        self.start_session(system, output_dir, log)
        self.case_seed = (int(self.config.get("seed", 0)) + int(digest(manifest["id"].encode())[:8], 16)) % (2 ** 31)
        self.turn_index = 0
        log("input.jsonl", {"kind": "system", "text": system, "query": manifest["query"],
                            "sha256": manifest["system_sha256"], "manifest": str(prepared_dir / "manifest.json")})
        settings = {**self.settings, "input_route": "official_vlm_plus_causal_streaming_asr",
                    "case_seed": self.case_seed,
                    "asr_health": health, "tts": False, "playback_speed": 1.0,
                    "asr_application_adaptation": "always-on source audio replaces demo push-to-talk boundaries; official Qwen3-ASR streaming state and unpadded finalize",
                    "offline_full_video_transcript": False}
        log("runlog.jsonl", {"event": "case_start", "id": manifest["id"], "settings": settings})
        audio_work, model_work = queue.Queue(), queue.Queue()
        cancelled = threading.Event()
        origin = time.monotonic()
        expected = len(audio) // 2
        duration = max(float(manifest["duration_sec"]), expected / 16000)
        coverage = {"frames_expected": len(manifest["frames"]), "frames_received": 0,
                    "source_audio_present": manifest["audio_present"],
                    "source_audio_samples_expected": expected,
                    "source_audio_samples_received": 0, "source_audio_samples_processed": 0,
                    "uncovered_tail_samples": expected}
        def asr_worker():
            try:
                while True:
                    item = audio_work.get()
                    if item is None:
                        break
                    index, start, end, raw, final = item
                    send = log("input.jsonl", {"kind": "asr_audio_send", "index": index,
                        "source_start_s": start, "source_end_s": end, "sample_start": index * 16000,
                        "sample_count": len(raw) // 2, "sha256": digest(raw),
                        "raw_source_file": str(prepared_dir / manifest["audio_file"]), "final": final})
                    response = requests.post(endpoint + "/chunk?session=" + session_id, data=raw,
                                headers={"Content-Type": "application/octet-stream", "X-Final": "1" if final else "0"}, timeout=240)
                    response.raise_for_status()
                    result = response.json()
                    self.case_usage["asr_calls"] += 1
                    received = log("raw-output.jsonl", {"kind": "asr_result", "raw": result,
                                      "source_start_s": start, "source_end_s": end,
                                      "send_monotonic_s": send["monotonic_s"]})
                    if expected_asr and result.get("runtime_identity") != health.get("runtime_identity"):
                        raise ValueError("ASR runtime changed during source replay; raw preserved, do not mix settings")
                    coverage["source_audio_samples_received"] = result["received_samples"]
                    coverage["source_audio_samples_processed"] = result["processed_samples"]
                    model_work.put(("transcript", {"text": result["text"], "raw": result,
                                   "source_end_s": end, "available_monotonic_s": received["monotonic_s"]}))
            except Exception:
                model_work.put(("error", traceback.format_exc()))
            finally:
                model_work.put(("asr_done", None))
        def produce():
            try:
                cursor = 0
                schedule = [(float(i), min(float(i + 1), duration)) for i in range(math.ceil(duration))] if audio else [(float(f["pts_sec"]), float(f["pts_sec"])) for f in manifest["frames"]]
                for index, (start, end) in enumerate(schedule):
                    if cancelled.wait(max(0, origin + end - time.monotonic())):
                        break
                    selected = []
                    while cursor < len(manifest["frames"]) and manifest["frames"][cursor]["pts_sec"] <= end:
                        selected.append(manifest["frames"][cursor])
                        cursor += 1
                    for frame in selected:
                        data = (prepared_dir / frame["file"]).read_bytes()
                        if digest(data) != frame["sha256"]:
                            raise ValueError("Source frame hash mismatch")
                    log("input.jsonl", {"kind": "source_release", "frames": selected,
                                      "source_start_s": start, "source_end_s": end, "index": index})
                    if selected:
                        model_work.put(("frames", selected))
                    if audio:
                        low, high = index * 16000, min(expected, (index + 1) * 16000)
                        if high > low:
                            audio_work.put((index, start, end, audio[low * 2:high * 2], high == expected))
                if not audio:
                    cancelled.wait(max(0, origin + duration - time.monotonic()))
            except Exception:
                model_work.put(("error", traceback.format_exc()))
            finally:
                audio_work.put(None)
                model_work.put(("video_done", None))
        producer = threading.Thread(target=produce, daemon=True)
        asr = threading.Thread(target=asr_worker, daemon=True) if audio else None
        producer.start()
        if asr:
            asr.start()
        video_done, asr_done = False, not bool(audio)
        pending, previous_transcript, pending_transcript, last_frame = [], "", None, None
        responses, model_latencies, error = [], [], None
        try:
            while True:
                try:
                    kind, item = model_work.get(timeout=0.05)
                    items = [(kind, item)]
                    while True:
                        try:
                            items.append(model_work.get_nowait())
                        except queue.Empty:
                            break
                    for kind, item in items:
                        if kind == "frames":
                            pending.extend(item)
                            last_frame = item[-1]
                        elif kind == "transcript":
                            if item["text"] != previous_transcript:
                                pending_transcript = item
                                previous_transcript = item["text"]
                        elif kind == "video_done":
                            video_done = True
                        elif kind == "asr_done":
                            asr_done = True
                        elif kind == "error":
                            raise RuntimeError(item)
                except queue.Empty:
                    pass
                enough = len(pending) >= self.minimum_frames
                flush_tail = video_done and bool(pending)
                speech_trigger = pending_transcript is not None and last_frame is not None
                if enough or flush_tail or speech_trigger:
                    current = pending if pending else [last_frame]
                    unique = list(pending)
                    pending = []
                    transcript = pending_transcript
                    pending_transcript = None
                    prompt = transcript["text"] if transcript else ""
                    send = log("input.jsonl", {"kind": "model_turn", "frames": current,
                        "new_source_frames": unique, "prompt": prompt,
                        "prompt_source": "completed_source_asr" if transcript else "autonomous_visual_tick",
                        "asr_available_monotonic_s": transcript["available_monotonic_s"] if transcript else None})
                    result = self.process_frames(prepared_dir, current, prompt, log)
                    self.turn_index += 1
                    coverage["frames_received"] += len(unique)
                    for response in result.get("responses", []):
                        responses.append(response)
                    model_latencies.append(time.monotonic() - send["monotonic_s"])
                if video_done and asr_done and model_work.empty() and not pending and pending_transcript is None:
                    break
        except Exception:
            error = traceback.format_exc()
            log("runlog.jsonl", {"event": "error", "traceback": error})
        finally:
            cancelled.set()
            producer.join(timeout=2)
            if asr:
                asr.join(timeout=2)
            if hasattr(self, "finish_session"):
                try:
                    self.finish_session(log)
                except Exception:
                    error = error or traceback.format_exc()
                    log("runlog.jsonl", {"event": "session_cleanup_error", "traceback": traceback.format_exc()})
        coverage["uncovered_tail_samples"] = expected - coverage["source_audio_samples_processed"]
        success = error is None and coverage["frames_received"] == coverage["frames_expected"] and coverage["uncovered_tail_samples"] == 0
        result = {"id": manifest["id"], "success": success,
                  "status": "completed" if success else "failed", "responses": responses,
                  "coverage": coverage, "settings": settings, "latency": {"model_turn_s": model_latencies,
                  "wall_elapsed_s": time.monotonic() - origin}, "usage": self.case_usage,
                  "billing": None, "error": error}
        log("runlog.jsonl", {"event": "case_complete", **result})
        for stream in streams.values():
            stream.close()
        (output_dir / "adapter-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
