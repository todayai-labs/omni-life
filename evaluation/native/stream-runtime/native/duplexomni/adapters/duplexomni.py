"""Official DuplexOmni Thinker AV protocol, text-only evaluation.

The model receives the official 480 ms chunk protocol with persistent
conversation_id and cached conversation history. No Talker/TTS or S2 model is
added. A separate producer releases canonical source chunks at real wall time.
Input logs losslessly store message deltas rather than duplicating all prior
audio/video in every request; request_sha256 covers the exact HTTP JSON bytes.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
import io
import json
import socket
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

import httpx


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _stamp(origin=None):
    now = time.monotonic()
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": now,
            "media_wall_s": None if origin is None else now - origin}


def _wav(pcm, sample_rate):
    target = io.BytesIO()
    with wave.open(target, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return target.getvalue()


def _parse_control(raw):
    """Only the official tts text field is an assistant response, not ASR."""
    clean = raw.strip()
    fenced = clean.startswith("```")
    if fenced and clean.endswith("```"):
        lines = clean.splitlines()
        if len(lines) >= 3 and lines[0].lower() in ("```", "```json", "```python"):
            clean = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(clean)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(clean)
        except (ValueError, SyntaxError):
            if clean.startswith("{") or fenced:
                return {}, "unparsed_structured_output"
            return {"tts": raw}, "plain_text"
    if not isinstance(parsed, dict):
        return {}, "non_dict_control_output"
    return parsed, "official_tts_text_field"


def _parse_text(raw):
    parsed, mode = _parse_control(raw)
    text = parsed.get("tts", "")
    return text if isinstance(text, str) else "", mode


def assemble_episodes(chunks):
    """Observable contiguous nonempty text episodes, not claimed native EOT."""
    episodes, current = [], None

    def close(reason, chunk):
        nonlocal current
        if current is not None:
            current["closure_reason"] = reason
            current["closure_media_s"] = chunk["receive"]["media_wall_s"] if chunk else current["end_media_s"]
            episodes.append(current)
            current = None

    for chunk in chunks:
        control, mode = _parse_control(chunk["raw"])
        stop = "[STOP]" in str(control.get("tts_control", ""))
        if stop:
            close("native_STOP_control", chunk)
        text = control.get("tts", "")
        if not isinstance(text, str):
            text = ""
        if not text:
            close("empty_tts_chunk" if not stop else "native_STOP_control", chunk)
            continue
        received = chunk["receive"]["media_wall_s"]
        if current is None:
            current = {"response_id": chunk["response_id"], "text": "", "start_media_s": received,
                       "end_media_s": received, "onset_source": "first_text_receive",
                       "source_seen_through_s": chunk["source_end_s"],
                       "onset_granularity": "complete_HTTP_response",
                       "unit": "observable_nonempty_tts_episode_not_native_end_of_turn",
                       "raw_output_indices": [], "fragment_receive_media_s": []}
        current["text"] += text
        current["end_media_s"] = received
        current["raw_output_indices"].append(chunk["index"])
        current["fragment_receive_media_s"].append(received)
    close("source_EOF_not_native_EOT", None)
    return episodes


class Adapter:
    @classmethod
    def load(cls, config):
        self = cls()
        self.config = dict(config)
        self.endpoint = config.get("endpoint", "http://127.0.0.1:19995")
        self.model_path = config["model_path"]
        # The official server does not support token streaming. We therefore
        # report text_receive, never claim a TTFT that was not observable.
        with httpx.Client(timeout=30) as client:
            health = client.get(self.endpoint + "/health")
            health.raise_for_status()
            self.health = health.json()
        if self.health.get("model") != self.model_path:
            raise ValueError("Duplex server model does not match pinned model_path")
        return self

    def run_case(self, prepared_dir, output_dir):
        return asyncio.run(self._run_case(Path(prepared_dir), Path(output_dir)))

    async def _run_case(self, prepared, output):
        output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((prepared / "manifest.json").read_text())
        system = (prepared / manifest["system_file"]).read_text()
        if _sha(system.encode()) != manifest["system_sha256"]:
            raise ValueError("System hash mismatch")
        if manifest["sample_rate_hz"] != 16000 or manifest["channels"] != 1:
            raise ValueError("Canonical source audio must be 16k mono")
        if any(q["start_time_sec"] != 0 for q in manifest["query"]):
            raise ValueError("Later queries require an explicit timed query adapter")
        source = b""
        if manifest["audio_present"]:
            source = (prepared / manifest["audio_file"]).read_bytes()
            if _sha(source) != manifest["audio_sha256"]:
                raise ValueError("Source audio hash mismatch")
            if len(source) // 2 != manifest["audio_samples"]:
                raise ValueError("Source audio sample count mismatch")
        frame_data = []
        for frame in manifest["frames"]:
            raw = (prepared / frame["file"]).read_bytes()
            if _sha(raw) != frame["sha256"]:
                raise ValueError("Source frame hash mismatch")
            frame_data.append((frame, raw))
        logs = {name: (output / (name + ".jsonl")).open("x")
                for name in ("input", "raw-output", "runlog")}

        def log(name, event):
            logs[name].write(json.dumps(event, ensure_ascii=False) + "\n")
            logs[name].flush()

        session_id = "omnilife-" + str(uuid.uuid4())
        messages = [{"role": "system", "content": system}]
        settings = {"model": "DuplexOmni", "checkpoint": self.model_path,
                    "adapter_host": socket.gethostname(),
                    "deployment_host": self.config.get("host"),
                    "physical_gpu": self.config.get("physical_gpu"),
                    "runtime_source": self.config.get("runtime_source"),
                    "official_repo_revision": "33bfba1a821b09c5aa66790944f9098584979d34",
                    "server_health": self.health, "chunk_ms": 480,
                    "audio_sample_rate_hz": 16000, "frame_source": "canonical_1fps_actual_PTS",
                    "temperature": 0.01, "top_p": 1.0, "max_tokens": 999, "stream": False,
                    "backend_default_seed": 0,
                    "protocol": "official_S1_chunk_cached_conversation",
                    "external_asr": False, "talker_started": False, "tts_started": False,
                    "s2_model_started": False, "synthetic_audio_added": False,
                    "input_log_encoding": "lossless_message_deltas_and_exact_request_sha256",
                    "text_onset_observable": "HTTP_text_response_receive_not_token_TTFT"}
        settings["response_unit"] = "observable_contiguous_nonempty_tts_emit_episode"
        settings["episode_closure"] = "empty_tts_chunk_or_native_STOP_or_source_EOF; no punctuation or gap threshold"
        settings["native_normal_end_of_turn_observable"] = False
        log("input", {"event": "system", "messages_append": messages,
                      "session_id": session_id, "source_manifest": str(prepared / "manifest.json"),
                      "source_manifest_sha256": _sha((prepared / "manifest.json").read_bytes()),
                      **_stamp()})
        origin = time.monotonic()
        queue = asyncio.Queue()
        responses = []
        output_chunks = []
        durations = []
        counters = {"frames_expected": len(frame_data), "frames_received": 0,
                    "source_audio_samples_expected": manifest["audio_samples"],
                    "source_audio_samples_received": 0, "uncovered_tail_samples": 0}
        usage = {"reported_prompt_tokens_sum": 0, "completion_tokens": 0,
                 "note": "prompt token sum includes cached repeated history, not unique tokens"}

        async def producer():
            # Release segment-end causal packets without padding or stretching.
            events = []
            samples_per_chunk = 7680
            for begin in range(0, len(source) // 2, samples_per_chunk):
                end = min(begin + samples_per_chunk, len(source) // 2)
                events.append({"start": begin / 16000, "end": end / 16000,
                               "audio_begin": begin, "audio_end": end})
            if not events:
                # The native S1 state still advances at 480 ms when the source
                # has no audio track. Empty audio_input is not a synthetic WAV.
                begin = 0.0
                while begin < manifest["duration_sec"] - 1e-9:
                    end = min(begin + 0.48, manifest["duration_sec"])
                    events.append({"start": begin, "end": end, "audio_begin": 0, "audio_end": 0})
                    begin = end
            # Frames beyond the real audio track still must arrive as video-only.
            if manifest["audio_present"]:
                for frame, _ in frame_data:
                    if frame["pts_sec"] > events[-1]["end"]:
                        events.append({"start": frame["pts_sec"], "end": frame["pts_sec"],
                                       "audio_begin": 0, "audio_end": 0})
            next_frame = 0
            for index, event in enumerate(events):
                await asyncio.sleep(max(0, origin + event["end"] - time.monotonic()))
                picked = []
                while next_frame < len(frame_data) and frame_data[next_frame][0]["pts_sec"] <= event["end"] + 1e-9:
                    picked.append(frame_data[next_frame])
                    next_frame += 1
                await queue.put((index, event, picked, _stamp(origin)))
            await queue.put(None)

        status, error = "completed", None
        producer_task = asyncio.create_task(producer())
        try:
            async with httpx.AsyncClient(timeout=self.config.get("request_timeout_s", 240)) as client:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    index, event, frames, ready = item
                    pcm = source[event["audio_begin"] * 2:event["audio_end"] * 2]
                    content = [{"type": "text", "text": "{'audio_input': '"}]
                    if pcm:
                        content.append({"type": "input_audio", "input_audio": {
                            "data": base64.b64encode(_wav(pcm, 16000)).decode(), "format": "wav"}})
                    content.append({"type": "text", "text": "', 'from_s2': ''"})
                    for frame, raw in frames:
                        content.extend([{"type": "text", "text": ", 'input_video': '"},
                                        {"type": "image_url", "image_url": {
                                            "url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}},
                                        {"type": "text", "text": "'"}])
                    content.append({"type": "text", "text": "}"})
                    user_message = {"role": "user", "content": content}
                    messages.append(user_message)
                    request = {"model": self.model_path, "messages": messages,
                               "max_tokens": 999, "temperature": 0.01, "stream": False,
                               "vllm_xargs": {"conversation_id": session_id}}
                    request_bytes = _json_bytes(request)
                    sent = _stamp(origin)
                    log("input", {"event": "request", "index": index,
                                  "messages_append": [user_message],
                                  "request_fields": {k: v for k, v in request.items() if k != "messages"},
                                  "request_sha256": _sha(request_bytes),
                                  "request_bytes": len(request_bytes), "history_messages": len(messages),
                                  "source_segment": event, "frame_pts_sec": [f[0]["pts_sec"] for f in frames],
                                  "ready": ready, "send": sent})
                    start = time.monotonic()
                    result = await client.post(self.endpoint + "/v1/chat/completions",
                                               content=request_bytes, headers={"Content-Type": "application/json"})
                    received = _stamp(origin)
                    log("raw-output", {"event": "http_response", "index": index,
                                       "http_status": result.status_code, "headers": dict(result.headers),
                                       "raw_body": result.text, "send": sent, "receive": received})
                    result.raise_for_status()
                    payload = result.json()
                    raw = payload["choices"][0]["message"].get("content") or ""
                    messages.append({"role": "assistant", "content": raw})
                    text, parse_mode = _parse_text(raw)
                    control, _ = _parse_control(raw)
                    output_chunks.append({"index": index, "raw": raw,
                                          "response_id": payload.get("id", f"chunk-{index}"),
                                          "receive": received, "source_end_s": event["end"]})
                    elapsed = time.monotonic() - start
                    durations.append(elapsed)
                    counters["frames_received"] += len(frames)
                    counters["source_audio_samples_received"] += len(pcm) // 2
                    usage["reported_prompt_tokens_sum"] += payload.get("usage", {}).get("prompt_tokens", 0)
                    usage["completion_tokens"] += payload.get("usage", {}).get("completion_tokens", 0)
                    log("runlog", {"event": "chunk_received", "index": index,
                                   "parse_mode": parse_mode, "source_end_s": event["end"],
                                   "inference_s": elapsed, "queue_wait_s": sent["monotonic_s"] - ready["monotonic_s"],
                                   "text": text, "native_control": control, **received})
                    if text:
                        responses.append({"response_id": payload.get("id", f"chunk-{index}"),
                                          "text": text, "start_media_s": received["media_wall_s"],
                                          "end_media_s": received["media_wall_s"],
                                          "onset_source": "first_text_receive",
                                          "source_seen_through_s": event["end"],
                                          "onset_granularity": "complete_HTTP_response",
                                          "raw_output_index": index})
        except Exception as exc:
            status, error = "failed", repr(exc)
            log("runlog", {"event": "error", "error": error, **_stamp(origin)})
        finally:
            if not producer_task.done():
                producer_task.cancel()
            await asyncio.gather(producer_task, return_exceptions=True)
            for handle in logs.values():
                handle.close()
        counters["uncovered_tail_samples"] = counters["source_audio_samples_expected"] - counters["source_audio_samples_received"]
        complete = counters["frames_expected"] == counters["frames_received"] and counters["uncovered_tail_samples"] == 0
        fragment_responses = responses
        responses = assemble_episodes(output_chunks)
        result = {"success": status == "completed" and complete, "status": status,
                  "error": error, "case_id": manifest["id"], "responses": responses,
                  "response_fragments": fragment_responses,
                  "coverage": counters, "settings": settings, "usage": usage, "billing": None,
                  "latency": {"wall_duration_s": time.monotonic() - origin,
                              "source_duration_s": manifest["duration_sec"],
                              "chunk_roundtrip_s": durations, "token_TTFT": None}}
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
