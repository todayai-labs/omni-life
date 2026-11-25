"""Chronological AV adapters for MOSS-VL and MOSS Video Preview.

The waveform is delivered causally to the official MOSS SenseVoice sidecar;
finalized utterances enter the same persistent session's text-input queue.
Preview's ASR connection is an explicitly named adaptation, not native audio.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import queue
import re
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

CONTROL = re.compile(r"(<\|[^<>|]+\|>|\[DONE\]|\[ERROR\])")
GENERIC_REMINDER = "继续执行最初任务；仅在原任务条件满足时回应，否则不输出。"


def reminder_options(config):
    """None means the untouched baseline; reminder experiments must opt in."""
    interval = config.get("reminder_interval_s")
    kind = config.get("reminder_kind")
    if interval is None:
        if kind is not None:
            raise ValueError("reminder_kind requires reminder_interval_s")
        return None, None
    interval = float(interval)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("reminder_interval_s must be finite and positive")
    if kind not in {"repeat_query", "generic"}:
        raise ValueError("reminder_kind must be repeat_query or generic")
    return interval, kind


def reminder_schedule(duration_s, interval_s):
    """Source-clock offsets strictly before EOF; no initial/EOF/drain reminder."""
    duration_s, interval_s = float(duration_s), float(interval_s)
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be finite and positive")
    return tuple(n * interval_s for n in range(1, math.ceil(duration_s / interval_s))
                 if n * interval_s < duration_s)


def reminder_text(kind, original_query):
    if kind == "repeat_query":
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("Repeated reminders require the original nonempty query")
        return f"提示：你的任务是 {original_query}"
    if kind == "generic":
        return GENERIC_REMINDER
    raise ValueError(f"Unsupported reminder_kind: {kind}")


async def run_reminders(*, duration_s, interval_s, kind, original_query, logger,
                        send, route, clock=time.monotonic, sleep=asyncio.sleep):
    """Add text through the native user-input queue, without driving decoding.

    `send(text, eof_clock)` must recheck EOF after obtaining any transport lock
    and return the monotonic instant at which actual sending/enqueueing begins.
    A missed deadline or failed send is an infrastructure failure, never a
    silently omitted experimental intervention. Cancellation is handled by the
    caller together with the AV producers.
    """
    offsets = reminder_schedule(duration_s, interval_s)
    text = reminder_text(kind, original_query)
    eof_clock = logger.start + duration_s
    logger.event("input", "reminder_plan", interval_s=interval_s, reminder_kind=kind,
                 planned_media_s=list(offsets), text=text, role="user", route=route,
                 task_progress_reset=False, forced_generate=False)
    for index, offset in enumerate(offsets):
        await sleep(max(0, logger.start + offset - clock()))
        before = clock()
        if before >= eof_clock:
            raise RuntimeError(f"Reminder {index} missed source EOF; no late prompt sent")
        logger.event("input", "reminder_send_attempt", reminder_index=index, reminder_kind=kind,
                     planned_media_s=offset, attempt_media_s=before - logger.start,
                     attempt_lateness_s=before - logger.start - offset, text=text,
                     role="user", route=route, payload={"type": "prompt", "text": text})
        try:
            send_started = await send(text, eof_clock)
        except Exception as exc:
            logger.event("runlog", "reminder_send_failed", reminder_index=index,
                         planned_media_s=offset, reminder_kind=kind, error=repr(exc))
            raise
        after = clock()
        logger.event("input", "reminder_sent", reminder_index=index, reminder_kind=kind,
                     planned_media_s=offset, actual_media_s=send_started - logger.start,
                     lateness_s=send_started - logger.start - offset,
                     send_complete_media_s=after - logger.start,
                     delivery_wait_s=send_started - before,
                     send_latency_s=after - send_started, text=text, role="user", route=route)


def request(url, data=None, binary=False):
    headers = {"Content-Type": "application/octet-stream" if binary else "application/json"}
    encoded = data if binary else json.dumps(data or {}).encode()
    with urllib.request.urlopen(urllib.request.Request(url, data=encoded, headers=headers), timeout=120) as response:
        return json.loads(response.read())


class CaseLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.start = None
        self.lock = threading.Lock()
        self.handles = {name: (self.path / f"{name}.jsonl").open("w")
                        for name in ("input", "raw-output", "runlog")}
        self.tokens = []

    def event(self, log, kind, **payload):
        now = time.monotonic()
        item = {"event": kind, "wall_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": now, "media_s": now - self.start if self.start is not None else None,
                **payload}
        with self.lock:
            self.handles[log].write(json.dumps(item, ensure_ascii=False) + "\n")
            self.handles[log].flush()
        return item

    def token(self, text, **payload):
        item = self.event("raw-output", "model_text", text=str(text), **payload)
        self.tokens.append(item)

    def parsed(self):
        responses = []
        text, onset, ending = "", None, None

        def flush():
            nonlocal text, onset, ending
            if text.strip():
                responses.append({"response_id": f"r{len(responses):04d}", "text": text,
                                  "start_media_s": onset, "end_media_s": ending,
                                  "onset_source": "first_text_receive"})
            text, onset, ending = "", None, None

        for item in self.tokens:
            for part in CONTROL.split(item["text"]):
                if not part:
                    continue
                if CONTROL.fullmatch(part):
                    if part in {"<|round_start|>", "<|round_end|>", "<|silence|>",
                                "<|eot_id|>", "<|im_end|>", "[DONE]", "[ERROR]"}:
                        flush()
                    continue
                if onset is None and part.strip():
                    onset = item["media_s"]
                text += part
                ending = item["media_s"]
        flush()
        return responses

    def close(self):
        for handle in self.handles.values():
            handle.close()


class SystemProcessor:
    """Replace Preview's fixed default system text at its public API boundary.

    Does not modify model weights/generation/cache. The exact injected system
    message is saved in every case's input ledger.
    """
    def __init__(self, processor, system):
        self.processor, self.system = processor, system

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        messages = [dict(m, content=self.system) if m.get("role") == "system" else m
                    for m in messages]
        return self.processor.apply_chat_template(messages, *args, **kwargs)


class Adapter:
    @classmethod
    def load(cls, config):
        self = cls()
        self.config = dict(config)
        self.reminder_interval_s, self.reminder_kind = reminder_options(config)
        self.kind = config.get("kind", "moss-vl-realtime")
        self.asr_url = config.get("asr_url", "http://127.0.0.1:18120")
        with urllib.request.urlopen(self.asr_url + "/health", timeout=30) as response:
            self.asr_health = json.loads(response.read())
        if not self.asr_health["ready"]:
            raise RuntimeError("Official SenseVoice sidecar is not ready")
        self.settings = {"kind": self.kind, "seed": config.get("seed", 42),
                         "seed_effective": self.kind == "moss-preview-realtime",
                         "input_route": "official_asr_pipeline" if self.kind == "moss-vl-realtime" else "asr_adapted_preview",
                         "audio_sample_rate": 16000, "source_audio_packet_s": 0.1,
                         "asr": self.asr_health, "decode": "greedy", "tts": False,
                         "frame_pts": "prepared-v1 actual source PTS",
                         "video_prefix_replay": False,
                         "post_video_drain_s": float(config.get("drain_s", 5.0)),
                         "model_path": config.get("model_path"),
                         "endpoint": config.get("endpoint", "ws://127.0.0.1:18100/v1/realtime")}
        self.settings["max_tokens_per_second"] = 12 if self.kind == "moss-vl-realtime" else 86400
        self.settings["max_new_tokens"] = 86400
        if self.reminder_interval_s is not None:
            self.settings["reminder_interval_s"] = self.reminder_interval_s
            self.settings["reminder_kind"] = self.reminder_kind
            self.settings["reminder_clock"] = "source_replay_monotonic"
            self.settings["reminder_route"] = ("official_ws_prompt_to_user_queue"
                if self.kind == "moss-vl-realtime" else "official_preview_new_prompts_user_queue")
        self.settings["native_cap_note"] = (
            "MOSS max_tokens_per_turn is the official tokens/sec pacing parameter, not a per-turn length cap. "
            "Preview uses its official API and script default of 86400; realtime stopping is native silence/stop, "
            "and offline max_length stopping is not patched.")
        if self.kind == "moss-preview-realtime":
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            torch.manual_seed(self.settings["seed"])
            self.processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True,
                                                          frame_extract_num_threads=1)
            self.model = AutoModelForCausalLM.from_pretrained(config["model_path"], trust_remote_code=True,
                         device_map={"": config.get("device", "cuda:0")}, torch_dtype=torch.bfloat16,
                         attn_implementation="flash_attention_2")
            self.model.eval()
            if not callable(getattr(self.model, "real_time_generate", None)):
                raise RuntimeError("Selected Preview checkpoint lacks its native streaming API")
        return self

    def run_case(self, prepared_dir, output_dir):
        return asyncio.run(self._run(Path(prepared_dir), Path(output_dir)))

    async def _run(self, prepared, output):
        manifest = json.loads((prepared / "manifest.json").read_text())
        system = (prepared / manifest.get("system_file", "SYSTEM.txt")).read_text()
        original_query = manifest["query"][0]["text"]
        query = "" if self.kind == "moss-vl-realtime" else original_query
        logger = CaseLog(output)
        sid = f"{self.kind}-{manifest['id']}-{uuid.uuid4().hex}"
        logger.event("input", "source_manifest", prepared_dir=str(prepared), manifest=manifest)
        logger.event("input", "initial_task", system_prompt=system, prompt=query,
                     original_query=original_query,
                     system_sha256=hashlib.sha256(system.encode()).hexdigest())
        received_frames, received_samples = 0, 0
        error = None
        asr_latencies = []
        ws, receiver, generator = None, None, None
        replay_tasks = []
        ws_input_lock = asyncio.Lock()
        images, prompts, tokens = queue.Queue(), queue.Queue(), queue.Queue()
        gen_error = []
        model_done = threading.Event()
        transport_started = time.monotonic()
        asr_active = False
        try:
            await asyncio.to_thread(request, self.asr_url + f"/sessions/{sid}", {})
            asr_active = True
            if self.kind == "moss-vl-realtime":
                import websockets
                ws = await websockets.connect(self.settings["endpoint"], max_size=32 * 1024**2, open_timeout=180)
                start = {"type": "start", "system_prompt": system, "prompt": query,
                         "frame_queue_size": 256, "max_tokens_per_second": 12,
                         "max_new_tokens": 86400, "do_sample": False, "repetition_penalty": 1.0}
                logger.event("input", "session_start", payload=start)
                await ws.send(json.dumps(start, ensure_ascii=False))
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), 180))
                    logger.event("raw-output", "transport", payload=message)
                    if message.get("type") == "error":
                        raise RuntimeError(message)
                    if message.get("type") == "ready":
                        break

                async def receive():
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("type") == "output":
                            logger.token(message.get("text", ""), raw_payload=message)
                        else:
                            logger.event("raw-output", "transport", payload=message)
                        if message.get("type") == "error":
                            raise RuntimeError(message)
                        if message.get("type") == "session_end":
                            return
                receiver = asyncio.create_task(receive())
            else:
                from PIL import Image
                processor = SystemProcessor(self.processor, system)
                prompts.put(query)

                def generate():
                    try:
                        self.model.real_time_generate(images, prompts, tokens, processor,
                            max_tokens_per_turn=86400, max_new_tokens=86400,
                            do_sample=False, repetition_penalty=1.0)
                    except Exception as exc:
                        gen_error.append(repr(exc))
                    finally:
                        model_done.set()
                generator = threading.Thread(target=generate, daemon=True)
                generator.start()

                async def receive():
                    while not model_done.is_set() or not tokens.empty():
                        try:
                            token = tokens.get_nowait()
                            logger.token(token)
                            if str(token).startswith("[ERROR]"):
                                raise RuntimeError(str(token))
                        except queue.Empty:
                            await asyncio.sleep(0.01)
                receiver = asyncio.create_task(receive())

            logger.start = time.monotonic()
            logger.event("runlog", "source_replay_start", wall_time=time.time())

            async def inject(result):
                transcript = result.get("transcript")
                if transcript is None:
                    return
                logger.event("raw-output", "asr_result", **transcript)
                asr_latencies.append(transcript["decode_ms"])
                text = transcript.get("text", "").strip()
                if text:
                    # Preserve the actual transcript. Never add question scaffolding.
                    logger.event("input", "asr_text_to_model", text=text,
                                 causal_audio_end_s=transcript["audio_end_sample"] / 16000)
                    if ws is not None:
                        async with ws_input_lock:
                            await ws.send(json.dumps({"type": "prompt", "text": text}, ensure_ascii=False))
                    else:
                        prompts.put(text)

            async def video():
                nonlocal received_frames
                for frame in manifest["frames"]:
                    await asyncio.sleep(max(0, logger.start + frame["pts_sec"] - time.monotonic()))
                    path = prepared / frame["file"]
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != frame["sha256"]:
                        raise RuntimeError(f"Prepared frame checksum mismatch: {path}")
                    if ws is not None:
                        # Keep each frame's metadata + bytes together even when
                        # an independent reminder/ASR prompt becomes due.
                        async with ws_input_lock:
                            await ws.send(json.dumps({"type": "frame", "timestamp": frame["pts_sec"]}))
                            await ws.send(data)
                    else:
                        from PIL import Image
                        with Image.open(path) as image:
                            images.put(image.convert("RGB").copy())
                    received_frames += 1
                    logger.event("input", "video_frame", pts_sec=frame["pts_sec"],
                                 bytes=len(data), file=str(path), sha256=frame["sha256"])
                    if receiver.done():
                        await receiver
                        raise RuntimeError("Model output stream stopped before source EOF")

            async def audio():
                nonlocal received_samples
                if not manifest["audio_present"]:
                    logger.event("input", "source_audio_absent", synthetic_audio_added=False)
                    return
                path = prepared / manifest["audio_file"]
                with path.open("rb") as handle:
                    while data := handle.read(3200):
                        end_sample = received_samples + len(data) // 2
                        # Deliver only after every sample in this packet has occurred.
                        await asyncio.sleep(max(0, logger.start + end_sample / 16000 - time.monotonic()))
                        before = received_samples
                        result = await asyncio.to_thread(request, self.asr_url + f"/sessions/{sid}/pcm", data, True)
                        received_samples = result["source_samples_received"]
                        if received_samples != end_sample:
                            raise RuntimeError("ASR source sample counter mismatch")
                        logger.event("input", "audio_packet", file=str(path),
                                     start_sample=before, end_sample=end_sample, bytes=len(data),
                                     sha256=hashlib.sha256(data).hexdigest(), payload_route="official_sensevoice_sidecar")
                        await inject(result)

            async def send_reminder(text, eof_clock):
                if receiver.done():
                    await receiver
                    raise RuntimeError("Model output stream ended before reminder delivery")
                if ws is not None:
                    async with ws_input_lock:
                        if time.monotonic() >= eof_clock:
                            raise RuntimeError("Reminder lock wait crossed source EOF")
                        send_started = time.monotonic()
                        await ws.send(json.dumps({"type": "prompt", "text": text}, ensure_ascii=False))
                        return send_started
                else:
                    if time.monotonic() >= eof_clock:
                        raise RuntimeError("Reminder queue delivery crossed source EOF")
                    if model_done.is_set():
                        raise RuntimeError("Preview generator ended before reminder delivery")
                    send_started = time.monotonic()
                    prompts.put_nowait(text)
                    return send_started

            replay_tasks = [asyncio.create_task(video()), asyncio.create_task(audio())]
            if self.reminder_interval_s is not None:
                replay_tasks.append(asyncio.create_task(run_reminders(
                    duration_s=manifest["duration_sec"], interval_s=self.reminder_interval_s,
                    kind=self.reminder_kind, original_query=original_query, logger=logger,
                    send=send_reminder, route=self.settings["reminder_route"])))
            await asyncio.gather(*replay_tasks)
            await asyncio.sleep(max(0, logger.start + manifest["duration_sec"] - time.monotonic()))
            result = await asyncio.to_thread(request, self.asr_url + f"/sessions/{sid}/end", {})
            asr_active = False
            await inject(result)
            await asyncio.sleep(self.settings["post_video_drain_s"])
            if ws is not None:
                logger.event("input", "stop", payload={"type": "stop"})
                await ws.send('{"type":"stop"}')
                try:
                    await asyncio.wait_for(receiver, timeout=30)
                except asyncio.TimeoutError:
                    receiver.cancel()
                    logger.event("runlog", "stop_ack_timeout")
            else:
                self.model.stop_real_time_generate()
                await asyncio.to_thread(generator.join, 20)
                if generator.is_alive():
                    raise RuntimeError("Preview generation thread did not stop safely")
                await receiver
                if gen_error:
                    raise RuntimeError(gen_error[0])
        except Exception as exc:
            error = repr(exc)
            logger.event("runlog", "failure", error=error)
        finally:
            # Stop every AV/reminder producer before closing logs/transport.
            # In particular, gather does not cancel its siblings after an error.
            for task in replay_tasks:
                if not task.done():
                    task.cancel()
            if replay_tasks:
                await asyncio.gather(*replay_tasks, return_exceptions=True)
            if asr_active:
                try:
                    await asyncio.to_thread(request, self.asr_url + f"/sessions/{sid}/end", {})
                except Exception as exc:
                    logger.event("runlog", "asr_close_error", error=repr(exc))
            if self.kind == "moss-preview-realtime" and generator is not None and generator.is_alive():
                self.model.stop_real_time_generate()
                await asyncio.to_thread(generator.join, 20)
            if ws is not None:
                await ws.close()
            if receiver is not None:
                if not receiver.done():
                    receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            responses = logger.parsed()
            coverage = {"frames_expected": len(manifest["frames"]), "frames_received": received_frames,
                        "source_audio_samples_expected": manifest["audio_samples"],
                        "source_audio_samples_received": received_samples,
                        "uncovered_tail_samples": max(0, manifest["audio_samples"] - received_samples),
                        "source_audio_present": manifest["audio_present"]}
            success = error is None and received_frames == len(manifest["frames"]) and received_samples == manifest["audio_samples"]
            result = {"success": success, "status": "complete" if success else "failed", "error": error,
                      "responses": responses, "coverage": coverage, "settings": self.settings,
                      "latency": {"total_wall_s": time.monotonic() - transport_started,
                                  "asr_decode_ms": asr_latencies,
                                  "asr_latency_included_in_output_timestamps": True},
                      "usage": {"raw_text_chunks": len(logger.tokens)}, "billing": None}
            (output / "parsed-output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            logger.close()
        return result
