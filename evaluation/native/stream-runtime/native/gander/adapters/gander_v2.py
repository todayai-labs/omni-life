"""Official Gander Thinker: v2 integer-bounded source PCM transport.

Uses the upstream OnlineRunner and streaming_prefill/generate APIs unchanged.
No ASR, Talker, query synthesis, padded tail, prefix replay, or trailing silence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import threading
import time
import traceback


def _stamp(origin=None):
    now = time.monotonic()
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": now,
            "media_wall_s": now - origin if origin is not None else None}


def _safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


class Adapter:
    @classmethod
    def load(cls, config):
        import numpy as np
        import torch
        # The official deepspeed 0.19 + transformers 4.51 stack has a circular
        # import if Transformers loads first. Preloading changes no model logic.
        import deepspeed
        config = dict(config)
        seed = int(config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sys.path.insert(0, str(Path(config["repo_path"]) / "minicpm_ft"))
        from mcpmft.args import ModelArguments
        from mcpmft.infer.common import load_for_infer
        from mcpmft.infer.online import DuplexParams
        self = cls()
        self.config = config
        self.gpu_before_load = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
        device = config.get("device", "cuda:0")
        if isinstance(device, int) or str(device).isdigit():
            device = "cuda:0"  # physical placement must be CUDA_VISIBLE_DEVICES
        args = ModelArguments(model_name_or_path=config["model_path"],
                              processor_name_or_path=config["model_path"],
                              init_vision=True, init_audio=True, init_tts=False,
                              device_map=device, attn_implementation="sdpa")
        self.model_args = args
        checkpoint = config.get("checkpoint_path")
        if not checkpoint:
            raise ValueError("checkpoint_path must point to the released Gander thinker/")
        self.bundle = load_for_infer(args, checkpoint=checkpoint, init_token2wav=False)
        self.params = DuplexParams(chunk_ms=1000, speak_text_tokens_per_unit=8,
                                   max_new_speak_tokens_per_chunk=11,
                                   talker_speech_tokens_per_unit=50,
                                   generate_audio=False, temperature=0.7, top_p=0.8,
                                   top_k=20, sliding_window_mode="context_slate",
                                   context_max_units=128, context_previous_max_tokens=1500)
        self.repo_commit = subprocess.check_output(
            ["git", "-C", config["repo_path"], "rev-parse", "HEAD"], text=True).strip()
        return self

    def run_case(self, prepared_dir, output_dir):
        import numpy as np
        import torch
        from PIL import Image
        from mcpmft.infer.online import OnlineRunner
        prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("input.jsonl", "raw-output.jsonl", "runlog.jsonl"):
            if (output_dir / name).exists():
                raise FileExistsError("Do not overwrite an earlier attempt: " + str(output_dir / name))
        manifest = json.loads((prepared_dir / "manifest.json").read_text())
        case_seed = (int(self.config.get("seed", 0)) + int(hashlib.sha256(manifest["id"].encode()).hexdigest()[:8], 16)) % (2**31)
        random.seed(case_seed)
        np.random.seed(case_seed)
        torch.manual_seed(case_seed)
        system = (prepared_dir / manifest["system_file"]).read_text()
        frames = manifest["frames"]
        audio = (prepared_dir / manifest["audio_file"]).read_bytes() if manifest["audio_present"] else b""
        if manifest["audio_present"] and len(audio) // 2 != manifest["audio_samples"]:
            raise ValueError("Source audio sample count changed")
        if manifest.get("audio_sha256") and _hash(audio) != manifest["audio_sha256"]:
            raise ValueError("Source audio hash changed")
        if _hash(system.encode()) != manifest["system_sha256"]:
            raise ValueError("System prompt changed")
        duration = max(float(manifest["duration_sec"]), len(audio) / 32000)
        expected_samples = len(audio) // 2
        streams = {k: (output_dir / k).open("x", encoding="utf8") for k in
                   ("input.jsonl", "raw-output.jsonl", "runlog.jsonl")}
        lock = threading.Lock()
        origin = None
        def log(kind, event):
            item = {**_stamp(origin), **_safe(event)}
            with lock:
                streams[kind].write(json.dumps(item, ensure_ascii=False) + "\n")
                streams[kind].flush()
            return item
        settings = {"official_repo_commit": self.repo_commit, "model": asdict(self.model_args),
                    "checkpoint": self.config["checkpoint_path"], "native_params": asdict(self.params),
                    "audio_route": "native_encoder", "tts": False, "asr": False,
                    "dependency_import_order": "deepspeed before transformers",
                    "playback_speed": 1.0, "seed": int(self.config.get("seed", 0)), "case_seed": case_seed,
                    "versions": {p: importlib.metadata.version(p) for p in
                                 ("torch", "transformers", "numpy", "mcpmft")}}
        log("runlog.jsonl", {"event": "case_prepare", "id": manifest["id"], "settings": settings,
                            "gpu_before_load": self.gpu_before_load})
        log("input.jsonl", {"kind": "system", "text": system, "sha256": manifest["system_sha256"],
                            "query": manifest["query"], "source_manifest": str(prepared_dir / "manifest.json")})
        runner = OnlineRunner(self.bundle, self.params)
        runner.prepare(system_prompt=system)
        responses, pending_response, received_frames, received_samples = [], None, 0, 0
        timings = []
        work = queue.Queue()
        cancelled = threading.Event()
        origin = time.monotonic()
        def produce():
            if not manifest["audio_present"]:
                for index, frame in enumerate(frames):
                    pts = float(frame["pts_sec"])
                    if cancelled.wait(max(0, origin + pts - time.monotonic())):
                        break
                    release = log("input.jsonl", {"kind": "source_release", "index": index,
                                  "source_start_s": pts, "source_end_s": pts,
                                  "frames": [frame], "audio_samples": 0,
                                  "source_audio_absent": True})
                    work.put((index, pts, pts, [frame], b"", release))
                cancelled.wait(max(0, origin + duration - time.monotonic()))
                work.put(None)
                return
            cursor = 0
            for index in range(math.ceil(duration)):
                start, end = float(index), min(float(index + 1), duration)
                delay = origin + end - time.monotonic()
                if cancelled.wait(max(0, delay)):
                    break
                selected = []
                while cursor < len(frames) and float(frames[cursor]["pts_sec"]) <= end:
                    selected.append(frames[cursor])
                    cursor += 1
                low, high = index * 16000, min(expected_samples, (index + 1) * 16000)
                raw = audio[low * 2:high * 2]
                release = log("input.jsonl", {"kind": "source_release", "index": index,
                              "source_start_s": start, "source_end_s": end,
                              "frames": selected, "audio_sample_start": low,
                              "audio_samples": len(raw) // 2, "audio_sha256": _hash(raw),
                              "raw_audio_file": str(prepared_dir / manifest["audio_file"]) if audio else None,
                              "synthetic_padding_samples": 0})
                work.put((index, start, end, selected, raw, release))
            work.put(None)
        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        success, status, error = False, "failed", None
        try:
            with torch.inference_mode():
                while True:
                    unit = work.get()
                    if unit is None:
                        break
                    index, start, end, selected, raw, release = unit
                    images = []
                    for frame in selected:
                        path = prepared_dir / frame["file"]
                        data = path.read_bytes()
                        if _hash(data) != frame["sha256"]:
                            raise ValueError("Source frame hash changed: " + str(path))
                        with Image.open(path) as im:
                            images.append(im.convert("RGB"))
                    waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768 if raw else None
                    send = log("input.jsonl", {"kind": "model_prefill", "index": index,
                               "source_start_s": start, "source_end_s": end,
                               "frames": selected, "audio_samples": len(raw) // 2,
                               "source_release_monotonic_s": release["monotonic_s"]})
                    if not images and waveform is None:
                        # No synthetic clock-tick/audio is injected into empty source tail.
                        log("runlog.jsonl", {"event": "empty_source_interval", "index": index})
                        continue
                    prefill = runner.duplex.streaming_prefill(audio_waveform=waveform,
                                                              frame_list=images or None,
                                                              max_slice_nums=1,
                                                              batch_vision_feed=False)
                    received_frames += len(images)
                    received_samples += len(raw) // 2
                    if not prefill.get("success", False):
                        native_buffer = getattr(runner.duplex, "audio_buffer", None)
                        reason = str(prefill.get("reason", ""))
                        # Native APIs may accept PCM into their buffer without an
                        # entire acoustic unit being ready. Do not pad, generate
                        # logits from an incomplete unit, or call this encoded.
                        buffered = (reason.startswith("audio not enough:") and
                                    waveform is not None and native_buffer is not None and
                                    len(native_buffer) >= len(waveform) and
                                    np.array_equal(native_buffer[-len(waveform):], waveform))
                        log("raw-output.jsonl", {"kind": "native_buffer_only", "index": index,
                            "source_start_s": start, "source_end_s": end, "raw": prefill,
                            "buffered_samples": len(native_buffer) if native_buffer is not None else None,
                            "buffer_float32_sha256": _hash(native_buffer.astype("<f4").tobytes()) if native_buffer is not None else None,
                            "source_tail_present_exactly": buffered,
                            "source_tail_sha256": _hash(raw), "generated": False})
                        if buffered:
                            continue
                        raise RuntimeError("Native prefill rejected source media: " + str(prefill))
                    runner.record_unit_time(start, end)
                    output = runner.streaming_generate(max_new_speak_tokens_per_chunk=11,
                                                       decode_mode="sampling", temperature=0.7,
                                                       top_p=0.8, top_k=20)
                    recv = log("raw-output.jsonl", {"kind": "native_unit", "index": index,
                               "source_start_s": start, "source_end_s": end,
                               "raw": output, "prefill": prefill,
                               "send_monotonic_s": send["monotonic_s"]})
                    timings.append(recv["monotonic_s"] - send["monotonic_s"])
                    text = output.get("text") or ""
                    if text:
                        if pending_response is None:
                            pending_response = {"response_id": str(len(responses)), "text": "",
                                "start_media_s": recv["media_wall_s"], "end_media_s": recv["media_wall_s"],
                                "onset_source": "first_text_receive", "first_receive_utc": recv["utc"],
                                "first_receive_monotonic_s": recv["monotonic_s"], "unit_indexes": []}
                        pending_response["text"] += text
                        pending_response["end_media_s"] = recv["media_wall_s"]
                        pending_response["unit_indexes"].append(index)
                    if output.get("end_of_turn") and pending_response is not None:
                        responses.append(pending_response)
                        pending_response = None
            if pending_response is not None:
                pending_response["closed_by_source_eof"] = True
                responses.append(pending_response)
            success = received_samples == expected_samples and received_frames == len(frames)
            status = "completed" if success else "coverage_incomplete"
        except Exception:
            error = traceback.format_exc()
            log("runlog.jsonl", {"event": "exception", "traceback": error})
            if pending_response is not None:
                responses.append(pending_response)
        finally:
            cancelled.set()
            producer.join(timeout=2)
        native_buffer = getattr(runner.duplex, "audio_buffer", np.zeros(0, dtype=np.float32))
        native_buffer_samples = len(native_buffer)
        result = {"id": manifest["id"], "success": success, "status": status,
                  "responses": responses, "coverage": {"frames_expected": len(frames),
                     "frames_received": received_frames, "source_audio_present": manifest["audio_present"],
                     "source_audio_samples_expected": expected_samples,
                     "source_audio_samples_received": received_samples,
                     "uncovered_tail_samples": expected_samples - received_samples,
                     "native_audio_buffered_samples_end": native_buffer_samples,
                     "source_audio_samples_consumed_from_native_buffer": max(0, received_samples - native_buffer_samples),
                     "source_audio_samples_encoded_lower_bound": max(0, received_samples - native_buffer_samples),
                     "native_audio_buffer_float32_sha256": _hash(native_buffer.astype("<f4").tobytes()),
                     "native_first_prefix_padding_samples": int((self.params.first_chunk_ms - self.params.chunk_ms) * 16) if expected_samples else 0,
                     "last_encoded_source_cutoff_s_lower_bound": max(0, received_samples - native_buffer_samples) / 16000},
                  "settings": settings, "latency": {"model_unit_s": timings,
                     "wall_elapsed_s": time.monotonic() - origin},
                  "usage": {"unit_count": len(timings)}, "billing": None, "error": error}
        log("runlog.jsonl", {"event": "case_complete", **result})
        for stream in streams.values():
            stream.close()
        (output_dir / "adapter-result.json").write_text(json.dumps(_safe(result), ensure_ascii=False, indent=2))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("config", "prepared", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    adapter = Adapter.load(json.loads(Path(args.config).read_text()))
    result = adapter.run_case(Path(args.prepared), Path(args.output))
    print(json.dumps(_safe(result), ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)
