"""Causal audiovisual replay with final source-AV transaction coalescing.

No HTTP polling clock, no artificial audio, no shared-checkpoint source edits.
The released encoder can buffer a short final packet; that state is recorded.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from contextlib import nullcontext

from .common import Recorder, wait_release, write_json
from .terminal_av import TerminalPrefillObservation, terminal_av_packets


class Adapter:
    def run_case(self, prepared_dir, output_dir):
        from PIL import Image

        prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
        manifest = json.loads((prepared_dir / "manifest.json").read_text())
        seed = self.config.get("seed", 20261008)
        self.torch.manual_seed(seed)
        self.np.random.seed(seed)
        if self.session_count:
            runtime = getattr(self.duplex, "memory_runtime", None)
            if runtime is not None:
                runtime.reset_memory()
        system = (prepared_dir / manifest["system_file"]).read_text()
        prepared_prompt = self.duplex.prepare(prefix_system_prompt=system)
        official = getattr(self.duplex, "official_duplex", self.duplex)
        options, implementation = self.configure_length_penalty(official, 0.8)
        self.session_count += 1
        start = time.monotonic()
        recorder = Recorder(output_dir, start)
        recorder.log("input.jsonl", "prepare", task_prompt=system, actual_prepared_prompt=prepared_prompt,
                     audio_present=manifest["audio_present"], generate_audio=False, seed=seed)
        responses, current = [], None
        counts = {"frames_expected": len(manifest["frames"]), "frames_received": 0,
                  "source_audio_samples_expected": manifest["audio_samples"], "source_audio_samples_received": 0,
                  "uncovered_tail_samples": 0, "synthetic_audio_samples": 0}
        step_costs, total_generated = [], 0
        terminal_observations = []
        retained_source_tail_samples = 0
        for packet in terminal_av_packets(prepared_dir):
            wait_release(start, packet["send_at_s"])
            frames = []
            for frame in packet["frames"]:
                with Image.open(prepared_dir / frame["file"]) as im:
                    frames.append(im.convert("RGB"))
            pcm = packet["audio"]
            audio = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768.0 if pcm else None
            recorder.log("input.jsonl", "prefill_send", packet_index=packet["index"],
                         source_start_s=packet["start_s"], source_end_s=packet["end_s"],
                         expected_release_s=packet["send_at_s"], frames=packet["frames"],
                         audio_file=manifest["audio_file"], audio_samples=len(pcm) // 2 if pcm else 0,
                         audio_sha256=hashlib.sha256(pcm).hexdigest() if pcm else None,
                         send_lag_s=max(0, time.monotonic() - start - packet["send_at_s"]))
            coalescing = packet.get("terminal_av_coalescing")
            observer = TerminalPrefillObservation(official, len(frames)) if coalescing else None
            if coalescing:
                recorder.log("input.jsonl", "terminal_av_transport", packet_index=packet["index"],
                             policy=coalescing)
            with observer if observer is not None else nullcontext():
                try:
                    prefill = self.duplex.streaming_prefill(audio, frames, None, 1, False)
                except RuntimeError as error:
                    # Native source buffering is retained verbatim, not encoded or flushed.
                    buffered = getattr(official, "audio_buffer", self.np.array([]))
                    if (audio is not None and "audio not enough" in str(error)
                            and len(buffered) >= len(audio)
                            and self.np.array_equal(buffered[-len(audio):], audio)):
                        prefill = {"success": False, "reason": str(error),
                                   "source_packet_retained_in_native_buffer": True,
                                   "native_buffered_samples": len(buffered)}
                    else:
                        raise
            if observer is not None:
                observation = {**observer.record, "packet_index": packet["index"],
                               "coalescing": coalescing,
                               "model_buffered_source_tail_samples": (
                                   len(audio) if prefill.get("source_packet_retained_in_native_buffer") else 0),
                               "source_tail_encoded": False if prefill.get(
                                   "source_packet_retained_in_native_buffer") else None}
                terminal_observations.append(observation)
                recorder.log("runlog.jsonl", "terminal_prefill_observation", **observation)
                if not observation["terminal_frames_prefilled_to_kv"]:
                    raise RuntimeError("Terminal source frames lacked observed native vision/KV prefill")
                if (prefill.get("source_packet_retained_in_native_buffer")
                        and observation["audio_processor_samples_appended"] != 0):
                    raise RuntimeError("Retained terminal audio unexpectedly changed native processor count")
            counts["frames_received"] += len(frames)
            counts["source_audio_samples_received"] += len(pcm) // 2 if pcm else 0
            if prefill.get("source_packet_retained_in_native_buffer"):
                retained_source_tail_samples = len(audio)
            recorder.log("runlog.jsonl", "prefill_return", packet_index=packet["index"], result=prefill)
            # Audio insufficient is native buffering, not a fabricated tail flush.
            if not prefill.get("success", True):
                continue
            before_ids = len(official.total_ids)
            generated = self.duplex.streaming_generate(
                prompt_wav_path=None, decode_mode="sampling", max_new_speak_tokens_per_chunk=50, **options)
            received = time.monotonic() - start
            wave = generated.get("audio_waveform")
            delta = list(official.total_ids[before_ids:])
            raw = {k: v for k, v in generated.items() if k != "audio_waveform"}
            raw["raw_token_ids"] = delta
            raw["raw_token_text"] = self.tokenizer.decode(delta, skip_special_tokens=False)
            raw["audio_waveform"] = None
            if wave is not None and getattr(wave, "size", 0):
                waveform = self.np.asarray(wave, dtype="<f4")
                file = output_dir / f"native-output-audio-{packet['index']:05d}.f32"
                file.write_bytes(waveform.tobytes())
                raw["audio_waveform"] = {"file": file.name, "samples": waveform.size,
                                         "peak_abs": float(self.np.max(self.np.abs(waveform))),
                                         "sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                                         "model_control_silence": bool(not self.np.any(waveform))}
            recorder.log("raw-output.jsonl", "native_generate", packet_index=packet["index"], raw=raw)
            if raw["audio_waveform"] and not raw["audio_waveform"]["model_control_silence"]:
                raise RuntimeError("Text-only native setting unexpectedly produced audible audio; raw preserved")
            total_generated += len(delta)
            step_costs.append({k: raw.get(k) for k in ("cost_llm", "cost_all", "n_tokens")})
            text = generated.get("text", "")
            if text:
                if current is None:
                    current = {"response_id": f"r{len(responses):04d}", "text": "", "start_media_s": received,
                               "end_media_s": received, "onset_source": "native_text_chunk_receive",
                               "audio_duration_s": 0.0}
                current["text"] += text
                current["end_media_s"] = received
            if generated.get("end_of_turn") or generated.get("is_listen"):
                if current is not None:
                    responses.append(current)
                    current = None
            # Preserve exact observed response boundaries if a later step fails.
            recorder.log("response-snapshots.jsonl", "responses",
                         responses=responses + ([dict(current)] if current is not None else []))
        if current is not None:
            current["incomplete_turn_at_source_end"] = True
            responses.append(current)
        wait_release(start, manifest["duration_sec"])
        counts["native_audio_buffered_samples_end"] = len(getattr(official, "audio_buffer", []))
        counts["native_internal_first_chunk_padding"] = "checkpoint-owned 35ms, unmodified"
        counts["model_buffered_source_tail_samples"] = retained_source_tail_samples
        counts["terminal_av_prefill_observations"] = terminal_observations
        recorder.log("runlog.jsonl", "source_end", coverage=counts, responses=len(responses))
        result = {"success": True, "status": "success", "responses": responses, "coverage": counts,
                  "settings": {"mode": "native_duplex", "generate_audio": False, "seed": seed,
                               "length_penalty": 0.8, "length_penalty_implementation": implementation,
                               "max_new_speak_tokens_per_chunk": 50, "decode_mode": "sampling",
                               "audio_chunk_s": 1.0, "max_slice_nums": 1, "memory_minutes": 40,
                               "native_final_packet_policy": "send all source; retain model-owned buffered remainder; no padding",
                               "terminal_av_transport_policy": "coalesce only real EOF partial audio followed by real later visuals",
                               "terminal_av_observation": "read-only resampler forward hook and native KV/processor counters"},
                  "latency": {"session_wall_s": time.monotonic() - start,
                              "text_receive_granularity": "official native generation chunk, not token-level TTFT",
                              "native_steps": step_costs},
                  "usage": {"generated_raw_token_count": total_generated}, "billing": None,
                  "api_cost": {"value": 0, "currency": "USD", "scope": "no paid inference API; hardware cost not supplied"}}
        write_json(output_dir / "parsed-output.json", responses)
        return result
