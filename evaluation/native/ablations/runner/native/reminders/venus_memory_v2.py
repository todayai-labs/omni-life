"""Released Venus native duplex API with final source-AV transaction coalescing.

No HTTP polling clock, no artificial audio, no shared-checkpoint source edits.
The released encoder can buffer a short final packet; that state is recorded.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from contextlib import nullcontext

from .common import Recorder, wait_release, write_json
from .venus_tail import TerminalPrefillObservation, terminal_av_packets


ACTIVE_MEMORY_ANSWER_REJECTION = (
    "finish the active Memory answer before submitting another text_list question"
)


class _TaskReminders:
    """Optional input-only text appended to an existing real AV transaction."""
    def __init__(self, config, manifest, duplex):
        self.interval = float(config["reminder_interval_s"])
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("reminder_interval_s must be finite and positive")
        self.kind = config.get("reminder_kind", "repeat_query")
        if self.kind not in ("repeat_query", "generic"):
            raise ValueError("Unsupported reminder_kind")
        query = manifest.get("query")
        if (not isinstance(query, list) or len(query) != 1
                or query[0].get("start_time_sec", 0) != 0
                or not isinstance(query[0].get("text"), str)
                or not query[0]["text"].strip()):
            raise ValueError("Periodic reminders require the original single initial query")
        self.text = ("提示：你的任务是 " + query[0]["text"] if self.kind == "repeat_query"
                     else "继续执行最初任务；仅在原任务条件满足时回应，否则不输出。")
        self.duration = float(manifest["duration_sec"])
        self.next_tick = 1
        self.accepted = self.rejected = self.skipped = self.failed = 0

    def _skip(self, tick, recorder, reason, **fields):
        self.skipped += 1
        recorder.log("input.jsonl", "task_reminder_skipped", planned_media_s=tick,
                     reason=reason, reminder_kind=self.kind, **fields)

    def select(self, packet, actual_media_s, recorder):
        due = []
        while self.next_tick * self.interval < self.duration:
            tick = self.next_tick * self.interval
            if tick > packet["send_at_s"] + 1e-9:
                break
            self.next_tick += 1
            due.append(tick)
        if not due:
            return None
        # Never create a burst of old reminders or an additional model step.
        for tick in due[:-1]:
            self._skip(tick, recorder, "no_separate_native_AV_boundary", packet_index=packet["index"])
        tick = due[-1]
        if actual_media_s >= self.duration or packet["send_at_s"] >= self.duration:
            self._skip(tick, recorder, "source_EOF_reached", packet_index=packet["index"],
                       actual_media_s=actual_media_s)
            return None
        receipt = {"planned_media_s": tick, "actual_media_s": actual_media_s,
                   "send_delay_s": max(0, actual_media_s - tick), "packet_index": packet["index"],
                   "packet_release_s": packet["send_at_s"], "reminder_kind": self.kind,
                   "text": self.text, "api": "streaming_prefill.text_list"}
        recorder.log("input.jsonl", "task_reminder_attempt", **receipt)
        return receipt

    def result(self, receipt, recorder, prefill=None, error=None):
        if receipt is None:
            return
        accepted = error is None and prefill.get("success", True)
        if accepted:
            self.accepted += 1
        else:
            self.failed += 1
        recorder.log("input.jsonl", "task_reminder_delivery", **receipt,
                     accepted_by_native_prefill=bool(accepted),
                     delivery_status="accepted" if accepted else "failed",
                     reason=(str(error) if error is not None else prefill.get("reason", "")))

    def reject(self, receipt, recorder, error, actual_media_s):
        self.rejected += 1
        recorder.log("input.jsonl", "task_reminder_delivery", **receipt,
                     accepted_by_native_prefill=False, delivery_status="rejected",
                     reason=str(error), native_error_type=type(error).__name__,
                     native_rejection_media_s=actual_media_s,
                     retry_reminder=False, rejection_stage="before_source_AV_processing")

    def finish(self, recorder):
        while self.next_tick * self.interval < self.duration:
            tick = self.next_tick * self.interval
            self.next_tick += 1
            self._skip(tick, recorder, "no_native_AV_packet_before_source_EOF")
        summary = {"interval_s": self.interval, "kind": self.kind,
                   "planned_ticks": self.next_tick - 1, "accepted": self.accepted,
                   "rejected": self.rejected, "skipped": self.skipped, "failed": self.failed,
                   "policy": "same_official_AV_prefill_only_no_extra_generation_no_EOF_reminders"}
        recorder.log("runlog.jsonl", "task_reminder_summary", **summary)
        return summary


def _native_source_prefill(duplex, official, np, audio, frames, text_list):
    """Keep the released source buffering path, including the short audio tail."""
    try:
        return duplex.streaming_prefill(audio, frames, text_list, 1, False)
    except RuntimeError as error:
        buffered = getattr(official, "audio_buffer", np.array([]))
        if (audio is not None and "audio not enough" in str(error)
                and len(buffered) >= len(audio)
                and np.array_equal(buffered[-len(audio):], audio)):
            return {"success": False, "reason": str(error),
                    "source_packet_retained_in_native_buffer": True,
                    "native_buffered_samples": len(buffered)}
        raise


def _prefill_source_packet(duplex, official, np, audio, frames, reminder,
                           reminders, recorder, start, packet):
    """A rejected text question is dropped; source AV is delivered exactly once.

    The single special case matches the released Memory guard that raises before
    processing audio or video. Nothing is queued, regenerated, or replayed after
    successful source processing. All other errors remain fatal.
    """
    text_list = [reminder["text"]] if reminder is not None else None
    try:
        prefill = _native_source_prefill(duplex, official, np, audio, frames, text_list)
    except Exception as error:
        early_memory_rejection = (
            isinstance(error, RuntimeError)
            and reminder is not None
            and getattr(duplex, "memory_runtime", None) is not None
            and str(error) == ACTIVE_MEMORY_ANSWER_REJECTION
        )
        if not early_memory_rejection:
            if reminders is not None:
                reminders.result(reminder, recorder, error=error)
            raise
        reminders.reject(reminder, recorder, error, time.monotonic() - start)
        recorder.log("input.jsonl", "source_AV_after_native_text_rejection",
                     packet_index=packet["index"], text_list=None,
                     actual_media_s=time.monotonic() - start,
                     original_AV_not_consumed=True,
                     reason="released_Memory_guard_raised_before_AV_processing",
                     policy="drop_rejected_text_deliver_same_real_AV_once_no_extra_generate")
        try:
            return _native_source_prefill(duplex, official, np, audio, frames, None)
        except Exception as fallback_error:
            recorder.log("runlog.jsonl", "source_AV_delivery_failed_after_text_rejection",
                         packet_index=packet["index"], reason=str(fallback_error),
                         error_type=type(fallback_error).__name__)
            raise
    if reminders is not None:
        reminders.result(reminder, recorder, prefill=prefill)
    return prefill


class Adapter:
    @classmethod
    def load(cls, config):
        import numpy as np
        import torch
        from transformers import AutoModel

        self = cls()
        self.config = config
        self.np, self.torch = np, torch
        sys.path.insert(0, config["repo_path"])
        from demos.model.decoding import configure_length_penalty
        self.configure_length_penalty = configure_length_penalty
        self.model = AutoModel.from_pretrained(
            config["model_path"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).eval().cuda()
        self.model.use_memory(memory_minutes=40)
        self.duplex = self.model.as_duplex(generate_audio=False)
        self.tokenizer = self.duplex.tokenizer
        self.session_count = 0
        return self

    def run_case(self, prepared_dir, output_dir):
        from PIL import Image

        prepared_dir, output_dir = Path(prepared_dir), Path(output_dir)
        manifest = json.loads((prepared_dir / "manifest.json").read_text())
        reminders = (_TaskReminders(self.config, manifest, self.duplex)
                     if self.config.get("reminder_interval_s") is not None else None)
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
                reminder = (reminders.select(packet, time.monotonic() - start, recorder)
                            if reminders is not None else None)
                prefill = _prefill_source_packet(
                    self.duplex, official, self.np, audio, frames, reminder,
                    reminders, recorder, start, packet)
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
            if reminders is not None and getattr(self.duplex, "memory_runtime", None) is not None:
                raw["raw_token_ids_scope"] = "official_main_duplex_total_ids_delta_only"
                raw["returned_text_scope"] = "complete_native_returned_text_including_MemoryQA_if_routed"
                raw["memory_qa_generated_token_count"] = None
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
        if current is not None:
            current["incomplete_turn_at_source_end"] = True
            responses.append(current)
        wait_release(start, manifest["duration_sec"])
        reminder_summary = reminders.finish(recorder) if reminders is not None else None
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
        if reminder_summary is not None:
            result["settings"]["task_reminders"] = reminder_summary
            if getattr(self.duplex, "memory_runtime", None) is not None:
                result["usage"] = {
                    "generated_raw_token_count": None,
                    "official_main_duplex_raw_token_count": total_generated,
                    "memory_qa_generated_token_count": None,
                    "scope": "main_duplex_counter_observed; MemoryQA_and_combined_token_counts_unknown",
                }
        return result
