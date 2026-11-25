"""Shared native duplex probe for unchanged MiniCPM/Venus load and decoding."""
from __future__ import annotations

import importlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

from strict_boundary import prepare_no_text, prefill_no_text
from tts_phase import (PhaseLedger, PlaybackProducer, Responses, TaskTail, save_new,
                       original_query, save_generic_system, failure_kind)


class NativeTTSAdapter:
    module_name = None
    model_id = None

    @classmethod
    def load(cls, config):
        if config.get("reminder_interval_s") is not None:
            raise ValueError("Use the original B0 config, not a reminder condition")
        self = cls()
        self.base = importlib.import_module(cls.module_name).Adapter.load(config)
        self.config = dict(config)
        self.tail_module = importlib.import_module("adapters.venus_tail")
        self.parse_action = importlib.import_module("scoring").parse_action
        return self

    def _settings(self, options, implementation):
        base, config = self.base, self.config
        original_transport_metadata = {
            "native_final_packet_policy": "send all source; retain model-owned buffered remainder; no padding",
            "terminal_av_transport_policy": "coalesce only real EOF partial audio followed by real later visuals",
            "terminal_av_observation": "read-only resampler forward hook and native KV/processor counters"}
        if self.model_id == "minicpm-o-4.5-local":
            return {**original_transport_metadata, "model_id": self.model_id, "mode": "native_duplex",
                "model_path": config["model_path"], "seed": config["seed"],
                "generate_audio": False, "decode_mode": "greedy",
                "max_new_speak_tokens_per_chunk": 20, "force_listen_count": 0,
                "audio_route": "native_audio_encoder", "init_tts": True,
                "tts_output_enabled": False, "official_defaults": base.duplex.official_duplex._default_duplex_params,
                "official_checkpoint_revision": config["checkpoint_revision"],
                "native_api": "MiniCPMO.as_duplex(generate_audio=False)",
                "max_slice_nums": 1, "audio_chunk_s": 1.0}
        return {**original_transport_metadata, "model_id": self.model_id, "mode": "native_duplex", "generate_audio": False,
            "seed": config.get("seed", 20261008), "length_penalty": 0.8,
            "length_penalty_implementation": implementation, "decode_mode": "sampling",
            "max_new_speak_tokens_per_chunk": 50, "memory_minutes": 40,
            "max_slice_nums": 1, "audio_chunk_s": 1.0,
            "native_api": "Realtime-Venus-Omni.Memory.as_duplex(generate_audio=False)"}

    def run_case(self, prepared, output, task_pcm, task_metadata):
        from PIL import Image

        prepared, output = Path(prepared), Path(output)
        manifest = json.loads((prepared / "manifest.json").read_text())
        base, np = self.base, self.base.np
        seed = self.config.get("seed", 20261008)
        base.torch.manual_seed(seed)
        np.random.seed(seed)
        if base.session_count:
            runtime = getattr(base.duplex, "memory_runtime", None)
            if runtime is not None:
                runtime.reset_memory()  # between source CASES only, never between phases
        rendered_system = (prepared / manifest["system_file"]).read_text()
        query = original_query(manifest)
        if task_metadata.get("text") != query:
            raise ValueError("Task WAV query differs from prepared query")
        system = save_generic_system(output, rendered_system, query)
        ledger = PhaseLedger(output, task_metadata["duration_sec"])
        responses, task_tail = Responses(ledger, self.parse_action), TaskTail()
        ledger.event("input.jsonl", "initial_task", system_prompt=system,
                     original_rendered_system_logging_only=rendered_system,
                     original_query_logging_only=manifest["query"], task_audio=task_metadata,
                     text_query_sent_to_model=False, source_manifest_logging_only=manifest)
        official = getattr(base.duplex, "official_duplex", base.duplex)
        assert system == ""
        prepared_prompt = prepare_no_text(base.duplex,self.model_id,output)
        options, implementation = base.configure_length_penalty(official, 0.8)
        base.session_count += 1
        ledger.event("input.jsonl", "native_prepare", actual_prepared_prompt=prepared_prompt,
                     phases_share_model_session=True, generate_audio=False)
        settings = self._settings(options, implementation)
        settings.update(condition="TTS-audio-only-v2", source_query_text_sent=False,
                        task_tts_input_only=True, no_ack_wait=True)
        producer = PlaybackProducer(ledger, task_pcm,
            lambda: self.tail_module.terminal_av_packets(prepared), manifest["duration_sec"])
        source_frames = source_samples = generated_tokens = 0
        frames_prefill_success = task_prefill_success = source_prefill_success = 0
        native_steps, terminal_observations = [], []
        error = None
        started = time.monotonic()
        producer.start()
        try:
            while (packet := producer.queue.get()) is not None:
                kind = packet["kind"]
                if kind == "source_frame_available":
                    continue  # no additional native step; original AV grouping retained
                if kind == "source_start":
                    continue
                task_count = 0
                if kind == "task":
                    pcm = packet["audio"]
                    buffered = len(getattr(official, "audio_buffer", []))
                    next_size = official.processor.get_streaming_chunk_size()
                    # Native FIRST_CHUNK padding is checkpoint-owned and unmodified.
                    # Later insufficient task packets are held, never submitted then reset.
                    # With real later audio, preserve the former causal tail coalescing.
                    # Without source audio, call the REAL native API: MiniCPM may
                    # buffer and continue VISION; Memory may genuinely reject it.
                    if (manifest["audio_present"] and getattr(official, "audio_chunk_idx", 0) > 0
                            and buffered + len(pcm) // 2 < next_size):
                        task_tail.hold(pcm)
                        ledger.event("input.jsonl", "task_audio_tail_held", input_origin="task",
                            pending_task_samples=len(task_tail.pending) // 2,
                            native_buffer_samples=buffered, native_next_chunk_samples=next_size,
                            added_silence_samples=0, discarded_samples=0)
                        continue
                    task_count = len(pcm) // 2
                    frames = []
                else:
                    pcm, task_count = task_tail.source_audio(packet["audio"], manifest["audio_present"])
                    frames = []
                    for frame in packet["frames"]:
                        with Image.open(prepared / frame["file"]) as image:
                            frames.append(image.convert("RGB").copy())
                audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0 if pcm else None
                sent = time.monotonic()
                processor = getattr(official.processor, "_streaming_mel_processor", None)
                processed_before = getattr(processor, "total_samples_processed", None)
                ledger.event("input.jsonl", "native_prefill_send", input_origin=kind,
                    source_packet_index=packet.get("index"), frame_metadata=packet.get("frames", []),
                    task_audio_samples=task_count,
                    source_audio_samples=len(packet["audio"] or b"") // 2 if kind == "source" else 0,
                    task_wav_file=task_metadata["wav_file"], source_audio_file=manifest.get("audio_file"),
                    send_lag_s=max(0, sent - packet["release_at"]), source_frames_not_sent_in_task=(kind == "task"))
                coalescing = packet.get("terminal_av_coalescing")
                observer = self.tail_module.TerminalPrefillObservation(official, len(frames)) if coalescing else None
                # These are native-API submission counts, not accepted/encoded counts.
                task_tail.received += task_count
                if kind == "source":
                    source_frames += len(frames)
                    source_samples += len(packet["audio"] or b"") // 2
                with observer if observer is not None else nullcontext():
                    try:
                        prefill = prefill_no_text(base.duplex,audio,frames,ledger,kind)
                    except RuntimeError as exc:
                        buffer = getattr(official, "audio_buffer", np.array([]))
                        if (kind == "source" and audio is not None and "audio not enough" in str(exc)
                                and len(buffer) >= len(audio) and np.array_equal(buffer[-len(audio):], audio)
                                and producer.source_done.is_set()):
                            prefill = {"success": False, "reason": str(exc),
                                "source_packet_retained_in_native_buffer": True,
                                "native_buffered_samples": len(buffer)}
                        else:
                            ledger.event("runlog.jsonl", "native_prefill_exception", input_origin=kind,
                                error=repr(exc), failure_kind=failure_kind(exc),
                                actual_native_API_called=True,
                                native_buffered_audio_samples=len(buffer),
                                task_audio_samples_submitted=task_count,
                                no_native_reset_or_bypass=True)
                            raise
                ledger.event("runlog.jsonl", "native_prefill_return", input_origin=kind, result=prefill)
                processed_after = getattr(processor, "total_samples_processed", None)
                ledger.event("runlog.jsonl", "native_input_observation", input_origin=kind,
                    native_prefill_success=bool(prefill.get("success", True)),
                    task_audio_samples_submitted=task_count,
                    source_audio_samples_submitted=len(packet["audio"] or b"") // 2 if kind == "source" else 0,
                    audio_processor_samples_before=processed_before,
                    audio_processor_samples_after=processed_after,
                    native_buffered_audio_samples=len(getattr(official, "audio_buffer", [])),
                    submission_is_not_proof_of_audio_encoding=True)
                if observer is not None:
                    observation = {**observer.record, "source_packet_index": packet["index"],
                                   "coalescing": coalescing}
                    terminal_observations.append(observation)
                    ledger.event("runlog.jsonl", "terminal_prefill_observation", **observation)
                    if not observation["terminal_frames_prefilled_to_kv"]:
                        raise RuntimeError("Terminal source frames lacked native vision/KV prefill")
                if not prefill.get("success", True):
                    if kind == "task":
                        buffer = getattr(official, "audio_buffer", np.array([]))
                        retained = (audio is not None and "audio not enough" in str(prefill.get("reason", ""))
                            and len(buffer) >= len(audio) and np.array_equal(buffer[-len(audio):], audio))
                        if self.model_id == "minicpm-o-4.5-local" and retained:
                            ledger.event("runlog.jsonl", "task_native_buffer_retained", input_origin="task",
                                result=prefill, task_tail_samples=len(audio),
                                native_buffered_audio_samples=len(buffer),
                                subsequent_source_VISION_allowed=True, extra_generate=False,
                                flush_requested=False, added_silence_samples=0)
                            continue
                        raise RuntimeError("Native task streaming_prefill failed: " + str(prefill.get("reason", "")))
                    if task_count:
                        raise RuntimeError("task_audio_tail_retained_unencoded_at_source_EOF; no text fallback")
                    continue
                task_prefill_success += task_count
                if kind == "source":
                    frames_prefill_success += len(frames)
                    source_prefill_success += len(packet["audio"] or b"") // 2
                before_ids = len(official.total_ids)
                # MiniCPM's UNCHANGED OfficialTextDuplex wrapper enforces greedy/20.
                # Venus receives exactly the original sampling/50 and length-penalty options.
                generated = base.duplex.streaming_generate(prompt_wav_path=None,
                    decode_mode="sampling", max_new_speak_tokens_per_chunk=50, **options)
                received = time.monotonic()
                delta = list(official.total_ids[before_ids:])
                raw = {k: v for k, v in generated.items() if k != "audio_waveform"}
                raw["raw_token_ids"] = delta
                raw["raw_token_text"] = base.tokenizer.decode(delta, skip_special_tokens=False)
                waveform = generated.get("audio_waveform")
                raw["audio_waveform"] = None
                if waveform is not None and getattr(waveform, "size", 0):
                    waveform = np.asarray(waveform, dtype="<f4")
                    filename = f"native-output-audio-{len(native_steps):05d}.f32"
                    with (output / filename).open("xb") as handle:
                        handle.write(waveform.tobytes())
                    raw["audio_waveform"] = {"file": filename, "samples": waveform.size,
                        "peak_abs": float(np.max(np.abs(waveform))),
                        "model_control_silence": not bool(np.any(waveform))}
                ledger.event("raw-output.jsonl", "native_generate", at=received,
                    input_origin=kind, source_packet_index=packet.get("index"), raw=raw)
                if raw["audio_waveform"] and not raw["audio_waveform"]["model_control_silence"]:
                    raise RuntimeError("Text-only native setting unexpectedly produced audible audio")
                generated_tokens += len(delta)
                native_steps.append({"input_origin": kind, "phase": ledger.phase_at(received),
                                     **{k: raw.get(k) for k in ("cost_llm", "cost_all", "n_tokens")}})
                responses.add(generated.get("text", ""), received,
                              ended=bool(generated.get("end_of_turn") or generated.get("is_listen")))
            if producer.error is not None:
                raise producer.error
            if task_tail.pending:
                raise RuntimeError("task_audio_tail_undelivered; no padding, discard or text fallback")
            if source_frames != len(manifest["frames"]) or source_samples != manifest["audio_samples"]:
                raise RuntimeError("Source input coverage incomplete")
            if task_tail.received != len(task_pcm) // 2:
                raise RuntimeError("Task TTS input coverage incomplete")
        except BaseException as exc:
            error = repr(exc)
            ledger.event("runlog.jsonl", "failure", error=error)
        finally:
            producer.stop_event.set()  # producer cleanup only; no native break/cancel/EOS
            producer.join(timeout=30)
            if producer.is_alive():
                error = error or "Playback producer failed to stop"
            task_responses, source_responses = responses.finish()
            clocks = ledger.clock_record()
            coverage = {"frames_expected": len(manifest["frames"]), "frames_received": source_frames,
                "source_audio_samples_expected": manifest["audio_samples"],
                "source_audio_samples_received": source_samples,
                "uncovered_tail_samples": max(0, manifest["audio_samples"] - source_samples),
                "source_audio_present": manifest["audio_present"], "source_synthetic_audio_samples": 0,
                "task_audio_samples_expected": len(task_pcm) // 2,
                "task_audio_samples_received": task_tail.received,
                "task_audio_samples_in_successful_prefill": task_prefill_success,
                "source_audio_samples_in_successful_prefill": source_prefill_success,
                "frames_in_successful_prefill": frames_prefill_success,
                "coverage_semantics": "received=submitted to native API; successful prefill is separate; buffer/processor observations retained",
                "pending_task_tail_samples": len(task_tail.pending) // 2,
                "native_audio_buffered_samples_end": len(getattr(official, "audio_buffer", [])),
                "native_first_chunk_padding": "checkpoint-owned and unmodified",
                "terminal_av_prefill_observations": terminal_observations}
            result = {"success": error is None, "status": "success" if error is None else "failed",
                "error": error, "responses": source_responses, "task_responses": task_responses,
                "failure_kind": failure_kind(error),
                "coverage": coverage, "settings": settings, "clocks": clocks,
                "latency": {"session_wall_s": time.monotonic() - started, "native_steps": native_steps,
                    "text_receive_granularity": "official native generation chunk, not token-level TTFT"},
                "usage": {"generated_raw_token_count": generated_tokens}, "billing": None,
                "api_cost": {"value": None, "scope": "No paid model API; hardware cost not supplied"},
                "content_semantics_judge": "not_called"}
            save_new(output / "transport-result.json", result)
            ledger.close()
        return result
