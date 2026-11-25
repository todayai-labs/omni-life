"""Terminal AV transport coalescing; no model, padding, or artificial flush.

All normal common.packets outputs are passed through unchanged. Only a real
partial audio EOF followed by real video frames is held until those frames are
observable and submitted as one final native AV transaction.
"""
import hashlib
from contextlib import AbstractContextManager

from .common import packets


def terminal_av_packets(prepared, chunk_s=1.0):
    """Preserve source bytes/order; combine only partial EOF + later visuals."""
    width = round(chunk_s * 16000) * 2
    tail = None
    later = []
    for packet in packets(prepared, chunk_s=chunk_s):
        audio = packet["audio"]
        if tail is None:
            if audio is not None and 0 < len(audio) < width:
                tail = packet
            else:
                yield packet
        else:
            if audio is not None:
                raise ValueError("Nonterminal partial audio packet cannot be coalesced")
            later.append(packet)
    if tail is None:
        return
    if not later:
        yield tail
        return
    combined = dict(tail)
    combined["frames"] = list(tail["frames"])
    for packet in later:
        combined["frames"].extend(packet["frames"])
    combined["send_at_s"] = max([tail["send_at_s"]] + [p["send_at_s"] for p in later])
    if any(f["pts_sec"] > combined["send_at_s"] for f in combined["frames"]):
        raise ValueError("Terminal AV packet contains an unobservable frame")
    combined["terminal_av_coalescing"] = {
        "policy": "real_partial_audio_eof_and_later_visuals_in_one_final_native_transaction",
        "original_packet_indices": [tail["index"]] + [p["index"] for p in later],
        "source_audio_eof_s": tail["end_s"],
        "source_audio_tail_samples": len(tail["audio"]) // 2,
        "original_audio_release_s": tail["send_at_s"],
        "final_media_available_s": combined["send_at_s"],
        "source_tail_hold_s": combined["send_at_s"] - tail["send_at_s"],
        "added_audio_samples": 0,
        "dropped_audio_samples": 0,
        "dropped_frames": 0,
    }
    yield combined


def _cache_length(official):
    decoder = official.decoder
    getter = getattr(decoder, "get_cache_length", None)
    if callable(getter):
        return int(getter())
    cache = getattr(decoder, "cache", None)
    if cache is None:
        return 0
    if hasattr(cache, "key_cache"):
        return int(cache.key_cache[0].shape[-2]) if cache.key_cache else 0
    return int(cache[0][0].shape[-2])


def _audio_processed(official):
    mel = getattr(official.processor, "_streaming_mel_processor", None)
    total = getattr(mel, "total_samples_processed", None)
    return int(total) if total is not None else None


class TerminalPrefillObservation(AbstractContextManager):
    """Read-only standard forward hook + state observations, terminal packet only.

    The hook returns None, never alters a tensor, and is removed even on error.
    No flags, cache, buffer, sampling setting, or official model code is changed.
    """
    def __init__(self, official, frames_expected):
        self.official = official
        self.frames_expected = frames_expected
        self.record = {"frames_expected": frames_expected, "resampler_calls": 0,
                       "resampler_output_shapes": [], "observation_only": True}
        self.handle = None

    def __enter__(self):
        self.record["kv_cache_tokens_before"] = _cache_length(self.official)
        self.record["audio_processor_samples_before"] = _audio_processed(self.official)
        self.record["native_audio_buffered_samples_before"] = len(self.official.audio_buffer)
        self.handle = self.official.model.resampler.register_forward_hook(self._observe)
        return self

    def _observe(self, module, inputs, output):
        self.record["resampler_calls"] += 1
        self.record["resampler_output_shapes"].append([int(n) for n in output.shape])
        return None

    def __exit__(self, exc_type, exc, traceback):
        if self.handle is not None:
            self.handle.remove()
        after = _cache_length(self.official)
        self.record["kv_cache_tokens_after"] = after
        self.record["kv_cache_tokens_appended"] = after - self.record["kv_cache_tokens_before"]
        processed = _audio_processed(self.official)
        self.record["audio_processor_samples_after"] = processed
        before = self.record["audio_processor_samples_before"]
        self.record["audio_processor_samples_appended"] = (
            processed - before if processed is not None and before is not None else None)
        self.record["native_audio_buffered_samples_after"] = len(self.official.audio_buffer)
        self.record["native_audio_buffer_sha256_after"] = hashlib.sha256(
            self.official.audio_buffer.astype("<f4", copy=False).tobytes()).hexdigest()
        shapes = self.record["resampler_output_shapes"]
        self.record["observed_image_embeddings"] = sum(s[0] for s in shapes if len(s) == 3)
        self.record["observed_vision_embedding_tokens"] = sum(
            s[0] * s[1] for s in shapes if len(s) == 3)
        minimum = (1 + self.record["observed_vision_embedding_tokens"]
                   + 2 * self.record["observed_image_embeddings"])
        self.record["terminal_frames_prefilled_to_kv"] = (
            self.record["observed_image_embeddings"] == self.frames_expected
            and self.record["observed_vision_embedding_tokens"] > 0
            and self.record["kv_cache_tokens_appended"] >= minimum)
        self.record["exception_type"] = exc_type.__name__ if exc_type else None
        return False
