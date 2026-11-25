"""Validate the source clock and the serialized reminder send near audio EOF."""

import math


def eof_witness(reminder, scheduled, duration, ledger):
    """Prove the reminder went out before the remaining original PCM and EOF."""
    if not 0 <= scheduled < duration <= reminder["actual_media_s"]:
        return None
    drift = reminder["actual_media_s"] - scheduled
    if drift > 2 or not math.isclose(reminder.get("drift_s", math.inf), drift, abs_tol=1e-9):
        return None

    index = next((i for i, ev in enumerate(ledger) if ev is reminder), None)
    if index is None or not ledger or ledger[-1]["kind"] != "audio_stream_end":
        return None
    if sum(ev["kind"] == "audio_stream_end" for ev in ledger) != 1:
        return None
    after = ledger[index + 1:-1]
    if after and after[0]["kind"] == "audio":
        audio = after[0]
    else:
        return None
    if len(after) != 1:
        return None
    if not (audio["samples"] > 0 and audio["start_s"] <= scheduled < audio["end_s"]
            and abs(audio["end_s"] - duration) <= 1 / 16000):
        return None
    eof = ledger[-1]["elapsed_s"]
    if not (reminder["send_completed_media_s"] <= reminder["elapsed_s"] <= audio["elapsed_s"] <= eof):
        return None
    return {"scheduled_media_s": scheduled, "source_duration_s": duration,
            "reminder_started_s": reminder["actual_media_s"], "reminder_finished_s": reminder["send_completed_media_s"],
            "tail_audio_send_elapsed_s": audio["elapsed_s"], "tail_audio_samples": audio["samples"],
            "audio_stream_end_elapsed_s": eof}


def valid(reminder, scheduled, duration, ledger):
    started = reminder["actual_media_s"]
    finished = reminder["send_completed_media_s"]
    elapsed = reminder["elapsed_s"]
    if not all(math.isfinite(value) for value in (scheduled, duration, started, finished, elapsed)):
        return False
    if not (0 <= scheduled < duration and scheduled - 0.002 <= started <= finished <= elapsed + 0.001):
        return False
    return started < duration or eof_witness(reminder, scheduled, duration, ledger) is not None
