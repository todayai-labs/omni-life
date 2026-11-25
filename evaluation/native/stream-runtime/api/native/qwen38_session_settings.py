"""Verify the observed Qwen3.8 Streaming provider settings from native receipts."""
import hashlib
import json


# Frozen from 144 historical and 33 current-route B0 session.updated receipts.
PROVIDER_SETTINGS = {
    "audio": {"input": {"format": {"sample_rate": 16000, "type": "pcm"}}},
    "enable_search": False,
    "input_audio_transcription": {"model": "qwen3-asr-flash-realtime"},
    "modalities": ["text"],
    "model": "qwen3.8-omni-flash-realtime",
    "object": "realtime.session",
    "output_audio_format": "pcm",
    "seed": 0,
    "temperature": 0.0,
    "turn_detection": {
        "create_response": True,
        "interrupt_response": True,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 800,
        "threshold": 0.5,
        "type": "semantic_vad",
    },
    "voice": "Tina",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


SETTINGS_SHA256 = hashlib.sha256(canonical(PROVIDER_SETTINGS).encode()).hexdigest()


def mismatches(session):
    return [name for name, selected in PROVIDER_SETTINGS.items()
            if name not in session or canonical(session[name]) != canonical(selected)]


def verify_echoes(events):
    updates = [event["session"] for event in events if event.get("type") == "session.updated"]
    if not updates:
        raise ValueError("missing provider session.updated")
    for session in updates:
        drift = mismatches(session)
        if drift:
            raise ValueError("provider session.updated settings differ: " + ", ".join(drift))
    return {"settings_sha256": SETTINGS_SHA256, "fields_checked": len(PROVIDER_SETTINGS),
            "session_updates_checked": len(updates)}


def verify_raw(attempt, expected_sha):
    raw = (attempt / "raw-output.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"native raw receipt digest mismatch: {attempt}")
    return verify_echoes(json.loads(line)["event"] for line in raw.splitlines() if line.strip())
