"""Deterministic temporal matching and silence accounting for saved responses.

Semantic decisions are read from the released binary judge records. Action tags
are parsed for provenance only; action equality is not a success criterion.
"""
from __future__ import annotations
import math
import re
from typing import Any, Callable
ACTIVE_ACTIONS = ("warn", "inform", "count", "feedback", "summarize", "answer")
ACTIONS = ACTIVE_ACTIONS + ("wait",)
_TAG = re.compile(r"^\s*\[([A-Za-z_-]+)\]\s*")

def parse_action(text: str, action: str | None = None) -> dict[str, Any]:
    """Parse an explicit leading tag or explicitly supplied parsed action.

    Untagged natural language is deliberately not guessed from keywords. A
    conflicting field/tag, unknown tag, multiple tags, or wait-with-body is
    invalid. Even a pure '[wait]' string is an emitted intervention, not silence.
    """
    text = text if isinstance(text, str) else ""
    tag_match = _TAG.match(text)
    tag = tag_match.group(1).lower() if tag_match else None
    body = text[tag_match.end() :].strip() if tag_match else text.strip()
    field = action.strip().lower().strip("[]") if isinstance(action, str) else None
    field = field or None
    reason = None
    if field is not None and field not in ACTIONS:
        reason = "unknown_action_field"
    elif tag is not None and tag not in ACTIONS:
        reason = "unknown_action_tag"
    elif field is not None and tag is not None and field != tag:
        reason = "conflicting_action_field_and_tag"
    elif _TAG.match(body):
        reason = "multiple_leading_action_tags"
    else:
        parsed = field or tag
        if parsed is None:
            reason = "missing_explicit_action"
        elif parsed == "wait" and body:
            reason = "wait_has_spoken_content"
        else:
            return {"action": parsed, "content": body, "valid": True, "reason": None}
    return {"action": None, "content": body, "valid": False, "reason": reason}

def _ratio(numerator: float, denominator: float) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator > 0 else None,
    }

def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None

def _ordered_matches(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    allowed: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> list[tuple[int, int]]:
    """Monotonic maximum-cardinality matching, deterministic earliest-GT ties.

    Edges may not cross. Among optimal matchings choose the earliest available
    GT, then earliest eligible response. In time/joint modes edges exist only
    when an observed onset is inside the *current* GT window: neither nearest
    future GT nor text-similarity repair is allowed. Action-only intentionally
    ignores time and measures sequence agreement, never latency alignment.
    """
    rows, columns = len(expected), len(predicted)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    edges = [[allowed(gt, response) for response in predicted] for gt in expected]
    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            diagonal = 1 + table[i + 1][j + 1] if edges[i][j] else -1
            table[i][j] = max(diagonal, table[i + 1][j], table[i][j + 1])
    pairs = []
    i = j = 0
    while i < rows and j < columns:
        if edges[i][j] and table[i][j] == 1 + table[i + 1][j + 1]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i][j + 1] == table[i][j]:
            j += 1
        else:
            i += 1
    return pairs

def _union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged

def _silence_metrics(
    silent: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    raw_responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Speech duration must come from audio, never a text receive span.

    Prefer transport's global FIFO simulated playback intervals. A legacy
    positive audio_duration_s without playback deltas supports only explicitly
    labelled contiguous simulation. Text-only outputs make speech occupancy
    unavailable but still violate a silent window at their observed text time.
    """
    response_intervals, audio_intervals = [], []
    unavailable_audio, unavailable_intervals, audio_sources = [], [], set()
    for response in predictions:
        index = response["source_index"]
        raw = raw_responses[index]
        intervals = []
        if response["text"].strip():
            text_times = [
                _number(delta.get("media_time_s"))
                for delta in raw.get("text_deltas", [])
                if delta.get("text")
            ]
            text_times = [
                value for value in text_times if value is not None and value >= 0
            ]
            if text_times:
                intervals.append((min(text_times), max(text_times)))
            elif response["timing_valid"]:
                intervals.append((response["start_media_s"], response["end_media_s"]))
            else:
                unavailable_intervals.append(index)
        if response["audio_duration_s"] > 0:
            played = []
            chunks = raw.get("audio_deltas", [])
            missing_playback_chunk = False
            for delta in chunks:
                if delta.get("control_silence") is True:
                    continue
                start = _number(
                    delta.get(
                        "audible_start_media_s", delta.get("playback_start_media_s")
                    )
                )
                end = _number(
                    delta.get("audible_end_media_s", delta.get("playback_end_media_s"))
                )
                if start is not None and end is not None and start >= 0 and end > start:
                    played.append((start, end))
                else:
                    missing_playback_chunk = True
            if played:
                audio_sources.add("transport_simulated_fifo_playback")
                if missing_playback_chunk:
                    unavailable_audio.append(index)
                    unavailable_intervals.append(index)
            elif response["timing_valid"]:
                played = [
                    (
                        response["start_media_s"],
                        response["start_media_s"] + response["audio_duration_s"],
                    )
                ]
                audio_sources.add("simulated_contiguous_from_audio_duration")
            else:
                unavailable_audio.append(index)
                unavailable_intervals.append(index)
            intervals.extend(played)
            audio_intervals.extend(played)
        else:
            unavailable_audio.append(index)
        response_intervals.extend(
            {"response_index": index, "start_media_s": start, "end_media_s": end}
            for start, end in intervals
        )

    violated = []
    for window in silent:
        overlaps = [
            interval["response_index"]
            for interval in response_intervals
            if (
                max(window["start"], interval["start_media_s"])
                < min(window["end"], interval["end_media_s"])
                or interval["start_media_s"] == interval["end_media_s"]
                and window["start"] <= interval["start_media_s"] < window["end"]
            )
        ]
        if overlaps:
            violated.append(
                {
                    "annotation_index": window["source_index"],
                    "response_indices": sorted(set(overlaps)),
                }
            )
    specificity = _ratio(len(silent) - len(violated), len(silent))
    specificity.update(
        {
            "violated_windows": violated,
            "unavailable_response_indices": sorted(set(unavailable_intervals)),
        }
    )
    if unavailable_intervals:
        specificity.update(numerator=None, value=None)
    merged_audio = _union_intervals(audio_intervals)
    overlap_s = sum(
        max(0.0, min(end, window["end"]) - max(start, window["start"]))
        for start, end in merged_audio
        for window in silent
    )
    occupancy = _ratio(
        overlap_s, sum(window["end"] - window["start"] for window in silent)
    )
    occupancy.update(
        {
            "source": sorted(audio_sources),
            "audio_playback_interval_union": [
                list(interval) for interval in merged_audio
            ],
            "unavailable_response_indices": sorted(set(unavailable_audio)),
            "known_overlap_s_lower_bound": overlap_s,
            "not_measured_on_physical_soundcard": True,
        }
    )
    if unavailable_audio:
        occupancy.update(numerator=None, value=None)
    return {
        "silent_window_specificity": specificity,
        "silent_speech_occupancy": occupancy,
    }

def score_case(
    annotations: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """Score one complete replay. Invalid predictions remain counted as FP.

    Required GT keys: start/end/action; text_zh/text_en are optional references.
    Response keys: response_id/text/start_media_s/end_media_s, optionally action
    and audio_duration_s. Inputs must be utterance-grouped, not text deltas.
    Missing/invalid timing stays an FP in timing and joint event metrics.
    GT intervals are nonoverlapping; ends beyond video duration are clipped and
    recorded. Unannotated gaps are unknown, never silently treated as wait.
    """
    duration = _number(duration_s)
    if duration is None or duration <= 0:
        raise ValueError("duration_s must be positive and finite")
    windows = []
    for source_index, annotation in enumerate(annotations):
        start, end = _number(annotation.get("start")), _number(annotation.get("end"))
        action = parse_action("", annotation.get("action"))["action"]
        if (
            start is None
            or end is None
            or start < 0
            or end <= start
            or start >= duration
        ):
            raise ValueError(f"invalid GT interval at annotation {source_index}")
        if action not in ACTIONS:
            raise ValueError(f"invalid GT action at annotation {source_index}")
        window = dict(
            annotation,
            source_index=source_index,
            start=start,
            end=min(end, duration),
            action=action,
        )
        window["original_end"] = end
        window["end_clipped_to_duration"] = end > duration
        windows.append(window)
    windows.sort(key=lambda item: (item["start"], item["source_index"]))
    if any(left["end"] > right["start"] for left, right in zip(windows, windows[1:])):
        raise ValueError("GT intervals must not overlap")
    active = [window for window in windows if window["action"] != "wait"]
    silent = [window for window in windows if window["action"] == "wait"]
    silence_s = sum(window["end"] - window["start"] for window in silent)

    predictions, empty_records, control_silence_records = [], [], []
    for source_index, response in enumerate(responses):
        text = response.get("text", "")
        text = text if isinstance(text, str) else ""
        audio_s = _number(response.get("audio_duration_s")) or 0.0
        if (
            response.get("onset_source") == "audio_control_silence"
            and response.get("has_audible_audio") is False
        ):
            control_silence_records.append(
                {
                    "source_index": source_index,
                    "response_id": response.get("response_id"),
                    "text": text,
                    "audio_duration_s": audio_s,
                }
            )
            continue
        if not text.strip() and audio_s <= 0:
            empty_records.append(
                {
                    "source_index": source_index,
                    "response_id": response.get("response_id"),
                }
            )
            continue
        parsed = parse_action(text, response.get("action"))
        start = _number(response.get("start_media_s"))
        end = _number(response.get("end_media_s"))
        valid_time = (
            start is not None and end is not None and start >= 0 and end >= start
        )
        prediction = {
            "source_index": source_index,
            "response_id": response.get("response_id", str(source_index)),
            "text": text,
            **parsed,
            "start_media_s": start,
            "end_media_s": end,
            "audio_duration_s": audio_s,
            "onset_source": response.get("onset_source", "unspecified_receive_side"),
            "timing_valid": valid_time,
        }
        predictions.append(prediction)
    predictions.sort(
        key=lambda item: (
            item["start_media_s"] if item["start_media_s"] is not None else math.inf,
            item["source_index"],
        )
    )

    violated = [p for p in predictions if p["timing_valid"] and any(
        w["start"] <= p["start_media_s"] < w["end"] for w in silent)]
    return {"annotations": windows, "predictions": predictions,
            "counts": {"wait_windows": len(silent)},
            "silent_window_specificity": _silence_metrics(silent, predictions, responses)["silent_window_specificity"],
            "silence": {"annotated_silence_s": silence_s, "false_interventions": len(violated)}}
