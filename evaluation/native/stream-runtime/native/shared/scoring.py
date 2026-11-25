"""Deterministic native-streaming metrics, not an OmniInteract/MMI scorer.

Each non-wait annotation is one expected intervention in [start, end). Each
response is already grouped by the adapter into one real emitted utterance.
Metrics use receive-side media-relative onset, not an inferred event timestamp
mentioned in model text. The scorer never uses GT to parse a predicted action.
No semantic judge is called: exact text and character overlap are diagnostics.
"""

from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
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


def normalize_content(text: str) -> str:
    """NFKC, case-fold, remove whitespace/punctuation; no semantic rewriting."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _ratio(numerator: float, denominator: float) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator > 0 else None,
    }


def _prf(tp: int, predicted: int, expected: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": predicted - tp,
        "fn": expected - tp,
        "precision": _ratio(tp, predicted),
        "recall": _ratio(tp, expected),
        "f1": _ratio(2 * tp, predicted + expected),
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


def _timely(gt: dict[str, Any], response: dict[str, Any]) -> bool:
    onset = response["start_media_s"]
    return response["timing_valid"] and gt["start"] <= onset < gt["end"]


def _action_time_quadrants(
    active: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    joint_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """GT-slot companion explanation, not a new PRF or semantic alignment.

    Preserve joint TPs first. On unused events, prefer time-only matches, then
    action-only sequence matches, then zip residual events in their scored
    order. Every event is used at most once. Later stages may cross earlier
    stage pairings: their correspondence is operational, not semantic evidence.
    A required GT slot with no paired output counts as time-wrong/action-wrong.
    Paired replies with invalid timing remain unscorable, not called wrong.
    Pure paired-response cells and extra predictions are retained separately.
    """
    assigned = [(i, j, "joint") for i, j in joint_pairs]
    joint_gt, joint_pred = {i for i, _ in joint_pairs}, {j for _, j in joint_pairs}
    remaining_gt = [i for i in range(len(active)) if i not in joint_gt]
    remaining_pred = [j for j in range(len(predictions)) if j not in joint_pred]
    for stage, allowed in (
        ("time_only", _timely),
        ("action_only", lambda gt, response: gt["action"] == response["action"]),
    ):
        matches = _ordered_matches(
            [active[i] for i in remaining_gt],
            [predictions[j] for j in remaining_pred],
            allowed,
        )
        added = [(remaining_gt[i], remaining_pred[j], stage) for i, j in matches]
        assigned.extend(added)
        used_gt, used_pred = {i for i, _, _ in added}, {j for _, j, _ in added}
        remaining_gt = [i for i in remaining_gt if i not in used_gt]
        remaining_pred = [j for j in remaining_pred if j not in used_pred]
    residual = list(zip(remaining_gt, remaining_pred))
    assigned.extend((i, j, "residual_order") for i, j in residual)
    used_gt, used_pred = {i for i, _ in residual}, {j for _, j in residual}
    missing = [active[i]["source_index"] for i in remaining_gt if i not in used_gt]
    extras = [
        predictions[j]["source_index"] for j in remaining_pred if j not in used_pred
    ]
    names = (
        "time_wrong_action_wrong",
        "time_correct_action_wrong",
        "time_wrong_action_correct",
        "time_correct_action_correct",
    )
    cells = {name: {"count": 0} for name in names}
    pairs = []
    for i, j, stage in sorted(assigned):
        gt, response = active[i], predictions[j]
        time_correct = _timely(gt, response) if response["timing_valid"] else None
        action_correct = gt["action"] == response["action"]
        cell = None
        if time_correct is not None:
            cell = f"time_{'correct' if time_correct else 'wrong'}_action_{'correct' if action_correct else 'wrong'}"
            cells[cell]["count"] += 1
        pairs.append(
            {
                "annotation_index": gt["source_index"],
                "response_index": response["source_index"],
                "response_id": response["response_id"],
                "assigned_stage": stage,
                "cell": cell,
                "gt_start_s": gt["start"],
                "gt_end_s": gt["end"],
                "response_start_media_s": response["start_media_s"],
                "response_end_media_s": response["end_media_s"],
                "expected_action": gt["action"],
                "predicted_action": response["action"],
                "time_correct": time_correct,
                "action_correct": action_correct,
                "timing_valid": response["timing_valid"],
                "time_status": (
                    "unknown"
                    if time_correct is None
                    else "correct" if time_correct else "outside_window"
                ),
                "action_valid": response["valid"],
                "action_invalid_reason": response["reason"],
                "onset_source": response["onset_source"],
            }
        )
    classifiable = sum(cell["count"] for cell in cells.values())
    unscorable = len(pairs) - classifiable
    for cell in cells.values():
        cell["ratio_over_paired"] = _ratio(cell["count"], classifiable)
        cell["ratio_over_positive_slots"] = _ratio(cell["count"], len(active))
    paired_response_cells = cells
    classified_slots = classifiable + len(missing)
    cells = {
        name: {
            "count": cell["count"]
            + (len(missing) if name == "time_wrong_action_wrong" else 0)
        }
        for name, cell in paired_response_cells.items()
    }
    for cell in cells.values():
        cell["ratio_over_positive_slots"] = _ratio(cell["count"], len(active))
        cell["ratio_over_classifiable_slots"] = _ratio(cell["count"], classified_slots)
    assert sum(cell["count"] for cell in cells.values()) + unscorable == len(active)
    assert sum(
        cell["count"] for cell in paired_response_cells.values()
    ) + unscorable + len(missing) == len(active)
    assert sum(
        cell["count"] for cell in paired_response_cells.values()
    ) + unscorable + len(extras) == len(predictions)
    assert cells["time_correct_action_correct"]["count"] == len(joint_pairs)
    return {
        "schema_version": "omnilife-gt-slot-quadrants/v2",
        "cells": cells,
        "paired_count": len(pairs),
        "positive_gt_count": len(active),
        "paired_response_cells": paired_response_cells,
        "classified_positive_slot_count": classified_slots,
        "classifiable_paired_count": classifiable,
        "unscorable_timing_pairs_count": unscorable,
        "prediction_count": len(predictions),
        "missing_interventions_count": len(missing),
        "extra_predictions_count": len(extras),
        "pairs": pairs,
        "missing_annotation_indices": missing,
        "extra_response_indices": extras,
        "unknown_timestamp_response_indices": [
            p["source_index"] for p in predictions if p["start_media_s"] is None
        ],
        "invalid_timing_response_indices": [
            p["source_index"] for p in predictions if not p["timing_valid"]
        ],
        "invalid_action_response_indices": [
            p["source_index"] for p in predictions if not p["valid"]
        ],
        "semantic_correctness": None,
    }


def _references(gt: dict[str, Any]) -> list[str]:
    values = [gt.get("text_zh", ""), gt.get("text_en", ""), gt.get("text", "")]
    return list(
        dict.fromkeys(
            normalize_content(value)
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )


def _exact(gt: dict[str, Any], response: dict[str, Any]) -> bool:
    reference = _references(gt)
    content = normalize_content(response["content"])
    return bool(reference and content and content in reference)


def _character_f1(reference: str, prediction: str) -> float:
    if not reference or not prediction:
        return 0.0
    overlap = sum((Counter(reference) & Counter(prediction)).values())
    return 2 * overlap / (len(reference) + len(prediction))


def _summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def quantile(fraction: float) -> float | None:
        if not ordered:
            return None
        offset = (len(ordered) - 1) * fraction
        lower = math.floor(offset)
        upper = math.ceil(offset)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (offset - lower)

    return {
        "count": len(values),
        "sum_s": sum(values),
        "mean_s": sum(values) / len(values) if values else None,
        "p50_s": quantile(0.50),
        "p95_s": quantile(0.95),
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
        "values_s": values,
    }


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

    action_pairs = _ordered_matches(
        active, predictions, lambda gt, response: gt["action"] == response["action"]
    )
    time_pairs = _ordered_matches(active, predictions, _timely)
    joint_pairs = _ordered_matches(
        active,
        predictions,
        lambda gt, response: _timely(gt, response)
        and gt["action"] == response["action"],
    )
    triple_pairs = _ordered_matches(
        active,
        predictions,
        lambda gt, response: _timely(gt, response)
        and gt["action"] == response["action"]
        and _exact(gt, response),
    )
    behavior_correct = sum(
        active[i]["action"] == predictions[j]["action"] for i, j in time_pairs
    )
    warn_expected = sum(gt["action"] == "warn" for gt in active)
    warn_matched = sum(active[i]["action"] == "warn" for i, _ in joint_pairs)
    weighted_matches = [
        {
            "annotation_index": active[i]["source_index"],
            "response_index": predictions[j]["source_index"],
            "response_id": predictions[j]["response_id"],
            "weight": max(
                0.0,
                1
                - (predictions[j]["start_media_s"] - active[i]["start"])
                / (active[i]["end"] - active[i]["start"]),
            ),
        }
        for i, j in joint_pairs
    ]
    soft_mass = sum(item["weight"] for item in weighted_matches)
    silence_metrics = _silence_metrics(silent, predictions, responses)

    def evidence(pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
        return [
            {
                "annotation_index": active[i]["source_index"],
                "response_index": predictions[j]["source_index"],
                "response_id": predictions[j]["response_id"],
            }
            for i, j in pairs
        ]

    def scored(pairs: list[tuple[int, int]]) -> dict[str, Any]:
        return {
            **_prf(len(pairs), len(predictions), len(active)),
            "matches": evidence(pairs),
        }

    time_matched_gt = {i for i, _ in time_pairs}
    content_diagnostics = []
    for i, j in time_pairs:
        gt, response = active[i], predictions[j]
        refs = _references(gt)
        predicted_content = normalize_content(response["content"])
        content_diagnostics.append(
            {
                "annotation_index": gt["source_index"],
                "response_index": response["source_index"],
                "response_id": response["response_id"],
                "reference_available": bool(refs),
                "normalized_exact": _exact(gt, response) if refs else None,
                "character_unigram_f1_proxy": max(
                    (_character_f1(ref, predicted_content) for ref in refs),
                    default=None,
                ),
            }
        )
    valid_content = [
        item for item in content_diagnostics if item["reference_available"]
    ]
    expected_content_proxy = {
        "normalized_exact": _ratio(
            sum(item["normalized_exact"] for item in valid_content), len(active)
        ),
        "character_unigram_f1_proxy": _ratio(
            sum(item["character_unigram_f1_proxy"] for item in valid_content),
            len(active),
        ),
        "missing_reference_annotation_indices": [
            gt["source_index"] for gt in active if not _references(gt)
        ],
        "unmatched_expected_window_contribution": 0,
        "semantic_judge_score": None,
    }
    if expected_content_proxy["missing_reference_annotation_indices"]:
        expected_content_proxy["normalized_exact"]["value"] = None
        expected_content_proxy["character_unigram_f1_proxy"]["value"] = None

    per_action = {}
    for action in ACTIVE_ACTIONS:
        expected_count = sum(gt["action"] == action for gt in active)
        predicted_count = sum(response["action"] == action for response in predictions)
        true_positive = sum(active[i]["action"] == action for i, _ in joint_pairs)
        per_action[action] = _prf(true_positive, predicted_count, expected_count)
    macro_values = [
        item["f1"]["value"]
        for item in per_action.values()
        if item["f1"]["value"] is not None
    ]

    in_window = []
    silence_violations = []
    duplicate = []
    spillover = []
    early, late, outside_video = [], [], []
    occupancy: dict[int, int] = {}
    for response in predictions:
        if not response["timing_valid"]:
            continue
        onset, end = response["start_media_s"], response["end_media_s"]
        reference = {
            "response_index": response["source_index"],
            "response_id": response["response_id"],
        }
        current = next((gt for gt in active if gt["start"] <= onset < gt["end"]), None)
        if current:
            in_window.append(reference)
            index = current["source_index"]
            if occupancy.get(index, 0):
                duplicate.append({**reference, "annotation_index": index})
            occupancy[index] = occupancy.get(index, 0) + 1
            if end > current["end"]:
                spillover.append(
                    {
                        **reference,
                        "annotation_index": index,
                        "overrun_s": end - current["end"],
                    }
                )
        else:
            past = [gt for gt in active if gt["start"] <= onset]
            if active and not past:
                early.append(
                    {
                        **reference,
                        "signed_distance_to_first_start_s": onset - active[0]["start"],
                    }
                )
            elif past:
                late.append(
                    {
                        **reference,
                        "preceding_annotation_index": past[-1]["source_index"],
                        "distance_from_preceding_end_s": onset - past[-1]["end"],
                    }
                )
        if onset >= duration:
            outside_video.append(reference)
        if any(gt["start"] <= onset < gt["end"] for gt in silent):
            silence_violations.append(reference)

    onset_delay = [
        predictions[j]["start_media_s"] - active[i]["start"] for i, j in time_pairs
    ]
    signed_boundary_distance = [
        item["signed_distance_to_first_start_s"] for item in early
    ] + [item["distance_from_preceding_end_s"] for item in late]
    response_ids = [item["response_id"] for item in predictions]
    return {
        "schema_version": "omnilife-native-metrics/v1",
        "protocol": {
            "metrics_are": "independent_simple_deterministic_metrics_not_official_OmniInteract_or_OmniMMI_scores",
            "expected_event": "one_non_wait_annotation_window",
            "prediction_event": "one_adapter_grouped_utterance_with_nonempty_text_or_positive_audio_duration",
            "pure_wait_tag_is_an_emitted_intervention": True,
            "audio_control_silence": "all_zero_PCM_control_outputs_are_retained_but_not_actual_interventions",
            "time_clock": "transport_simulated_audio_playback_media_time_preferred_text_receive_time_fallback_never_model_reported_time",
            "text_only_fallback": "counted_as_text_intervention_not_claimed_as_audio_or_physical_soundcard_measurement",
            "timely": "GT_start <= response_onset < GT_end",
            "matching": "maximum_cardinality_monotonic_one_to_one_earliest_GT_then_earliest_response_ties",
            "action_only": "sequence_diagnostic_ignoring_time_not_used_for_latency_or_content_alignment",
            "time_and_joint": "current_window_only_no_future_or_nearest_GT_repair",
            "action_time_quadrants": "companion_common_one_to_one_pairing_preserve_joint_then_remaining_time_monotonic_then_remaining_action_monotonic_then_residual_order_zip_operational_not_semantic_correspondence_never_repairs_main_TP",
            "quadrant_time_wrong": "valid_timing_outside_paired_GT_window_invalid_or_missing_timing_has_null_cell_and_unknown_status_not_called_wrong",
            "quadrant_action_correct": "explicit_action_label_equality_only_not_semantic_content_correctness",
            "quadrant_population": "main_cells_unit_positive_non_wait_GT_slot_includes_missing_interventions_as_time_wrong_action_wrong_unknown_timing_excluded_paired_response_cells_retain_reply_only_diagnostic_extras_and_correct_silence_separate",
            "quadrant_missing": "missing_interventions_count_is_a_subset_of_main_time_wrong_action_wrong_not_a_fifth_category_no_actual_response_is_fabricated",
            "quadrant_rates": "main_cells_over_all_positive_slots_or_classifiable_paired_plus_missing_slots_paired_response_cells_ratio_over_paired_uses_classifiable_paired_count",
            "early_late": "boundary_diagnostics_only_early_before_first_window_other_outside_active_onsets_late_to_most_recent_past_window",
            "normalization": "NFKC_casefold_remove_unicode_whitespace_and_punctuation",
            "content": "normalized_exact_and_character_unigram_overlap_are_nonsemantic_diagnostic_proxies",
            "expected_content_proxy": "time_matched_content_proxy_sum_over_all_expected_active_windows_missing_interventions_zero_missing_references_unavailable",
            "macro_f1": "mean_supported_active_action_joint_F1_unknown_or_wait_predictions_count_as_global_FP_not_as_active_class",
            "behavior_accuracy": "correct_actions_among_first_one_to_one_time_matches_not_GT_repaired_joint_matches",
            "warn_miss_rate": "warn_windows_without_a_correct_action_and_timely_intervention_over_all_warn_windows",
            "silent_window_specificity": "wait_windows_not_intersecting_observed_text_or_simulated_audio_intervals_over_all_wait_windows",
            "silent_speech_occupancy": "union_of_digital_signal_audio_intervals_intersect_wait_seconds_over_wait_seconds_legacy_packet_fallback_text_only_duration_unavailable",
            "timeweighted_joint_f1": "2_sum(max(0,1_(onset_start)/(end_start)))_over_predicted_plus_positive_counts_for_joint_matches_only_not_IA_QTF1",
            "undefined_zero_denominators": None,
        },
        "duration_s": duration,
        "counts": {
            "annotation_windows": len(windows),
            "active_gt_events": len(active),
            "wait_windows": len(silent),
            "response_records": len(responses),
            "emitted_interventions": len(predictions),
            "audible_audio_interventions": sum(
                response["audio_duration_s"] > 0 for response in predictions
            ),
            "text_only_fallback_interventions": sum(
                response["audio_duration_s"] <= 0 for response in predictions
            ),
            "empty_bookkeeping_records": len(empty_records),
            "audio_control_silence_records": len(control_silence_records),
            "invalid_action_interventions": sum(
                not response["valid"] for response in predictions
            ),
            "invalid_timing_interventions": sum(
                not response["timing_valid"] for response in predictions
            ),
            "emitted_wait_tags": sum(
                response["action"] == "wait" for response in predictions
            ),
            "repeated_response_ids": len(response_ids) - len(set(response_ids)),
            "clipped_gt_ends": sum(
                window["end_clipped_to_duration"] for window in windows
            ),
        },
        "action_event": scored(action_pairs),
        "time_event": scored(time_pairs),
        "joint_action_time_event": scored(joint_pairs),
        "action_time_quadrants": _action_time_quadrants(
            active, predictions, joint_pairs
        ),
        "window_hit_rate": _ratio(len(time_pairs), len(active)),
        "joint_slot_success_rate": _ratio(len(joint_pairs), len(active)),
        "miss_rate": _ratio(len(active) - len(joint_pairs), len(active)),
        "behavior_accuracy_on_time_matches": _ratio(behavior_correct, len(time_pairs)),
        "warn_miss_rate": _ratio(warn_expected - warn_matched, warn_expected),
        "actiontime_softmass": soft_mass,
        "timeweighted_joint_f1": {
            **_ratio(2 * soft_mass, len(predictions) + len(active)),
            "matches": weighted_matches,
        },
        "timeweighted_joint_precision": _ratio(soft_mass, len(predictions)),
        "timeweighted_joint_recall": _ratio(soft_mass, len(active)),
        **silence_metrics,
        "strict_action_time_exact_content_proxy": scored(triple_pairs),
        "per_action_joint": per_action,
        "macro_action_joint_f1": {
            "value": sum(macro_values) / len(macro_values) if macro_values else None,
            "supported_actions": len(macro_values),
        },
        "in_window_rate": {
            **_ratio(len(in_window), len(predictions)),
            "responses": in_window,
        },
        "silence": {
            "annotated_silence_s": silence_s,
            "false_interventions": len(silence_violations),
            "false_interventions_per_minute": _ratio(
                len(silence_violations), silence_s / 60
            ),
            "responses": silence_violations,
        },
        "boundaries": {
            "early_before_first_window": early,
            "late_outside_active_window": late,
            "missed_gt_windows": [
                gt["source_index"]
                for i, gt in enumerate(active)
                if i not in time_matched_gt
            ],
            "duplicate_within_window": duplicate,
            "spillover_past_active_end": spillover,
            "onset_after_video_end": outside_video,
            "signed_outside_boundary_distance_s": signed_boundary_distance,
        },
        "response_delay_from_gt_start_s": _summary(onset_delay),
        "content_diagnostics_expected_active_windows": expected_content_proxy,
        "content_diagnostics_on_time_matches": {
            "semantic_judge_score": None,
            "semantic_judge_not_run_reason": "no_authorized_semantic_judge_no_semantic_correctness_claim",
            "normalized_exact": _ratio(
                sum(item["normalized_exact"] for item in valid_content),
                len(valid_content),
            ),
            "character_unigram_f1_proxy": {
                "value": (
                    sum(item["character_unigram_f1_proxy"] for item in valid_content)
                    / len(valid_content)
                    if valid_content
                    else None
                ),
                "count": len(valid_content),
            },
            "matches": content_diagnostics,
        },
        "annotations": windows,
        "predictions": predictions,
        "empty_records": empty_records,
        "control_silence_records": control_silence_records,
        "unannotated_duration_s": duration
        - sum(window["end"] - window["start"] for window in windows),
    }
