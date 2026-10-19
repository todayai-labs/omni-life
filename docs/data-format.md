# Data formats and scoring

## Dataset

`dataset/metadata/cases.jsonl` has one row per video episode. `case_id` is the stable annotation and result key. `video` is relative to a user-supplied video root; `annotation` is relative to `dataset/`. Video and annotation SHA-256 hashes identify the evaluated inputs.

`duration_s` is the source-media duration recorded in the frozen evaluation manifest, not inference wall time. `source_type` distinguishes the 277 synthetic and 123 real-world videos. Population and scenario names are defined in `dataset/metadata/populations.json` and `scenarios.json`.

Each original `dataset/annotations/**/*.labels.json` contains a standing `query` and ordered `annotation` intervals. Intervals use source-video seconds and the half-open convention `[start, end)`. The five actions are `inform`, `warn`, `feedback`, `answer`, and `wait`. Required responses have Chinese and English references expressing the same assistance requirement. `wait` intervals require no output. An assistant role describes the task context; it is not a verified physical-device label.

`dataset/qa/qa_zh.csv` contains 800 questions with stable `qa_id`, `case_id`, `question_type`, relative `video`, `question`, `option_*`, and semicolon-separated answer letters. Content and timing each have 400 questions.

## MiniCPM reference results

| File | Meaning |
|---|---|
| `streaming.jsonl` | 400 sessions: model, case ID, status, source-media duration, and normalized predictions |
| `qa.jsonl` | 800 raw answer strings, gold and parsed choices, format validity, and exact-match results |
| `judge/jobs.jsonl` | 92 unique frozen content judgments, their inputs, raw binary answers, and requested/returned model IDs |
| `judge/pairs.jsonl` | 100 links between an annotation, a recorded response, and a content-judge job |
| `matches.jsonl` | Final matches for every case at 0/1/2/5/10 seconds of added end-window tolerance |
| `metrics.json` | Aggregate counts and scores; percentage metrics use the 0–100 scale |

Streaming predictions are normalized records, not native transport dumps. `start_media_s` and `end_media_s` use the source timeline. Onset denotes client-side receipt of the first text content; it does not measure audible playback. Empty prediction lists are retained.

Content judgments compare a candidate response with the standing task and node-specific reference. The judge does not see the video or determine response timing. A positive judge label alone is not a successful intervention: temporal eligibility and one-to-one matching are applied afterward. Frozen automatic decisions are supplied for exact offline reproduction; rerunning an API alias can produce different decisions or resolve to a different model version.

## Metrics

- **Intervention Recall (`event_recall`):** timely, content-correct matches divided by all 555 required events.
- **Event F1 (`event_f1`):** `2 * TP / (emitted responses + required events)`. Wrong, duplicate, early, or otherwise unmatched outputs count against precision.
- **Full Correct (`strict_video_correct`):** successful assistance episodes with every required event matched, no extra output, and no response interval overlapping reference silence; denominator 290.
- **Silence Success (`silent_correct`):** completed fully silent episodes with no actual output; denominator 110.
- **Unwanted / min (`false_per_min`):** response onsets during reference silence divided by the duration of that silence in minutes, including quiet intervals within assistance episodes.
- **QA:** exact option-set matching; content and timing each use 400 questions, while Joint requires both answers for a video to be correct.

Failed sessions and invalid QA answers remain in the fixed denominators. Behavior-tag agreement is not required for semantic content credit. The separate `video_correct` field reports complete event coverage without the strict extra-output and silence-overlap checks. `counts.false` counts unwanted onsets; F1 false positives are in `event_confusion.fp`.

The MiniCPM reference has 54 emitted responses and 8 timely, content-correct matches: TP=8, FP=46, FN=547. Four assistance episodes are fully correct, 105 fully silent episodes succeed, and 23 outputs begin during 166.201195 minutes of reference silence.

Latency-tolerance scoring extends only each interval's end, keeps recorded outputs and content judgments fixed, and recomputes maximum-cardinality, order-preserving one-to-one matches. It does not extend output collection or simulate a faster model.
