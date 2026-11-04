# Dataset and evaluation protocol

Omni-Life contains **400 episodes: 277 synthetic and 123 real-world**, spanning five application domains. Of these, **290 assistance episodes contain 555 interventions** and **110 episodes require silence throughout**. The 685 `wait` intervals also include quiet periods within assistance episodes.

## Files and identities

| Path | Contents |
| --- | --- |
| `annotations/` | 400 original episode annotations |
| `qa/qa_zh.csv` | 800 Chinese questions: one content and one timing question per episode |
| `metadata/cases.jsonl` | Canonical case inventory, relative paths, source types, durations, and SHA-256 identities |
| `metadata/populations.json`, `metadata/scenarios.json`, `metadata/statistics.json` | Scenario categories and release counts |
| `metadata/quality-notes.json` | Observed quality limitations and frozen-data handling decisions |
| `metadata/release-manifest.json` | Frozen-file identities for integrity checks |
| `metadata/media-rights.json` | Review status of source videos and included display assets; pending entries are not clearances |

`case_id` joins annotations, metadata, QA, and [reference results](../results/README.md). `metadata/cases.jsonl` is authoritative for benchmark membership, `source_type`, and `duration_s`. Its `annotation` paths are relative to `dataset/`; `video` paths are relative to a separately supplied video root. Hashes identify inputs and do not establish ownership, consent, or redistribution rights.

Annotation files contain a standing `query` and ordered `annotation` intervals, measured in source-media seconds with half-open `[start, end)` boundaries. The actions are `inform`, `warn`, `feedback`, `answer`, and `wait`. The first four require a reference-compatible response; `wait` requires silence. All 555 intervention references contain Chinese and English text. The frozen tasks and QA use Chinese; English standing-query translations are available for 336 episodes.

QA rows contain `qa_id`, `case_id`, `question_type`, relative `video`, `question`, `option_*`, and semicolon-separated answer letters. Content and timing each have 400 questions. Assistant-role fields describe task context, not a verified physical device. Population categories describe scenarios, not verified diagnoses or personal attributes.

## Scoring

Normalized streaming predictions use `start_media_s` and `end_media_s` on the source timeline. Onset means client receipt of the **first text content**, not audible playback. Content judging compares a response with the standing task and node-specific reference, without seeing the video or judging timing. Temporal eligibility and maximum-cardinality, order-preserving one-to-one matching are applied separately afterward.

| Metric | Definition |
| --- | --- |
| Intervention Recall (`event_recall`) | Timely, content-correct matches / 555 interventions |
| Event F1 (`event_f1`) | `2 × TP / (emitted responses + required interventions)`; wrong, duplicate, early, or otherwise unmatched responses reduce precision |
| Full Correct (`strict_video_correct`) | Assistance episodes with all interventions matched, no extra output, and no output interval overlapping reference silence / 290 |
| Silence Success (`silent_correct`) | Completed fully silent episodes with no actual output / 110 |
| Unwanted / min (`false_per_min`) | Response onsets in reference silence / reference-silence minutes, including quiet intervals within assistance episodes |
| Content / Timing QA | Exact option-set accuracy, each over 400 questions |
| Joint QA | Episodes with both content and timing answers correct / 400 |

Percentages use a 0–100 scale. Failed sessions and invalid QA answers remain in the fixed denominators. Behavior-tag agreement is not required for semantic credit. `video_correct` measures complete event coverage without the strict extra-output and silence-overlap checks; `counts.false` counts unwanted onsets, whereas F1 false positives are in `event_confusion.fp`.

Tolerance scoring adds 0/1/2/5/10 seconds only to reference-window ends, keeps outputs and content judgments fixed, and recomputes matching. It neither extends output collection nor simulates a faster model. [Frozen result files](../results/README.md) permit exact offline reproduction; a new judge API call can produce different labels.

## Access, privacy, and rights

Benchmark material is covered by the [Omni-Life Research Evaluation License 1.0](LICENSE) for **noncommercial research evaluation only; model training is prohibited**. Code has a separate [Apache-2.0 license](../LICENSE).

Source videos are not included due to their large size and may be provided upon reasonable request. Email [mengze@today.ai](mailto:mengze@today.ai), subject **Omni-Life data access**, with your research contact and affiliation, if applicable.
