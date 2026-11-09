# Frozen reference results

`minicpm-o-4.5/` contains the released MiniCPM reference run. These files support offline reproduction of the published scores; they are normalized records, not native transport logs or a distribution of model weights.

| File | Records | Purpose |
|---|---:|---|
| `streaming.jsonl` | 400 | Session status, source duration, and normalized emitted responses |
| `qa.jsonl` | 800 | Raw answers, parsed choices, gold choices, and exact-match decisions |
| `judge/jobs.jsonl` | 92 | Unique content-judge inputs, frozen binary decisions, and requested/returned model IDs |
| `judge/pairs.jsonl` | 100 | Links between case annotations, model responses, and judge jobs |
| `matches.jsonl` | 400 | Case-level matches at 0/1/2/5/10-second end-window tolerances |
| `metrics.json` | 1 summary | Aggregate counts and scores |

## Reproduction

Run from the repository root:

```bash
python3 -m evaluation validate
python3 -m evaluation score --check
python3 -m evaluation test --verbose
```

These commands use only the Python standard library and local artifacts. They make no model or judge calls and require no videos, weights, GPU, or API credentials. `--check` compares the saved metrics and every case-level match with recomputed values.

The released input identities are in [`dataset/metadata/cases.jsonl`](../dataset/metadata/cases.jsonl). Keep the original annotations, questions, recorded outputs, and content judgments together. Editing labels, dropping unsuccessful cases, or filtering unwanted responses changes the benchmark.

## Interpretation and limits

- All 400 episodes and 800 questions remain in their fixed denominators, including failed sessions or invalid answers when present.
- The 54 emitted responses include nine whose onsets occur after the source video ends. These observations are retained; the scoring protocol determines eligibility without moving or deleting them.
- Response onset is receipt of the first text content on the source timeline. It is not audible playback latency.
- Content judging, temporal eligibility, and one-to-one matching are separate stages. A positive judge decision alone does not establish a successful intervention.
- Tolerance scoring changes only the accepted end boundary. It does not collect new replies or simulate faster inference.
- Frozen decisions permit exact offline reproduction. A new API run may differ, including when a model alias resolves to a different version. Reproducing these files does not independently validate source-video rights, safety, or model behavior on new inputs.

See [metric definitions](../dataset/README.md#scoring) and [observed data-quality notes](../dataset/metadata/quality-notes.json). Result files and embedded benchmark content are subject to the [data license](../dataset/LICENSE); code is licensed separately. For access or rights inquiries, follow the [data-access process](../dataset/README.md#access-privacy-and-rights) or contact **mengze@today.ai**.
