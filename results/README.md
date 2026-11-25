# Model results

[Main summary](summary.csv) · [Ablation summary](ablations.csv)

Each of the 12 model directories has 800 QA records and 400 Streaming rows. The five models with ablations include an `ablations.jsonl` file with 400 rows per tested condition, for 22 conditions in total.

| File | Contents |
| --- | --- |
| `qa.jsonl` | Output text, correctness and available latency for each Content or Timing question |
| `streaming.jsonl` | Full Correct decision, capture records and evidence route for each scene |
| `ablations.jsonl` | Per-condition Full Correct decisions and scoring evidence |

`null` means the available evidence could not resolve the score. Gemini still has 122 unresolved main scenes and 411 unresolved ablation scenes in this snapshot. `pending` counts those scenes. `full_correct_percent` is present when all 400 scenes have determined scores. The `score_source` field records whether a case used its original protocol or a supplemental capture, such as an explicit end-of-video control. The main and ablation summaries use the recorded decisions across those routes.

Rebuild the two tables with `python3 -m evaluation reproduce --dataset ./omni-life-data --check`. See the [evaluation guide](../evaluation/README.md) for setup and the [Hugging Face dataset](https://huggingface.co/datasets/mengze-hong/omni-life) for inputs and data terms.

## Human baseline

Two reviewers answered independently. Observed counts use original answers that remain compatible with the released benchmark. The 400-item estimates average each reviewer's observed rate and assume comparable difficulty for items without compatible answers.

| Track | Reviewer 1 observed | Reviewer 2 observed | Estimated on all 400 |
| --- | ---: | ---: | ---: |
| Content QA | 236/265 | 236/265 | 89.1% |
| Timing QA | 233/309 | 243/309 | 77.0% |
| Streaming Full Correct | 261/377 | 268/377 | 70.2% |
| Joint QA | 158/219 | 167/219 | |

The observed mean for Joint QA is 74.2% across the 219 paired scenes per reviewer.
