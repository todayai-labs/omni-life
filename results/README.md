# Model results

[Main summary](summary.csv) · [Ablation summary](ablations.csv)

Each of the 12 model directories has 800 QA records and 400 Streaming rows. The five models with ablations include an `ablations.jsonl` file with 400 rows per tested condition, for 22 conditions in total.

| File | Contents |
| --- | --- |
| `qa.jsonl` | Output text, correctness and available latency for each Content or Timing question |
| `streaming.jsonl` | Full Correct decision, capture records and evidence route for each scene |
| `ablations.jsonl` | Per-condition Full Correct decisions and scoring evidence |

`null` means the available evidence could not resolve the score. Gemini still has 122 unresolved main scenes and 411 unresolved ablation scenes in this snapshot. `pending` counts those scenes. `full_correct_percent` is present when all 400 scenes have determined scores. The evidence route is recorded per case, including runs that use supplemental terminal controls.

Rebuild the two tables with `python3 -m evaluation reproduce --dataset ./omni-life-data --check`. See the [evaluation guide](../evaluation/README.md) for setup and the [Hugging Face dataset](https://huggingface.co/datasets/mengze-hong/omni-life) for inputs and data terms.
