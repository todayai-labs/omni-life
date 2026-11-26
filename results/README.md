# Model results

[Main summary](summary.csv) · [Input ablations](ablations.csv) · [Paper analysis values](paper.json)

## Dataset versions

These results use **20261126-r3.3**. The two revised Content prompts were rerun for all 11 models. The other 798 QA answers per model were preserved and regraded under the current answer keys. Streaming and input-ablation evidence is carried over because the inputs and rubric are unchanged.

The [experiment versions](experiment-versions.json) bind the current results to Hugging Face revision `5f776e25cb1f1eb51b42bc2ddf82eab795544755`. The [r3.2 archive](https://github.com/todayai-labs/omni-life/tree/6a2eb02/results) uses the [previous dataset revision](https://huggingface.co/datasets/mengze-hong/omni-life/tree/1df079eadff3c0db96062c119183605d66821954).

## Main comparison

The model comparison uses Chinese QA with 400 items per track and 400 determined Streaming outcomes per model. The human row shows full-set rates with the adjustments documented below.

| Respondent | Streaming Full Correct | Content | Timing | Joint |
|---|---:|---:|---:|---:|
| AURA | 3.25% | 82.25% | 28.75% | 26.00% |
| DuplexOmni (S1) | 3.50% | 28.25% | 5.00% | 1.00% |
| Gander | 27.00% | 48.25% | 27.25% | 15.75% |
| JoyAI | 0.00% | 77.50% | 19.75% | 15.75% |
| MiniCPM-o 4.5 | 27.75% | 73.25% | 30.50% | 24.50% |
| MOSS Preview | 1.00% | 74.00% | 15.25% | 12.50% |
| MOSS-VL | 26.75% | 19.25% | 4.75% | 0.50% |
| Qwen3.5 Flash | 26.25% | 73.50% | 30.00% | 23.50% |
| Qwen3.5 Plus | 26.75% | 82.25% | 44.75% | 39.50% |
| Qwen3.8 Flash | 27.25% | 84.75% | 56.50% | 49.75% |
| Venus-Omni | 25.25% | 78.25% | 35.00% | 30.75% |
| Human (human-002, adjusted) | 70.25% (281/400) | 89.75% (359/400) | 74.75% (299/400) | 69.00% (276/400) |

| File | Contents |
| --- | --- |
| `<model>/qa.jsonl` | Recorded Chinese QA output, correctness and measured latency |
| `<model>/streaming.jsonl` | Scene verdict, original assessment, available captures and supplemental evidence |
| `<model>/ablations.jsonl` | Input condition, scene verdict, available original capture and supplemental evidence |
| `scene-groups.csv` | Scenario and service population for all 400 scenes |
| `judge-panel.csv` | Four original-protocol assessments for each of 1,600 model and scene pairs |
| `human-baseline.csv` | Observed reviewer counts by track |

`score_source` identifies the evidence used for each scene. The main comparison uses 4,329 original-protocol assessments and 71 supplemental assessments. The 18 input ablations use 6,967 original-protocol assessments and 233 supplemental assessments. Supplemental routes include explicit terminal controls, provider-specific media commits and zero-valued PCM for silent source clips. They can change model behavior. Event diagnostics and deadline replay use original captures. Judge sensitivity uses scenes with a decision from every judge within each model. An empty judge cell means the score is unresolved.

Run `python3 -m evaluation reproduce --dataset ./omni-life-data --check` to replay the available judged captures and verify the published summaries. Omit `--check` to print the computed results and paper analysis values as JSON. Where an accepted adjudication or failure certificate is the available evidence, verification uses that record. See the [evaluation guide](../evaluation/README.md) for dataset and inference setup.

## Human baseline

The adjusted human-002 row assigns credit to seven unverified Content items (six blank and one with an unverified prompt version) and to seven unsubmitted Streaming scenes. Five of the seven adjusted Content items have correct Timing answers. The 16 submitted Streaming follow-ups were scored separately: six correct and ten incorrect. Observed counts before adjustment are Streaming 274/393, Content 352/393, Timing 299/400, and Joint 271/393.

## Gemini 3.8 Live

Gemini 3.8 Live has complete Chinese and English QA. Its Streaming has **278/400 determined scenes** and 122 unresolved, so no full-set Streaming score is reported.

| Language | Content | Timing | Joint | QA answers |
|---|---:|---:|---:|---:|
| Chinese | 86.50% | 47.25% | 42.75% | 800 |
| English | 87.75% | 49.00% | 45.25% | 800 |

See [Gemini coverage](gemini-coverage.csv) for the exact counts.
