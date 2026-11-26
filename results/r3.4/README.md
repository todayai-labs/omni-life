# Omni-Life r3.4 results

The model comparison uses [20261126-r3.4](https://huggingface.co/datasets/mengze-hong/omni-life/tree/d309530b697922e3a1ea9d9ec3aeadec393fc154). The revised Content prompts were rerun. Timing and Streaming changes were applied to the original captures where inputs were unchanged. Each ranked model has 400 Streaming and 800 Chinese QA outcomes.

| Respondent | Streaming Full Correct | Content | Timing | Joint |
|---|---:|---:|---:|---:|
| Human (human-002) | 69.50% (278/400) | 90.00% (360/400) | 74.25% (297/400) | 68.75% (275/400) |
| MiniCPM-o 4.5 | 27.75% | 73.25% | 30.75% | 24.75% |
| Qwen3.8 Flash | 27.25% | 85.00% | 56.50% | 50.00% |
| Gander | 27.00% | 48.50% | 27.75% | 16.25% |
| MOSS-VL | 26.75% | 19.25% | 4.75% | 0.50% |
| Qwen3.5 Plus | 26.75% | 82.75% | 44.75% | 39.50% |
| Qwen3.5 Flash | 26.25% | 73.50% | 30.00% | 23.25% |
| Venus-Omni | 25.25% | 78.75% | 35.00% | 30.75% |
| AURA | 3.25% | 82.75% | 28.75% | 26.00% |
| DuplexOmni (S1) | 3.25% | 28.00% | 5.00% | 1.00% |
| MOSS Preview | 1.00% | 73.75% | 15.25% | 12.25% |
| JoyAI | 0.00% | 78.25% | 19.75% | 16.00% |

Seven revised Content items receive assumed-correct credit. [Scoring record](human-002.json). [Main model counts](summary.csv)

### Gemini 3.8 Live

| Language | Content | Timing | Joint | QA answers |
|---|---:|---:|---:|---:|
| Chinese | 86.75% | 47.25% | 43.00% | 800 |
| English | 87.75% | 49.00% | 45.25% | 800 |

Gemini Streaming coverage is 278/400 determined in the previous published run. [Gemini QA counts](gemini-qa.csv)

[Input ablations regraded for r3.4](ablations.csv) · [Previous results](https://huggingface.co/datasets/mengze-hong/omni-life/tree/aef4be03e7154495837f2567ea3bcdb293787e5c)

## Evaluation records

[Version and coverage manifest](experiment-versions.json) · [Streaming replay](streaming-replay.json) · [Ablation replay](ablation-replay.json)

Each ranked model directory contains its 800 Chinese QA records. The replay files contain the affected Streaming scenes and ablation outcomes. Unaffected Streaming and ablation records are in the [r3.3 archive](../README.md).
