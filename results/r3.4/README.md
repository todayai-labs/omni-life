# Omni-Life r3.4 results

The latest dataset is [20261127-r3.4.1](https://huggingface.co/datasets/mengze-hong/omni-life/tree/5deda5205700b978f2c00fcc0475988379cb8cdc). It extends the baby_formula Content clip to 14 seconds, updates its Streaming onset to 13 seconds, and adds direct links for two existing Timing videos. The table below reports scores from r3.4 while the targeted Content evaluation for r3.4.1 is pending.

The model comparison uses the [final r3.4 dataset](https://huggingface.co/datasets/mengze-hong/omni-life/tree/31aa5b31efd75f552c2fa48a5b8741d4815e3a98). All ranked models have 400 Streaming and 800 Chinese QA outcomes. The revised Content and Timing questions for pet_dog_bowl were rerun. The Streaming reference change was reviewed against existing outputs and did not change any Full Correct result. [Targeted outputs](pet-dog-bowl-reruns.jsonl). [Streaming replay](pet-dog-bowl-streaming-replay.json).

| Respondent | Intervention Full Correct (297) | Silence Success (103) | Overall Full Correct (400) | Content | Timing | Joint |
|---|---:|---:|---:|---:|---:|---:|
| Human (human-002) | 60.27% | 96.12% | 69.50% | 90.00% | 74.25% | 68.75% |
| MiniCPM-o 4.5 | 3.37% | 98.06% | 27.75% | 73.25% | 30.75% | 24.75% |
| Qwen3.8 Flash | 5.72% | 89.32% | 27.25% | 85.00% | 56.50% | 50.00% |
| Gander | 2.02% | 99.03% | 27.00% | 48.50% | 27.75% | 16.25% |
| MOSS-VL | 8.42% | 79.61% | 26.75% | 19.25% | 4.75% | 0.50% |
| Qwen3.5 Plus | 5.72% | 87.38% | 26.75% | 82.75% | 44.75% | 39.50% |
| Qwen3.5 Flash | 5.05% | 87.38% | 26.25% | 73.50% | 30.00% | 23.25% |
| Venus-Omni | 6.06% | 80.58% | 25.25% | 78.75% | 35.00% | 30.75% |
| AURA | 0.34% | 11.65% | 3.25% | 82.75% | 28.50% | 25.75% |
| DuplexOmni (S1) | 4.38% | 0.00% | 3.25% | 28.00% | 5.00% | 1.00% |
| MOSS Preview | 0.00% | 3.88% | 1.00% | 73.75% | 15.25% | 12.25% |
| JoyAI | 0.00% | 0.00% | 0.00% | 78.25% | 19.75% | 16.00% |

Intervention Full Correct requires every required response in the scene to be correct and timely, with no extraneous response. Silence Success is computed over silence-only scenes. Overall Full Correct pools both groups. [Main model counts](summary.csv).

The human row carries forward the previous QA scores, including the seven revised Content items that received assumed-correct credit. [Human scoring record](human-002.json).

### Gemini 3.8 Live

| Language | Content | Timing | Joint | QA answers |
|---|---:|---:|---:|---:|
| Chinese | 86.50% | 47.25% | 43.00% | 800 |
| English | 87.75% | 49.00% | 45.25% | 800 |

Gemini Streaming has 278 of 400 scenes determined. [Gemini QA counts](gemini-qa.csv).

## Evaluation records

[Version and coverage manifest](experiment-versions.json) · [Dataset patch](final-patch.json) · [Earlier Streaming replay](streaming-replay.json) · [Input ablations on the previous r3.4 revision](ablations.csv)

Each ranked model directory contains all 800 Chinese QA records. Gemini outputs are available for [Chinese](gemini-3.8-live/qa-zh.jsonl) and [English](gemini-3.8-live/qa-en.jsonl). Unchanged Streaming and ablation records are in the [r3.3 archive](../README.md).
