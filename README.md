# Omni-Life

**Benchmarking Omnimodal Real-Time AI Assistants in People's Daily Lives.**

Omni-Life evaluates whether an assistant follows a standing task, delivers the required help on time, and stays silent when no help is needed. It contains **400 video episodes** (277 synthetic and 123 real-world): **290 assistance episodes with 555 required events**, plus **110 fully silent episodes**. The same references support **800 QA questions**: 400 content and 400 timing questions.

This repository provides the non-video dataset, prompts, evaluation code, and reference results for **MiniCPM-o 4.5**. Videos and model weights are not included.

## Contents

| Path | Contents |
|---|---|
| [`dataset/`](dataset/) | Original annotations, QA questions, case identities, population and scenario metadata |
| [`prompts/`](prompts/) | Frozen streaming, QA, and content-judge instructions; construction prompt references |
| [`configs/`](configs/) | MiniCPM checkpoint revision and evaluated settings; judge model metadata |
| [`evaluation/`](evaluation/) | Replay adapter, output parsing, temporal matching, and metric computation |
| [`scripts/`](scripts/) | Media preparation, inference, result export, judging, scoring, and validation |
| [`results/minicpm-o-4.5/`](results/minicpm-o-4.5/) | 400 streaming records, 800 QA answers, frozen judge decisions, matches, and metrics |

See [data formats and scoring](docs/data-format.md) for field definitions and denominators.

## Reproduce the reference scores

Requires **Python 3.10+**, with no additional packages, videos, GPU, or API calls:

```bash
python3 scripts/validate_release.py
python3 scripts/score.py --check
```

`--check` verifies the released metrics and every case-level match, including the 0/1/2/5/10-second latency-tolerance settings.

| Model | Recall (%) | Event F1 (%) | Full Correct (%) | Silence Success (%) | Unwanted / min |
|---|---:|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 1.44 | 2.63 | 1.38 | 95.45 | 0.14 |

| Content QA (%) | Timing QA (%) | Joint QA (%) |
|---:|---:|---:|
| 53.50 | 25.75 | 13.00 |

Streaming records preserve normalized emitted responses and source-media timestamps. QA records retain raw answer text. Content-judge decisions are released separately from final temporal, one-to-one matches.

## Run new evaluations

Follow the [inference, judging, and scoring commands](docs/running.md). Inference requires the pinned checkpoint, its official dependencies, a CUDA GPU, FFmpeg, and the matching source videos. API credentials are read from environment variables. New outputs are written separately from the reference results.

Tests:

```bash
python3 -m unittest discover -s tests
```

## Citation

```bibtex
@misc{hong2026omnilife,
  title={Omni-Life: Benchmarking Omnimodal Real-Time AI Assistants in People's Daily Lives},
  author={Hong, Mengze and Shang, Ziwei and Liang, Di and Li, Zijian and Zhu, Liang and Shan, Hongyu and Zhang, Chen Jason},
  year={2026},
  howpublished={Technical report}
}
```
