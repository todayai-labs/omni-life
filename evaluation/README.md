# Evaluation toolkit

Run all commands from the repository root with **Python 3.10+**. The released scores, input validation, and tests use the standard library only; no videos, weights, GPU, credentials, or dependency installation are needed for offline reproduction.

## Structure and usage

| Files | Responsibility |
| --- | --- |
| `core/` | Output parsing, normalized export, scoring, and reference reproduction |
| `*.py` | Tools for inference, scoring, and validation, used through `python3 -m evaluation` |
| `tests/` | Offline protocol, integrity, media-review, and credential-handling tests |
| `configs/` | Frozen model, decoding, and content-judge settings |
| `prompts/` | Frozen streaming, QA, judging, and dataset-construction prompts |

| Command | Purpose |
| --- | --- |
| `score` | Recompute saved results; `--check` verifies reference metrics and every case-level match |
| `validate` | Validate dataset schemas, coverage, and reference artifacts |
| `check` | Check frozen identities, release hygiene, and local links |
| `media` | Check media-review records; `--require-cleared` requires documented clearance |
| `test` | Run offline tests; `--verbose` prints each result |
| `prepare` | Verify separately obtained videos, extract source-aligned frames/audio, and prepare tasks |
| `run` | Run streaming or QA with a local MiniCPM checkpoint |
| `export` | Convert a new run to normalized records and content-judge inputs |
| `judge` | Judge new text pairs through an explicitly configured HTTPS API |

Use `python3 -m evaluation <command> --help` for options. Start with:

```bash
python3 -m evaluation validate
python3 -m evaluation score --check
python3 -m evaluation check
python3 -m evaluation test --verbose
```

`media` checks records rather than granting permission. Before publishing included media, run `python3 -m evaluation media --require-cleared`; it fails while any included asset has pending review. Source-video requests are assessed per case, using `--case-id ID` and the [access process](../dataset/README.md#access-privacy-and-rights).

## Run a new evaluation

New inference generates a separate result and does not replace the frozen reference. Obtain source media through the [access process](../dataset/README.md#access-privacy-and-rights). Keep videos, weights, credentials, private evidence, and generated runs outside the files you commit.

### 1. Prepare model and media

Use [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/README.md) at revision **`503e754207c94da6bb26850b4469f367c9ea3582`**, with its official inference dependencies and a CUDA GPU. The adapter loads executable checkpoint code with `trust_remote_code=True`; inspect that code before execution. The root dependency list is a convenience list, not a tested environment lock. Install FFmpeg separately and make `ffmpeg` and `ffprobe` available on `PATH`.

```bash
python3 -m pip install -r requirements.txt
python3 -m evaluation prepare --videos ./videos --output ./prepared
```

Place videos using the relative paths in [`dataset/metadata/cases.jsonl`](../dataset/metadata/cases.jsonl). Preparation checks hashes, samples timestamped frames, extracts original audio, and inserts each written standing task into the frozen streaming prompt. It adds no task speech or artificial track. `--case-id ID` selects a subset; `--dry-run` checks source identities without extracting media.

[`configs/minicpm.json`](configs/minicpm.json) fixes seed 20261008, greedy decoding, 20 new speak tokens per chunk, `force_listen_count=0`, 1 Hz frames at width 640, and 16 kHz audio in 1-second chunks. Audio generation is disabled. Preserve these values when claiming the released protocol.

### 2. Run streaming and QA

```bash
python3 -m evaluation run --track streaming \
  --model-path ./weights/MiniCPM-o-4_5 --prepared ./prepared --output ./runs/streaming
python3 -m evaluation run --track qa \
  --model-path ./weights/MiniCPM-o-4_5 --prepared ./prepared --output ./runs/qa
```

Each run requires a new output directory. `--dry-run` validates prepared inputs without loading a model; `--case-id ID` selects cases. Omitted or failed cases receive no credit when exported against the full benchmark.

### 3. Export and judge

```bash
python3 -m evaluation export --streaming ./runs/streaming --qa ./runs/qa \
  --prepared ./prepared --output ./runs/portable
python3 -m evaluation judge --jobs ./runs/portable/judge/jobs-input.jsonl \
  --output ./runs/portable/judge/jobs.jsonl --dry-run
```

To make new judge calls, set `JUDGE_BASE_URL` to an HTTPS OpenAI-compatible base URL without embedded credentials, query parameters, or a fragment, and `JUDGE_API_KEY` in the environment. Repeat the last command without `--dry-run`; `JUDGE_MODEL` optionally chooses the requested model. Redirects are refused to prevent credential forwarding, and errors omit response bodies and configured endpoint values. Never write credentials into source or result files.

The judge sends task, reference, context, and candidate text to that provider. Review the provider's processing terms and the [data license](../dataset/LICENSE) before transmission. [`configs/judge.json`](configs/judge.json) records the historical requested alias `deepseek-v4-flash` separately from returned model `deepseek-v4-flash-0731` (temperature 0, thinking disabled, 32 output tokens). New alias calls can resolve differently and produce different labels; actual returned IDs are recorded. Use a new output file.

### 4. Score

```bash
python3 -m evaluation score --results ./runs/portable --output ./runs/portable/metrics.json
```

Scoring uses local references, normalized outputs, and saved judge decisions, retaining 400-video and 800-question denominators. It makes no model or judge call. See the [metric definitions](../dataset/README.md#scoring) and [frozen result layout](../results/README.md).

## Changes and private reports

Keep the released data, prompts, configurations, and reference results frozen. Submit a proposed correction separately with case IDs, rationale, affected fields, and evidence. Accepted semantic changes need a versioned change record, updated manifest, and regenerated affected results. Document protocol deviations and subsets in new results. New media requires actual rights/privacy review; record only opaque evidence references in `dataset/metadata/media-rights.json` and keep permission documents private.

Report credentials, security vulnerabilities, personal information, and source-permission concerns privately to [mengze@today.ai](mailto:mengze@today.ai). Include the affected commit/path/case and minimal reproduction using synthetic inputs where possible; do not send live credentials or publish private evidence in an issue. Credential scanning and integrity tests do not replace source-level rights or privacy review. Revoke an exposed credential before coordinating repository/history remediation.

## Licensing and external materials

| Material | Terms |
| --- | --- |
| Original evaluation scripts and software in `core/` and `tests/`; original configuration and dependency manifests; original technical documentation | [Apache License 2.0](../LICENSE), excluding embedded benchmark content and third-party material |
| Annotations, metadata, QA, benchmark prompts in `evaluation/prompts/`, and frozen results | [Omni-Life Research Evaluation License 1.0](../dataset/LICENSE), `LicenseRef-Omni-Life-Research-Evaluation-1.0`: noncommercial research evaluation, no model training |
| Source video/audio, third-party frames/images, and figures/artwork in `assets/` | Excluded from both grants; establish rights for the particular source and intended use |
| Today and other names, logos, trademarks, and brand assets | No branding, trademark, or endorsement permission is granted |
| External model code/weights, libraries, fonts, FFmpeg, and tools installed separately | Their upstream licenses and notices apply |

The root Apache-2.0 license text is unmodified. Data restrictions do not amend the software license, and the software license does not extend to data or excluded media. Permission covers only rights held by each contributor or relevant rights holder. The [pinned MiniCPM model card](https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/503e754207c94da6bb26850b4469f367c9ea3582/README.md) identifies its code and weights as Apache-2.0; Omni-Life does not relicense them. Reference results imply no endorsement by model or tool developers.

Synthetic/real-world source labels describe origin, not copyright ownership, participant consent, or redistribution clearance. Figures include project-supplied assets and paper excerpts; their inclusion grants no independent rights to underlying media or depicted people. The release has no completed source-by-source rights or consent register and must not be described as fully cleared or fully anonymized. Send attribution or rights concerns through the [private process](../dataset/README.md#access-privacy-and-rights).
