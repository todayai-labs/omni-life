# Run a new evaluation

Run commands from the repository root. These commands generate new results; use `scripts/score.py --check` for exact offline reproduction of the released reference.

## 1. Prepare the model and media

Use the [MiniCPM-o 4.5 checkpoint](https://huggingface.co/openbmb/MiniCPM-o-4_5) at the revision in `configs/minicpm.json`, together with its official inference dependencies and a CUDA GPU. Install FFmpeg and ensure `ffmpeg` and `ffprobe` are on `PATH`.

```bash
python3 -m pip install -r requirements-inference.txt
python3 scripts/prepare_media.py --videos ./videos --output ./prepared
```

Place source videos under `videos/` using the relative paths in `dataset/metadata/cases.jsonl`. Preparation checks their hashes, samples timestamped video frames, extracts original audio, and inserts each written standing task into the frozen streaming prompt. It adds no task speech or artificial audio track.

Use `--case-id ID` to prepare a subset or `--dry-run` to verify source identities without extracting media. Offline scoring is fully validated without media; decoding and inference require the dependencies above.

## 2. Run streaming and QA

```bash
python3 scripts/run_minicpm.py --track streaming \
  --model-path ./weights/MiniCPM-o-4_5 --prepared ./prepared --output ./runs/streaming
python3 scripts/run_minicpm.py --track qa \
  --model-path ./weights/MiniCPM-o-4_5 --prepared ./prepared --output ./runs/qa
```

Each command requires a new output directory. `--dry-run` validates prepared inputs without loading the model. `--case-id ID` selects a subset; omitted and failed cases still receive no credit when exported into the full benchmark.

## 3. Export and judge new responses

```bash
python3 scripts/export_run.py --streaming ./runs/streaming --qa ./runs/qa \
  --prepared ./prepared --output ./runs/portable
python3 scripts/run_judge.py --jobs ./runs/portable/judge/jobs-input.jsonl \
  --output ./runs/portable/judge/jobs.jsonl --dry-run
```

Export produces normalized streaming and QA records and content-judge inputs. Set `JUDGE_BASE_URL` to an HTTPS OpenAI-compatible API base and `JUDGE_API_KEY` in your environment, then repeat the judge command without `--dry-run`. Never put credentials in source files or result records. `JUDGE_MODEL` optionally selects the requested model.

The historical requested alias and returned version are recorded separately in `configs/judge.json`. New calls may differ from the frozen labels; each response records the actual returned model. A new judge output file is required.

## 4. Score

```bash
python3 scripts/score.py --results ./runs/portable --output ./runs/portable/metrics.json
```

Scoring uses the released references, normalized outputs, and judge decisions, retaining the full 400-video and 800-question denominators. It does not call a model or judge API.
