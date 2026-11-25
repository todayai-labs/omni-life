# Omni-Life evaluation

Run from the repository root with Python 3.10+. Use [Hugging Face](https://huggingface.co/datasets/mengze-hong/omni-life) as the data source. The [results directory](../results/README.md) contains the model runs and summary tables.

## Verify and score

```sh
hf download mengze-hong/omni-life --repo-type dataset \
  --include 'scenes.csv' 'qa/*' 'protocol/*' --local-dir ./omni-life-data
python3 -m evaluation reproduce --dataset ./omni-life-data --check
```

`reproduce` verifies the submitted options against the answer key and rebuilds the main and ablation tables from the recorded Streaming verdicts. The scoring commands use the standard library and run offline.

| Command | Input | Output |
| --- | --- | --- |
| `qa --dataset DIR --question ID` | Hugging Face files and a question ID | Prompt, response format and media reference |
| `qa --dataset DIR --responses FILE` | CSV with `qa_id` and `raw_output` or `selected_choices` | Content, Timing and Joint QA scores |
| `streaming PROTOCOL CAPTURES` | Scoring rules and adjudicated Streaming captures | Per-scene Full Correct decisions and coverage |
| `ablation --protocol PROTOCOL --baseline B0 --arm CONDITION=FILE` | A baseline and one or more adjudicated ablation capture files | Per-condition Full Correct and paired coverage |

```sh
python3 -m evaluation qa --dataset ./omni-life-data --question qa_c_syn_adult_bath
python3 -m evaluation streaming ./omni-life-data/protocol/streaming-scoring.json captures.json
```

For QA, pass `--lang en` for the English question bank. A single-choice response must select exactly one letter. A multiple-choice response must match the complete answer set. Mark a failed QA capture with `technical_success=false`.

Streaming captures are JSON maps keyed by case ID. Each capture supplies `capture_complete`, `eof_opportunity`, `semantic_review_complete` and an ordered `replies` list. Each reply contains text, a semantic `verdict` and either `phase: video` with `video_s` or `phase: post_video` with `turn: 1`. A grounded reply supplies the ordered reference fact IDs in `facts`. Use the released [semantic criteria](https://huggingface.co/datasets/mengze-hong/omni-life/blob/main/protocol/semantic-judge.json) to adjudicate content before scoring. A missing verdict is `null`. Paired ablations use scenes with determined scores in both runs.

## Run and export native captures

The `native/` directory contains the model adapters and the shared Streaming worker. Use `python3 -m evaluation --help` to see the inference and export commands.

| Command | Purpose |
| --- | --- |
| `native-run` / `api-run` | Verify or run the baseline |
| `native-ablation` / `api-ablation` | Verify or run a named ablation condition |
| `export-streaming` / `export-ablation` | Export local runs with timing for adjudication |

These drivers require a local prepared-input handoff bound to the recordings. Set `OMNI_LIFE_HANDOFF` to its root with `dataset/manifest.json`, `stream-runtime/prepared/`, the source identity ledger and the selected case lists. Ablations also use the bound structured task, reminder or task-audio inputs. Hugging Face provides source media. The exact inference protocol uses the prepared input cache, which is supplied separately.

```sh
export OMNI_LIFE_HANDOFF=/path/to/prepared-inputs
python3 -m evaluation native-run --model gander --all --config ./gander.json
python3 -m evaluation native-ablation --model minicpm-o-4.5-local --condition R5 --config ./minicpm.json
```

Verification is the default. Add `--execute` and an output root to run inference. Use the model dependencies, checkpoints and API environment variables for your deployment. The workers save each attempt's output and latency. Run Full Correct scoring after exporting and adjudicating the captures.
