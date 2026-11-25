# Omni-Life r3.2 experiments

This toolkit uses the November 25 [Hugging Face dataset](https://huggingface.co/datasets/mengze-hong/omni-life). The 297 assistance scenes have 500 required interventions, including 13 final count reports. Another 103 scenes require silence. The [historical toolkit](../README.md) reproduces the original paper-era reference.

## Reproduce the released main results

Run from the Git checkout with Python 3.10 or later. The dataset download below includes the recorded results and QA tables. It omits the source video archive.

```sh
hf download mengze-hong/omni-life --repo-type dataset \
  --include 'results/main/*' 'qa/*' 'scenes.csv' 'protocol/*' \
  --local-dir ./omni-life-data
python3 -m evaluation.r32.reproduce --dataset ./omni-life-data --check
```

The check replays 8,800 recorded QA answers and verifies 4,400 Streaming case decisions and every cell of the released tables. The original and combined columns retain separate evidence routes. Supplemental end-of-video or manual-commit runs can change model behavior.

## Score a new submission

```sh
python3 -m evaluation.r32.qa --dataset ./omni-life-data --question qa_c_syn_adult_bath
python3 -m evaluation.r32.qa --dataset ./omni-life-data --responses qa-answers.csv
python3 -m evaluation.r32.score_streaming ./omni-life-data/protocol/streaming-scoring.json b0.json
python3 -m evaluation.r32.ablation --protocol ./omni-life-data/protocol/streaming-scoring.json \
  --baseline b0.json --arm Q-structured=structured.json --arm R1=repeated-task.json
```

QA submissions use one `qa_id` per row with `raw_output` or `selected_choices`. An optional `technical_success=false` records a failed capture. Single-choice responses must select one letter. Multiple-choice responses must match the full answer set. `--lang en` selects the English questions.

Streaming files contain adjudicated capture records keyed by `case_id`. A record carries `capture_complete`, `eof_opportunity`, `semantic_review_complete`, and ordered `replies`. Each reply contains its text, semantic `verdict`, and either `phase=video` with `video_s` or `phase=post_video` with `turn=1`. A grounded reply supplies the ordered reference fact IDs in `facts`. Consult the dataset's semantic judgment criteria for additional fields. The scorer reports incomplete evidence explicitly. The ablation tool compares each arm with B0 on scenes with two determined scores and reports the paired denominator.

## Replay native Streaming inputs

The [native sources](native/) include the self-hosted B0 workers, the Qwen and Gemini WebSocket clients, and the original structured-instruction, reminder and audio-only ablation drivers. To replay the archived input protocol, set `OMNI_LIFE_HANDOFF` to a local input handoff with the signed r3-final dataset, `stream-runtime/prepared/`, the original `source-identities.json` and its receipt, and the corresponding case lists. Native ablations also use the recorded `ablations/conditions.json` and audio-input index from that handoff. The Hugging Face media download supplies source media; exact archived replay additionally requires the original prepared cache and manifests. Set model configurations and runtime dependencies for your machine.

```sh
export OMNI_LIFE_HANDOFF=/path/to/prepared-handoff
native=evaluation/r32/native
python3 "$native/stream-runtime/run_native.py" --model gander --all --config ./gander.json
python3 "$native/ablations/runner/native/run_arm.py" \
  --model minicpm-o-4.5-local --condition R5 --config ./minicpm.json
python3 "$native/stream-runtime/api/run.py" --model gemini-3.8-live \
  --case-ids asm_dumpbed_feedback_01
python3 "$native/ablations/runner/api/run.py" --model gemini-3.8-live --arm R1 \
  --case-ids p4_office_late
```

These commands check the listed inputs without calling models. Add `--execute` and the run output argument shown by `--help` for inference. Supply your own API credentials through the documented environment variables and use the configured service endpoint for your account. The native runtimes save per-attempt output and timing. `export_native.py` and `export_arm.py` prepare captures for semantic adjudication before Full Correct scoring. Record any change to the model checkpoint, timing, transport, or terminal controls with the resulting outputs.
