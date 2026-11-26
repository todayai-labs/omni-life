# Model results

[Main summary](summary.csv) · [Input ablations](ablations.csv) · [Paper analysis values](paper.json)

## Dataset versions

The published model scores and archived answers use **20261125-r3.2**. See [experiment versions](experiment-versions.json) for the frozen dataset revision, input checksums and r3.3 rerun roster.

**20261126-r3.3** changes two Content prompts with identical media and answer keys. To update the published Chinese comparison, obtain two new answers for each of the 11 models and recompute Content and Joint QA using the new answers. Timing QA, Streaming and input ablations use the same inputs and scoring rules in both revisions.

## Published comparison

The released comparison covers 11 models. Each has 400 Content QA answers, 400 Timing QA answers and 400 determined Streaming outcomes. The four models with input ablations have 400 determined outcomes for each of the 18 model and condition combinations. Gemini's Streaming coverage is incomplete.

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

Two reviewers answered independently. Observed counts use original answers compatible with the released benchmark. Full-set estimates average the observed rates and assume comparable difficulty for items without compatible answers.

| Track | Reviewer 1 observed | Reviewer 2 observed | Estimated on all 400 |
| --- | ---: | ---: | ---: |
| Content QA | 236/265 | 236/265 | 89.06% |
| Timing QA | 233/309 | 243/309 | 77.02% |
| Streaming Full Correct | 261/377 | 268/377 | 70.16% |
| Joint QA | 158/219 | 167/219 | |

The observed mean for Joint QA is 74.20% across 219 paired scenes per reviewer.
