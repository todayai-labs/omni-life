"""Verify the main and ablation results against the published inputs and stored verdicts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .qa import evaluate, load_questions


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def unique_records(path, key):
    with path.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    ids = {key(row) for row in records}
    if len(ids) != len(records):
        raise ValueError(f'Duplicate records in {path}')
    return records


def streaming_summary(records, scene_ids):
    if {row['case_id'] for row in records} != scene_ids or len(records) != len(scene_ids):
        raise ValueError('Streaming results do not cover the published scene roster')
    for row in records:
        verdict = row['full_correct']
        if verdict is not None and type(verdict) is not bool:
            raise ValueError(f'Invalid Full Correct decision for {row["case_id"]}')
        source = row['score_source']
        if (verdict is None) != (source == 'undetermined'):
            raise ValueError(f'Inconsistent Full Correct status for {row["case_id"]}')
        if source not in ('original', 'undetermined') and not any(
                e['route'] == source and e['full_correct'] is verdict for e in row['evidence']):
            raise ValueError(f'Missing score evidence for {row["case_id"]}')
    outcomes = Counter(row['full_correct'] for row in records)
    return dict(scenes=len(scene_ids), correct=outcomes[True], incorrect=outcomes[False], pending=outcomes[None],
                full_correct_percent=round(outcomes[True]*100/len(scene_ids), 2) if not outcomes[None] else None)


def recompute(dataset, results):
    questions, media = load_questions(dataset, 'zh')
    scene_ids = {row['case_id'] for row in table(dataset/'scenes.csv')}
    protocol = json.loads((dataset/'protocol/streaming-scoring.json').read_text(encoding='utf-8'))
    if len(scene_ids) != 400 or scene_ids != protocol['cases'].keys():
        raise ValueError('Published scene roster and protocol disagree')
    main = []
    ablations = []
    for directory in sorted(p.parent for p in results.glob('*/qa.jsonl')):
        answers = unique_records(directory/'qa.jsonl', lambda row: row['qa_id'])
        if {a['qa_id'] for a in answers} != questions.keys():
            raise ValueError(f'Incomplete QA roster for {directory.name}')
        for answer in answers:
            item = media[answer['qa_id']]
            if (answer['case_id'], answer['track']) != (item['case_id'], item['track']):
                raise ValueError(f'QA record points to a different input for {answer["qa_id"]}')
        graded = evaluate(questions, media, {a['qa_id']:a for a in answers})
        if any(graded['questions'][a['qa_id']] is not a['correct'] for a in answers):
            raise ValueError(f'Stored QA result disagrees for {directory.name}')
        runs = unique_records(directory/'streaming.jsonl', lambda row: row['case_id'])
        main.append(dict(model=directory.name, **{f'{t}_correct':group['correct'] for t,group in graded['groups'].items()},
                         **streaming_summary(runs, scene_ids)))
        path = directory/'ablations.jsonl'
        if path.is_file():
            arms = {}
            for row in unique_records(path, lambda row: (row['condition'], row['case_id'])):
                arms.setdefault(row['condition'], []).append(row)
            for arm, arm_runs in sorted(arms.items()):
                ablations.append(dict(model=directory.name, condition=arm, **streaming_summary(arm_runs, scene_ids)))
    if len(main) != 12 or len(ablations) != 22:
        raise ValueError('Expected 12 model results and 22 ablation conditions')
    return main, ablations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path, help='Local copy of the Hugging Face dataset')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--check', action='store_true', help='Check the committed summary tables')
    args = parser.parse_args()
    main_rows, ablation_rows = recompute(args.dataset, args.results)
    if args.check:
        for filename, computed in [('summary.csv',main_rows), ('ablations.csv',ablation_rows)]:
            formatted = [{k:'' if v is None else str(v) for k,v in row.items()} for row in computed]
            expected = {tuple(sorted(row.items())) for row in table(args.results/filename)}
            if expected != {tuple(sorted(row.items())) for row in formatted} or len(expected)!=len(computed):
                raise ValueError(f'The committed table disagrees with the records in {filename}')
        print('Verified 9,600 QA answers, 4,800 Streaming rows, and 8,800 ablation rows.')
    else:
        print(json.dumps({'main':main_rows, 'ablations':ablation_rows}, indent=2))


if __name__ == '__main__':
    main()
