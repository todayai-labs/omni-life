"""Verify the main and ablation results against the published inputs and stored verdicts."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path

from .qa import evaluate, indexed, load_questions
from .result_analysis import analyze, validate_basis


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


def streaming_summary(records, cases, name):
    if {row['case_id'] for row in records} != cases.keys() or len(records) != len(cases):
        raise ValueError(f'Incomplete Streaming roster for {name}')
    for row in records:
        cid = row['case_id']
        validate_basis(row, cases[cid], f'{name}/{cid}')
    outcomes = Counter(row['full_correct'] for row in records)
    original = sum(row['score_source'] == 'original' for row in records)
    return dict(scenes=len(cases), correct=outcomes[True], incorrect=outcomes[False], pending=outcomes[None],
                full_correct_percent=round(outcomes[True] * 100 / len(cases), 2),
                original_protocol=original, supplemental=len(cases)-original)


def recompute(dataset, results):
    questions, media = load_questions(dataset, 'zh')
    scene_ids = {row['case_id'] for row in table(dataset/'scenes.csv')}
    protocol_file = (dataset/'protocol/streaming-scoring.json').read_bytes()
    cases = json.loads(protocol_file)['cases']
    if len(scene_ids) != 400 or scene_ids != cases.keys():
        raise ValueError('Published scene roster and protocol disagree')
    main, ablations = [], []
    main_records, ablation_records = {}, {}
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
        main_records[directory.name] = runs
        main.append(dict(model=directory.name, **{f'{t}_correct':group['correct'] for t,group in graded['groups'].items()},
                         **streaming_summary(runs, cases, directory.name)))
        path = directory/'ablations.jsonl'
        if path.is_file():
            arms = {}
            for row in unique_records(path, lambda row: (row['condition'], row['case_id'])):
                arms.setdefault(row['condition'], []).append(row)
            for arm, arm_runs in sorted(arms.items()):
                ablation_records[directory.name, arm] = arm_runs
                ablations.append(dict(model=directory.name, condition=arm,
                                      **streaming_summary(arm_runs, cases, f'{directory.name}/{arm}')))
    if len(main) != 11 or len(ablations) != 18:
        raise ValueError('Expected 11 model results and 18 ablation conditions')
    generated = analyze(cases, indexed(results/'scene-groups.csv', 'case_id'), main_records, ablation_records,
                        table(results/'judge-panel.csv'), table(results/'human-baseline.csv'))
    generated['streaming_protocol_sha256'] = sha256(protocol_file).hexdigest()
    return main, ablations, generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path, help='Local copy of the Hugging Face dataset')
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--check', action='store_true', help='Check the committed summary tables')
    args = parser.parse_args()
    versions = args.results/'experiment-versions.json'
    if versions.is_file():
        published = json.loads(versions.read_text(encoding='utf-8'))['published_results']
        for name, digest in published['dataset_sha256'].items():
            if sha256((args.dataset/name).read_bytes()).hexdigest() != digest:
                parser.error(f'{name} differs from the archived inputs. Use {published["version"]} '
                             f'(Hugging Face revision {published["revision"]}) to reproduce these results.')
    main_rows, ablation_rows, paper = recompute(args.dataset, args.results)
    if args.check:
        for filename, computed in [('summary.csv',main_rows), ('ablations.csv',ablation_rows)]:
            formatted = [{k:'' if v is None else str(v) for k,v in row.items()} for row in computed]
            expected = {tuple(sorted(row.items())) for row in table(args.results/filename)}
            if expected != {tuple(sorted(row.items())) for row in formatted} or len(expected)!=len(computed):
                raise ValueError(f'The committed table disagrees with the records in {filename}')
        if paper != json.loads((args.results/'paper.json').read_text(encoding='utf-8')):
            raise ValueError('The paper analyses disagree with the published evidence')
        print('Verified 8,800 QA answers, 4,400 Streaming scenes, 7,200 input-ablation scenes and 1,600 judge-panel rows.')
    else:
        print(json.dumps({'main':main_rows, 'ablations':ablation_rows,'paper':paper}, indent=2))


if __name__ == '__main__':
    main()
