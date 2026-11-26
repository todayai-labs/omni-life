"""Rebuild paper analyses from the released adjudications and scene metadata."""
from collections import Counter, defaultdict
from copy import deepcopy

from .score_streaming import score

JUDGES = ('primary', 'deepseek-v4-flash', 'kimi-k3', 'gpt-6-astra')
DEADLINES = (0, 1, 2, 5, 10)


def obligations(case):
    return case['required_events'] + ([case['terminal_count']] if 'terminal_count' in case else [])


def validate_basis(row, case, label):
    """Check capture-based verdicts, then confirm the evidence selection."""
    original = row['original']
    result = original.get('alignment')
    capture = original.get('scoring_capture')
    if capture is not None and score(case, capture) != result:
        raise ValueError(f'Original alignment differs for {label}')
    selected = original['full_correct']
    certificate = original.get('failure_certificate')
    if certificate is not None:
        if selected is not False or result['full_correct'] is not None:
            raise ValueError(f'Certificate contradicts the capture verdict for {label}')
        if 'minimum' in certificate and certificate['observed'] >= certificate['minimum']:
            raise ValueError(f'Invalid minimum-turn certificate for {label}')
        if 'maximum' in certificate:
            observed = next(certificate[key] for key in ('observed', 'observed_at_least', 'inside_lower_bound')
                            if key in certificate)
            if observed <= certificate['maximum']:
                raise ValueError(f'Invalid maximum-turn certificate for {label}')
    source = 'original'
    if selected is None:
        for evidence in row['evidence']:
            if type(evidence['full_correct']) is bool:
                selected = evidence['full_correct']
                source = evidence['route']
                break
    if type(selected) is not bool or selected is not row['full_correct'] or source != row['score_source']:
        raise ValueError(f'No consistent selected verdict for {label}')


def replay_deadlines(case, original, credited):
    alignment, capture = original.get('alignment'), original.get('scoring_capture')
    hits = [credited] * len(DEADLINES)
    if (not alignment or not capture or original['full_correct'] is None
            or not capture.get('capture_complete') or not capture.get('semantic_review_complete')
            or 'events' not in alignment or credited == len(obligations(case))
            or any(r['verdict'] == 'pending' for r in capture['replies'])
            or sum(event['status'] == 'hit' for event in alignment['events']) != credited):
        return hits
    for i, delay in enumerate(DEADLINES[1:], 1):
        extended = deepcopy(case)
        for event in obligations(extended):
            event['response_windows_s'] = [
                [lo, min(max(case['media_end_s'], hi), hi + delay)]
                for lo, hi in event['response_windows_s']
            ]
        hits[i] = sum(event['status'] == 'hit' for event in score(extended, capture).get('events', []))
    return hits


def analyze(protocol, metadata, main, ablations, judge_rows, human_rows):
    if metadata.keys() != protocol.keys():
        raise ValueError('Scene grouping does not cover the protocol')
    results = {'models': {}, 'input_ablations': {}, 'judge_panel': {}, 'human': {}, 'actions': {}}
    actions = defaultdict(Counter)
    for model, records in main.items():
        groups = defaultdict(lambda: {'correct': 0, 'total': 0})
        failure = Counter(dict.fromkeys(('pass', 'miss', 'extra', 'both', 'silent_fail', 'other_fail'), 0))
        event = Counter(dict.fromkeys(('hit', 'miss', 'unknown', 'extra_replies'), 0))
        deadlines = [0] * len(DEADLINES)
        silence_correct = 0
        source = Counter()
        for row in records:
            cid = row['case_id']
            case = protocol[cid]
            required = obligations(case)
            original = row['original']
            original_score = original['full_correct']
            alignment = original.get('alignment') or {}
            marks = {event['event_id']: event['status'] for event in alignment.get('events', [])}
            final = row['full_correct']
            source[row['score_source']] += 1
            group_keys = [('populations', metadata[cid]['population'])]
            if required:
                group_keys.append(('scenarios', metadata[cid]['scenario']))
            elif final:
                silence_correct += 1
            for key in group_keys:
                groups[key]['total'] += 1
                groups[key]['correct'] += int(final)

            known_miss = original_score is False and 'miss' in marks.values()
            known_extra = original_score is False and bool(alignment.get('extra_response_indices'))
            if final:
                failure['pass'] += 1
            elif not required:
                failure['silent_fail'] += 1
            elif known_miss and known_extra:
                failure['both'] += 1
            elif known_miss:
                failure['miss'] += 1
            elif known_extra:
                failure['extra'] += 1
            else:
                failure['other_fail'] += 1

            hits = 0
            for required_event in required:
                # Event analyses use the original comparable capture. A scene-level
                # supplemental decision need not resolve its individual obligations.
                status = ('hit' if original_score is True else
                          marks.get(required_event['event_id']) if original_score is False else None)
                status = status if status in {'hit', 'miss'} else 'unknown'
                actions[required_event['action']][status] += 1
                event[status] += 1
                hits += status == 'hit'
            if original_score is False and 'extra_response_indices' in alignment:
                event['extra_replies'] += len(alignment['extra_response_indices'])
            elif original_score is not True:
                event['extra_replies'] += original['logged_replies']

            sweep = replay_deadlines(case, original, hits)
            if len(sweep) != len(DEADLINES) or sweep[0] != hits or sweep != sorted(sweep) or sweep[-1] > len(required):
                raise ValueError(f'Invalid deadline replay for {model}/{cid}')
            deadlines = [a + b for a, b in zip(deadlines, sweep)]
        if sum(event[name] for name in ('hit', 'miss', 'unknown')) != 500:
            raise ValueError(f'Wrong obligation count for {model}')
        results['models'][model] = {
            'silence_correct': silence_correct, 'event': dict(event), 'deadline_hits': deadlines,
            'failures': dict(failure), 'score_sources': dict(source),
            'scenarios': {label: counts for (group, label), counts in groups.items() if group == 'scenarios'},
            'populations': {label: counts for (group, label), counts in groups.items() if group == 'populations'},
        }

    for (model, condition), records in ablations.items():
        base = {row['case_id']: row['full_correct'] for row in main[model]}
        passed = sum(row['full_correct'] for row in records)
        gained = sum(row['full_correct'] and not base[row['case_id']] for row in records)
        lost = sum(base[row['case_id']] and not row['full_correct'] for row in records)
        results['input_ablations'].setdefault(model, {})[condition] = {
            'correct': passed, 'total': len(records), 'gained': gained, 'lost': lost,
            'silence_correct': sum(row['full_correct'] for row in records if not obligations(protocol[row['case_id']])),
            'supplemental_scenes': sum(row['score_source'] != 'original' for row in records),
        }

    judge_by_model = defaultdict(dict)
    for row in judge_rows:
        model, cid = row['model'], row['case_id']
        if model not in main or cid in judge_by_model[model]:
            raise ValueError(f'Unknown or duplicate judge row for {model}/{cid}')
        judge_by_model[model][cid] = row
    if len(judge_by_model) != 4:
        raise ValueError('Expected four models in the judge panel')
    for model, records in judge_by_model.items():
        if records.keys() != protocol.keys():
            raise ValueError(f'Incomplete judge panel for {model}')
        values = {}
        for judge in JUDGES:
            observed = Counter(row[judge] for row in records.values())
            if set(observed) - {'True', 'False', ''}:
                raise ValueError(f'Invalid judge decision for {model}/{judge}')
            values[judge] = {'correct': observed['True'], 'incorrect': observed['False'],
                             'pending': observed[''], 'total': len(records)}
        paired = [row for row in records.values() if all(row[judge] for judge in JUDGES)]
        results['judge_panel'][model] = {
            'full_bank': values, 'paired_scenes': len(paired),
            'paired_correct': {judge: sum(row[judge] == 'True' for row in paired) for judge in JUDGES},
        }

    for row in human_rows:
        track = row['track']
        if track in results['human']:
            raise ValueError(f'Duplicate human baseline track: {track}')
        if 'reviewer' in row:
            counts = [[int(row['correct']), int(row['observed'])]]
        else:
            counts = [[int(row[f'reviewer_{i}_correct']), int(row[f'reviewer_{i}_observed'])] for i in (1, 2)]
        if any(not 0 <= correct <= observed <= 400 or observed == 0 for correct, observed in counts):
            raise ValueError(f'Invalid human baseline counts for {track}')
        results['human'][track] = {'counts': counts,
                                  'observed_mean_percent': round(sum(100 * h / n for h, n in counts) / len(counts), 2)}
    results['actions'] = {action: {key: counts[key] for key in ('hit', 'miss', 'unknown')}
                          for action, counts in actions.items()}
    return results
