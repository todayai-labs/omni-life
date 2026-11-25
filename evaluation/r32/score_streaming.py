"""Replay time matching on independently judged, time-stamped outputs. Python 3.9+."""
import argparse
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from functools import lru_cache
from pathlib import Path

def valid_time(event, reply, onset):
    if reply['phase']=='post_video':
        return event.get('allow_post_video',False) and reply.get('turn')==1
    t=reply['video_s']
    return t>=onset and any(lo <= t < hi or (hi == reply['_label_end'] and lo <= t <= reply['_media_end'])
                            for lo,hi in event['response_windows_s'])

def options(event, replies, first):
    """Yield completed ways to consume this event's ordered grounded facts."""
    facts=event['facts']
    ids=[f['id'] for f in facts]
    states={0:()}
    for j in range(first,len(replies)):
        r=replies[j]
        claims=r.get('facts',[])
        if r.get('verdict')!='grounded' or not claims:continue
        for k, prior in list(states.items()):
            n=k+len(claims)
            if n>len(ids) or claims!=ids[k:n]:continue
            if not valid_time(event,r,facts[n-1]['observable_onset_s']):continue
            path=prior+(j,)
            if n==len(ids): yield path
            elif n not in states: states[n]=path

def align(events,replies):
    @lru_cache(None)
    def solve(i,first):
        if i==len(events):return (0,0,())
        hits,used,rest=solve(i+1,first)
        best=(hits,used,((),)+rest)
        for path in options(events[i],replies,first):
            h,u,r=solve(i+1,path[-1]+1)
            candidate=(1+h,len(path)+u,(path,)+r)
            if candidate[:2]>best[:2]:best=candidate
        return best
    return solve(0,0)[2]

def score(case,record):
    if record is None or record.get('capture_complete') is not True:
        return {'status':'missing_capture','full_correct':None,'count_accuracy':None}
    replies = [dict(r) for r in record.get('replies', [])]
    media_end = case.get('media_end_s', case['annotation_end_s'])
    precision = record.get('timestamp_resolution_s')
    if precision is not None and precision != 0.1:
        raise ValueError('Only documented nearest-tenth legacy timestamps are supported.')
    rounded_eof = float(Decimal(str(media_end)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    for r in replies:
        r['_at_eof'] = False
        r['_label_end'] = case['annotation_end_s']
        r['_media_end'] = media_end
        if r.get('phase') != 'video':
            continue
        t = r['video_s']
        if not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) or t < 0:
            raise ValueError('Invalid video time.')
        if precision and abs(t * 10 - round(t * 10)) > 1e-7:
            raise ValueError('Timestamp disagrees with declared precision.')
        # A rounded tick below the true end may still denote an earlier observation.
        # Never advance that timestamp, or grant access to a not-yet-observable fact.
        if t > media_end:
            if precision and t == rounded_eof:
                r['video_s'] = media_end
            else:
                raise ValueError('Video response occurs after end of source media.')
        r['_at_eof'] = r['video_s'] == media_end
    eof=record.get('eof_opportunity') is True
    post_turns=0
    last=(-1,-1.); allowed={'grounded','optional','extraneous','pending'}
    for r in replies:
        if r.get('verdict') not in allowed:raise ValueError('Missing or invalid semantic verdict.')
        if r.get('phase')=='video':
            t=r['video_s']
            if not math.isfinite(t) or t<0 or t>case['annotation_end_s']+.25:raise ValueError('Invalid video time.')
            key=(0,t)
        elif r.get('phase')=='post_video':
            post_turns+=1
            if not eof or r.get('turn')!=1 or post_turns>1:raise ValueError('Only one captured post-video response is allowed.')
            key=(1,0.)
        else:raise ValueError('Missing capture phase.')
        if key<last:raise ValueError('Replies are not in recorded order.')
        last=key
    if record.get('semantic_review_complete') is not True or any(r['verdict']=='pending' for r in replies):
        return {'status':'pending_semantics','full_correct':None,'count_accuracy':None}
    ordinary=case['required_events']
    terminal=case.get('terminal_count')
    # A final statement can also give the final increment. Require both semantic
    # facts in this utterance, in order, and permit no other cross-event reuse.
    turns=[]
    for j,r in enumerate(replies):
        parts=r.get('facts',[])
        count_fusion = terminal and ordinary and parts == [
            ordinary[-1]['facts'][0]['id'], terminal['facts'][0]['id']]
        explicit_fusion = r['phase'] == 'post_video' and parts in case.get('allowed_eof_fusions', [])
        if count_fusion or explicit_fusion:
            turns.extend([dict(r, facts=[fact], source=j) for fact in parts])
        else:turns.append(dict(r,source=j))
    events=ordinary+([terminal] if terminal else [])
    paths=align(events,turns)
    consumed=set();used_parts=set();results=[]; total_result=None
    for event,matches in zip(events,paths):
        match_sources=sorted(set(turns[k]['source'] for k in matches))
        consumed.update(match_sources);used_parts.update(matches)
        status='hit' if matches else ('unobservable_eof' if event.get('allow_post_video') and not eof else 'miss')
        lead=replies[match_sources[0]] if matches else {}
        latency=round(lead['video_s']-event['observable_onset_s'],6) if lead.get('phase')=='video' else None
        results.append(dict(event_id=event['event_id'],status=status,replies=match_sources,latency_s=latency))
        if event is terminal:total_result=status
    # Score the historical terminal-only dimension independently of the full track.
    if terminal:
        final_id=terminal['facts'][0]['id']
        declared=[j for j,r in enumerate(replies) if r.get('addresses_final_count') or final_id in r.get('facts',[])]
        if declared:
            j=declared[-1]
            count_result=any(turns[path[-1]]['source']==j for path in options(terminal,turns,0))
        else:count_result=False if eof else None
    else:count_result=None
    def grounded_optional(j,r):
        if r['verdict']!='optional' or r.get('phase')!='video':return False
        idx=r.get('optional_index')
        windows=case.get('optional_progress',[])
        return (isinstance(idx,int) and 0<=idx<len(windows) and
                (windows[idx]['start']<=r['video_s']<windows[idx]['end'] or
                 (windows[idx]['end']==r['_label_end'] and windows[idx]['start']<=r['video_s']<=r['_media_end'])))
    extras=set(t['source'] for k,t in enumerate(turns) if k not in used_parts)
    seen_optional=set()
    for j in sorted(extras.copy()):
        r=replies[j]
        if grounded_optional(j,r) and r['optional_index'] not in seen_optional:
            seen_optional.add(r['optional_index']); extras.remove(j)
    extras=sorted(extras)
    partial=any(r['status']=='unobservable_eof' for r in results)
    # Even when a missing EOF leaves some units unknown, an established miss
    # or extra response independently establishes a failed whole scene.
    advice = case.get('scene_requirement_zh')
    advice_met = None
    if advice and not any(r['status'] == 'miss' for r in results):
        if any('provides_help' not in replies[j] for j in consumed):
            if not any(replies[j].get('provides_help') is True for j in consumed):
                return {'status':'pending_semantics','full_correct':None,'count_accuracy':count_result}
        advice_met = any(replies[j].get('provides_help') is True for j in consumed)
    failed=bool(extras) or any(r['status']=='miss' for r in results) or advice_met is False
    full=False if failed else None if partial else True
    return dict(status='unobservable_eof' if partial else 'scored',full_correct=full,
                count_accuracy=count_result,events=results,extra_response_indices=extras)

def load_records(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['records'] if isinstance(data.get('records'), dict) else data


def evaluate(protocol,records):
    cases=protocol['cases']
    if set(records)-set(cases):raise ValueError('Received an unknown case_id.')
    results={};groups={}
    for c,case in cases.items():
        s=score(case,records.get(c));results[c]=s
        group='count' if 'terminal_count' in case else 'silence' if not case['required_events'] else 'ordinary'
        for key in [group,'all']:
            v=groups.setdefault(key,Counter());v['population']+=1
            if s['full_correct'] is not None:v['full_correct_known']+=1;v['full_correct_hits']+=int(s['full_correct'])
            if s['count_accuracy'] is not None:v['count_accuracy_known']+=1;v['count_accuracy_hits']+=int(s['count_accuracy'])
            v[s['status']]+=1
    for key,v in groups.items():
        v['full_correct_rate_known_only']=v['full_correct_hits']/v['full_correct_known'] if v['full_correct_known'] else None
        v['full_correct_rate_entire_population']=v['full_correct_rate_known_only'] if v['full_correct_known']==v['population'] else None
        if key=='count':v['final_count_accuracy_known_only']=v['count_accuracy_hits']/v['count_accuracy_known'] if v['count_accuracy_known'] else None
    return {'groups':groups,'cases':results}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol',type=Path);p.add_argument('records',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args()
    result=evaluate(json.loads(args.protocol.read_text(encoding='utf-8')),load_records(args.records))
    formatted=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(formatted+'\n')
    else:print(formatted)
