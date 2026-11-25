"""Offline native Streaming audit, conditional budget, and recoverable plan for four APIs."""
import argparse
import os
import collections
import functools
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


OLDQ = os.environ.get("OMNI_LIFE_QWEN_ARCHIVE_ENDPOINT", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
NEWQ = os.environ.get("OMNI_LIFE_QWEN_CURRENT_ENDPOINT", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
OLDG = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
G = "gemini-3.8-live"
DESK = Path("experiments")
PREP = Path("prepared")
WORK = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_bytes())


def lines(p):
    with p.open() as f:
        return [json.loads(x) for x in f]


def sha(data):
    return hashlib.sha256(data).hexdigest()


@functools.lru_cache(maxsize=65536)
def fsha(path):
    return sha(path.read_bytes())


def require(val, reason):
    if not val:
        raise ValueError(reason)


def equal_media(m):
    return (m['source_video_sha256'],m['duration_sec'],m['audio_present'],m['audio_samples'],m.get('audio_sha256'),
            tuple((f['pts_sec'],f['sha256']) for f in m['frames']))


def qsetup(model,text):
    return {'model':model,'instructions':text,'modalities':['text'],'audio':{'input':{'format':{'type':'pcm','sample_rate':16000}}},
            'turn_detection':{'type':'semantic_vad'},'temperature':0,'seed':0,'enable_search':False,'tools':[]}


def gsetup(text):
    return {'model':'models/'+G,'systemInstruction':{'parts':[{'text':text}]},
            'generationConfig':{'responseModalities':['AUDIO'],'temperature':0},'outputAudioTranscription':{},
            'contextWindowCompression':{'slidingWindow':{}},'proactivity':{'proactiveAudio':True}}


def qbase(a,m,text,s,endpoint=OLDQ):
    expected=qsetup(m,text)
    session=read(a/'session_config.json')
    require(session['endpoint']==endpoint+'?model='+m,'Qwen route/model query mismatch')
    q=lines(a/'input-events.jsonl')
    require(q and q[0].get('event')=={'type':'session.update','session':expected}==session['requested'],'session.update mismatch')
    raw=[r['event'] for r in lines(a/'raw-output.jsonl')]
    echo=next(x['session'] for x in raw if x.get('type')=='session.updated')
    require(echo.get('model')==m and echo.get('instructions')==text and echo.get('modalities')==['text']
            and echo.get('temperature')==0 and echo['turn_detection']['type']=='semantic_vad'
            and all(echo['audio']['input']['format'].get(k)==v for k,v in expected['audio']['input']['format'].items()),'provider echo mismatch')
    optional={k:echo.get(k,'not_echoed') for k in ('tools','seed','enable_search')}
    require(all(v==expected[k] or v=='not_echoed' for k,v in optional.items()),'optional model setting mismatch')
    require(s['audio_chunk_seconds']==0.2 and s['playback_rate']==1 and s['observation_seconds']==2 and s['max_input_drift_seconds']==2
            and not s.get('empty_audio_control_probe') and not any(s[k] for k in ('task_tts','padding','manual_commit','manual_response_create'))
            and s['generation_trigger']=='provider-native semantic VAD only','timing/forced trigger mismatch')
    return q,raw,optional


def qaudit(a,m,f,folder,text,endpoint=OLDQ):
    s=read(a/'input-spec.json')
    q,raw,optional=qbase(a,m,text,s,endpoint)
    man=Path(s['source_manifest'])
    require(equal_media(read(man))==equal_media(f),'prepared manifest mismatch')
    schedule=s['schedule']
    require(len(q)==len(schedule)+1,'native send count mismatch')
    runtime_attribution='per_attempt_runtime_snapshot'
    if (a/'runtime-source.json').is_file():
        source=read(a/'runtime-source.json')['source']
    else:
        link=read(a/'reused-probe.json'); original=Path(link['source_attempt'])
        require(link['no_new_model_call'] and original.parent.name==a.parent.name,'probe archive attribution')
        for name in ('input-spec.json','input-events.jsonl','raw-output.jsonl','status.json','transport-result.json'):
            require(fsha(a/name)==fsha(original/name),'probe archive byte mismatch')
        source=(original.parents[5]/'code/snapshots/initial-probe-runner.py').read_text()
        runtime_attribution='initial_probe_code_archive_and_original_postsend_records'

    require('await ws.send(json.dumps(outgoing))' in source and 'base64.b64encode(chunk)' in source
            and 'audio[2 * item["sample_start"]:2 * item["sample_end"]]' in source
            and 'base64.b64encode((source / item["file"]).read_bytes())' in source
            and source.index('await ws.send(json.dumps(outgoing))') < source.index('"send_completed_media_s": after_send'),'missing post-send provenance')
    prev=0; frames=[]; latest=-1000
    for row,expected in zip(q[1:],schedule):
        require(all(row.get(k)==v for k,v in expected.items()) and row['send_completed_media_s']>=row['media_s']
                and row['media_s']>=latest and row['send_completed_media_s']-row['send_due_s']<=2
                and Path(row['path'])==man.parent/row['file'],'packet timing/path/schedule mismatch')
        latest=row['media_s']
        if row['kind']=='audio':
            require(row['event_type']=='input_audio_buffer.append' and row['sample_start']==prev and 0<row['sample_end']-prev<=3200
                    and row['sample_end']==min(prev+3200,f['audio_samples']) and row['file']==f['audio_file'],'PCM interval mismatch')
            prev=row['sample_end']
        else:
            i=len(frames); frm=f['frames'][i]
            require(row['kind']=='video' and row['event_type']=='input_image_buffer.append' and row['frame_index']==i and
                    row['source_pts_s']==frm['pts_sec'] and fsha(Path(row['path']))==frm['sha256']==fsha(folder/frm['file']),'JPEG hash/PTS mismatch')
            frames.append(frm['pts_sec'])
    require(prev==f['audio_samples'] and len(frames)==len(f['frames']),'truncated input')
    if f['audio_present']:
        require(fsha(man.parent/f['audio_file'])==f['audio_sha256']==fsha(folder/f['audio_file']),'PCM archive hash mismatch')
    t=read(a/'transport-result.json'); st=read(a/'status.json'); lat=read(a/'latency.json')
    require(st['status']==t['status']=='completed' and st['observation_completed_media_s']>=float(f['duration_sec'])+2 and
            t['sent_frames']==len(frames) and t['sent_audio_samples']==prev and t['session_acknowledged'] and not t['provider_errors']
            and not any(x.get('type') in ('error','receive_error') for x in raw),'native terminal/horizon mismatch')
    seen={r['response_id']:r['usage'] for r in lat['usage']}
    return {'runtime_sha256':sha(source.encode()),'runtime_attribution':runtime_attribution,'echo':optional,'usage_events':len(seen),'usage_units':{
        k:sum(u.get(k,0) for u in seen.values()) for k in ('input_tokens','output_tokens','total_tokens')},
        'monetary_charge':lat['api_cost'],'raw_events':len(raw),'media_packets':len(schedule)}


def gaudit(a,m,f,folder,text,endpoint=OLDG,resource=None):
    s=read(a/'input-spec.json'); q=lines(a/'input-events.jsonl'); t=read(a/'transport-result.json')
    route=read(a/'route.json') if (a/'route.json').exists() else read(a.parents[2]/'config.json')
    require(route['endpoint']==endpoint,'Gemini route mismatch')
    expected=gsetup(text)
    if resource: expected['model']=resource
    require(q[0]==dict(q[0],kind='setup',setup=expected) and q[0]['setup']==s['setup'] and equal_media(s['manifest'])==equal_media(f),'Gemini actual setup or source mismatch')
    require(s['no_forced_turns'] and s['no_synthetic_audio'] and s['tail_wait_s']==2 and s['audio_chunk_s']==0.2 and s['playback_speed']==1,'Gemini timing mismatch')
    require('setupComplete' in read(a/'setup-response.json'),'missing native setupComplete')
    audio=(folder/f['audio_file']).read_bytes() if f['audio_present'] else b''
    if audio:require(sha(audio)==f['audio_sha256'],'Gemini frozen PCM mismatch')
    nf=offset=ended=0; last=-1
    for r in q[1:]:
        require(r['elapsed_s']>=last,'Gemini input clock order'); last=r['elapsed_s']
        if r['kind']=='video':
            fr=f['frames'][nf]; nf+=1
            require(r['pts_s']==fr['pts_sec'] and Path(r['path']).name==fr['file'] and r['sha256']==fr['sha256']==fsha(folder/fr['file']),'Gemini outgoing JPEG mismatch')
        elif r['kind']=='audio':
            count=r['samples']
            require(0<count<=3200 and count==min(3200,f['audio_samples']-offset) and
                    abs(r['start_s']-offset/16000)<1e-6 and abs(r['end_s']-(offset+count)/16000)<1e-6
                    and r['sha256']==sha(audio[2*offset:2*(offset+count)]),'Gemini outgoing PCM mismatch')
            offset+=count
        elif r['kind']=='audio_stream_end':
            ended+=1; require(offset==f['audio_samples'],'premature audio end')
        else: raise ValueError('unexpected Gemini user/control event')
    c=t['coverage']
    require(offset==f['audio_samples']==c['audio_samples_sent'] and nf==len(f['frames'])==c['video_frames_sent']
            and c['complete'] and ended==int(f['audio_present']) and t['status']=='success' and not t['error'],'Gemini incomplete send/transport')
    require(t['latency']['total_session_seconds']+0.01 >=float(f['duration_sec'])+2,'Gemini causal horizon truncated')
    raw=lines(a/'raw-output.jsonl')
    require(raw and not any('error' in x.get('event',{}) for x in raw),'Gemini raw error')
    u=t['usage'][-1] if t['usage'] else {}
    return {'usage_events':len(t['usage']),'usage_units':{k:u.get(k,0) for k in ('promptTokenCount','responseTokenCount','totalTokenCount')},
            'monetary_charge':t['billing']['api_cost'],'raw_events':len(raw),'media_packets':nf+(offset+3199)//3200}


def quantities(ids, frozen):
    items=[frozen[x] for x in ids]
    return {'sessions':len(items),'video_seconds':round(sum(float(f['duration_sec']) for f in items),6),
            'frames':sum(len(f['frames']) for f in items),'pcm_seconds':round(sum(f['audio_samples'] for f in items)/16000,6),
            'audio_present':sum(bool(f['audio_present']) for f in items),'silent_source':sum(not f['audio_present'] for f in items),
            'two_second_tails_seconds':2*len(items)}


def annotate_failures(model,home,rows,frozen,prompts,signatures):
    exact=collections.defaultdict(list)
    for p in home.glob('cases/*/attempt-*/transport-result.json'):
        t=read(p)
        if t['status'] in ('completed','success') or not (p.parent/'input-spec.json').is_file(): continue
        s=read(p.parent/'input-spec.json')
        f=s['manifest'] if model==G else read(Path(s['source_manifest']))
        text=s['setup']['systemInstruction']['parts'][0]['text'] if model==G else s['instructions']
        matches=signatures.get((text,equal_media(f)),[])
        for cid in matches:
            if rows[cid]['decision']=='reuse_success':continue
            errors=t.get('provider_errors',[])
            category='previous_failed_attempt'
            if model!=G and not f['audio_present'] and any('append image before append audio' in str(z).lower() for z in errors):
                try:
                    _,raw,_=qbase(p.parent,model,prompts[cid],s)
                    if any(x.get('type')=='error' and 'append image before append audio' in str(x).lower() for x in raw):
                        category='native_rejected_exact_silent_source'
                except (OSError,ValueError,StopIteration,KeyError):pass
            exact[cid].append({'attempt':str(p.parent),'category':category,
                               'sent_frames':t.get('sent_frames'),'sent_audio_samples':t.get('sent_audio_samples')})
    for cid,failures in exact.items():
        rows[cid]['exact_input_failed_attempts']=failures
        if any(x['category']=='native_rejected_exact_silent_source' for x in failures):
            rows[cid]['decision']='retain_terminal_native_failure'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',type=Path,default=WORK/'launch/api-stream-candidates.json')
    parser.add_argument('--prepared',type=Path,default=PREP)
    parser.add_argument('--prompts',type=Path,default=WORK/'aligned/protocols/streaming-prompts.jsonl')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    data=read(args.candidates)
    frozen={p.parent.name:read(p) for p in args.prepared.glob('*/manifest.json')}
    prompts={r['id']:r['text'] for r in lines(args.prompts)}
    require(len(frozen)==400 and set(frozen)==set(prompts),'incomplete frozen release')
    signatures=collections.defaultdict(list)
    for cid,f in frozen.items():
        text=(args.prepared/cid/f.get('system_file','SYSTEM.txt')).read_bytes()
        require(text==prompts[cid].encode(),'prepared system mismatch')
        signatures[(prompts[cid],equal_media(f))].append(cid)
    summary={}; allrows={}
    for model,orig in data.items():
        examples=orig['candidate_attempts']; home=Path(next(iter(examples.values()))['attempt']).parents[2]
        rows={cid:{'decision':'pending','system_sha256':sha(prompts[cid].encode()),
                   'manifest_sha256':fsha(args.prepared/cid/'manifest.json'),
                   'media_identity_sha256':sha(json.dumps(equal_media(frozen[cid]),separators=(',',':')).encode()),
                   'audio_present':bool(frozen[cid]['audio_present'])} for cid in frozen}
        homes=sorted(home.glob('cases/*/attempt-*/input-spec.json'))
        # Include alternate attempts and renamed IDs, which a same-ID first-hit screen can miss.
        for p in homes:
            a=p.parent; terminal=a/('transport-result.json' if model==G else 'status.json')
            if not terminal.is_file() or read(terminal)['status'] not in ('completed','success'):continue
            s=read(p); man=s['manifest'] if model==G else read(Path(s['source_manifest']))
            text=s['setup']['systemInstruction']['parts'][0]['text'] if model==G else s['instructions']
            for cid in signatures.get((text,equal_media(man)),[]):
                if rows[cid]['decision']=='reuse_success': continue
                try: evidence=(gaudit if model==G else qaudit)(a,model,frozen[cid],args.prepared/cid,prompts[cid])
                except (AssertionError,ValueError,StopIteration,KeyError,IndexError,FileNotFoundError) as e:
                    rows[cid].setdefault('rejected_candidate',[]).append({'attempt':str(a),'reason':str(e) or type(e).__name__});continue
                rows[cid].update(decision='reuse_success',attempt=str(a),native_evidence=evidence)
        annotate_failures(model,home,rows,frozen,prompts,signatures)
        selected=[r['attempt'] for r in rows.values() if r['decision']=='reuse_success']
        require(len(selected)==len(set(selected)),'one historical session was selected for two target cases')
        by=collections.defaultdict(list)
        for c,r in rows.items():by[r['decision']].append(c)
        accepted=[rows[c]['native_evidence'] for c in by['reuse_success']]
        usage=collections.Counter()
        for r in accepted:usage.update(r['usage_units'])
        summary[model]={'prescreen_candidates':orig['candidate_count'],'accepted':quantities(by['reuse_success'],frozen),
                        'native_failed_exact':quantities(by['retain_terminal_native_failure'],frozen),'pending':quantities(by['pending'],frozen),
                        'prior_route':OLDG if model==G else OLDQ,'other_route':'Vertex AI us-central1 v1 (regional API key in archived QA runner)' if model==G else NEWQ,
                        'observed_accounting':{'accepted_sessions_with_usage':sum(r['usage_events']>0 for r in accepted),
                        'usage_event_count':sum(r['usage_events'] for r in accepted),'usage_units':dict(usage),
                        'case_cost_receipts':sum(r['monetary_charge'] is not None for r in accepted)},
                        'other_completed_attempt_matching_renamed_id':sum(Path(rows[c]['attempt']).parent.name!=c for c in by['reuse_success']),
                        'failed_first_screen_candidates':{c:rows[c].get('rejected_candidate') for c in examples if rows[c]['decision']!='reuse_success'}}
        allrows[model]=rows
        print(model,[(k,summary[model][k]['sessions']) for k in ('accepted','native_failed_exact','pending')],flush=True)
    for name,value in [('cases.json',allrows),('budget.json',summary)]:
        (args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    main()
