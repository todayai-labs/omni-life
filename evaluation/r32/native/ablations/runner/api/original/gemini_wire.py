"""Record completed WebSocket sends and compare native setup, media and reminder packets to frozen inputs."""
import base64
import hashlib
import json
from pathlib import Path
from reminder_clock import valid

def read(path):
    return json.loads(path.read_bytes())

def lines(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def outgoing(message):
    if set(message)=={'setup'}:
        return {'kind':'setup','setup':message['setup']}
    assert set(message)=={'realtimeInput'}
    item=message['realtimeInput']
    assert len(item)==1
    kind,data=next(iter(item.items()))
    if kind in ('video','audio'):
        content=base64.b64decode(data['data'],validate=True)
        return {'kind':kind,'mime':data['mimeType'],'bytes':len(content),'sha256':digest(content)}
    if kind=='text':return {'kind':'reminder','text':data}
    assert kind=='audioStreamEnd' and data is True
    return {'kind':'audio_stream_end'}

class ObservedSocket:
    def __init__(self,socket,stream):
        self.socket=socket
        self.stream=stream

    def __getattr__(self,name):
        return getattr(self.socket,name)

    def __aiter__(self):
        return self.socket.__aiter__()

    async def send(self,message):
        await self.socket.send(message)
        self.stream.write(json.dumps(outgoing(json.loads(message)),ensure_ascii=False)+'\n')
        self.stream.flush()

class ObservedConnection:
    def __init__(self,connection,stream):
        self.connection=connection
        self.stream=stream

    async def __aenter__(self):
        return ObservedSocket(await self.connection.__aenter__(),self.stream)

    async def __aexit__(self,*exc):
        return await self.connection.__aexit__(*exc)

def verify(attempt,prepared,mf,expected_setup,planned,*,main=False):
    spec=read(attempt/'input-spec.json');native=lines(attempt/'input-events.jsonl');out=read(attempt/'transport-result.json')
    assert spec['setup']==native[0]['setup']==expected_setup and native[0]['kind']=='setup'
    assert spec['playback_speed']==1 and spec['audio_chunk_s']==0.2 and spec['tail_wait_s']==2 and spec['no_forced_turns'] and spec['no_synthetic_audio']
    assert 'setupComplete' in read(attempt/'setup-response.json')
    assert out['coverage']['complete'] and out['status']=='success' and not out['error']
    assert out['latency']['total_session_seconds']+0.01>=mf['duration_sec']+2
    pcm=(prepared/mf['audio_file']).read_bytes() if mf['audio_present'] else b''
    assert len(pcm)==mf['audio_samples']*2 and (not pcm or digest(pcm)==mf['audio_sha256'])
    schedule=[(f['pts_sec'],1,'video',f) for f in mf['frames']]
    for start in range(0,len(pcm),6400):
        end=min(len(pcm),start+6400);schedule.append((end/32000,0,'audio',(start,end)))
    for index,(t,body) in enumerate(planned,1):
        schedule.append((t,2,'reminder',(index,body)))
    schedule.sort(key=lambda item:(item[0],item[1]))
    if pcm:schedule.append((mf['duration_sec'],3,'audio_stream_end',None))
    assert len(native)==len(schedule)+1
    wire=None if main else lines(attempt/'actual-outgoing.jsonl')
    if wire is not None:assert len(wire)==len(native) and wire[0]=={'kind':'setup','setup':expected_setup}
    last=0
    for idx,(row,(when,_,kind,item)) in enumerate(zip(native[1:],schedule),1):
        assert row['kind']==kind and row['elapsed_s']>=last and row['elapsed_s']+0.002>=when
        last=row['elapsed_s']
        if kind=='video':
            payload=(prepared/item['file']).read_bytes();h=digest(payload)
            assert h==item['sha256'] and row['pts_s']==when and Path(row['path']).name==item['file']
            if main:assert row['sha256']==h
            expect={'kind':kind,'mime':'image/jpeg','bytes':len(payload),'sha256':h}
        elif kind=='audio':
            start,end=item;payload=pcm[start:end];h=digest(payload)
            assert abs(row['start_s']-start/32000)<1e-6 and abs(row['end_s']-end/32000)<1e-6 and row['samples']==len(payload)//2
            if main:assert row['sha256']==h
            expect={'kind':kind,'mime':'audio/pcm;rate=16000','bytes':len(payload),'sha256':h}
        elif kind=='reminder':
            index,body=item
            assert row['scheduled_media_s']==when and row['text']==body and row['reminder_index']==index
            assert row['payload']=={'realtimeInput':{'text':body}} and row['delivery_status']=='sent_without_individual_provider_ack'
            assert valid(row,when,mf['duration_sec'],native)
            expect={'kind':'reminder','text':body}
        else:
            expect={'kind':'audio_stream_end'}
        if wire is not None:assert wire[idx]==expect
    assert out['coverage']['video_frames_sent']==len(mf['frames']) and out['coverage']['audio_samples_sent']==mf['audio_samples']
    assert out['responses']==read(attempt/'parsed-output.json')
    inbound=lines(attempt/'raw-output.jsonl')
    assert all('error' not in e['event'] for e in inbound)
    if not main:
        notices=[]
        for i,record in enumerate(inbound):
            if 'goAway' in record['event']:
                body=record['event']['goAway']
                notices.append({'kind':'goAway','raw_event_index':i,'receive_media_s':record['receive_media_s'],
                                'time_left':body.get('timeLeft') if isinstance(body,dict) else None})
        assert out['connection_advisories']==notices
    return {'video_frames':len(mf['frames']),'audio_samples':mf['audio_samples'],'reminders_sent':len(planned),
            'media_packets':len(schedule),'source_seconds':mf['duration_sec'],'tail_seconds':2,
            'actual_wire_source':'original post-send digests' if main else 'independent post-send wire observer',
            'input_events_sha256':digest((attempt/'input-events.jsonl').read_bytes()),
            'wire_sha256':None if main else digest((attempt/'actual-outgoing.jsonl').read_bytes())}
