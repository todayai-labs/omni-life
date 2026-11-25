"""Gemini native input-design ablations, with shared quota gate and causal AV.

B0 references the completed baseline. Other arms change only the initial query
or add realtimeInput.text. No model deployment, media encoding or task audio.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import getpass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import types

BASE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gemini-3.8-live'
ARMS = ('B0', 'Q-structured', 'R1', 'R5', 'N1')
REPEAT = '提示：你的任务是 {query}'
NEUTRAL = '继续执行最初任务；仅在原任务条件满足时回应，否则不输出。'
STOP = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def safe(value, key=''):
    if isinstance(value, str):
        return value.replace(key, '[REDACTED]') if key else value
    if isinstance(value, dict):
        return {name: safe(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [safe(item, key) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError('Frozen input/settings changed: ' + str(path))
    else:
        save(path, value)


def snapshot(source, target):
    value = Path(source).read_bytes()
    target = Path(target)
    if target.exists():
        if target.read_bytes() != value:
            raise ValueError('Frozen source changed: ' + str(target))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)


def load_native(path):
    module = types.ModuleType('gemini_native_baseline')
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def clean_manifest(original, query):
    fields = ('id', 'duration_sec', 'sample_rate_hz', 'channels', 'audio_present',
              'audio_samples', 'audio_file', 'source_video')
    result = {field: original[field] for field in fields if field in original}
    result['query'] = query
    result['frames'] = [{'file': f['file'], 'pts_sec': f['pts_sec']} for f in original['frames']]
    return result


def make_schedule(manifest, pcm, arm, query):
    # Original audio release is packet-end time, with an unpadded 200ms tail.
    schedule = [(frame['pts_sec'], 'video', frame) for frame in manifest['frames']]
    for offset in range(0, len(pcm), 6400):
        chunk = pcm[offset:offset + 6400]
        schedule.append(((offset + len(chunk)) / 32000, 'audio', (offset, chunk)))
    interval = {'R1': 1, 'N1': 1, 'R5': 5}.get(arm)
    if interval:
        index = 1
        while interval * index < manifest['duration_sec']:
            text = NEUTRAL if arm == 'N1' else REPEAT.format(query=query)
            schedule.append((interval * index, 'reminder', {'text': text, 'index': index}))
            index += 1
    priority = {'audio': 0, 'video': 1, 'reminder': 2}
    return sorted(schedule, key=lambda item: (item[0], priority[item[1]]))


def load_inputs(args):
    labels = {}
    for path in sorted(args.data.rglob('*.labels.json')):
        ident = path.stem.removesuffix('.labels')
        if ident in labels or Path(ident).name != ident:
            raise ValueError('Duplicate/unsafe source ID')
        labels[ident] = {'gt': read(path), 'path': str(path)}
    if len(labels) != 400:
        raise ValueError('Expected exactly 400 original cases')
    mapping = read(args.structured)['cases']
    structured = {row['id']: row for row in mapping}
    if len(mapping) != 400 or set(structured) != set(labels):
        raise ValueError('Structured query map does not match the 400 source cases')
    template = args.sp.read_text()
    if template.count('{{USER_REQUEST}}') != 1:
        raise ValueError('Expected one task placeholder')
    sources = {}
    for ident, source in labels.items():
        gt, approved = source['gt'], structured[ident]
        query = gt['query']
        manifest_path = args.prepared / ident / 'manifest.json'
        original = read(manifest_path)
        if (original['id'] != ident or original['query'] != query or len(query) != 1
                or query[0].get('start_time_sec') != 0 or approved['original_query'] != query[0]['text']
                or approved.get('eligible_for_inference') is not True):
            raise ValueError('Original/approved query differs: ' + ident)
        if ((args.prepared / ident / original['system_file']).read_text().strip()
                != template.replace('{{USER_REQUEST}}', query[0]['text']).strip()):
            raise ValueError('Baseline prepared SYSTEM differs: ' + ident)
        if (original.get('synthetic_audio_added') or original['sample_rate_hz'] != 16000
                or original['channels'] != 1 or not original['frames']
                or not isinstance(approved['structured_query'], str) or not approved['structured_query'].strip()):
            raise ValueError('Invalid source media or structured query: ' + ident)
        sources[ident] = {'gt': gt, 'labels': source['path'], 'manifest': clean_manifest(original, query),
            'source_manifest': str(manifest_path), 'query': query[0]['text'],
            'structured_query': approved['structured_query'], 'template': template}
    return sources


def setup_for(source, arm, supplied=None):
    setup = json.loads(json.dumps(supplied)) if supplied else {
        'model': 'models/' + MODEL,
        'generationConfig': {'responseModalities': ['AUDIO'], 'temperature': 0},
        'outputAudioTranscription': {}, 'contextWindowCompression': {'slidingWindow': {}},
        'proactivity': {'proactiveAudio': True}}
    query = source['structured_query'] if arm == 'Q-structured' else source['query']
    setup['systemInstruction'] = {'parts': [{'text': source['template'].replace('{{USER_REQUEST}}', query)}]}
    expected = ('models/' + MODEL, {'responseModalities': ['AUDIO'], 'temperature': 0},
                {}, {'slidingWindow': {}}, {'proactiveAudio': True})
    actual = tuple(setup.get(name) for name in ('model', 'generationConfig', 'outputAudioTranscription',
                    'contextWindowCompression', 'proactivity'))
    if actual != expected or set(setup) != {'model', 'generationConfig', 'outputAudioTranscription',
                                          'contextWindowCompression', 'proactivity', 'systemInstruction'}:
        raise ValueError('Setup source differs from accepted native baseline settings')
    return setup, query


async def native_session(key, setup, attempt, manifest, prepared, arm, query, native, gate):
    """Streaming branch of native run_eval_v2.session; only text scheduling added."""
    def media():
        frames = {frame['file']: (prepared / frame['file']).read_bytes() for frame in manifest['frames']}
        pcm = (prepared / manifest['audio_file']).read_bytes() if manifest['audio_present'] else b''
        if len(pcm) != int(manifest['audio_samples']) * 2:
            raise ValueError('Original source audio sample count differs')
        return frames, pcm
    frame_bytes, pcm = await asyncio.to_thread(media)
    expected_audio, expected_frames = len(pcm) // 2, len(manifest['frames'])
    await asyncio.to_thread(save, attempt / 'input-spec.json', {
        'setup': setup, 'manifest': manifest, 'prepared': str(prepared), 'arm': arm,
        'playback_speed': 1, 'audio_chunk_s': .2, 'frame_rule': 'original frozen source PTS',
        'tail_wait_s': 2, 'no_forced_turns': True, 'no_synthetic_audio': True,
        'reminder_schedule': [{'scheduled_media_s': t, **item}
                              for t, kind, item in make_schedule(manifest, pcm, arm, query) if kind == 'reminder']})
    logs = await asyncio.to_thread(native.DurableLogs, attempt)
    responses, usage, current = [], [], None
    connection_advisories = []
    t0 = receiver = first_audio = first_text = None
    sent_audio = sent_frames = sent_reminders = interrupted = 0
    max_drift = 0.0
    status, error, latency = 'failed', None, {}
    def record(kind, **fields):
        logs.check()
        logs.emit('input-events.jsonl', {'utc': utc(), 'elapsed_s': time.monotonic() - t0 if t0 else None,
                                       'kind': kind, **fields})
    def finish(reason):
        nonlocal current
        if current is not None:
            current.update(response_id='r%d' % len(responses), finish_reason=reason,
                           interrupted=reason == 'interrupted', complete=reason == 'turnComplete',
                           generation_complete='generation_complete_media_s' in current,
                           turn_complete=reason == 'turnComplete')
            responses.append(current)
            current = None
    try:
        async with native.websockets.connect(native.ENDPOINT, additional_headers={'x-goog-api-key': key},
                open_timeout=30, close_timeout=3, max_size=None, ping_interval=20, ping_timeout=30) as ws:
            record('setup', setup=setup)
            await ws.send(json.dumps({'setup': setup}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), 30))
            await asyncio.to_thread(save, attempt / 'setup-response.json', safe(ack, key))
            if 'setupComplete' not in ack:
                raise RuntimeError('Provider rejected setup: ' + json.dumps(ack))
            t0 = time.monotonic()
            async def receive():
                nonlocal current, first_audio, first_text, interrupted
                raw_index = -1
                async for payload in ws:
                    raw_index += 1
                    elapsed = time.monotonic() - t0
                    event = json.loads(payload)
                    logs.check()
                    logs.emit('raw-output.jsonl', {'utc': utc(), 'receive_media_s': elapsed, 'event': safe(event, key)})
                    if 'goAway' in event:
                        body = event['goAway']
                        connection_advisories.append({'kind': 'goAway', 'raw_event_index': raw_index,
                            'receive_media_s': elapsed, 'time_left': body.get('timeLeft') if isinstance(body, dict) else None})
                    if 'error' in event:
                        raise RuntimeError('Provider error: ' + json.dumps(event))
                    if 'usageMetadata' in event:
                        usage.append(event['usageMetadata'])
                    content = event.get('serverContent', {})
                    parts = content.get('modelTurn', {}).get('parts', [])
                    audio_parts = [part['inlineData'] for part in parts
                                   if part.get('inlineData', {}).get('mimeType', '').startswith('audio/')]
                    text = content.get('outputTranscription', {}).get('text', '')
                    if not text:
                        text = ''.join(part.get('text', '') for part in parts if not part.get('thought'))
                    if audio_parts or text:
                        if current is None:
                            current = {'text': '', 'start_media_s': elapsed, 'end_media_s': elapsed,
                                       'timing_valid': True, 'output_audio_bytes': 0}
                        current['text'] += text
                        current['end_media_s'] = elapsed
                        if text:
                            if first_text is None:
                                first_text = elapsed
                            current.setdefault('first_text_media_s', elapsed)
                        if audio_parts:
                            if first_audio is None:
                                first_audio = elapsed
                            current.setdefault('first_audio_media_s', elapsed)
                            current['start_media_s'] = current['first_audio_media_s']
                            current['output_audio_bytes'] += sum(len(base64.b64decode(part['data'])) for part in audio_parts)
                    if content.get('generationComplete') and current is not None:
                        current['generation_complete_media_s'] = elapsed
                    if content.get('interrupted'):
                        interrupted += 1
                        finish('interrupted')
                    if content.get('turnComplete'):
                        finish('turnComplete')
            receiver = asyncio.create_task(receive())
            async def check():
                logs.check()
                if receiver.done():
                    if receiver.cancelled():
                        raise RuntimeError('Receiver cancelled before session completion')
                    if receiver.exception():
                        raise receiver.exception()
                    raise RuntimeError('Receiver closed before session completion')
            for timestamp, kind, item in make_schedule(manifest, pcm, arm, query):
                await asyncio.sleep(max(0, t0 + timestamp - time.monotonic()))
                await check()
                before = time.monotonic() - t0
                drift = before - timestamp
                max_drift = max(max_drift, drift)
                if drift > 2:
                    raise RuntimeError('Playback drift exceeds 2 seconds; transport failure')
                if kind == 'video':
                    outgoing = {'realtimeInput': {'video': {'mimeType': 'image/jpeg',
                                'data': base64.b64encode(frame_bytes[item['file']]).decode()}}}
                elif kind == 'audio':
                    offset, chunk = item
                    outgoing = {'realtimeInput': {'audio': {'mimeType': 'audio/pcm;rate=16000',
                                'data': base64.b64encode(chunk).decode()}}}
                else:
                    outgoing = {'realtimeInput': {'text': item['text']}}
                await ws.send(json.dumps(outgoing))
                after = time.monotonic() - t0
                if kind == 'video':
                    sent_frames += 1
                    record('video', pts_s=timestamp, path=str(prepared / item['file']), drift_s=drift)
                elif kind == 'audio':
                    sent_audio += len(chunk) // 2
                    record('audio', start_s=offset / 32000, end_s=timestamp, samples=len(chunk) // 2, drift_s=drift)
                else:
                    sent_reminders += 1
                    record('reminder', payload=outgoing, text=item['text'], reminder_index=item['index'],
                        scheduled_media_s=timestamp, actual_media_s=before, send_completed_media_s=after,
                        delivery_status='sent_without_individual_provider_ack', drift_s=drift)
            await asyncio.sleep(max(0, t0 + manifest['duration_sec'] - time.monotonic()))
            if pcm:
                await ws.send(json.dumps({'realtimeInput': {'audioStreamEnd': True}}))
                record('audio_stream_end')
            await asyncio.sleep(2)
            await check()
            latency = {'first_audio_media_s': first_audio, 'first_text_media_s': first_text,
                       'max_playback_drift_s': max_drift, 'total_session_seconds': time.monotonic() - t0}
            if sent_audio != expected_audio or sent_frames != expected_frames:
                raise RuntimeError('Incomplete source coverage')
            status = 'success'
    except asyncio.CancelledError:
        error = {'error_type': 'CancelledError', 'error': 'Session cancelled or wall-clock timeout'}
        raise
    except Exception as problem:
        error = {'error_type': type(problem).__name__, 'error': str(problem).replace(key, '[REDACTED]')}
    finally:
        if receiver is not None:
            if status == 'success' and receiver.done() and not receiver.cancelled() and receiver.exception() is not None:
                status = 'failed'
                error = {'error_type': type(receiver.exception()).__name__,
                         'error': str(receiver.exception()).replace(key, '[REDACTED]')}
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
        finish('session_end' if status == 'success' else 'failed_session_end')
        if error and any(word in error['error'].lower() for word in ('quota', 'resource_exhausted', 'rate limit', '429')):
            error['shared_cooldown_s'] = await gate.penalize()
            error['quota_event_counted'] = True
        transport = {'status': status, 'responses': safe(responses, key), 'latency': latency, 'usage': safe(usage, key),
            'billing': {'api_cost': None, 'reason': 'Provider monetary charge unavailable; complete usage events retained'},
            'coverage': {'source_audio_present': manifest['audio_present'], 'audio_samples_sent': sent_audio,
                'video_frames_sent': sent_frames, 'audio_samples_expected': expected_audio,
                'video_frames_expected': expected_frames, 'complete': sent_audio == expected_audio and sent_frames == expected_frames},
            'reminders': {'planned': sum(kind == 'reminder' for _, kind, _ in make_schedule(manifest, pcm, arm, query)),
                          'sent': sent_reminders, 'provider_acknowledged': None, 'interrupted_events': interrupted},
            'connection_advisories': connection_advisories, 'error': error, 'finished_utc': utc()}
        try:
            await asyncio.to_thread(logs.finish)
        except Exception as problem:
            transport.update(status='failed', error={'error_type': type(problem).__name__,
                'error': str(problem).replace(key, '[REDACTED]'), 'local_recovery_journal': logs.recovery})
        await asyncio.to_thread(save, attempt / 'parsed-output.json', safe(responses, key))
        await asyncio.to_thread(save, attempt / 'transport-result.json', transport)
    return transport


def baseline_references(args, sources):
    result = read(args.source / 'results.json')
    rows = {row['id']: row for row in result['results']}
    if len(result['results']) != 400 or set(rows) != set(sources) or result.get('failures'):
        raise ValueError('B0 must reference the completed 400-case baseline')
    references = []
    for ident, row in rows.items():
        attempt = args.source / 'cases' / ident / Path(row['attempt']).name
        if row['status'] != 'success' or read(attempt / 'transport-result.json')['status'] != 'success':
            raise ValueError('Baseline selected attempt is not successful')
        references.append({'id': ident, 'status': 'success', 'attempt': str(attempt),
                           'baseline_result': str(attempt / 'result.json'), 'no_new_model_call': True})
    freeze(args.root / 'models' / MODEL / 'B0/run/results.json', {'model': MODEL, 'arm': 'B0',
        'planned_cases': 400, 'terminal': 400, 'baseline_scoring': str(args.baseline_scoring), 'results': references})


async def pipeline(args, key):
    global STOP
    STOP = asyncio.Event()
    native = load_native(args.native_source)
    native.STOP = STOP
    gate = native.ConnectionGate(1.0, 3)
    native.CONNECTION_GATE = gate
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, STOP.set)
    sources = await asyncio.to_thread(load_inputs, args)
    baseline_references(args, sources)
    selected = sorted(sources) if args.caseids is None else args.caseids.split(',')
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(sources):
        raise ValueError('Case list must be unique original IDs')
    arms = args.arms.split(',')
    if len(arms) != len(set(arms)) or not set(arms) <= set(ARMS):
        raise ValueError('Unknown or duplicate arm')
    snapshot(__file__, args.root / 'code/run_ablation.py')
    snapshot(args.native_source, args.root / 'code/native_runner.py')
    snapshot(args.structured, args.root / 'source/structured-queries.json')
    snapshot(args.sp, args.root / 'source/SYSTEM-template.txt')
    supplied = read(args.setup_source) if args.setup_source else None
    if supplied and 'setup' in supplied:
        supplied = supplied['setup']
    fixed = {'model': MODEL, 'original_source_cases': sorted(sources), 'prepared': str(args.prepared),
        'data': str(args.data), 'sp': str(args.sp), 'baseline': str(args.source),
        'baseline_scoring': str(args.baseline_scoring), 'source_code': str(args.native_source),
        'api_version': 'v1alpha', 'endpoint': native.ENDPOINT,
        'playback_speed': 1, 'audio_chunk_s': .2, 'tail_receive_s': 2, 'temperature': 0,
        'proactive_audio': True, 'context_compression': 'official slidingWindow',
        'periodic_text': REPEAT.replace('{query}', '{原query}'), 'neutral_text': NEUTRAL,
        'case_policy': 'Exactly 400 unique IDs per full condition; successful probe cases are accepted and never repeated'}
    freeze(args.root / 'config.json', fixed)
    freeze(args.root / 'environment' / ('run-' + str(os.getpid()) + '.json'), {
        'utc': utc(), 'pid': os.getpid(), 'python': sys.version, 'platform': platform.platform(),
        'websockets': importlib.metadata.version('websockets'), 'concurrency': args.concurrency,
        'selected_arms': arms, 'selected_caseids': selected, 'credential_route': 'stdin/getpass; process memory only'})
    semaphore = asyncio.Semaphore(args.concurrency)
    progress_lock = threading.Lock()
    def event(kind, **fields):
        with (args.root / 'runlog.jsonl').open('a') as handle:
            handle.write(json.dumps(safe({'utc': utc(), 'event': kind, **fields}, key), ensure_ascii=False) + '\n')
    def progress_unlocked():
        table = {}
        for arm in ARMS:
            run = args.root / 'models' / MODEL / arm / 'run'
            if arm == 'B0':
                table[arm] = {'planned': 400, 'success': 400, 'reused': True}
                continue
            done = [read(path) for path in run.glob('cases/*/result.json')]
            counts = Counter(row['status'] for row in done)
            table[arm] = {'planned': 400, 'success': counts['success'], 'failed': counts['failed'],
                          'pending': 400 - counts['success'], 'selected_for_this_run': arm in arms}
            save(run / 'progress.json', {'utc': utc(), 'arm': arm, **table[arm]})
        value = {'utc': utc(), 'pid': os.getpid(), 'concurrency': args.concurrency, 'models': {MODEL: table},
                 'quota_events': gate.quota_events, 'stop_requested': STOP.is_set()}
        save(args.root / 'progress.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    def progress():
        with progress_lock:
            progress_unlocked()
    async def worker(arm, ident):
        if STOP.is_set():
            return
        async with semaphore:
            run = args.root / 'models' / MODEL / arm / 'run'
            folder = run / 'cases' / ident
            folder.mkdir(parents=True, exist_ok=True)
            canonical = folder / 'result.json'
            if canonical.exists() and read(canonical)['status'] == 'success':
                return
            source = sources[ident]
            setup, actual_query = setup_for(source, arm, supplied)
            for retry in range(args.max_attempts):
                if STOP.is_set():
                    break
                attempts = sorted(folder.glob('attempt-*'))
                if len(attempts) >= args.max_attempts:
                    break
                try:
                    await gate.wait(STOP)
                except native.StopRequested:
                    break
                attempt = folder / ('attempt-%03d' % (max([int(p.name.split('-')[-1]) for p in attempts], default=0) + 1))
                attempt.mkdir()
                await asyncio.to_thread(save, attempt / 'ground-truth.json', source['gt'])
                await asyncio.to_thread(save, attempt / 'source.json', {
                    'id': ident, 'arm': arm, 'source_manifest': source['source_manifest'],
                    'source_labels': source['labels'], 'original_query': source['query'],
                    'actual_query': actual_query, 'query_route': 'initial final SP only; optional scheduled realtime text'})
                event('case_started', arm=arm, case=ident, attempt=str(attempt))
                start = time.monotonic()
                try:
                    manifest = clean_manifest(source['manifest'], [{**source['gt']['query'][0], 'text': actual_query}])
                    transport = await asyncio.wait_for(native_session(key, setup, attempt, manifest,
                        args.prepared / ident, arm, source['query'], native, gate), source['manifest']['duration_sec'] + 120)
                except Exception as problem:
                    transport_path = attempt / 'transport-result.json'
                    transport = read(transport_path) if transport_path.exists() else {
                        'status': 'failed', 'responses': [], 'error': {'error_type': type(problem).__name__,
                            'error': str(problem).replace(key, '[REDACTED]')}, 'coverage': {'complete': False}}
                result = {'id': ident, 'video_id': ident, 'arm': arm, 'status': transport['status'],
                    'attempt': str(attempt), 'elapsed_s': time.monotonic() - start,
                    'responses': len(transport.get('responses', [])), 'coverage': transport.get('coverage'),
                    'latency': transport.get('latency'), 'usage': transport.get('usage'),
                    'billing': transport.get('billing'), 'error': transport.get('error'),
                    'finished_utc': utc(), 'semantic_content_status': 'pending; no local answer-based retry'}
                await asyncio.to_thread(save, attempt / 'result.json', result)
                await asyncio.to_thread(save, canonical, result)
                event('case_finished', arm=arm, case=ident, result=result)
                await asyncio.to_thread(progress)
                if result['status'] == 'success':
                    break
                detail = json.dumps(result.get('error'), ensure_ascii=False).lower()
                if any(word in detail for word in ('unauthorized', 'permission', 'api key', 'authentication', '403', '401', 'invalid_argument')):
                    STOP.set()
                    event('queue_paused', reason='credential/configuration failure', error=result['error'])
                    break
                if not (result.get('error') or {}).get('quota_event_counted'):
                    await gate.penalize(retry, count_event=False)
    await asyncio.to_thread(progress)
    tasks = [worker(arm, ident) for ident in selected for arm in arms if arm != 'B0']
    await asyncio.gather(*tasks)
    await asyncio.to_thread(progress)
    for arm in arms:
        if arm == 'B0':
            continue
        run = args.root / 'models' / MODEL / arm / 'run'
        results = [read(path) for path in sorted(run.glob('cases/*/result.json'))]
        save(run / 'results.json', {'model': MODEL, 'arm': arm, 'planned_cases': 400, 'results': results,
            'success': sum(row['status'] == 'success' for row in results), 'semantic_content_status': 'pending',
            'stop_requested': STOP.is_set(), 'selected_cases_this_run': selected})
    event('runner_finished', stop_requested=STOP.is_set(), quota_events=gate.quota_events)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=BASE / 'assets/prepared')
    parser.add_argument('--sp', type=Path, required=True)
    parser.add_argument('--source', '--baseline-root', dest='source', type=Path,
                        default=BASE / 'server-records/gemini-3.8-live/runs/streaming')
    parser.add_argument('--baseline-scoring', type=Path, default=BASE / 'reports/content-judge')
    parser.add_argument('--native-source', type=Path, default=BASE / 'run_eval_v2.py')
    parser.add_argument('--setup-source', type=Path)
    parser.add_argument('--structured', '--mapping', dest='structured', type=Path,
                        default=BASE.parent / 'streaming-five-actions-20261012/ablations/results/input-design-full-20261013/records/design/structured-queries-final.json')
    parser.add_argument('--arms', default='Q-structured,R1,R5,N1')
    parser.add_argument('--caseids', help='Probe-only selected IDs; omitted means all 400. Accepted cases persist in the formal folders.')
    parser.add_argument('--probe', action='store_true', help='Use three fixed source cases unless --caseids is explicit; successful cases persist.')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--credential-stdin', action='store_true')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    if args.caseids is not None and not args.probe:
        parser.error('--caseids requires --probe; formal runs must select all 400 source cases')
    if args.probe and args.caseids is None:
        args.caseids = 'elder_stand_f_pos,deaf_sign_home,adult_candle_neg'
    if not (args.source / 'results.json').is_file():
        candidate = args.source / 'gemini-3.8-live/runs/streaming'
        if (candidate / 'results.json').is_file():
            args.source = candidate
    if args.concurrency not in (1, 2) or args.max_attempts < 1:
        parser.error('Use verified initial concurrency 1 or optional 2, and positive attempt budget')
    return args


def main():
    args = arguments()
    args.root.mkdir(parents=True, exist_ok=True)
    key = sys.stdin.readline().rstrip('\r\n') if args.credential_stdin else getpass.getpass('Gemini credential (memory only): ')
    if not key:
        raise SystemExit('Empty credential')
    lock = (args.root / 'evaluation-queue.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another ablation runner owns this experiment') from None
    if args.detach:
        pid = os.fork()
        if pid:
            print(json.dumps({'detached_pid': pid, 'root': str(args.root)}), flush=True)
            return 0
        os.setsid()
        logs = args.root / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / 'runner.log').open('a') as handle:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
        with open(os.devnull) as handle:
            os.dup2(handle.fileno(), 0)
    save(args.root / 'launch.json', {'utc': utc(), 'pid': os.getpid(), 'command': sys.argv,
        'concurrency': args.concurrency, 'credential_saved': False, 'phase': 'streaming_only'})
    try:
        asyncio.run(pipeline(args, key))
        return 0
    except Exception as problem:
        save(args.root / 'runner-error.json', {'utc': utc(), 'type': type(problem).__name__,
                                             'error': str(problem).replace(key, '[REDACTED]')})
        raise
    finally:
        key = ''
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
