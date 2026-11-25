"""Check the Hugging Face dataset, with optional media archive verification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

from .choice_format import parse_selected
from .qa import choices, indexed, load_questions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(value):
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and '..' not in path.parts and '\\' not in value


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def verify_archive(path, checksum, videos, input_hashes):
    expected, name = checksum.read_text(encoding='utf-8').split()
    require(name == 'media.zip' and re.fullmatch('[0-9a-f]{64}', expected), 'Invalid media.zip.sha256')
    with path.open('rb') as stream:
        require(sha256(stream) == expected, 'Media archive SHA-256 mismatch')
    members = videos | input_hashes.keys()
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(members) and set(names) == members,
                'The media archive does not match the scene and QA indexes')
        for name, digest in input_hashes.items():
            with archive.open(name) as stream:
                require(sha256(stream) == digest, f'QA media SHA-256 mismatch for {name}')


def validate(dataset, archive):
    scenes = indexed(dataset / 'scenes.csv', 'case_id')
    protocol = read_json(dataset / 'protocol/streaming-scoring.json')
    cases = protocol['cases']
    require(len(scenes) == 400 and scenes.keys() == cases.keys(), 'Scene roster and protocol disagree')
    require({row['sample'] for row in scenes.values()} == {str(i) for i in range(1, 401)},
            'Scene numbers must cover 1 through 400')
    zh, media = load_questions(dataset, 'zh')
    en, _ = load_questions(dataset, 'en')
    require(zh.keys() == en.keys(), 'Chinese and English QA IDs differ')

    sources = Counter()
    videos = set()
    for case_id, row in scenes.items():
        video, label = row['video_path'], row['labels_path']
        require(relative_path(video) and video.startswith(('data/real/', 'data/synthetic/'))
                and video not in videos, f'Invalid or duplicate video path in {case_id}')
        require(label == f'streaming-labels/{case_id}.json' and relative_path(label),
                f'Unexpected label path in {case_id}')
        require(read_json(dataset / label) == cases[case_id],
                f'Scene label and aggregate protocol differ for {case_id}')
        case = cases[case_id]
        require(case['case_id'] == case_id and str(case['sample']) == row['sample']
                and case['task_zh'] == row['task_zh'] and row['task_en'].strip(),
                f'Scene metadata or task differs for {case_id}')
        sources[PurePosixPath(video).parts[1]] += 1
        videos.add(video)

    input_kinds = Counter()
    hashes = {}
    for question_id, item in media.items():
        scene = scenes[item['case_id']]
        mode = item['choice_mode']
        kind, path = item['input_kind'], item['input_path']
        require(mode in {'single', 'multiple'} and item['sample'] == scene['sample'],
                f'Invalid QA metadata for {question_id}')
        require(relative_path(path), f'Invalid QA media path for {question_id}')
        if kind == 'full_video':
            require(path == scene['video_path'] and not item['cutoff_s'], f'Invalid full video for {question_id}')
        else:
            limit = cases[item['case_id']]['annotation_end_s']
            require(kind in {'prefix_video', 'cutoff_frame'} and path.startswith('review/')
                    and 0 <= float(item['cutoff_s']) <= limit, f'Invalid cutoff media for {question_id}')
        digest = item['input_sha256']
        require(re.fullmatch('[0-9a-f]{64}', digest) and hashes.setdefault(path, digest) == digest,
                f'Invalid or inconsistent media digest for {question_id}')
        input_kinds[kind] += 1
        answers = []
        option_sets = []
        for language, question in (('zh', zh[question_id]), ('en', en[question_id])):
            options = choices(question)
            require(question['lang'] == language and question['video'] == scene['video_path']
                    and question['question'].strip() and len(options) >= 2, f'Invalid question in {question_id}')
            answers.append(parse_selected(question['answer'], mode, options))
            option_sets.append(options.keys())
        require(answers[0] and answers[0] == answers[1] and option_sets[0] == option_sets[1],
                f'Bilingual QA options or answer key differ for {question_id}')

    silence = sum(c['mode'] == 'conditional_silence' for c in cases.values())
    terminal = sum('terminal_count' in c for c in cases.values())
    events = sum(len(c['required_events']) for c in cases.values()) + terminal
    require(sources == {'synthetic': 277, 'real': 123} and (events, silence, terminal) == (500, 103, 13)
            and input_kinds == {'full_video': 522, 'prefix_video': 262, 'cutoff_frame': 16},
            'Published dataset counts do not match the indexes')
    if archive:
        verify_archive(archive, dataset / 'media.zip.sha256', videos, hashes)
    print(f'Verified {len(scenes)} Streaming labels, {events} interventions and {len(media)} bilingual QA inputs.')
    if archive:
        print(f'Verified media archive SHA-256 and coverage of {len(videos | hashes.keys())} files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--media-archive', type=Path, help='Also verify the ZIP checksum and indexed media paths')
    args = parser.parse_args()
    validate(args.dataset, args.media_archive)
