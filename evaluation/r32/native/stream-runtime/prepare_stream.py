"""Build the 400-case Streaming model inputs from the signed r3 dataset and original 1 Hz native AV cache."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

RELEASE_SHA = "f6ef9aa92591c760fb2f0a7a0f5b368bf399ebeea9d581ad0233935a96ca0fdb"
SEPARATOR = "当前用户交给你的任务：\n"


def require(value, why):
    if not value:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_verified(source, target, expected):
    require(sha(source) == expected, f"Input checksum mismatch in {source}")
    if sys.platform == "darwin" and ctypes.CDLL(None).clonefile(os.fsencode(source), os.fsencode(target), 0) == 0:
        return
    shutil.copyfile(source, target)


def make(args, stage):
    signed = args.dataset / "manifest.json"
    require(sha(signed) == RELEASE_SHA, "Expected signed r3-final dataset")
    dataset = load(signed)
    protocol = args.dataset / "protocol/streaming-scoring.json"
    require(sha(protocol) == dataset["files_sha256"]["protocol/streaming-scoring.json"], "Bad r3 scoring protocol")
    rules = load(protocol)["cases"]
    lines = [json.loads(line) for line in args.old_prompts.read_text().splitlines()]
    prompts = {r["id"]: r["text"] for r in lines}
    require(len(lines) == len(prompts) == len(dataset["streaming_labels"]) == 400, "Unexpected case count")
    headers = {text.split(SEPARATOR)[0] for text in prompts.values() if text.count(SEPARATOR) == 1}
    require(len(headers) == 1 and set(prompts) == set(rules), "Archived system instructions differ between cases")
    rows, updates, frames, audio_samples = [], [], 0, 0
    out = stage / "prepared"
    out.mkdir()
    for item in dataset["streaming_labels"]:
        cid = item["case_id"]
        original, dest = args.prepared_source / cid, out / cid
        dest.mkdir()
        old = load(original / "manifest.json")
        require(old["id"] == cid and old["source_video_sha256"] == item["video_sha256"], "Wrong frozen original: " + cid)
        old_sp = original / old["system_file"]
        original_prompt = prompts[cid]
        require(sha(old_sp) == old["system_sha256"] and old_sp.read_text() == original_prompt and original_prompt.count(SEPARATOR) == 1, "Archived prepared SP differs from deployment: " + cid)
        header, old_task = original_prompt.split(SEPARATOR)
        label_path = args.dataset / item["labels_path"]
        source = args.dataset / item["video_path"]
        task_audio = args.dataset / item["task_audio"]
        require(sha(source) == item["video_sha256"] and sha(label_path) == item["revised_labels_sha256"] and sha(task_audio) == item["task_audio_sha256"], "Signed source differs: " + cid)
        q = load(label_path)["query"]
        require(len(q) == 1 and q[0]["start_time_sec"] == 0 and rules[cid]["task_zh"] == q[0]["text"], "Ambiguous r3 task: " + cid)
        new_sp = header + SEPARATOR + q[0]["text"] + "\n"
        (dest / "SYSTEM.txt").write_text(new_sp)
        if new_sp != original_prompt:
            updates.append(cid)
        times = [f["pts_sec"] for f in old["frames"]]
        require(times and times[0] >= 0 and times == sorted(set(times)) and times[-1] <= old["duration_sec"] + 0.001, "Invalid video clock: " + cid)
        require(old["sample_rate_hz"] == 16000 and old["channels"] == 1 and not old["synthetic_audio_added"], "Invalid PCM protocol")
        for frame in old["frames"]:
            require(Path(frame["file"]).name == frame["file"], cid)
            copy_verified(original / frame["file"], dest / frame["file"], frame["sha256"])
        if old["audio_present"]:
            pcm = original / old["audio_file"]
            require(pcm.stat().st_size == old["audio_samples"] * 2, f"PCM sample count: {cid}")
            copy_verified(pcm, dest / old["audio_file"], old["audio_sha256"])
        else:
            require(old["audio_samples"] == 0, "Missing original audio: " + cid)
        frames += len(times)
        audio_samples += old["audio_samples"]
        man = dict(old, source_video=item["video_path"], system_file="SYSTEM.txt", system_sha256=sha(dest/"SYSTEM.txt"),
                   query=q, dataset_version=dataset["version"], labels_sha256=item["revised_labels_sha256"])
        write(dest / "manifest.json", man)
        # Fields that classify ground truth stay outside model input files. Never pass the inventory to a model as text.
        def rel(field):
            return str(Path(item[field]).relative_to("data"))
        rows.append(dict(id=cid, case_id=cid, video=rel("video_path"), video_bytes=source.stat().st_size,
                         video_sha256=item["video_sha256"], labels=rel("labels_path"), labels_sha256=item["revised_labels_sha256"],
                         task_audio=rel("task_audio"), task_audio_sha256=item["task_audio_sha256"], query=q,
                         system_sha256=man["system_sha256"], duration_sec=old["duration_sec"], audio_present=old["audio_present"],
                         allow_post_video=any(e.get("allow_post_video") for e in rules[cid]["required_events"]) or bool((rules[cid].get("terminal_count") or {}).get("allow_post_video")),
                         prepared_manifest_sha256=sha(dest / "manifest.json")))
    require(len(rows) == len({r["id"] for r in rows}) == 400, "Duplicate input")
    write(stage / "source-identities.json", dict(version=dataset["version"], case_count=len(rows), data="../dataset/data",
          release_manifest_sha256=RELEASE_SHA, archived_prompt_index_sha256=sha(args.old_prompts), cases=rows))
    write(stage / "build-receipt.json", dict(version=dataset["version"], cases=len(rows), frames=frames, pcm_samples=audio_samples,
          original_audio_cases=sum(r["audio_present"] for r in rows), first_post_video_opportunity_scored_cases=sum(r["allow_post_video"] for r in rows),
          changed_vs_1120_prompts=sorted(updates), video_clock_origin_s=0, original_pcm_jpeg_bytes_verified=True,
          source_identities_sha256=sha(stage/"source-identities.json"), network_calls=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset","prepared-source","old-prompts","out"):
        p.add_argument("--"+key, type=Path, required=True)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    names = ("prepared", "source-identities.json", "build-receipt.json")
    require(all(not (args.out / name).exists() for name in names), "Use a new materialization directory")
    with tempfile.TemporaryDirectory(prefix=".building-r3-stream-", dir=args.out.parent) as stage:
        stage=Path(stage)
        make(args, stage)
        for name in names:
            (stage/name).rename(args.out/name)
    print((args.out/"build-receipt.json").read_text())


if __name__ == "__main__":
    main()
