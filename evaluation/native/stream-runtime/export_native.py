"""Export real raw model captures for score_migrated_streaming.py; preserve unknown or late timing for review."""
import argparse
import math
from pathlib import Path
import json
from run_native import ROOT, PACKAGE, VERSION, require, load, write, sha, stable_digest, vendor


def elapsed(event):
    t = event.get("media_s")
    return t if type(t) in (float,int) and math.isfinite(t) and t >= 0 else None


def record(attempt, result, cid, duration, annotation_end, batch, native_manifest, scoring_media_end=None):
    successful = (result.get("success") is True and batch.coverage_gate(result,native_manifest) == "success")
    events_path = attempt / "runlog.jsonl"
    events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()] if events_path.is_file() else []
    begins = [e for e in events if e.get("event")=="r3_eof_capture_begin" and e.get("receiver_open") is True]
    ends = [e for e in events if e.get("event")=="r3_eof_capture_end" and e.get("receiver_open") is True]
    begin = elapsed(begins[0]) if len(begins)==1 else None
    end = elapsed(ends[0]) if len(ends)==1 else None
    eof = successful and begin is not None and end is not None and duration <= begin < end
    replies, unresolved = [], []
    post = False
    for i, raw in enumerate(result.get("responses", [])):
        text, ts = raw.get("text"), raw.get("start_media_s")
        if not isinstance(text,str) or not text.strip() or type(ts) not in (float,int) or not math.isfinite(ts) or ts < 0:
            unresolved.append(i)
            continue
        if ts <= min(duration, scoring_media_end if scoring_media_end is not None else duration) and ts <= annotation_end + .25 and not post:
            replies.append(dict(phase="video", video_s=ts, text=text, verdict="pending"))
        elif eof and begin <= ts <= end and not post:
            replies.append(dict(phase="post_video", turn=1, text=text, verdict="pending"))
            post = True
        else:
            unresolved.append(i)
    if any(a["video_s"] > b["video_s"] for a,b in zip(replies,replies[1:]) if a["phase"]==b["phase"]=="video"):
        unresolved.append("nonmonotonic_native_onset")
    return dict(capture_complete=successful and not unresolved, eof_opportunity=eof, semantic_review_complete=False,
                replies=replies, raw_responses=result.get("responses",[]), unresolved_response_indices=unresolved,
                source_attempt=str(attempt), transport_sha256=sha(attempt/"transport-result.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Run directory containing binding.json and raw/")
    p.add_argument("--out", type=Path, required=True)
    a=p.parse_args()
    b=load(a.run/"binding.json")
    require(b["dataset_version"]==VERSION and b["prepared_index_sha256"]==sha(ROOT/"source-identities.json"),"Different input binding")
    require(a.run.name==stable_digest(b)[:20],"Wrong native run directory")
    src,provenance,batch=vendor(b["model"])
    require(provenance["files"]==b["runtime_files"],"Archived native runtime differs")
    rules=load(PACKAGE/"dataset/protocol/streaming-scoring.json")["cases"]
    records={}
    for cid in b["selected_ids"]:
        folders=sorted((a.run/"raw/cases"/cid).glob("attempt-*"))
        attempts=[f for f in folders if (f/"case-status.json").is_file() and load(f/"case-status.json").get("status")=="success"]
        if not attempts:
            continue
        f=attempts[0]
        man=load(ROOT/"prepared"/cid/"manifest.json")
        require(load(f/"input-spec.json")["manifest_sha256"]==sha(ROOT/"prepared"/cid/"manifest.json"),"Bad case input: " + cid)
        records[cid]=record(f,load(f/"transport-result.json"),cid,man["duration_sec"],rules[cid]["annotation_end_s"],batch,man,rules[cid].get("media_end_s"))
    write(a.out,dict(dataset_version=VERSION,model=b["model"],records=records))
    print(json.dumps(dict(selected=len(b["selected_ids"]),captured=len(records),require_timing_review=sum(not r["capture_complete"] for r in records.values()))))


if __name__ == "__main__":
    main()
