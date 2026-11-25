"""Run or check selected final-version Qwen/Gemini B0 native Streaming sessions."""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get("OMNI_LIFE_HANDOFF", HERE.parents[1])).resolve()
RUNTIME = PACKAGE / "stream-runtime"
DATA = PACKAGE / "dataset"
MODELS = ("qwen3.5-omni-flash-realtime","qwen3.5-omni-plus-realtime","qwen3.8-omni-flash-realtime","gemini-3.8-live")
VERSION = "20261124-r3-final"
RELEASE = "f6ef9aa92591c760fb2f0a7a0f5b368bf399ebeea9d581ad0233935a96ca0fdb"
PINS = {"native/qwen.py":"789be187a0e5b047df853991d1ff20b166e3d884de29e8e6947763a8fec7d168",
        "native/gemini.py":"13036b882341fd1bf6cba9e304bcf32cd4819c198127fa0512043216f254c085",
        "native/audit.py":"4b08a0d278232f206e4bd6c14781ccf8cdbc9f97286f015e5fc1a480332bffe0",
        "native/qwen38_session_settings.py":"83e22f2323a799f8015dd843a5233e0464d8a561f6fb37b30b2265da51fd1e22",
        "historical-media-routes.json":"92add266cf10d99e78b4d9e5ef7bdf2de1daed6929c8e7520ae5844bfa9ee813"}


def require(value, error):
    if not value:
        raise ValueError(error)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    out = path.with_name("." + path.name + "." + uuid4().hex)
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    out.replace(path)


def module(name):
    source = HERE/"native"/(name+".py")
    spec = importlib.util.spec_from_file_location("original_"+name,source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inputs(model, chosen, *, queue_path=None):
    for rel,expected in PINS.items():
        require(digest((RUNTIME/"api" if rel == "historical-media-routes.json" else HERE)/rel) == expected,"Different historical source: "+rel)
    require(digest(DATA/"manifest.json") == RELEASE,"Dataset release mismatch")
    build = read(RUNTIME/"build-receipt.json")
    identity = RUNTIME/"source-identities.json"
    require(digest(identity) == build["source_identities_sha256"], "Native input identity checksum mismatch")
    inventory = read(identity)
    require(inventory["release_manifest_sha256"] == RELEASE and inventory["version"]==VERSION, "Wrong Streaming input release")
    archive = read(RUNTIME/"api/historical-media-routes.json")["models"][model]
    auditor = module("audit")
    roster = {r["id"]:r for r in inventory["cases"]}
    dataset = read(DATA/"manifest.json")
    official = {r["case_id"]:r for r in dataset["streaming_labels"]}
    require(roster.keys()==official.keys()==archive.keys() and len(roster)==400,"Invalid scene register")
    protocol = DATA/"protocol/streaming-scoring.json"
    require(digest(protocol)==dataset["files_sha256"]["protocol/streaming-scoring.json"],"Wrong timing protocol")
    rules=read(protocol)["cases"]
    queued = (queue_path or PACKAGE/"streaming/rerun"/(model+".txt")).read_text().splitlines()
    ids = chosen or queued
    require(len(set(queued))==len(queued) and set(queued)<=roster.keys() and len(set(ids))==len(ids) and set(ids)<=set(queued),
            "Duplicate, unknown or non-pending case")
    plans = []
    for cid in ids:
        info, orig, final = roster[cid],archive[cid],official[cid]
        folder = RUNTIME/"prepared"/cid
        manifest_path = folder/"manifest.json"
        require(digest(manifest_path)==info["prepared_manifest_sha256"], "Wrong prepared manifest: "+cid)
        man=read(manifest_path)
        sp=(folder/man["system_file"]).read_text()
        require(man["id"]==cid and man["dataset_version"]==VERSION and not man["synthetic_audio_added"] and
                info["video_sha256"]==man["source_video_sha256"]==final["video_sha256"] and
                digest(DATA/final["labels_path"])==info["labels_sha256"]==final["revised_labels_sha256"],"Wrong final video or task: "+cid)
        require(info["query"]==man["query"]==read(DATA/final["labels_path"])["query"] and len(man["query"])==1 and
                sp.endswith("当前用户交给你的任务：\n"+man["query"][0]["text"]+"\n") and
                digest(folder/"SYSTEM.txt")==man["system_sha256"]==info["system_sha256"],"Wrong text task: "+cid)
        signature = auditor.sha(json.dumps(auditor.equal_media(man),separators=(",",":")).encode())
        require(signature==orig["original_media_identity_sha256"], "Original native AV schedule changed: "+cid)
        require(all(digest(folder/f["file"])==f["sha256"] for f in man["frames"]),"JPEG checksum mismatch: "+cid)
        if man["audio_present"]:
            pcm=folder/man["audio_file"]
            require(pcm.stat().st_size==2*man["audio_samples"] and digest(pcm)==man["audio_sha256"],"PCM checksum mismatch: "+cid)
        plans.append((cid,folder,man,sp,dict(case_id=cid,dataset_version=VERSION,release_manifest_sha256=RELEASE,model=model,
                                             route=orig["baseline_route"],prepared_manifest_sha256=info["prepared_manifest_sha256"],
                                             source_video_sha256=info["video_sha256"],labels_sha256=info["labels_sha256"],
                                             system_sha256=info["system_sha256"],media_identity_sha256=signature,
                                             audio_samples=man["audio_samples"],frames=len(man["frames"]),
                                             media_end_s=rules[cid].get("media_end_s",rules[cid]["annotation_end_s"]))))
    return auditor,plans


class SocketBoundary(Exception):
    pass


def native_transport(model, audit, profile, offline):
    gemini = model==MODELS[-1]
    require(profile=="gemini-developer" if gemini else profile in {"qwen-current","qwen-archive"},"Wrong native route for selected model")
    endpoint,env = {"qwen-current":(audit.NEWQ,"OMNILIFE_QWEN_CURRENT_KEY"),"qwen-archive":(audit.OLDQ,"OMNILIFE_QWEN_ARCHIVE_KEY"),
                    "gemini-developer":(audit.OLDG,"OMNILIFE_GEMINI_DEVELOPER_KEY")}[profile]
    crossed = []
    if offline:
        ws=ModuleType("websockets")
        ws.__version__="15.offline"
        def intercept(url,**kwargs):
            require(url == (endpoint if gemini else endpoint+"?model="+model),"Wrong socket endpoint")
            crossed.append(True)
            raise SocketBoundary("offline validation intercepted native socket")
        ws.connect=intercept
        with patch.dict(sys.modules,{"websockets":ws}):
            native=module("gemini" if gemini else "qwen")
    else:
        require(importlib.metadata.version("websockets").split(".")[0]=="15","Historical transport requires websockets 15.x")
        native=module("gemini" if gemini else "qwen")
    native.ENDPOINT=endpoint
    if not gemini:
        original=native.verify_session
        settings=module("qwen38_session_settings")
        expected=dict(settings.PROVIDER_SETTINGS,model=model)
        def checked(requested,returned):
            result=original(requested,returned)
            mismatches=[k for k,v in expected.items() if settings.canonical(returned.get(k)) != settings.canonical(v)]
            result["mismatches"]=sorted(set(result["mismatches"]+mismatches))
            result["passed"]=not result["mismatches"]
            return result
        native.verify_session=checked
    return native,env,crossed


def convert(responses, horizon, evidence, cid, final_end):
    replies=[]
    unresolved=[]
    for r in responses:
        stamp=r.get("start_media_s")
        text=r.get("text") or r.get("transcript") or ""
        if not text.strip() or type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp < 0:
            unresolved.append({"response_id":r.get("response_id"),"reason":"missing transcript or source-clock onset"})
        elif stamp > final_end:
            unresolved.append({"response_id":r.get("response_id"),"reason":"native arrival after EOF needs independent first-turn attribution"})
        else:
            replies.append(dict(phase="video",video_s=stamp,text=text,verdict="pending",facts=[]))
    if any(a["video_s"]>b["video_s"] for a,b in zip(replies,replies[1:])):
        unresolved.append({"reason":"native output order and measured onsets disagree"})
    capture={"capture_complete":True,"eof_opportunity":evidence, "semantic_review_complete":not replies,"replies":replies}
    return dict(case_id=cid,native_raw_responses=responses,observation_completed_media_s=horizon,unresolved=unresolved,
                normalized=None if unresolved else capture)


async def attempt_one(model,plan,root,profile,offline,auditor):
    cid,folder,man,sp,bind=plan
    native,env,crossed=native_transport(model,auditor,profile,offline)
    key="offline-placeholder" if offline else os.environ.get(env,"").strip()
    require(key and "\n" not in key,"Missing credential in "+env)
    run=root/"models"/model/"b0"
    gemini=model==MODELS[-1]
    scope=run/"cases"/cid
    if gemini:
        target=scope/("attempt-"+uuid4().hex)
        target.mkdir(parents=True)
        save(target/"route.json",dict(profile=profile,endpoint=native.ENDPOINT))
        cfg=auditor.gsetup(sp)
        result=await native.session(key,cfg,target,man,folder,qa_row=None)
    else:
        result=await native.run_case(model,cid,key,RUNTIME/"prepared",root,"b0")
        target=Path(result["attempt"])
        require(target.is_relative_to(scope),"Native session wrote to an unexpected directory")
    if offline:
        require(crossed==[True],"Native transport failed before reaching the socket boundary: "+cid)
        spec=read(target/"input-spec.json")
        require(spec["setup"]==auditor.gsetup(sp) if gemini else spec["instructions"]==sp and not spec["task_tts"] and not spec["padding"],
                "Offline transport specification differs: "+cid)
        return {"case_id":cid,"model":model,"profile":profile,"frames":len(man["frames"]),"audio_samples":man["audio_samples"],
                "system_sha256":bind["system_sha256"],"socket_boundary_verified":True}
    if result["status"]!=("success" if gemini else "completed"):
        return dict(case_id=cid,binding=dict(bind,route=profile),attempt=str(target),
                    native_status=result["status"],native_error=result.get("error"),terminal_failure=True,
                    normalized=None,unresolved=[{"reason":"native technical failure, original evidence retained"}])
    if gemini:
        verified=auditor.gaudit(target,model,man,folder,sp,native.ENDPOINT,cfg["model"])
        horizon=result["latency"]["total_session_seconds"]
        responses=result["responses"]
        # AUDIO proactivity is explicitly enabled, and the original receiver remains
        # active after source completion; audible-source runs send audioStreamEnd.
        opportunity=man["audio_present"] and horizon>=bind["media_end_s"]+1.99
        mechanism="proactive audio, end-of-audio event and active receiver through the audited observation tail"
    else:
        verified=auditor.qaudit(target,model,man,folder,sp,native.ENDPOINT)
        horizon=read(target/"status.json")["observation_completed_media_s"]
        responses=read(target/"parsed-output.json")["responses"]
        # Plain semantic VAD does not explicitly grant a new EOF turn after its last
        # media packet. Retain the monitored tail without crediting an assumed turn.
        opportunity=False
        mechanism="semantic VAD; active receiver through tail, no explicit new EOF turn"
    trace=convert(responses,horizon,bool(opportunity),cid,bind["media_end_s"])
    trace.update(binding=dict(bind,route=profile),attempt=str(target),native_evidence=verified,eof_evidence=mechanism)
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--case-ids",nargs="+",help="Defaults to the per-model rerun list")
    p.add_argument("--route",choices=("historical","qwen-current","qwen-archive","gemini-developer"),default="historical")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--out",type=Path,help="Live-only, new results root")
    p.add_argument("--report",type=Path,help="Offline-only input/connection-boundary receipt")
    args=p.parse_args()
    if args.execute and (args.out is None or args.report):
        p.error("--execute requires --out; --report is offline only")
    auditor,plans=inputs(args.model,args.case_ids)
    if not args.execute:
        with TemporaryDirectory() as tmp:
            result=[asyncio.run(attempt_one(args.model,plan,Path(tmp),plan[-1]["route"] if args.route=="historical" else args.route,True,auditor)) for plan in plans]
        receipt=dict(dataset_version=VERSION,release_manifest_sha256=RELEASE,model=args.model,cases=len(result),live_connections=0,checked=result)
        if args.report:
            save(args.report,receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!="checked"},ensure_ascii=False))
        return
    root=args.out.resolve()/VERSION
    run=root/"models"/args.model/"b0"
    run.mkdir(exist_ok=True,parents=True)
    with (run/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for plan in plans:
            cid,_,_,_,binding=plan
            profile=binding["route"] if args.route=="historical" else args.route
            proof=run/"verified"/(cid+".json")
            bound=dict(binding,route=profile)
            if proof.exists():
                require(read(proof)["binding"]==bound,"Previous accepted run has different inputs or route: "+cid)
                continue
            failed=run/"failed"/(cid+".json")
            if failed.exists():
                require(read(failed)["binding"]==bound,"Historical failed attempt has different bindings")
                continue
            previous=run/"cases"/cid
            require(not previous.exists(),"Existing unverified native attempt at "+str(previous)+"; inspect before deliberately resubmitting into a separate output root.")
            trace=asyncio.run(attempt_one(args.model,plan,root,profile,False,auditor))
            proof.parent.mkdir(exist_ok=True)
            if trace.get("terminal_failure"):
                failed=run/"failed"/(cid+".json")
                failed.parent.mkdir(exist_ok=True)
                save(failed,trace)
            else:
                save(proof,trace)
            print(json.dumps({"case_id":cid,"normalized":trace["normalized"] is not None}),flush=True)
            export(run,args.model)
            if trace.get("terminal_failure") and any(token in str(trace.get("native_error","")).lower()
                    for token in ("quota","429","401","403","accessdenied","invalid_api_key","unauthorized","resource_exhausted")):
                raise SystemExit("Provider authorization or quota failure. Receipt retained: "+str(failed))
        export(run,args.model)


def export(run,model):
    records={}
    pending={}
    for path in sorted((run/"verified").glob("*.json")):
        trace=read(path)
        require(trace["binding"]["model"]==model and trace["binding"]["release_manifest_sha256"]==RELEASE and trace["case_id"]==path.stem,"Foreign receipt")
        if trace["normalized"] is None:
            pending[path.stem]=trace["unresolved"]
        else:
            records[path.stem]=trace["normalized"]
    for path in sorted((run/"failed").glob("*.json")):
        pending[path.stem]=read(path)["unresolved"]
    save(run/"native-new.json",dict(dataset_version=VERSION,release_manifest_sha256=RELEASE,model=model,records=records))
    save(run/"time-or-transcript-review.json",pending)


if __name__=="__main__":
    main()
