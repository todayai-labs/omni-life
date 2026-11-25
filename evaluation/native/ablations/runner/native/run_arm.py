"""Run archived self-hosted Streaming ablations on the final data; default performs local input checks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get("OMNI_LIFE_HANDOFF", HERE.parents[2])).resolve()
ABL = PACKAGE / "ablations"
TASK_AUDIO = ABL / "runner/native/task-audio.json"
STREAM = PACKAGE / "stream-runtime"
sys.path.insert(0, str(HERE.parents[2] / "stream-runtime"))
import run_native as native
from prepare_stream import load, sha, require, write, SEPARATOR

STRICT = "TTS-audio-only-v2"
REMINDERS = {"N1", "R1", "R5"}
MODELS = ("minicpm-o-4.5-local","realtime-venus-omni","moss-vl-realtime")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exact_file(src, dst):
    if dst.exists():
        require(sha(dst)==sha(src), "Existing input differs: " + str(dst))
    else:
        shutil.copyfile(src,dst)


def same_json(path, obj):
    if path.exists():
        require(load(path)==obj,"Changed run file: "+str(path))
    else:
        write(path,obj)


def sample_overlay(base, row, text, out):
    cid = row["id"]
    original = base / cid
    dst = out / cid
    dst.mkdir(parents=True,exist_ok=True)
    m = load(original/"manifest.json")
    old = (original/m["system_file"]).read_text()
    query = m["query"]
    require(old.count(query[0]["text"])==1 and old.split(SEPARATOR)[1].strip()==query[0]["text"],"Task suffix differs: "+cid)
    updated = old.replace(query[0]["text"],text)
    sp = dst/m["system_file"]
    if sp.exists():
        require(sp.read_text()==updated, "Changed structured system")
    else:
        sp.write_text(updated)
    for f in [x["file"] for x in m["frames"]]+([m["audio_file"]] if m["audio_present"] else []):
        dst_file = dst/f
        if not dst_file.exists():
            dst_file.symlink_to(os.path.relpath((original/f).resolve(),dst.resolve()))
        require(dst_file.resolve()==(original/f).resolve(), "Changed structured media: "+cid)
    new = dict(m, original_query=query, query=[dict(query[0],text=text)], system_sha256=sha(sp))
    same_json(dst/"manifest.json",new)
    return new


def materialize_runtime(model, source, out, reminders):
    root = out/"runtime"
    adapters=root/"adapters"
    adapters.mkdir(parents=True,exist_ok=True)
    files = [*source.glob("*.py"),*(source/"adapters").glob("*.py")]
    for f in files:
        alt = None
        if reminders and model=="moss-vl-realtime" and f.name=="moss_av.py":
            alt=HERE/"reminders/moss_av.py"
        elif reminders and f.name=="venus_v2.py":
            alt=HERE/"reminders"/("venus_memory_v2.py" if model=="realtime-venus-omni" else "venus_v2.py")
        exact_file(alt or f,root/f.relative_to(source))
    if not reminders:
        exact_file(native.NATIVE_SCORER,root/"scoring.py")
    return root


def validate_condition(c,m):
    require(c["task_zh"]==m["query"][0]["text"] and c["strict_tts_system_text"]=="", "Wrong r3 task or TTS system")
    task=c["task_zh"]
    for field in c["fields"]:
        a,b=field["source_span"]
        require(task[a:b]==field["source_text"],"Structured span differs from final task")
    # The archived editor sometimes adds a full stop after a source span. Use the frozen rendered string verbatim.
    labels = [f["label"]+"："+f["source_text"] for f in c["fields"]]
    rendered = c["structured_query"].splitlines()
    punctuation = "。，；、,;！!？? "
    require(len(labels)==len(rendered) and all(a.rstrip(punctuation)==b.rstrip(punctuation) for a,b in zip(labels,rendered)),
            "Changed structured field")
    require(c["reminder_text"]=="提示：你的任务是 "+task,"Changed repeated reminder")


def ready(a, refs, rows, config, src, out):
    arm=a.condition
    prepared=STREAM/"prepared"
    effective=src
    cfg=dict(config)
    reminder=None
    jobs=None
    strict_dir=HERE/"strict"/a.model
    if arm in REMINDERS:
        cfg.update(reminder_kind="generic" if arm=="N1" else "repeat_query",reminder_interval_s=5 if arm=="R5" else 1)
        effective=materialize_runtime(a.model,src,out,reminders=True)
        sys.path.insert(0,str(effective))
        try:
            # Each command launches its own interpreter; adapters are imported from this arm's runtime.
            import importlib
            reminder=importlib.import_module("adapters."+("moss_av" if a.model=="moss-vl-realtime" else "venus_v2"))
        finally:
            sys.path.pop(0)
    elif arm==STRICT:
        strict=module("strict_r3_driver",strict_dir/"run_full.py")
        helper=module("tts_phase_r3",strict_dir/"tts_phase.py")
        jobs=strict.source_rows(STREAM/"source-identities.json",prepared)
        require(len(jobs)==400,"Strict worker requires the complete inventory")
        effective=materialize_runtime(a.model,src,out,reminders=False)
        scoring=strict.load_scoring(effective)
    elif arm=="Q-structured":
        prepared=out/"prepared-structured"
    else:
        raise ValueError("Unknown ablation")
    for row in rows:
        cid=row["id"]
        m=load(STREAM/"prepared"/cid/"manifest.json")
        c=refs[cid]
        validate_condition(c,m)
        if arm=="Q-structured":
            updated = sample_overlay(STREAM/"prepared",row,c["structured_query"],prepared)
            require(updated["query"][0]["text"]==c["structured_query"],"Structured worker input mismatch")
        elif arm in REMINDERS:
            if a.model=="moss-vl-realtime":
                interval,kind=reminder.reminder_options(cfg)
                rendered=reminder.reminder_text(kind,m["query"][0]["text"])
                times=reminder.reminder_schedule(m["duration_sec"],interval)
                require(all(0<t<m["duration_sec"] for t in times),"EOF reminder")
            else:
                controller=reminder._TaskReminders(cfg,m,SimpleNamespace())
                interval=controller.interval
                rendered=controller.text
            expected=c["generic_reminder"] if arm=="N1" else c["reminder_text"]
            require(rendered==expected and interval==(5 if arm=="R5" else 1),"Wrong reminder")
        elif arm==STRICT:
            job=strict.gt_job(cid,SimpleNamespace(prepared_root=prepared,data_root=PACKAGE/"dataset/data"),helper,scoring)
            pcm,entry=helper.load_task_audio(TASK_AUDIO,cid,m)
            require(job["effective"]=="" and len(pcm)==entry["audio_samples"]*2 and sha(PACKAGE/c["task_audio"])==entry["source_mp3_sha256"]==c["task_audio_sha256"],"Wrong strict input")
    same_json(out/"deployment.json",cfg)
    python=a.python or cfg.get("python") or sys.executable
    args=[python,"-u"]
    ids=[row["id"] for row in rows]
    if arm==STRICT:
        args += [strict_dir/"run_full.py","--model",a.model,"--runtime-dir",effective,"--config",out/"deployment.json",
                 "--source-index",STREAM/"source-identities.json","--prepared-root",prepared,"--tts-index",TASK_AUDIO,
                 "--data-root",PACKAGE/"dataset/data","--output-root",out/"raw","--lane-count","1","--lane-index","0","--only",*ids]
    else:
        args += [native.WORKER,"--config",out/"deployment.json","--inventory",STREAM/"source-identities.json",
                 "--adapter-root",effective,"--prepared",prepared,"--data",PACKAGE/"dataset/data","--run",out/"raw","--scorer",native.NATIVE_SCORER,
                 "--max-attempts","2","--only",*ids]
    return list(map(str,args)),effective


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--condition",choices=sorted(REMINDERS|{"Q-structured",STRICT}),required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--caseids-file",type=Path)
    group.add_argument("--all",action="store_true")
    p.add_argument("--config",type=Path)
    p.add_argument("--python")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--run-root",type=Path)
    a=p.parse_args()
    require((ABL/a.model/a.condition/"rerun-model.txt").is_file(),"This model/condition was not in the experiment")
    src,base_prov,batch=native.vendor(a.model)
    prov=load(HERE/"provenance.json")
    require(sha(ABL/"conditions.json")==prov["conditions_sha256"] and sha(TASK_AUDIO)==prov["task_audio_index_sha256"],"Changed condition inputs")
    for f,record in prov["runtime_files"].items():
        require(sha(HERE/f)==record["file_sha256"],"Changed archived ablation code: "+f)
    conditions=load(ABL/"conditions.json")
    a.caseids_file=a.caseids_file or ABL/a.model/a.condition/"rerun-model.txt"
    rows=native.selected(a,batch)
    cfg=load(a.config or STREAM/"native/configs"/(a.model+".json"))
    require(cfg["model_id"]==a.model and cfg["adapter_module"]==base_prov["adapter_module"],"Wrong model configuration")
    key=dict(dataset_version=native.VERSION,model=a.model,condition=a.condition,selected_ids=[r["id"] for r in rows],
             prepared_index_sha256=sha(STREAM/"source-identities.json"),base_runtime=base_prov["files"],source_sha256=sha(HERE/"provenance.json"),
             config_sha256=native.stable_digest(cfg))
    report=dict(dataset_version=native.VERSION,model=a.model,condition=a.condition,cases=len(rows),selected_sha256=native.stable_digest(key["selected_ids"]),
                boundary={"Q-structured":"stored structured SYSTEM and query","R1":"native one-second task reminder",
                          "R5":"native five-second task reminder","N1":"native one-second generic reminder",
                          STRICT:"native empty text system, official PCM loader and source phase"}[a.condition])
    if not a.execute:
        with tempfile.TemporaryDirectory(prefix="r3-native-arm-") as tmp:
            ready(a,conditions,rows,cfg,src,Path(tmp))
        print(json.dumps(report,ensure_ascii=False))
        return
    require(a.run_root is not None,"Provide --run-root for a new run")
    out=a.run_root.resolve()/native.VERSION/a.model/a.condition/native.stable_digest(key)[:20]
    with native.run_lock(out):
        if not (out/"binding.json").exists():
            require(not (out/"raw").exists(),"Cannot reuse unbound raw outputs")
        same_json(out/"binding.json",key)
        args,cwd=ready(a,conditions,rows,cfg,src,out)
    print(json.dumps(dict(report,run=str(out)),ensure_ascii=False),flush=True)
    subprocess.run(args,cwd=cwd,check=True)


if __name__=="__main__":
    main()
