"""Export a bound GPU ablation run into the pending Full Correct semantic-review format."""
import argparse
from pathlib import Path
from run_arm import HERE, ABL, PACKAGE, STREAM, STRICT, TASK_AUDIO, native, load, sha, write, require, module
from export_native import record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    binding=load(a.run/"binding.json")
    model,condition=binding["model"],binding["condition"]
    require(binding["dataset_version"]==native.VERSION and a.run.name==native.stable_digest(binding)[:20] and
            binding["source_sha256"]==sha(HERE/"provenance.json") and binding["prepared_index_sha256"]==sha(STREAM/"source-identities.json"),
            "Unbound or changed ablation inputs")
    _,prov,batch=native.vendor(model)
    require(binding["base_runtime"]==prov["files"],"Different base model runtime")
    strict=condition==STRICT
    if strict:
        code=HERE/"strict"/model
        driver=module("r3_export_strict",code/"run_full.py")
        helper=module("r3_export_tts",code/"tts_phase.py")
    raw=a.run/"raw"
    records={}
    rules=load(PACKAGE/"dataset/protocol/streaming-scoring.json")["cases"]
    for cid in binding["selected_ids"]:
        source=STREAM/"prepared"/cid
        if condition=="Q-structured":
            source=a.run/"prepared-structured"/cid
        man=load(source/"manifest.json")
        # The original strict worker also records infrastructure attempts. Honor its explicit success field.
        attempts=sorted((raw/"cases"/cid).glob("attempt-*"))
        if strict:
            attempts=sorted({*attempts,*raw.glob("lanes/*/cases/"+cid+"/attempt-*")})
        for f in attempts:
            if not (f/"case-status.json").is_file():
                continue
            status=load(f/"case-status.json")
            if status.get("state" if strict else "status")!="success":
                continue
            result=load(f/"transport-result.json")
            spec=load(f/"input-spec.json")
            if strict:
                require(spec["condition"]==STRICT and not spec["concrete_task_text_sent"] and spec["effective_generic_system"]=="" and spec["source_manifest"]==man,
                        "Strict input receipt differs: "+cid)
                pcm,entry=helper.load_task_audio(TASK_AUDIO,cid,man)
                require(entry["wav_sha256"]==spec["task_audio"]["wav_sha256"] and driver.coverage_error(result,man,pcm) is None,
                        "Incomplete original TTS coverage: "+cid)
            else:
                require(spec["manifest_sha256"]==sha(source/"manifest.json"),"Wrong ablation source: "+cid)
            records[cid]=record(f,result,cid,man["duration_sec"],rules[cid]["annotation_end_s"],batch,man,rules[cid].get("media_end_s"))
            break
    write(a.out,dict(dataset_version=native.VERSION,model=model,condition=condition,records=records))
    print(f"{len(records)}/{len(binding['selected_ids'])} captures exported")


if __name__=="__main__":
    main()
