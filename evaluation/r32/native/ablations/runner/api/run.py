"""Run selected Qwen/Gemini input-design arms on verified final-version inputs."""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

HERE=Path(__file__).resolve().parent
PACKAGE=Path(os.environ.get("OMNI_LIFE_HANDOFF", HERE.parents[2])).resolve()
AROOT=PACKAGE/"ablations"
ORIGINAL=HERE/"original"
B0=HERE.parents[2]/"stream-runtime/api"
QWEN="qwen3.8-omni-flash-realtime"
GEMINI="gemini-3.8-live"
ARMS=("Q-structured","R1","R5","N1")
PINS={"qwen_arm.py":"3adefed2008ae55d280ef40c6c24fdc35b6a853f464d9f94624ad1d9726241b0",
      "qwen_arm_wire.py":"3321ea45f76716cc93168fa1fc8f01e13f96dade43716113e060017a2c5d0c88",
      "gemini_arm.py":"592fd38f257cc3209f169b984d3df2af429e7ded6e69d9f8d8fa08ccb515f3ea",
      "gemini_native.py":"08ef02a771ee2af85055152dc29d0afa61db844032e229216e8ae0d741d1deae",
      "gemini_wire.py":"0cfafcb829de9a835724f1a6cd18f5075c9c8e6c65607a6fd07c4c77d361ff1f",
      "reminder_clock.py":"20a52068ab20c14bd939546d7dec70d9fb04a22e051621ea33c89cbb03714128"}
CONDITIONS="7ffac9c64ff9d56981455944da22376693208581a007f75aa8f5bdc2dbd3a447"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base=load("final_native_base",B0/"run.py")
require=base.require


def let(name,**attrs):
    mod=ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def planned(arm,cond,man):
    task=man["query"][0]["text"]
    require(task==cond["task_zh"],"Ablation task differs from final source")
    if arm=="Q-structured":
        return []
    step=5 if arm=="R5" else 1
    text=cond["generic_reminder"] if arm=="N1" else cond["reminder_text"]
    return [(i,text) for i in range(step,math.ceil(man["duration_sec"]),step)]


def get_qwen_wire(audit,folder,man,sp,row):
    def checked_input(prepared,cid,witness):
        require(prepared==base.RUNTIME/"prepared" and cid==row["case_id"] and witness["media_identity_sha256"]==row["media_identity_sha256"],"Mismatched original wire input")
        return man,sp
    deps={"audit":audit,"resume":let("resume",profile=lambda model,profile:(audit.NEWQ,""),PREP=folder.parent,
                                    read=base.read,fsha=base.digest,verify_input=checked_input),
          "check_qwen38_ablation_inputs":let("check_qwen38_ablation_inputs",BASE=B0,LEGACY=ORIGINAL/"qwen_arm.py",
                                            LEGACY_SHA=PINS["qwen_arm.py"],MAPPING=AROOT/"conditions.json",MODEL=QWEN),
          "qwen38_session_settings":base.module("qwen38_session_settings")}
    with patch.dict(sys.modules,deps):
        return load("qwen_condition_wire",ORIGINAL/"qwen_arm_wire.py")


async def qwen(arm,plan,root,offline,audit,cond,mapping):
    cid,folder,man,sp,binding=plan
    ticks=planned(arm,cond,man)
    native,env,crossed=base.native_transport(QWEN,audit,"qwen-current",offline)
    legacy=load("qwen_original_conditions",ORIGINAL/"qwen_arm.py")
    wire=get_qwen_wire(audit,folder,man,sp,binding)
    runner=legacy.build_run_case(native,arm,mapping,AROOT/"conditions.json",B0/"native/qwen.py")
    instructions=legacy.render_instructions(sp,man,cid,arm,mapping)[0]
    spec=dict(source_media_identity_sha256=binding["media_identity_sha256"],system_sha256=audit.sha(instructions.encode()),
              reminder_ticks_seconds=[i for i,_ in ticks])
    _,_,expected=wire.expected_messages(native,legacy.__dict__,mapping,runner,arm,cid,binding,spec)
    require([(r["source_time_s"],r["text"]) for r in expected if "text" in r]==ticks,"Wrong Qwen reminder wire plan")
    key="offline-placeholder" if offline else os.environ.get(env,"").strip()
    require(key and "\n" not in key,"Missing source credential: "+env)
    logdir=root/"models"/QWEN/arm/"wire"
    logdir.mkdir(parents=True,exist_ok=True)
    logpath=logdir/(cid+".jsonl")
    with logpath.open("x") as log:
        wire.bind(runner,native.websockets,expected,log)
        result=await runner(QWEN,cid,key,folder.parent,root,"conditional")
    target=Path(result["attempt"])
    actual=base.read(target/"input-spec.json")
    require(actual["arm"]==arm and actual["instructions"]==instructions and
            [(r["send_due_s"],r["text"]) for r in actual["runtime_reminders_scheduled"]]==ticks,"Incorrect materialized Qwen arm")
    if offline:
        require(crossed==[True],"Qwen socket boundary missing")
        return result_info(binding,arm,"qwen-current",instructions,ticks)
    if result["status"]!="completed":
        return dict(result_info(binding,arm,"qwen-current",instructions,ticks),terminal_failure=True,attempt=str(target),status=result["status"],error=result.get("error"))
    auditlog=audit.lines(logpath)
    verified=wire.acceptance(target,man,instructions,expected,auditlog,arm)
    horizon=base.read(target/"status.json")["observation_completed_media_s"]
    raw=base.read(target/"parsed-output.json")["responses"]
    capture=base.convert(raw,horizon,False,cid,binding["media_end_s"])
    return dict(result_info(binding,arm,"qwen-current",instructions,ticks),verified=verified,attempt=str(target),capture=capture)


def gemini_inbound(audit, target):
    events = [item["event"] for item in audit.lines(target/"raw-output.jsonl")]
    meaningful = [event for event in events if event.get("serverContent") or event.get("voiceActivity")
                  or any(((event.get("usageMetadata") or {}).get(k) or 0) > 0 for k in ("promptTokenCount","totalTokenCount"))]
    return dict(post_setup_events=len(events), media_processing_events=len(meaningful))


async def gemini(arm,plan,root,offline,audit,cond,mapping):
    cid,folder,man,sp,binding=plan
    ticks=planned(arm,cond,man)
    runner=load("original_gemini_conditions",ORIGINAL/"gemini_arm.py")
    with patch.dict(sys.modules,{"reminder_clock":load("ablation_eof_clock",ORIGINAL/"reminder_clock.py")}):
        wire=load("gemini_conditional_wire",ORIGINAL/"gemini_wire.py")
    # The original selective dispatcher builds the template from the source
    # instructions with precisely one sighting of the unmodified task.
    task=cond["task_zh"]
    require(sp.count(task)==1 and "{{USER_REQUEST}}" not in sp,"Ambiguous original Gemini task span")
    setup,query=runner.setup_for(dict(query=task,structured_query=cond["structured_query"],template=sp.replace(task,"{{USER_REQUEST}}")),arm)
    instructions=setup["systemInstruction"]["parts"][0]["text"]
    require(setup==audit.gsetup(instructions),"Gemini session setting differs")
    native,env,crossed=base.native_transport(GEMINI,audit,"gemini-developer",offline)
    ws=native.websockets
    with patch.dict(sys.modules,{"websockets":ws}):
        helper=load("source_gemini_native",ORIGINAL/"gemini_native.py")
    require(helper.ENDPOINT==audit.OLDG,"Wrong Gemini endpoint")
    helper.STOP=asyncio.Event()
    gate=helper.ConnectionGate(1.0,3)
    key="offline-placeholder" if offline else os.environ.get(env,"").strip()
    require(key and "\n" not in key,"Missing source credential: "+env)
    target=root/"models"/GEMINI/arm/"run/cases"/cid/("attempt-"+uuid4().hex)
    target.mkdir(parents=True)
    await gate.wait(helper.STOP)
    with (target/"actual-outgoing.jsonl").open("x") as log:
        helper.websockets=SimpleNamespace(connect=lambda *a,**kw:wire.ObservedConnection(ws.connect(*a,**kw),log))
        result=await asyncio.wait_for(runner.native_session(key,setup,target,man,folder,arm,query,helper,gate),
                                      timeout=float(man["duration_sec"])+150)
    actual=base.read(target/"input-spec.json")
    require(actual["arm"]==arm and actual["setup"]==setup and [(v["scheduled_media_s"],v["text"]) for v in actual["reminder_schedule"]]==ticks,
            "Incorrect materialized Gemini arm")
    if offline:
        require(crossed==[True],"Gemini socket boundary missing")
        return result_info(binding,arm,"gemini-developer",instructions,ticks)
    if result["status"]!="success":
        return dict(result_info(binding,arm,"gemini-developer",instructions,ticks),terminal_failure=True,attempt=str(target),status=result["status"],error=result.get("error"))
    verified=wire.verify(target,folder,man,setup,ticks)
    inbound=gemini_inbound(audit,target)
    verified.update(inbound)
    horizon=result["latency"]["total_session_seconds"]
    capture=base.convert(result["responses"],horizon,bool(man["audio_present"] and horizon>=binding["media_end_s"]+1.99),cid,binding["media_end_s"])
    transport_only=not inbound["media_processing_events"]
    if transport_only:
        capture["unresolved"].append({"reason":"No post-setup provider evidence of media processing"})
        capture["normalized"]=None
    return dict(result_info(binding,arm,"gemini-developer",instructions,ticks),verified=verified,
                transport_only=transport_only,attempt=str(target),capture=capture)


def result_info(binding,arm,profile,sp,ticks):
    return dict(case_id=binding["case_id"],arm=arm,model=binding["model"],dataset_version=base.VERSION,
                release_manifest_sha256=base.RELEASE,conditions_sha256=CONDITIONS,route=profile,media_identity_sha256=binding["media_identity_sha256"],
                prepared_manifest_sha256=binding["prepared_manifest_sha256"],effective_system_sha256=hashlib.sha256(sp.encode()).hexdigest(),
                scheduled_reminders=len(ticks),reminder_schedule_sha256=hashlib.sha256(json.dumps(ticks,ensure_ascii=False,separators=(",",":")).encode()).hexdigest())


def run(model,arm,chosen,root,offline):
    queue=AROOT/model/arm/"rerun-model.txt"
    audit,plans=base.inputs(model,chosen,queue_path=queue)
    conditions=base.read(AROOT/"conditions.json")
    require(len(conditions)==400 and base.digest(AROOT/"conditions.json")==CONDITIONS,"Wrong final conditions")
    require(not sys.flags.optimize,"The original native verifiers require normal Python assertions")
    for name,expected in PINS.items():
        require(base.digest(ORIGINAL/name)==expected,"Original ablation code differs: "+name)
    mapping={k:dict(original_query=v["task_zh"],structured_query=v["structured_query"]) for k,v in conditions.items()}
    pool=root/"models"/model/arm
    pool.mkdir(parents=True,exist_ok=True)
    results=[]
    with (pool/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for plan in plans:
            cid=plan[0]
            receipt=pool/"receipts"/(cid+".json")
            if receipt.exists():
                old=base.read(receipt)
                require(old["case_id"]==cid and old["model"]==model and old["arm"]==arm and old["release_manifest_sha256"]==base.RELEASE and
                        old["conditions_sha256"]==CONDITIONS and old["prepared_manifest_sha256"]==plan[-1]["prepared_manifest_sha256"],"Existing ablation result differs")
                results.append(old)
                continue
            require(not (pool/"run/cases"/cid).exists() and not (pool/"wire"/(cid+".jsonl")).exists(),
                    "Unverified previous attempt, inspect before resubmission: "+cid)
            fn=qwen if model==QWEN else gemini
            result=asyncio.run(fn(arm,plan,root,offline,audit,conditions[cid],mapping))
            results.append(result)
            if not offline:
                receipt.parent.mkdir(exist_ok=True)
                base.save(receipt,result)
                export(pool,model,arm)
                if result.get("terminal_failure") and any(word in str(result.get("error","")).lower() for word in
                        ("quota","429","401","403","accessdenied","invalid_api_key","unauthorized","resource_exhausted")):
                    raise SystemExit("Provider authorization or quota failure. Receipt retained: "+str(receipt))
        return results


def export(pool,model,arm):
    records={}
    unresolved={}
    for path in sorted((pool/"receipts").glob("*.json")):
        row=base.read(path)
        if row.get("terminal_failure"):
            unresolved[path.stem]="technical_failure"
        elif row["capture"]["normalized"] is None:
            unresolved[path.stem]=row["capture"]["unresolved"]
        else:
            records[path.stem]=row["capture"]["normalized"]
    base.save(pool/"arm-new.json",dict(dataset_version=base.VERSION,release_manifest_sha256=base.RELEASE,model=model,condition=arm,records=records))
    base.save(pool/"unresolved.json",unresolved)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",choices=(QWEN,GEMINI),required=True)
    ap.add_argument("--arm",choices=ARMS,required=True)
    ap.add_argument("--case-ids",nargs="+")
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--report",type=Path)
    args=ap.parse_args()
    if args.execute:
        if args.out is None or args.report: ap.error("--execute requires --out; --report is for offline checking")
        rows=run(args.model,args.arm,args.case_ids,args.out.resolve()/base.VERSION/"ablations",False)
    else:
        with TemporaryDirectory() as temporary:
            rows=run(args.model,args.arm,args.case_ids,Path(temporary),True)
        if args.report:
            base.save(args.report,dict(model=args.model,arm=args.arm,live_connections=0,checked=rows))
    print(json.dumps(dict(model=args.model,arm=args.arm,cases=len(rows),offline=not args.execute,scheduled_reminders=sum(r["scheduled_reminders"] for r in rows),
                          technical_failures=sum(bool(r.get("terminal_failure")) for r in rows))))
