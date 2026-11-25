"""Verify the planned and actual outbound Qwen Streaming application messages."""
import base64
from collections import Counter
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from audit import lines, qbase, require
from check_qwen38_ablation_inputs import BASE, LEGACY, LEGACY_SHA, MAPPING, MODEL
from qwen38_session_settings import mismatches, verify_echoes
import resume

ENDPOINT=resume.profile(MODEL,"qwen-current")[0]
WIRE={"audio":"input_audio_buffer.append","video":"input_image_buffer.append","reminder":"conversation.item.create"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compiler(arm):
    require(resume.fsha(LEGACY)==LEGACY_SHA,"legacy source identity changed")
    native=resume.transport("qwen")
    native.ENDPOINT=ENDPOINT
    scope={"__file__":str(LEGACY),"__name__":"original_qwen_reminders"}
    exec(compile(LEGACY.read_bytes(),str(LEGACY),"exec"),scope)
    mapping=resume.read(MAPPING)
    runner=scope["build_run_case"](native,arm,mapping,MAPPING,BASE/"native/qwen.py")
    # The native runner invokes this check after session.updated and before media playback.
    native_verify=native.verify_session
    def verify_session(requested,returned):
        outcome=native_verify(requested,returned)
        drift=mismatches(returned)
        if drift:
            return {**outcome,"passed":False,"mismatches":sorted(set(outcome["mismatches"]+drift))}
        return {**outcome,"provider_session_settings":verify_echoes([{"type":"session.updated","session":returned}])}
    runner.__globals__["verify_session"]=verify_session
    return native,scope,mapping,runner


def expected_messages(native,scope,mapping,runner,arm,cid,b0row,plan):
    manifest,old=resume.verify_input(resume.PREP,cid,b0row)
    instructions,design=scope["render_instructions"](old,manifest,cid,arm,mapping)
    require(sha(instructions.encode())==plan["system_sha256"] and b0row["media_identity_sha256"]==plan["source_media_identity_sha256"],"plan drift")
    scheduled=runner.__globals__["schedule"](manifest)
    require([x for x in scheduled if x["kind"]!="reminder"]==native.schedule(manifest),"media schedule drift")
    require([x["send_due_s"] for x in scheduled if x["kind"]=="reminder"]==plan["reminder_ticks_seconds"],"tick list drift")
    root=resume.PREP/cid
    pcm=(root/manifest["audio_file"]).read_bytes() if manifest["audio_present"] else b""
    expected=[dict(type="session.update",exact=native.session_config(MODEL,instructions))]
    for row in scheduled:
        kind=row["kind"]
        entry={"type":WIRE[kind],"source_time_s":row["send_due_s"]}
        if kind=="reminder":
            entry["text"]=row["text"]
        else:
            raw=(pcm[2*row["sample_start"]:2*row["sample_end"]] if kind=="audio" else (root/row["file"]).read_bytes())
            entry.update(payload_sha256=sha(raw),payload_bytes=len(raw))
        expected.append(entry)
    return manifest,instructions,expected


def wire_match(event, expected):
    require(expected["type"]==event["type"],"unplanned application event or changed order")
    if "exact" in expected:
        require(event==expected["exact"],"session setup mismatch")
        return {"session_sha256":sha(json.dumps(event,ensure_ascii=False,sort_keys=True).encode())}
    if "text" in expected:
        item=event["item"]
        require(item["type"]=="message" and item["role"]=="user" and item.get("id") and event.get("event_id") and
                item["content"]==[{"type":"input_text","text":expected["text"]}],"reminder payload mismatch")
        return {"text":expected["text"],"item_id":item["id"],"event_id":event["event_id"]}
    name="audio" if event["type"]==WIRE["audio"] else "image"
    data=base64.b64decode(event[name],validate=True)
    require(bool(data) and len(data)==expected["payload_bytes"] and sha(data)==expected["payload_sha256"],"actual media differs from frozen source")
    return {"payload_sha256":expected["payload_sha256"],"payload_bytes":len(data),"event_id":event["event_id"]}


class WireConnection:
    def __init__(self,native_ws,expected,ledger):
        self.ws=native_ws
        self.expected=expected
        self.cursor=0
        self.ledger=ledger

    def __aiter__(self):
        return self.ws.__aiter__()

    async def send(self,payload):
        event=json.loads(payload)
        require(self.cursor<len(self.expected),"extra outgoing event")
        wanted=self.expected[self.cursor]
        checked=wire_match(event,wanted)
        before=time.monotonic()
        await self.ws.send(payload)
        after=time.monotonic()
        row=dict(index=self.cursor,type=event["type"],source_time_s=wanted.get("source_time_s"),
                 sent_start_monotonic=before,sent_complete_monotonic=after,**checked)
        print(json.dumps(row,ensure_ascii=False),file=self.ledger,flush=True)
        self.cursor+=1


def bind(runner, original_ws, expected, ledger):
    @asynccontextmanager
    async def connect(endpoint,**kwargs):
        require(endpoint==ENDPOINT+"?model="+MODEL and kwargs["proxy"] is None,"unexpected route or proxy")
        async with original_ws.connect(endpoint,**kwargs) as ws:
            yield WireConnection(ws,expected,ledger)
    runner.__globals__["websockets"]=SimpleNamespace(connect=connect,__version__=original_ws.__version__)


def acceptance(attempt, manifest, instructions, expected, wire, arm):
    spec=resume.read(attempt/"input-spec.json")
    sent,raw,_=qbase(attempt,MODEL,instructions,spec,ENDPOINT)
    settings=verify_echoes(raw)
    status=resume.read(attempt/"status.json")
    transport=resume.read(attempt/"transport-result.json")
    require(status["status"]==transport["status"]=="completed" and not any(r.get("type")=="error" or r.get("code") for r in raw),"native result incomplete")
    require(len(wire)==len(expected)==len(sent),"actual wire, post-send ledger and planned counts differ")
    require([r["index"] for r in wire]==list(range(len(expected))),"wire sequence mismatch")
    scheduled=spec["schedule"]
    require(len(scheduled)==len(expected)-1,"input schedule mismatch")
    for e,w,r in zip(expected[1:],wire[1:],sent[1:]):
        require(w["type"]==e["type"] and r["send_due_s"]==w["source_time_s"]==e["source_time_s"] and
                0<=r["send_completed_media_s"]-r["send_due_s"]<=2 and
                r["media_s"]<=r["send_completed_media_s"],"wire timing mismatch")
        if e["type"]==WIRE["reminder"]:
            require(e["text"]==w["text"]==r["text"] and w["item_id"]==r["item_id"] and w["event_id"]==r["event_id"],"reminder send receipt mismatch")
        else:
            require(r["event_id"]==w["event_id"] and r["event_type"]==w["type"] and
                    w["payload_bytes"]==e["payload_bytes"] and w["payload_sha256"]==e["payload_sha256"],"native media mismatch")
    reminder=[w for w in wire if w["type"]==WIRE["reminder"]]
    confirmations=[e for e in raw if e.get("type")=="conversation.item.created" and
            e.get("item",{}).get("role")=="user" and any(p.get("type")=="input_text" for p in e["item"].get("content",[]))]
    recorded=resume.read(attempt/"runtime-reminders.json")
    require(recorded["accepted"]==recorded["scheduled_sent"]==len(reminder)==len(confirmations) and
            not recorded["rejected"] and not recorded["unconfirmed"],"native reminder acknowledgements incomplete")
    require(len({x["event_id"] for x in confirmations})==len(confirmations),"duplicate provider acknowledgement")
    ack_ids=set()
    for index,(w,row) in enumerate(zip(reminder,recorded["items"])):
        ack=row["provider_ack"]
        require(ack in confirmations and row["acknowledgement"]=="accepted" and row["event_id"]==w["event_id"] and
                ack["event_id"] not in ack_ids and ack["item"]["content"]==[{"type":"input_text","text":w["text"]}],
                "ambiguous or incorrect native reminder acknowledgement")
        if row["acknowledgement_match"]=="native_item_id":
            require(ack["item"]["id"]==w["item_id"],"native item ID mismatch")
        else:
            require(row["acknowledgement_match"]=="ordered_exact_user_text_echo" and
                    ack["event_id"]==confirmations[index]["event_id"],"ordered rewritten-ID text ack mismatch")
        ack_ids.add(ack["event_id"])
    require(transport["sent_audio_samples"]==manifest["audio_samples"] and transport["sent_frames"]==len(manifest["frames"])
            and status["observation_completed_media_s"]>=manifest["duration_sec"]+2,"native media/observation interval incomplete")
    snap=resume.read(attempt/"runtime-source.json")
    require(sha(snap["source"].encode())==LEGACY_SHA and sha(snap["native_source"].encode())==resume.fsha(BASE/"native/qwen.py"),"runtime source drift")
    require(bool(reminder)==(arm!="Q-structured" and manifest["duration_sec"]>(5 if arm=="R5" else 1)),"unexpected reminder occurrence")
    return dict(wire_counts=dict(Counter(w["type"] for w in wire)),acknowledged_reminders=len(reminder),
                provider_session_settings=settings,
                session_updated_model=MODEL,temperature=0,seed_requested=0,turn_detection="semantic_vad",
                manual_response_create_count=0,manual_commit_count=0,received_events=len(raw),
                observed_duration_with_tail_s=status["observation_completed_media_s"])
