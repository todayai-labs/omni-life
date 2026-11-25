#!/usr/bin/env python3
"""Native Qwen input-design ablations; source AV and baseline settings unchanged."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import getpass
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

MODEL = "qwen3.5-omni-flash-realtime"
ARMS = ("B0", "Q-structured", "R1", "R5", "N1")
BASELINE = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parents[1]
PREPARED = BASELINE.parent / "gemini-live-five-actions-20261013/assets/prepared"
MAPPING = BASELINE.parent / "streaming-five-actions-20261012/ablations/results/input-design-full-20261013/records/design/structured-queries-final.json"
NATIVE = BASELINE / "code/run_streaming.py"
PROBES = ("elder_stand_f_pos", "deaf_sign_home", "orbit_leash_11")


def load_native(path: Path):
    spec = importlib.util.spec_from_file_location("qwen_native_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def query_text(manifest):
    query = manifest.get("query", [])
    if len(query) != 1 or float(query[0].get("start_time_sec", 0)) != 0:
        raise ValueError("Exactly one initial source query is required.")
    return query[0]["text"]


def read_mapping(path):
    rows = read_json(path)["cases"]
    mapping = {row["id"]: row for row in rows}
    if len(rows) != 400 or len(mapping) != 400:
        raise ValueError("Approved mapping must contain 400 unique IDs.")
    if any(not row.get("eligible_for_inference") for row in rows):
        raise ValueError("Mapping contains an unapproved structured query.")
    return mapping


def render_instructions(original, manifest, case, arm, mapping):
    query = query_text(manifest)
    row = mapping[case]
    if row["original_query"] != query:
        raise ValueError(f"Approved query does not match prepared source: {case}")
    model_query = row["structured_query"] if arm == "Q-structured" else query
    if arm == "Q-structured":
        marker = "当前用户交给你的任务：\n" + query
        if original.count(marker) != 1:
            raise ValueError("Initial task boundary must occur exactly once in SYSTEM.txt.")
        original = original.replace(marker, "当前用户交给你的任务：\n" + model_query, 1)
    return original, {"arm": arm, "original_query": query, "model_query": model_query,
                      "initial_query_changed": arm == "Q-structured"}


def reminder_schedule(manifest, arm):
    duration = float(manifest["duration_sec"])
    query = query_text(manifest)
    if arm not in ("R1", "R5", "N1"):
        return []
    step = 5 if arm == "R5" else 1
    times = range(step, math.ceil(duration), step)
    text = "继续执行最初任务；仅在原任务条件满足时回应，否则不输出。" if arm == "N1" else "提示：你的任务是 " + query
    return [{"kind": "reminder", "send_due_s": float(t), "text": text,
             "reminder_index": index} for index, t in enumerate(times) if t < duration]


def native_text_event(text):
    return {"type": "conversation.item.create", "event_id": "evt-" + uuid.uuid4().hex,
            "item": {"id": "item-" + uuid.uuid4().hex, "type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": text}]}}


async def dispatch_reminder(ws, item, origin, reminders, write_event, input_file, native):
    event = native_text_event(item["text"])
    before = time.monotonic() - origin
    record = {**item, "event_id": event["event_id"], "item_id": event["item"]["id"],
              "media_s": before, "sent_utc": native.utc(), "event": event,
              "acknowledgement": "pending", "provider_ack": None, "provider_rejection": None}
    reminders.append(record)
    await ws.send(json.dumps(event, ensure_ascii=False))
    after = time.monotonic() - origin
    record.update(send_completed_media_s=after, send_duration_s=after - before,
                  input_drift_s=before - item["send_due_s"])
    write_event(input_file, dict(record))
    if after - item["send_due_s"] > 2:
        raise RuntimeError("Causal reminder send drift exceeded 2 seconds.")


def observe_reminder(event, media_s, received_at, reminders):
    kind = event.get("type")
    if kind == "conversation.item.created":
        item = event.get("item", {})
        matches = [row for row in reminders if row["item_id"] == item.get("id")]
        method = "native_item_id"
        if not matches and item.get("role") == "user":
            texts = [part.get("text") for part in item.get("content", []) if part.get("type") in ("input_text", "text")]
            matches = [row for row in reminders if row["acknowledgement"] == "pending" and row["text"] in texts][:1]
            method = "ordered_exact_user_text_echo"
        for row in matches:
            row.update(acknowledgement="accepted", provider_ack=event,
                       ack_media_s=media_s, ack_utc=received_at, acknowledgement_match=method)
    elif kind == "error":
        error = event.get("error", {})
        related_id = error.get("event_id") or event.get("client_event_id")
        text = json.dumps(error, ensure_ascii=False).lower()
        for row in reminders:
            if related_id == row["event_id"] or (
                    not related_id and row["acknowledgement"] == "pending" and
                    "conversation.item.create" in text):
                row.update(acknowledgement="rejected", provider_rejection=event,
                           rejection_media_s=media_s, rejection_utc=received_at)


def finish_reminders(attempt, result, reminders, native, credential):
    rejected = [row for row in reminders if row["acknowledgement"] == "rejected"]
    accepted = sum(row["acknowledgement"] == "accepted" for row in reminders)
    native.save(attempt / "runtime-reminders.json", {"scheduled_sent": len(reminders),
        "accepted": accepted, "rejected": len(rejected), "unconfirmed": len(reminders) - accepted - len(rejected),
        "items": reminders, "text_delivery": "native conversation.item.create; no response.create or audio commit"}, credential)
    result.update(runtime_reminders_sent=len(reminders), runtime_reminders_accepted=accepted,
                  runtime_reminders_rejected=len(rejected),
                  runtime_reminders_unconfirmed=len(reminders) - accepted - len(rejected))
    if rejected:
        result.update(status="failed", failure_category="runtime_text_input_rejected", retryable=False,
                      arm_unsupported=True)
    elif result["status"] == "completed" and len(reminders) > accepted:
        result.update(status="failed", failure_category="runtime_text_ack_unconfirmed", retryable=False,
                      runtime_text_needs_attention=True, error="Native runtime text acknowledgement incomplete.")


def compatible_attempt(attempt, case, prepared, arm, mapping):
    """Never reuse a prototype with a different prompt or reminder condition."""
    try:
        recorded = read_json(Path(attempt) / "input-spec.json")
        source = prepared / case
        manifest = read_json(source / "manifest.json")
        original = (source / manifest.get("system_file", "SYSTEM.txt")).read_text(encoding="utf-8")
        instructions, _ = render_instructions(original, manifest, case, arm, mapping)
        return (recorded.get("arm") == arm and recorded.get("prepared") == str(source)
                and recorded.get("instructions") == instructions and recorded.get("query") == manifest["query"]
                and recorded.get("runtime_reminders_scheduled") == reminder_schedule(manifest, arm))
    except (OSError, ValueError, KeyError):
        return False


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Baseline source anchor changed: {old[:100]!r}")
    return source.replace(old, new, 1)


def build_run_case(native, arm, mapping, mapping_path, native_path):
    """Small fail-closed source fork: keep the native receiver/parser/media code."""
    source = inspect.getsource(native.run_case)
    source = replace_once(source, 'output_root / "models" / model / scope / "cases" / case',
                          'output_root / "models" / model / ARM / "run" / "cases" / case')
    source = replace_once(source, '    if scope == "run":\n', '    if False:  # no cross-condition probe reuse\n')
    source = replace_once(source, '            if previous.get("status") == "completed":\n',
        '            if previous.get("status") == "completed" and compatible_attempt(status_file.parent, case, prepared, ARM, MAPPING_ROWS):\n')
    source = replace_once(source, '    parser = ResponseParser()\n', '    parser = ResponseParser()\n    reminders = []\n')
    old = '        instructions = (source / manifest.get("system_file", "SYSTEM.txt")).read_text(encoding="utf-8")\n'
    source = replace_once(source, old, old + '        instructions = render_instructions(instructions, manifest, case, ARM, MAPPING_ROWS)[0]\n')
    source = replace_once(source, '                        parser.feed(event, media_s, received_at)\n',
                          '                        parser.feed(event, media_s, received_at)\n                        observe_reminder(event, media_s, received_at, reminders)\n')
    source = replace_once(source, '                if item["kind"] == "audio":\n                    chunk =',
        '                if item["kind"] == "reminder":\n                    await dispatch_reminder(ws, item, origin, reminders, write_event, input_file, NATIVE_MODULE)\n                    continue\n                if item["kind"] == "audio":\n                    chunk =')
    source = replace_once(source, '        parsed = parser.export()\n',
                          '        finish_reminders(attempt, result, reminders, NATIVE_MODULE, credential)\n        parsed = parser.export()\n')

    def schedule(manifest):
        events = native.schedule(manifest) + reminder_schedule(manifest, arm)
        return sorted(events, key=lambda row: (row["send_due_s"], {"audio": 0, "video": 1, "reminder": 2}[row["kind"]]))

    def save(path, value, credential=""):
        if path.name == "input-spec.json":
            manifest = read_json(value["source_manifest"])
            original = (Path(value["prepared"]) / manifest.get("system_file", "SYSTEM.txt")).read_text(encoding="utf-8")
            _, design = render_instructions(original, manifest, value["case"], arm, mapping)
            value = {**value, **design, "query_delivery": "session.instructions; native runtime text for R1/R5/N1",
                     "structured_mapping": str(mapping_path), "native_source": str(native_path),
                     "runtime_reminders_scheduled": reminder_schedule(manifest, arm), "probe_scope": None}
        elif path.name == "runtime-source.json":
            value = {**value, "native_source_file": str(native_path),
                     "native_source": native_path.read_text(encoding="utf-8"), "executed_run_case": source}
        native.save(path, value, credential)

    namespace = dict(native.__dict__)
    namespace.update(__file__=str(Path(__file__).resolve()), ARM=arm, MAPPING_ROWS=mapping,
        NATIVE_MODULE=native, render_instructions=render_instructions, observe_reminder=observe_reminder,
        dispatch_reminder=dispatch_reminder, finish_reminders=finish_reminders,
        compatible_attempt=compatible_attempt, schedule=schedule, save=save)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["run_case"]


def condition_root(root, arm):
    return root / "models" / MODEL / arm / "run"


def full_probe_guard(root, arm):
    if arm == "B0":
        return
    for receipt in condition_root(root, arm).glob("cases/*/probe-receipt.json"):
        row = read_json(receipt)
        if row.get("source_audio_present") and row.get("status") == "completed" and (
                arm == "Q-structured" or row.get("runtime_reminders_accepted", 0) > 0):
            return
    raise ValueError(f"{arm}: complete a source-AV probe and verify native runtime text acknowledgement before full run.")


async def run(args, credential):
    native = load_native(args.native_source)
    mapping = read_mapping(args.mapping)
    arms = args.arms.split(",")
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Use unique supported conditions: " + ",".join(ARMS))
    if not 1 <= args.concurrency <= 2:
        raise ValueError("Baseline stable global concurrency cap is 2.")
    if args.caseids and not args.probe:
        raise ValueError("--caseids requires --probe; each formal condition must cover all 400 source cases.")
    all_cases = sorted(path.parent.name for path in args.prepared.glob("*/manifest.json"))
    if len(all_cases) != 400 or set(all_cases) != set(mapping):
        raise ValueError("Prepared source and approved mapping must cover identical 400 IDs.")
    cases = args.caseids.split(",") if args.caseids else list(PROBES) if args.probe else all_cases
    if len(set(cases)) != len(cases) or any(case not in all_cases for case in cases):
        raise ValueError("Case IDs must be unique members of the canonical 400.")
    for case in all_cases:
        manifest = read_json(args.prepared / case / "manifest.json")
        if query_text(manifest) != mapping[case]["original_query"]:
            raise ValueError(f"Source query differs from approved mapping: {case}")
    if not args.probe:
        for arm in arms:
            full_probe_guard(args.root, arm)
    functions = {arm: build_run_case(native, arm, mapping, args.mapping, args.native_source) for arm in arms if arm != "B0"}
    native.save(args.root / "protocol.json", {"model": MODEL, "arms": list(ARMS), "case_count": 400,
        "source": str(args.prepared), "data_provenance": str(args.data) if args.data else None,
        "sp_provenance": str(args.sp) if args.sp else None, "source_policy": "Reuse original prepared SYSTEM.txt and media; never re-encode or synthesize audio.",
        "mapping": str(args.mapping), "baseline_root": str(args.baseline_root), "native_source": str(args.native_source),
        "endpoint": native.ENDPOINT, "global_concurrency_cap": 2, "playback_rate": 1, "tail_seconds": 2,
        "audio_chunk_seconds": .2, "task_tts": False, "padding": False,
        "generation_trigger": "provider-native semantic VAD; no manual response.create",
        "native_text_event": "conversation.item.create user input_text", "reminder_text_n1": "继续执行最初任务；仅在原任务条件满足时回应，否则不输出。",
        "repeat_text_prefix": "提示：你的任务是 ", "probe_successes_are_formal_cases": True,
        "audio_less_cases": "actually run each condition; no cross-condition failure reuse"}, credential)
    native.save(args.root / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "websockets": native.websockets.__version__, "native_source": str(args.native_source),
        "argv": sys.argv, "route": "official WebSocket; proxy=None"}, credential)
    semaphore = asyncio.Semaphore(args.concurrency)
    paused = set()
    stop = asyncio.Event()
    selected = {arm: {} for arm in arms}
    active = {}
    cooling_until = 0.0
    queue_log = (args.root / "queue-runlog.jsonl").open("a", encoding="utf-8")

    def log(event, **fields):
        queue_log.write(json.dumps(native.redact({"utc": native.utc(), "event": event, **fields}, credential), ensure_ascii=False) + "\n")
        queue_log.flush()

    def progress(state="running"):
        models = {}
        for arm in arms:
            rows = list(selected[arm].values())
            completed = sum(row.get("status") == "completed" for row in rows)
            terminal = sum(bool(row.get("terminal")) for row in rows)
            models[arm] = {"planned": len(cases), "terminal": terminal, "successful": completed,
                          "failed": terminal - completed, "pending": len(cases) - terminal,
                          "active": [case for (condition, case) in active if condition == arm]}
            folder = condition_root(args.root, arm)
            folder.mkdir(parents=True, exist_ok=True)
            native.save(folder / "progress.json", {"utc": native.utc(), **models[arm]}, credential)
        native.save(args.root / "progress.json", {"utc": native.utc(), "pid": os.getpid(), "status": state,
            "model": MODEL, "probe": args.probe, "concurrency": args.concurrency, "conditions": models,
            "active_sessions": list(active.values()), "paused_arms": sorted(paused)}, credential)

    async def worker(arm, case):
        nonlocal cooling_until
        folder = condition_root(args.root, arm) / "cases" / case
        folder.mkdir(parents=True, exist_ok=True)
        selection = folder / "final-selection.json"
        async with semaphore:
            if stop.is_set() or arm in paused:
                return
            if shutil.disk_usage(args.root).free < 5 * 1024**3:
                log("storage_pause", free_bytes=shutil.disk_usage(args.root).free)
                stop.set()
                return
            if selection.exists():
                old = read_json(selection)
                if old.get("terminal") and (arm == "B0" or compatible_attempt(old.get("attempt", ""), case, args.prepared, arm, mapping)):
                    selected[arm][case] = old
                    progress()
                    return
            if arm == "B0":
                path = args.baseline_root / "models" / MODEL / "run/cases" / case / "final-selection.json"
                old = read_json(path)
                if not old.get("terminal"):
                    raise ValueError(f"B0 source is not terminal: {case}")
                result = {**old, "model": MODEL, "case": case, "arm": arm, "terminal": True,
                          "reference_only": True, "source_selection": str(path), "reused_identical_condition": True}
                native.save(selection, result, credential)
                selected[arm][case] = result
                progress()
                return
            active[arm, case] = {"arm": arm, "case": case, "started_utc": native.utc()}
            progress()
            result = None
            try:
                prior = [read_json(path) for path in folder.glob("attempt-*/status.json")]
                count = sum(row.get("status") == "failed" and native.classify_failure(row)[0] == "native_audio_buffer_state_error" for row in prior)
                for retry in range(max(1, 3 - count)):
                    delay = cooling_until - time.monotonic()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if stop.is_set() or arm in paused:
                        return
                    result = await functions[arm](MODEL, case, credential, args.prepared, args.root, "run")
                    if result["status"] == "skipped_completed":
                        result = {**read_json(Path(result["attempt"]) / "status.json"), "reused_same_condition_success": True}
                    if result["status"] != "failed":
                        break
                    category, retryable = native.classify_failure(result)
                    if result.get("arm_unsupported"):
                        category, retryable = "runtime_text_input_rejected", False
                        paused.add(arm)
                    elif result.get("runtime_text_needs_attention"):
                        category, retryable = "runtime_text_ack_unconfirmed", False
                        paused.add(arm)
                    result.update(failure_category=category, retryable=retryable)
                    log("attempt_failed", arm=arm, case=case, category=category, attempt=result["attempt"])
                    if category in ("credential_or_permission", "configuration_rejected", "local_transport_unavailable"):
                        stop.set()
                        break
                    if not retryable or retry + 1 == max(1, 3 - count):
                        if category in ("temporary_transport_or_server", "rate_limit"):
                            stop.set()
                        break
                    if category == "rate_limit":
                        cooling_until = max(cooling_until, time.monotonic() + 30 * 2**retry)
                    else:
                        await asyncio.sleep(5 * 2**retry)
                terminal = result.get("failure_category") not in ("temporary_transport_or_server", "rate_limit", "local_transport_unavailable", "credential_or_permission", "configuration_rejected", "runtime_text_ack_unconfirmed")
                result = {**result, "arm": arm, "terminal": terminal, "selected_utc": native.utc()}
                if selection.exists():
                    history = folder / "selection-history"
                    history.mkdir(exist_ok=True)
                    native.save(history / ("before-recovery-" + uuid.uuid4().hex + ".json"), read_json(selection), credential)
                native.save(selection, result, credential)
                selected[arm][case] = result
                if args.probe:
                    native.save(folder / "probe-receipt.json", {**result, "actual_probe": True}, credential)
                log("case_selected", arm=arm, case=case, status=result["status"], terminal=terminal)
                print(json.dumps({"arm": arm, "case": case, "status": result["status"], "terminal": terminal}), flush=True)
            finally:
                active.pop((arm, case), None)
                progress()

    log("queue_start", arms=arms, cases=cases, concurrency=args.concurrency, probe=args.probe)
    try:
        await asyncio.gather(*(worker(arm, case) for case in cases for arm in arms))
        progress("needs_attention" if stop.is_set() or paused else "finished")
        log("queue_end", stopped=stop.is_set(), paused_arms=sorted(paused))
    finally:
        queue_log.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=ROOT)
    cli.add_argument("--prepared", type=Path, default=PREPARED)
    cli.add_argument("--data", type=Path, help="Source dataset provenance, not a media override.")
    cli.add_argument("--sp", type=Path, help="Final source SP provenance; prepared SYSTEM.txt remains the input.")
    cli.add_argument("--mapping", type=Path, default=MAPPING)
    cli.add_argument("--native-source", type=Path, default=NATIVE)
    cli.add_argument("--baseline-root", type=Path, default=BASELINE)
    cli.add_argument("--arms", default="Q-structured,R1,R5,N1")
    cli.add_argument("--caseids", help="Comma-separated canonical IDs. Probe defaults to the three baseline probes.")
    cli.add_argument("--concurrency", type=int, default=2)
    cli.add_argument("--probe", action="store_true")
    cli.add_argument("--credential-stdin", action="store_true")
    cli.add_argument("--detach", action="store_true")
    args = cli.parse_args()
    credential = sys.stdin.readline().strip() if args.credential_stdin else getpass.getpass("Qwen API credential: ")
    if not credential:
        cli.error("A nonempty credential is required.")
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "evaluation-queue.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        cli.error("A queue is already running; successful cases must not be duplicated.")
    if args.detach:
        pid = os.fork()
        if pid:
            print(json.dumps({"status": "launched", "pid": pid, "model": MODEL, "arms": args.arms.split(","), "concurrency": args.concurrency}), flush=True)
            os._exit(0)
        os.setsid()
        logs = args.root / "logs"
        logs.mkdir(exist_ok=True)
        output = (logs / "evaluation-queue.log").open("a", buffering=1)
        os.dup2(output.fileno(), 1)
        os.dup2(output.fileno(), 2)
        sys.stdout = sys.stderr = output
        null = open(os.devnull)
        os.dup2(null.fileno(), 0)
    caffeinate = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-s", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if sys.platform == "darwin" else None
    native = load_native(args.native_source)
    if (args.root / "launch.json").exists():
        history = args.root / "logs/launch-history"
        history.mkdir(parents=True, exist_ok=True)
        native.save(history / ("launch-" + uuid.uuid4().hex + ".json"), read_json(args.root / "launch.json"), credential)
    native.save(args.root / "launch.json", {"pid": os.getpid(), "started_utc": native.utc(),
        "command": sys.argv, "concurrency": args.concurrency, "arms": args.arms.split(","),
        "probe": args.probe, "caffeinate_pid": caffeinate.pid if caffeinate else None}, credential)
    try:
        asyncio.run(run(args, credential))
    finally:
        if caffeinate and caffeinate.poll() is None:
            caffeinate.terminate()
        lock.close()


if __name__ == "__main__":
    main()
