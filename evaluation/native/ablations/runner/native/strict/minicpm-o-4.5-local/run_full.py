#!/usr/bin/env python3
"""Resumable, case-locked TTS-only native AV evaluation; no semantic judge."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter

MODULES = {"minicpm-o-4.5-local": "tts_minicpm", "realtime-venus-omni": "tts_venus"}
RESOURCE_FIELDS = {"physical_gpu", "endpoint", "asr_url"}
sys.dont_write_bytecode = True


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable record: {path}")
    temporary = path.parent / f".{path.name}-{os.getpid()}-{time.time_ns()}.pending"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"Record appeared concurrently; pending evidence retained: {path}")
    # All publishers hold the corresponding case/plan/snapshot lock.
    temporary.rename(path)


def refresh(path, value):
    """Refresh our generated pointer, retaining all prior versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        shutil.copy2(path, history / f"{path.stem}-{time.time_ns()}{path.suffix}")
    temporary = path.parent / f".{path.name}-{os.getpid()}-{time.time_ns()}.tmp"
    save_new(temporary, value)
    os.replace(temporary, path)


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing frozen definition differs: {path}")
    else:
        save_new(path, value)


def event(directory, name, **fields):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    row = {"utc": utc(), "event": name, "pid": os.getpid(), **fields}
    with (directory / "runlog.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)


@contextlib.contextmanager
def lock(path, nonblocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def source_rows(inventory, prepared, expected=400):
    container = read(inventory)
    rows = container.get("cases") if isinstance(container, dict) else container
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Formal evaluation requires all {expected} source cases")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or i in {".", ".."}
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", i) for i in ids):
        raise ValueError("Duplicate or unsafe source IDs")
    result = []
    for ident in sorted(ids):
        manifest = read(Path(prepared) / ident / "manifest.json")
        if manifest.get("id") != ident:
            raise ValueError(f"{ident}: prepared/source ID mismatch")
        duration = manifest.get("duration_sec")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError(f"{ident}: invalid source duration")
        result.append({"id": ident, "duration_sec": duration})
    return result


def duration_partitions(rows, lanes):
    if lanes < 1:
        raise ValueError("At least one lane is required")
    groups, totals = [[] for _ in range(lanes)], [0.0] * lanes
    for row in sorted(rows, key=lambda r: (-r["duration_sec"], r["id"])):
        lane = min(range(lanes), key=lambda n: (totals[n], len(groups[n]), n))
        groups[lane].append(row["id"])
        totals[lane] += row["duration_sec"]
    return groups


def worker_lane_name(args):
    """Replicas share a frozen parent partition, never its worker lock/logs."""
    if args.replica_index is not None:
        return f"replica-{args.lane_index:02d}-{args.replica_index:02d}"
    return f"{'probe' if args.probe_only else 'lane'}-{args.lane_index:02d}"


def worker_definition(args, assigned, partitions, config):
    definition = {"lane_index": args.lane_index, "lane_count": args.lane_count,
        "assigned_case_ids": assigned if args.probe_only else partitions[args.lane_index],
        "probe_only": args.probe_only, "config": config}
    if args.replica_index is not None:
        definition.update(parent_lane_index=args.lane_index, replica_index=args.replica_index)
    return definition


def selected_outcome(case):
    completed = []
    for status_path in sorted(Path(case).glob("attempt-*/case-status.json")):
        status = read(status_path)
        if status.get("state") == "success" or status.get("terminal"):
            completed.append((status_path.parent, status))
    successes = [item for item in completed if item[1]["state"] == "success"]
    if len(successes) > 1:
        raise ValueError(f"Multiple successful attempts: {case}")
    selected = successes[0] if successes else (completed[-1] if completed else None)
    if selected:
        attempt, status = selected
        record = read(attempt / "scored-result.json")
        if record["status"] != status["state"]:
            raise ValueError(f"Case status/record mismatch: {attempt}")
        if status["state"] == "success":
            native = read(attempt / "transport-result.json")
            if not native.get("success") or read(attempt / "parsed-output.json") != native.get("responses"):
                raise ValueError(f"Successful case native/parsed evidence differs: {attempt}")
            for name in ("raw-output.jsonl", "input.jsonl", "input-spec.json", "ground-truth.json"):
                if not (attempt / name).is_file():
                    raise ValueError(f"Successful case evidence missing: {attempt / name}")
            if record.get("metrics") is None:
                raise ValueError(f"Successful case has no metrics: {attempt}")
        return {**record, "selected_attempt": str(attempt)}
    return None


def attempts(case):
    return sorted(path for path in Path(case).glob("attempt-*") if path.is_dir())


def recover_interrupted(case):
    """Caller MUST hold the case lock: no active worker can own these attempts."""
    for attempt in attempts(case):
        if (attempt / "case-status.json").exists():
            continue
        save_new(attempt / "case-status.json", {"state": "interrupted", "terminal": False,
                 "failure_kind": "infrastructure", "at": utc(),
                 "error": "previous case-lock owner ended before terminal status"})
        save_new(attempt / "interruption-recovery.json", {"at": utc(), "previous_files_preserved": True})


def new_attempt(case):
    case = Path(case)
    number = max((int(path.name.split("-")[-1]) for path in attempts(case)), default=0) + 1
    path = case / f"attempt-{number:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def coverage_error(result, manifest, task_pcm):
    if not result.get("success"):
        return result.get("error") or "native adapter returned unsuccessful replay"
    cov = result.get("coverage", {})
    expected = {"frames_expected": len(manifest["frames"]), "frames_received": len(manifest["frames"]),
        "source_audio_samples_expected": manifest["audio_samples"],
        "source_audio_samples_received": manifest["audio_samples"],
        "task_audio_samples_expected": len(task_pcm) // 2,
        "task_audio_samples_received": len(task_pcm) // 2,
        "uncovered_tail_samples": 0, "pending_task_tail_samples": 0,
        "source_synthetic_audio_samples": 0}
    for key, value in expected.items():
        if cov.get(key) != value:
            return f"input coverage mismatch: {key} expected {value}, got {cov.get(key)}"
    return None


def capture_environment():
    packages = sorted(({"name": item.metadata.get("Name"), "version": item.version,
                        "location": str(item.locate_file(""))}
                       for item in importlib.metadata.distributions()), key=lambda item: str(item["name"]))
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                       "--format=csv"], text=True, timeout=15)
    except Exception as exc:
        gpu = {"unavailable": str(exc)}
    return {"at": utc(), "pid": os.getpid(), "hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": packages, "gpu": gpu, "judge_called": False}


def gpu_monitor(lane, stop):
    with (lane / f"gpu-{time.time_ns()}.csv").open("x", encoding="utf-8") as stream:
        stream.write("utc,index,memory_used_mib,utilization_percent,power_watts\n")
        while not stop.is_set():
            try:
                values = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], text=True, timeout=10)
                for value in values.strip().splitlines():
                    stream.write(f"{utc()},{value}\n")
                stream.flush()
            except Exception as exc:
                with (lane / "gpu-errors.jsonl").open("a", encoding="utf-8") as stream_error:
                    stream_error.write(json.dumps({"at": utc(), "error": str(exc)}) + "\n")
            stop.wait(10)


def freeze_runtime(run, runtime, code):
    with lock(run / ".snapshot.lock", nonblocking=False):
        for root, target, sources in (
            (runtime, run / "snapshot/runtime", list((runtime / "adapters").rglob("*.py")) + [runtime / "scoring.py"]),
            (code, run / "snapshot/tts-code", [code / f"{name}.py" for name in
                 ("tts_phase", "tts_native", "tts_minicpm", "tts_venus", "strict_boundary")]),
        ):
            for source in sources:
                destination = target / source.relative_to(root)
                content = source.read_bytes()
                if destination.exists():
                    if destination.read_bytes() != content:
                        raise ValueError(f"Frozen runtime code differs: {destination}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as stream:
                        stream.write(content)
    return run / "snapshot/runtime", run / "snapshot/tts-code"


def load_scoring(path):
    spec = importlib.util.spec_from_file_location("tts_only_frozen_scoring", Path(path) / "scoring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gt_job(ident, args, helper, scoring):
    prepared = args.prepared_root / ident
    manifest = read(prepared / "manifest.json")
    original = (prepared / manifest["system_file"]).read_text(encoding="utf-8")
    query = helper.original_query(manifest)
    effective = helper.build_tts_only_system(original, query)
    if effective != "":
        raise ValueError(f"{ident}: generic system prompt contains the concrete task")
    relative = Path(manifest["source_video"]).relative_to("data")
    gt = read(args.data_root / relative.with_suffix(".labels.json"))
    if gt["query"] != manifest["query"]:
        raise ValueError(f"{ident}: GT/source task differs")
    expected = scoring.score_case(gt["annotation"], [], manifest["duration_sec"])
    return {"id": ident, "prepared": prepared, "manifest": manifest, "original": original,
            "effective": effective, "gt": gt, "expected_metrics": expected}


def case_pass(metrics):
    counts = metrics["counts"]
    joint = metrics["joint_action_time_event"]
    return (joint["tp"] == counts["active_gt_events"] and joint["fp"] == 0
            and metrics["silent_window_specificity"]["numerator"] == counts["wait_windows"]
            and counts["invalid_action_interventions"] == 0 and counts["invalid_timing_interventions"] == 0)


def ratio(n, d):
    return {"numerator": n, "denominator": d, "value": n / d if d else None}


def aggregate_records(records, expected_jobs):
    """All GT denominators include failed/missing videos. Only successes earn TP."""
    planned = {job["id"]: job for job in expected_jobs}
    selected = {record["id"]: record for record in records}
    if len(selected) != len(records) or not set(selected) <= set(planned):
        raise ValueError("Duplicate/foreign canonical results")
    tracks = {}
    for name, active_track in (("Intervention", True), ("Silent", False)):
        ids = [ident for ident, job in planned.items()
               if (job["expected_metrics"]["counts"]["active_gt_events"] > 0) == active_track]
        successes = [selected[ident] for ident in ids if ident in selected and selected[ident]["status"] == "success"]
        active = sum(planned[ident]["expected_metrics"]["counts"]["active_gt_events"] for ident in ids)
        waits = sum(planned[ident]["expected_metrics"]["counts"]["wait_windows"] for ident in ids)
        joint = sum(record["metrics"]["joint_action_time_event"]["tp"] for record in successes)
        wait_correct = sum(record["metrics"]["silent_window_specificity"]["numerator"] for record in successes)
        observed = [selected[ident] for ident in ids if ident in selected]
        emitted = sum((record.get("metrics") or record.get("failure_diagnostics") or {})
                      .get("counts", {}).get("emitted_interventions", 0) for record in observed)
        tracks[name] = {"videos": len(ids), "terminal_videos": sum(ident in selected for ident in ids),
            "successful_replays": len(successes),
            "video_correct": ratio(sum(case_pass(record["metrics"]) for record in successes), len(ids)),
            "action_correct": ratio(joint, active), "wait_window_correct": ratio(wait_correct, waits),
            "observed_emitted_interventions": emitted,
            "precision": ratio(joint, emitted), "recall": ratio(joint, active),
            "f1": ratio(2 * joint, active + emitted),
            "silent_no_reaction_correct": ratio(sum(record["metrics"]["counts"]["emitted_interventions"] == 0
                       for record in successes), len(ids)) if not active_track else None}
    return {"planned_cases": len(planned), "terminal_cases": len(selected),
        "successful_replays": sum(record["status"] == "success" for record in records),
        "failed_replays": sum(record["status"] == "failed" for record in records),
        "missing_case_ids": sorted(set(planned) - set(selected)),
        "complete": len(selected) == len(planned), "tracks": tracks,
        "failure_kind_counts": dict(Counter(record.get("failure_kind") for record in records if record["status"] != "success")),
        "semantic_content_correctness": "pending_judge_not_called",
        "fixed_denominator_policy": "all source GT videos/nodes; failures earn zero correct/silent credit",
        "observed_response_metrics": "partial failed replies retained as diagnostics; unobserved outputs never fabricated"}


def model_report(run, jobs):
    with lock(run / ".aggregate.lock", nonblocking=False):
        records = [selected_outcome(run / "cases" / job["id"]) for job in jobs]
        records = [record for record in records if record is not None]
        summary = aggregate_records(records, jobs)
        refresh(run / "model-progress.json", {"at": utc(), **summary})
        if summary["complete"]:
            refresh(run / "summary.json", {"at": utc(), **summary})
            refresh(run / "all-metrics.json", {"summary": summary, "case_results": records})
        return summary


def failure_result(error, kind="native_unknown"):
    return {"success": False, "status": "failed", "error": str(error), "failure_kind": kind,
            "responses": [], "task_responses": [], "coverage": {}, "latency": {}, "usage": {},
            "billing": None, "api_cost": None, "content_semantics_judge": "pending_not_called"}


def evaluate_attempt(job, attempt, adapter, pcm, task, scoring, max_attempts, adapter_load=None):
    """No GT is passed into Adapter.run_case; scorer sees GT only afterwards."""
    save_new(attempt / "attempt-start.json", {"at": utc(), "pid": os.getpid(), "id": job["id"]})
    save_new(attempt / "ground-truth.json", job["gt"])
    save_new(attempt / "input-spec.json", {"prepared": str(job["prepared"]),
        "source_manifest": job["manifest"], "original_system_logging_only": job["original"],
        "effective_generic_system": job["effective"], "task_audio": task,
        "condition": "TTS-audio-only-v2", "concrete_task_text_sent": False, "judge_called": False})
    loading = adapter is None
    try:
        if adapter is None:
            adapter = adapter_load()
        loading = False
        result = adapter.run_case(job["prepared"], attempt, pcm, task)
    except Exception as exc:
        kind = "infrastructure" if loading or isinstance(exc, (ConnectionError, TimeoutError, OSError)) else "native_unknown"
        result = failure_result(repr(exc), kind)
        (attempt / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
    error = coverage_error(result, job["manifest"], pcm)
    if error and result.get("success"):
        result = {**result, "success": False, "status": "failed", "error": error,
                  "failure_kind": "protocol_setup_or_coverage"}
        save_new(attempt / "coverage-failure.json", {"error": error, "original_result_preserved": True})
    if not (attempt / "transport-result.json").exists():
        save_new(attempt / "transport-result.json", result)
    for name, content in (("raw-output.jsonl", ""), ("input.jsonl", "")):
        if not (attempt / name).exists():
            with (attempt / name).open("x", encoding="utf-8") as stream:
                stream.write(content)
    if not (attempt / "parsed-output.json").exists():
        save_new(attempt / "parsed-output.json", result.get("responses", []))
    diagnostic = scoring.score_case(job["gt"]["annotation"], result.get("responses", []), job["manifest"]["duration_sec"])
    success = bool(result.get("success")) and error is None
    kind = None if success else result.get("failure_kind", "native_unknown")
    attempt_number = int(attempt.name.split("-")[-1])
    terminal = success or kind != "infrastructure" or attempt_number >= max_attempts
    record = {"id": job["id"], "status": "success" if success else "failed", "terminal": terminal,
        "failure_kind": kind, "error": error, "failure_counts_as_incorrect": not success,
        "metrics": diagnostic if success else None, "failure_diagnostics": None if success else diagnostic,
        "expected_gt_counts": job["expected_metrics"]["counts"], "task_response_count": len(result.get("task_responses", [])),
        "source_response_count": len(result.get("responses", [])), "coverage": result.get("coverage"),
        "latency": result.get("latency"), "usage": result.get("usage"),
        "billing": result.get("billing"), "api_cost": result.get("api_cost"),
        "settings": result.get("settings"), "clocks": result.get("clocks"),
        "semantic_content_correctness": "pending_judge_not_called"}
    save_new(attempt / "scored-result.json", record)
    save_new(attempt / "case-status.json", {"state": record["status"], "terminal": terminal,
              "failure_kind": kind, "error": error, "at": utc()})
    return adapter, record


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODULES)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--tts-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True, help="canonical model run folder")
    parser.add_argument("--lane-index", type=int, required=True)
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--replica-index", type=int, help="independent worker for this frozen parent lane, starting at 1")
    parser.add_argument("--only", nargs="+", help="explicit missing IDs; source denominator remains 400")
    parser.add_argument("--probe-only", action="store_true", help="up to 12 explicit IDs before full lanes; same canonical cases")
    parser.add_argument("--max-infrastructure-attempts", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    parser.add_argument("--tts-wait-timeout-sec", type=float, default=86400)
    parser.add_argument("--poll-sec", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not (0 <= args.lane_index < args.lane_count) or args.max_infrastructure_attempts < 1:
        parser.error("invalid lane or infrastructure-attempt limit")
    if args.only and len(set(args.only)) != len(args.only):
        parser.error("--only must contain unique explicit IDs")
    if args.probe_only and (not args.only or len(args.only) > 12):
        parser.error("--probe-only requires 1-12 explicit --only IDs")
    if args.replica_index is not None and (args.replica_index < 1 or args.probe_only):
        parser.error("--replica-index must be at least 1 and cannot be used with --probe-only")
    return args


def main(args):
    config = read(args.config)
    config = config.get("config", config)
    if config.get("model_id") != args.model or config.get("reminder_interval_s") is not None:
        raise ValueError("Model/config mismatch or non-B0 reminder setting")
    rows = source_rows(args.source_index, args.prepared_root)
    partitions = duration_partitions(rows, args.lane_count)
    assigned = partitions[args.lane_index]
    if args.only:
        if not set(args.only) <= {row["id"] for row in rows}:
            raise ValueError("Unknown explicit --only source case")
        # An explicit repair worker can never silently steal another lane's IDs.
        if not args.probe_only and not set(args.only) <= set(assigned):
            raise ValueError("--only contains case IDs owned by another static lane")
        assigned = list(args.only)
    run = args.output_root.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lane = run / "lanes" / worker_lane_name(args)
    lane.mkdir(parents=True, exist_ok=True)
    with lock(lane / ".worker.lock"):
        with lock(run / ".plan.lock", nonblocking=False):
            frozen(run / "run-plan.json", {"condition": "TTS-audio-only-v2", "model_id": args.model,
                "source_case_ids": [row["id"] for row in rows], "fixed_denominator": 400,
                "source_index": str(args.source_index.resolve()), "prepared_root": str(args.prepared_root.resolve()),
                "data_root": str(args.data_root.resolve()), "task_audio_index": str(args.tts_index.resolve()),
                "baseline_runtime": str(args.runtime_dir.resolve()), "lane_count": args.lane_count,
                "partitions": partitions, "config": {k: v for k, v in config.items() if k not in RESOURCE_FIELDS},
                "max_infrastructure_attempts": args.max_infrastructure_attempts, "judge_called": False})
        frozen(lane / "lane-definition.json", worker_definition(args, assigned, partitions, config))
        runtime, tts_code = freeze_runtime(run, args.runtime_dir.resolve(), Path(__file__).parent.resolve())
        sys.path.insert(0, str(runtime))
        sys.path.insert(0, str(tts_code))
        helper = importlib.import_module("tts_phase")
        scoring = load_scoring(runtime)
        jobs = [gt_job(row["id"], args, helper, scoring) for row in rows]
        lookup = {job["id"]: job for job in jobs}
        save_new(lane / f"environment-{time.time_ns()}.json", capture_environment())
        event(lane, "lane_started", model=args.model, assigned_ids=assigned)
        adapter = None
        adapter_type = None
        stop = threading.Event()
        monitor = threading.Thread(target=gpu_monitor, args=(lane, stop), daemon=True)
        monitor.start()
        waiting_since = None
        try:
            while True:
                pending = [ident for ident in assigned if selected_outcome(run / "cases" / ident) is None]
                if not pending:
                    break
                index = read(args.tts_index) if args.tts_index.is_file() else {"cases": {}}
                ready = [ident for ident in pending if ident in index.get("cases", {})]
                if not ready:
                    if waiting_since is None:
                        waiting_since = time.monotonic()
                        event(lane, "waiting_for_task_audio", pending_ids=pending)
                    if time.monotonic() - waiting_since > args.tts_wait_timeout_sec:
                        raise TimeoutError("Task audio never became ready; model cases remain pending")
                    refresh(lane / "progress.json", {"at": utc(), "state": "waiting_for_task_audio",
                            "terminal": len(assigned) - len(pending), "assigned": len(assigned)})
                    time.sleep(args.poll_sec)
                    continue
                waiting_since = None
                made_progress = False
                for ident in ready:
                    if shutil.disk_usage(run).free < args.min_free_gb * 1024**3:
                        raise RuntimeError("Storage guard: less than required free capacity; no source data removed")
                    case = run / "cases" / ident
                    try:
                        with lock(case / ".case.lock"):
                            if selected_outcome(case) is not None:
                                continue
                            recover_interrupted(case)
                            job = lookup[ident]
                            pcm, task = helper.load_task_audio(args.tts_index, ident, job["manifest"])
                            attempt = new_attempt(case)
                            if adapter_type is None:
                                adapter_type = importlib.import_module(MODULES[args.model]).Adapter
                            def create_adapter():
                                begin = time.monotonic()
                                loaded = adapter_type.load(config)
                                event(lane, "model_loaded", seconds=time.monotonic() - begin)
                                save_new(lane / f"environment-loaded-{time.time_ns()}.json", capture_environment())
                                return loaded
                            adapter, record = evaluate_attempt(job, attempt, adapter, pcm, task, scoring,
                                args.max_infrastructure_attempts, create_adapter)
                            event(lane, "case_finished", id=ident, attempt=attempt.name,
                                  status=record["status"], terminal=record["terminal"], failure_kind=record["failure_kind"])
                            made_progress = True
                    except BlockingIOError:
                        event(lane, "case_owned_by_another_live_worker", id=ident)
                summary = model_report(run, jobs)
                refresh(lane / "progress.json", {"at": utc(), "state": "running",
                    "terminal": sum(selected_outcome(run / "cases" / ident) is not None for ident in assigned),
                    "assigned": len(assigned), "model_terminal": summary["terminal_cases"]})
                if not made_progress:
                    time.sleep(args.poll_sec)
            summary = model_report(run, jobs)
            refresh(lane / "progress.json", {"at": utc(), "state": "complete", "terminal": len(assigned), "assigned": len(assigned)})
            event(lane, "lane_completed", model_terminal=summary["terminal_cases"], fixed_denominator=400)
            return 0
        except BaseException as exc:
            event(lane, "lane_failed", error_type=type(exc).__name__, error=str(exc))
            refresh(lane / "progress.json", {"at": utc(), "state": "failed", "error": str(exc)})
            (lane / f"failure-{time.time_ns()}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            stop.set()
            monitor.join(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main(arguments()))
