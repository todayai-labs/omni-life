"""One persistent official model per process; resumable 400-case AV replay.

The model adapter never receives GT. Scoring happens only after a finished
session and full source delivery. Failed transports are not model silence.
"""
import argparse
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save(path, obj, replace=False):
    path = Path(path)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)
    else:
        with path.open("x") as f:
            f.write(text)


def log(root, event, **fields):
    record = {"utc": utc(), "monotonic_s": time.monotonic(), "event": event, **fields}
    with (root / "runlog.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def environment():
    def command(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30)
        except Exception as e:
            return str(e)
    return {"utc": utc(), "host": platform.node(), "python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "installed_packages": sorted([{ "name": d.metadata["Name"], "version": d.version,
                                             "location": str(d.locate_file("")) }
                                           for d in importlib.metadata.distributions()], key=lambda x: str(x["name"])),
            "pip_freeze_diagnostic": command([sys.executable, "-m", "pip", "freeze"]),
            "nvidia_smi": command(["nvidia-smi"])}


def resources(output, done):
    with (output / "gpu.csv").open("x") as f:
        f.write("utc,monotonic_s,index,memory_used_mib,utilization_percent,power_watts\n")
        while not done.is_set():
            try:
                text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw",
                                                "--format=csv,noheader,nounits"], text=True, timeout=10)
                for line in text.strip().splitlines():
                    f.write(f"{utc()},{time.monotonic()},{line}\n")
                f.flush()
            except Exception as e:
                with (output / "resource-errors.jsonl").open("a") as err:
                    err.write(json.dumps({"utc": utc(), "error": str(e)}) + "\n")
            done.wait(1)


def ratio(n, d):
    return {"numerator": n, "denominator": d, "value": n / d if d else None}


def aggregate(rows, planned):
    successful = [r for r in rows if r.get("status") == "success"]
    tracks = {}
    for name, positive in (("Intervention", True), ("Silent", False)):
        selected = [r for r in successful if (r["metrics"]["counts"]["active_gt_events"] > 0) == positive]
        active = sum(r["metrics"]["counts"]["active_gt_events"] for r in selected)
        predicted = sum(r["metrics"]["counts"]["emitted_interventions"] for r in selected)
        tp = sum(r["metrics"]["joint_action_time_event"]["tp"] for r in selected)
        passed = sum(r["metrics"]["joint_action_time_event"]["tp"] == r["metrics"]["counts"]["active_gt_events"]
                     and r["metrics"]["joint_action_time_event"]["fp"] == 0
                     and r["metrics"]["silent_window_specificity"]["numerator"] == r["metrics"]["counts"]["wait_windows"]
                     and r["metrics"]["counts"]["invalid_action_interventions"] == 0
                     and r["metrics"]["counts"]["invalid_timing_interventions"] == 0 for r in selected)
        tracks[name] = {"scored_cases": len(selected), "case_all_correct": ratio(passed, len(selected)),
                        "active_nodes": active, "emitted_interventions": predicted,
                        "joint_node_success": ratio(tp, active),
                        "joint_event": {"tp": tp, "fp": predicted - tp, "fn": active - tp,
                                        "precision": tp / predicted if predicted else None,
                                        "recall": tp / active if active else None,
                                        "f1": 2 * tp / (active + predicted) if active + predicted else None},
                        "silent_cases_no_reaction": ratio(sum(r["metrics"]["counts"]["emitted_interventions"] == 0 for r in selected), len(selected)) if not positive else None}
    return {"planned_cases": planned, "scored_cases": len(successful), "status_counts": dict(Counter(r["status"] for r in rows)),
            "all_400_success": len(successful) == planned == 400, "tracks": tracks,
            "semantic_content_score": None, "semantic_content_status": "not judged; not mixed into action/time score",
            "metric_scope": "strict original GT windows; no tolerance ablation; denominators expose incomplete coverage",
            "case_metrics": rows}


def coverage_gate(result, manifest):
    if not result.get("success"):
        return result.get("status", "transport_failed")
    coverage = result.get("coverage", {})
    for key, expected in (("frames_expected", len(manifest["frames"])), ("frames_received", len(manifest["frames"])),
                          ("source_audio_samples_expected", manifest["audio_samples"]),
                          ("source_audio_samples_received", manifest["audio_samples"])):
        if coverage.get(key) != expected:
            return "incomplete_input:" + key
    if coverage.get("uncovered_tail_samples", 0) != 0:
        return "incomplete_source_audio_tail"
    return "success"


def freeze_adapter_package(root, config):
    """Freeze helpers as well as the main adapter for new runs and resumption."""
    destination = root / "snapshot/runtime"
    if not destination.exists():
        package = Path(__file__).parent / config["adapter_module"].split(".")[0]
        if not package.is_dir():
            raise ValueError("Adapter package missing: " + str(package))
        target = destination / package.name
        target.mkdir(parents=True)
        receipts = []
        for source in sorted(package.rglob("*.py")):
            data = source.read_bytes()
            frozen = target / source.relative_to(package)
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes(data)
            receipts.append({"path": str(frozen.relative_to(destination)),
                             "sha256": hashlib.sha256(data).hexdigest(), "source": str(source)})
        provenance = {"utc": utc(), "files": receipts, "policy": "run imports its immutable adapter/helper package"}
        repo = config.get("repo_path")
        if repo:
            for key, command in (("official_repo_commit", ["git", "-C", repo, "rev-parse", "HEAD"]),
                                 ("official_repo_dirty_tracked", ["git", "-C", repo, "status", "--porcelain", "-uno"]),
                                 ("official_repo_diff", ["git", "-C", repo, "diff", "HEAD"])):
                try:
                    provenance[key] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
                except Exception as error:
                    provenance[key] = {"unavailable": str(error)}
        save(root / "snapshot/runtime-provenance.json", provenance)
    else:
        provenance = json.loads((root / "snapshot/runtime-provenance.json").read_text())
        for receipt in provenance["files"]:
            if digest(destination / receipt["path"]) != receipt["sha256"]:
                raise ValueError("Frozen adapter/helper changed: " + receipt["path"])
    sys.path.insert(0, str(destination))


def select_cases(cases, only):
    """Reject misspelled/empty probe selections instead of silently dropping IDs."""
    if only is None:
        if len(cases) != 400:
            raise ValueError("Formal run requires all 400")
        return cases
    if not only or len(only) != len(set(only)):
        raise ValueError("Probe selection must contain unique explicit case IDs")
    unknown = sorted(set(only) - {r["id"] for r in cases})
    if unknown:
        raise ValueError("Unknown probe case IDs: " + ", ".join(unknown))
    return [r for r in cases if r["id"] in only]


def main():
    p = argparse.ArgumentParser()
    for name in ("config", "inventory", "prepared", "data", "run", "scorer"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--max-attempts", type=int, default=2)
    args = p.parse_args()
    root = Path(args.run)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads(Path(args.config).read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    rows = select_cases(inventory["cases"], args.only)
    snapshot = {"config": cfg, "config_sha256": digest(args.config), "inventory_sha256": digest(args.inventory),
                "source_prompt_policy": "approved 20261010 source query and frozen user streaming prompt", "planned_cases": len(rows)}
    if (root / "config.json").exists():
        if json.loads((root / "config.json").read_text()) != snapshot:
            raise ValueError("Cannot resume a run after configuration changed")
    else:
        save(root / "config.json", snapshot)
        save(root / "environment.json", environment())
        save(root / "source-identities.json", inventory)
        (root / "snapshot").mkdir()
        for path in (Path(__file__), Path(args.scorer)):
            (root / "snapshot" / path.name).write_bytes(path.read_bytes())
    # Older already-running experiments retain their original startup snapshot.
    # New runs import a complete frozen package, including shared ASR helpers.
    if (root / "snapshot/runtime").exists() or not (root / "runlog.jsonl").exists():
        freeze_adapter_package(root, cfg)
    module = importlib.import_module(cfg["adapter_module"])
    (root / "snapshot" / (cfg["model_id"] + "-adapter.py")).write_bytes(Path(module.__file__).read_bytes()) if not (root / "snapshot" / (cfg["model_id"] + "-adapter.py")).exists() else None
    spec = importlib.util.spec_from_file_location("frozen_scorer", root / "snapshot" / Path(args.scorer).name)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    attempts, outcomes = {}, {}
    for row in rows:
        directory = root / "cases" / row["id"]
        prior_dirs = sorted(directory.glob("attempt-*"))
        prior = []
        for attempt_dir in prior_dirs:
            status = attempt_dir / "case-status.json"
            if not status.exists():
                save(status, {"status": "interrupted", "utc": utc(), "note": "prior process ended without terminal status; original artifacts preserved"})
            prior.append(status)
        attempts[row["id"]] = len(prior_dirs)
        for status in prior:
            state = json.loads(status.read_text())
            if state.get("status") == "success":
                outcomes[row["id"]] = json.loads((status.parent / "scored-result.json").read_text())
                break
        if prior and row["id"] not in outcomes and len(prior) >= args.max_attempts:
            outcomes[row["id"]] = {"id": row["id"], "status": "failed", "metrics": None, "attempts": len(prior)}
    if len(outcomes) == len(rows) and (root / "seal.json").exists():
        log(root, "already_terminal_no_model_load", cases=len(rows))
        return
    log(root, "load_start", model=cfg["model_id"], pid=os.getpid())
    adapter = module.Adapter.load(cfg)
    log(root, "load_complete", model=cfg["model_id"])
    while len(outcomes) < len(rows):
        ready = [r for r in rows if r["id"] not in outcomes and (Path(args.prepared) / r["id"] / "manifest.json").is_file()]
        if not ready:
            save(root / "progress.json", {"model": cfg["model_id"], "pid": os.getpid(), "utc": utc(),
                                          "planned": len(rows), "terminal": len(outcomes), "success": sum(r["status"] == "success" for r in outcomes.values()),
                                          "state": "waiting_for_prepared_source"}, replace=True)
            time.sleep(5)
            continue
        for row in ready:
            ident = row["id"]
            attempts[ident] += 1
            output = root / "cases" / ident / f"attempt-{attempts[ident]:03d}"
            output.mkdir(parents=True, exist_ok=False)
            prepared = Path(args.prepared) / ident
            manifest = json.loads((prepared / "manifest.json").read_text())
            if manifest["source_video_sha256"] != row["video_sha256"]:
                raise ValueError("Source hash mismatch: " + ident)
            save(output / "input-spec.json", {"prepared": str(prepared), "manifest_sha256": digest(prepared / "manifest.json"),
                                              "manifest": manifest, "system": (prepared / manifest["system_file"]).read_text()})
            done = threading.Event()
            resource = threading.Thread(target=resources, args=(output, done), daemon=True)
            resource.start()
            log(root, "case_start", id=ident, attempt=attempts[ident], output=str(output))
            started = time.monotonic()
            try:
                result = adapter.run_case(prepared, output)
                save(output / "transport-result.json", result)
                status = coverage_gate(result, manifest)
                if status != "success":
                    raise RuntimeError(status)
                labels = Path(args.data) / row["labels"]
                if digest(labels) != row["labels_sha256"]:
                    raise ValueError("GT changed after source freeze")
                gt = json.loads(labels.read_text())
                save(output / "ground-truth.json", gt)
                metrics = scoring.score_case(gt["annotation"], result["responses"], manifest["duration_sec"])
                scored = {"id": ident, "case_id": row["case_id"], "status": "success", "metrics": metrics,
                          "latency": result.get("latency"), "coverage": result["coverage"], "usage": result.get("usage"),
                          "billing": result.get("billing"), "elapsed_s": time.monotonic() - started, "attempt": attempts[ident],
                          "source_audio_present": manifest["audio_present"]}
                save(output / "scored-result.json", scored)
                outcomes[ident] = scored
                save(output / "case-status.json", {"status": "success", "utc": utc()})
                log(root, "case_complete", id=ident, responses=len(result["responses"]), elapsed_s=scored["elapsed_s"])
            except Exception as error:
                save(output / "case-status.json", {"status": "failed", "utc": utc(), "error": str(error), "traceback": traceback.format_exc()})
                log(root, "case_failed", id=ident, attempt=attempts[ident], error=str(error))
                if attempts[ident] >= args.max_attempts:
                    outcomes[ident] = {"id": ident, "status": "failed", "metrics": None, "error": str(error)}
            finally:
                done.set()
                resource.join(timeout=12)
            summary = aggregate(list(outcomes.values()), len(rows))
            save(root / "results.json", summary, replace=True)
            save(root / "progress.json", {"model": cfg["model_id"], "pid": os.getpid(), "utc": utc(), "planned": len(rows),
                                          "terminal": len(outcomes), "success": summary["scored_cases"], "state": "running"}, replace=True)
    summary = aggregate(list(outcomes.values()), len(rows))
    save(root / "results.json", summary, replace=True)
    seal = [{"path": str(f.relative_to(root)), "sha256": digest(f)} for f in sorted(root.rglob("*"))
            if f.is_file() and str(f.relative_to(root)) not in {".worker.lock", "runlog.jsonl", "results.json", "progress.json", "seal.json"}]
    save(root / "seal.json", {"utc": utc(), "complete_400": summary["all_400_success"], "artifacts": seal})
    save(root / "progress.json", {"model": cfg["model_id"], "utc": utc(), "planned": len(rows), "terminal": len(outcomes),
                                  "success": summary["scored_cases"], "state": "complete" if summary["all_400_success"] else "incomplete_or_probe"}, replace=True)
    log(root, "run_finished", all_400_success=summary["all_400_success"], scored=summary["scored_cases"])


if __name__ == "__main__":
    main()
