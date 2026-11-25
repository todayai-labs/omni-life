"""Run or verify the Streaming worker for a selected set of benchmark cases."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_stream import RELEASE_SHA, SEPARATOR, require, sha, load, write

CODE = Path(__file__).resolve().parent
WORKER = CODE / "native/shared/batch_native.py"
NATIVE_SCORER = CODE / "native/shared/scoring.py"
PACKAGE = Path(os.environ.get("OMNI_LIFE_HANDOFF", CODE.parent)).resolve()
ROOT = PACKAGE / "stream-runtime"
VERSION = "20261124-r3-final"


def stable_digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def vendor(model):
    models = load(CODE / "native/provenance.json")
    require(model in models, "Choose an archived self-hosted model")
    src = CODE / "native" / model
    record = models[model]
    for name, hashes in record["files"].items():
        path = {"batch_native.py": WORKER, "scoring.py": NATIVE_SCORER}.get(name, src / name)
        require(sha(path) == hashes["file"], "Changed native implementation: " + str(path))
    spec = importlib.util.spec_from_file_location("native_stream_batch", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return src, record, module


def verify(rows, dataset):
    for row in rows:
        cid = row["id"]
        folder = ROOT / "prepared" / cid
        require(sha(folder / "manifest.json") == row["prepared_manifest_sha256"], "Bad manifest: " + cid)
        man = load(folder / "manifest.json")
        require(man["id"] == cid and man["query"] == row["query"] and man["source_video_sha256"] == row["video_sha256"],
                "Wrong input identity: " + cid)
        sp = folder / man["system_file"]
        require(sha(sp) == row["system_sha256"] == man["system_sha256"] and
                sp.read_text().split(SEPARATOR)[-1].strip() == row["query"][0]["text"], "Wrong task: " + cid)
        for key in ("video", "labels", "task_audio"):
            require(sha(dataset / "data" / row[key]) == row[key+"_sha256"], "Changed source: " + row[key])
        for frame in man["frames"]:
            require(sha(folder / frame["file"]) == frame["sha256"], "Changed frame: " + cid)
        if man["audio_present"]:
            pcm = folder / man["audio_file"]
            require(pcm.stat().st_size == man["audio_samples"]*2 and sha(pcm) == man["audio_sha256"], "Changed PCM: " + cid)


def selected(args, batch):
    dataset = PACKAGE / "dataset"
    index = ROOT / "source-identities.json"
    require(sha(dataset/"manifest.json") == RELEASE_SHA, "Wrong r3 release")
    inv = load(index)
    require(inv["release_manifest_sha256"] == RELEASE_SHA and sha(index) == load(ROOT/"build-receipt.json")["source_identities_sha256"],
            "Wrong r3 source identity ledger")
    ids_path = args.caseids_file or PACKAGE / "streaming/rerun" / (args.model+".txt")
    ids = None if args.all else [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    rows = batch.select_cases(inv["cases"], ids)
    verify(rows, dataset)
    return rows


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "binding.lock").open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--caseids-file", type=Path, help="One source case ID per line; default: model's rerun list")
    selection.add_argument("--all", action="store_true")
    p.add_argument("--config", type=Path, help="Deployment config from the existing model host")
    p.add_argument("--python", help="Python binary in the corresponding model environment")
    p.add_argument("--run-root", type=Path, help="Parent for isolated, versioned outputs")
    p.add_argument("--execute", action="store_true", help="Start the real model worker; default verifies inputs offline")
    p.add_argument("--max-attempts", type=int, default=2)
    a = p.parse_args()
    source, provenance, batch = vendor(a.model)
    rows = selected(a, batch)
    require(a.max_attempts > 0, "Set positive --max-attempts")
    cfgpath = a.config or ROOT/"native/configs"/(a.model+".json")
    if not a.config:
        require(sha(cfgpath) == provenance["archived_config_sha256"], "Archived config changed")
    cfg = load(cfgpath)
    require(cfg["model_id"] == a.model and cfg["adapter_module"] == provenance["adapter_module"], "Deployment and adapter model differ")
    if a.model == "joyai-vl-interaction":
        cfg["allowed_local_image_roots"] = [str(ROOT/"prepared")]
    ids = [r["id"] for r in rows]
    key = dict(dataset_version=VERSION, model=a.model, selected_ids=ids, prepared_index_sha256=sha(ROOT/"source-identities.json"),
               runtime_files=provenance["files"], config_sha256=stable_digest(cfg))
    output = (a.run_root or PACKAGE/"new-runs").resolve() / VERSION / a.model / stable_digest(key)[:20]
    python = a.python or cfg.get("python") or sys.executable
    args = [str(python), "-u", str(WORKER), "--config", str(output/"deployment.json"),
            "--adapter-root", str(source),
            "--inventory", str(ROOT/"source-identities.json"), "--prepared", str(ROOT/"prepared"), "--data", str(PACKAGE/"dataset/data"),
            "--scorer", str(NATIVE_SCORER), "--run", str(output/"raw"), "--max-attempts", str(a.max_attempts), "--only", *ids]
    status = dict(dataset_version=VERSION, model=a.model, cases=len(ids), selected_sha256=stable_digest(ids),
                  signed_inputs_verified=True, native_entry_sha256=sha(WORKER), run=str(output),
                  executed=a.execute)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if a.execute:
        require(a.run_root is not None, "Provide --run-root for actual model outputs")
        with run_lock(output):
            binding = output / "binding.json"
            if binding.exists():
                require(load(binding) == key and load(output/"deployment.json") == cfg, "Different existing run binding")
            else:
                require(not (output/"raw").exists(), "Cannot resume unbound native output")
                config_out = output/"deployment.json"
                if config_out.exists():
                    require(load(config_out) == cfg, "Mismatched partial initialization")
                else:
                    write(config_out, cfg)
                write(binding, key)
        subprocess.run(args, cwd=source, check=True)


if __name__ == "__main__":
    main()
