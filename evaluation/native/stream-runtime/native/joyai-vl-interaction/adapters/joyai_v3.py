"""Official JoyAI persistent streaming application + causal source-only ASR.

The upstream application's chunk memory, asynchronous summaries, output
normalizer and decoding parameters are retained. No fresh prefix replay runner,
TTS, synthetic source audio or reference annotations are used.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

try:
    from .asr_pipeline_v3 import SourceAlignedASRAdapter
except ImportError:
    from asr_pipeline_v3 import SourceAlignedASRAdapter


class Request:
    def __init__(self, payload, session_id):
        self.payload = payload
        self.headers = {"x-streaming-session": session_id}

    async def json(self):
        return self.payload


class Adapter(SourceAlignedASRAdapter):
    minimum_frames = 1

    @classmethod
    def load(cls, config):
        import requests
        self = cls()
        self.config = dict(config)
        for base in (config["main_api_base"], config["summary_api_base"]):
            requests.get(base + "/models", timeout=30).raise_for_status()
        repo = Path(config["repo_path"])
        source = repo / "services/webinfer/live_adapter.py"
        sys.path.insert(0, str(source.parent))
        spec = importlib.util.spec_from_file_location("omnilife_official_joyai", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.module = module
        native_config = module.AdapterConfig(main_api_base=config["main_api_base"],
            main_model=config.get("served_model_name", "joyai-vl-interaction"),
            summarizer_api_base=config["summary_api_base"], summarizer_model="omni-summary",
            longterm_api_base=config["summary_api_base"], longterm_model="omni-summary",
            allowed_local_image_roots=tuple(config.get("allowed_local_image_roots", ("prepared",))),
            per_session_dirs=False,
            # The standing instruction arrives in SYSTEM before any source speech.
            force_silence_before_query=False,
            frame_save_dir=str(Path(__file__).resolve().parents[2] / "joyai-frames"))
        self.native_config = native_config
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        async def initialize():
            self.official = module.StreamingInferAdapter(native_config)
            self.official.start_background_tasks()
        asyncio.run_coroutine_threadsafe(initialize(), self.loop).result()
        self.settings = {"official_repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "task_role": "system_only", "user_text_source": "completed_source_asr",
            "native_application_config": asdict(native_config), "model_path": config["model_path"],
            "model_api_base": config["main_api_base"], "summary_path": config["summary_path"],
            "vllm_version": "0.22.0", "seed": int(config.get("seed", 0)),
            "gpu_at_adapter_attach": subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], text=True),
            "native_delivery": "official whole-turn HTTP completion; onset is application receipt, not estimated server TTFT",
            "helper_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), Path(__file__).with_name("asr_pipeline_v3.py"), source,
                          source.with_name("memory_summarizer.py"),
                          source.parent / "scripts/start_model.sh",
                          source.parent / "scripts/start_summary_model.sh"]}}
        # Instrument SDK boundaries without altering official memory construction.
        original = self.official.main_client.chat.completions.create
        async def main_tap(**kwargs):
            # The official wrapper strips outer whitespace while parsing its override.
            messages = kwargs["messages"]
            if (not messages or messages[0]["role"] != "system"
                    or sum(m["role"] == "system" for m in messages) != 1
                    or messages[0]["content"] != self.system.strip()):
                raise ValueError("Native JoyAI request lost the frozen system message")
            messages[0] = {**messages[0], "content": self.system}
            kwargs["seed"] = self.case_seed + self.turn_index
            self.active_log("input.jsonl", {"kind": "official_main_model_http", "payload": kwargs})
            result = await original(**kwargs)
            self.active_log("raw-output.jsonl", {"kind": "official_main_model_http", "raw": result.model_dump()})
            self.case_usage["model_calls"] += 1
            if result.usage:
                self.case_usage["prompt_tokens"] += result.usage.prompt_tokens
                self.case_usage["completion_tokens"] += result.usage.completion_tokens
            if not result.choices or any(c.finish_reason == "error" for c in result.choices):
                raise RuntimeError("Official JoyAI backend returned no valid completion")
            return result
        self.official.main_client.chat.completions.create = main_tap
        clients = {id(self.official.summarizer._client): self.official.summarizer._client,
                   id(self.official.summarizer._longterm_client): self.official.summarizer._longterm_client}
        for client in clients.values():
            original_summary = client.chat.completions.create
            def summary_tap(_original=original_summary, **kwargs):
                active_log, usage = self.active_log, self.case_usage
                kwargs["seed"] = self.case_seed + 100000 + usage.get("summary_calls", 0)
                usage["summary_calls"] = usage.get("summary_calls", 0) + 1
                active_log("input.jsonl", {"kind": "official_summary_http", "payload": kwargs})
                result = _original(**kwargs)
                active_log("raw-output.jsonl", {"kind": "official_summary_http", "raw": result.model_dump()})
                return result
            client.chat.completions.create = summary_tap
        return self

    def start_session(self, system, output_dir, log):
        self.system, self.active_log = system, log
        self.session_id = self.manifest["id"] + "-" + uuid.uuid4().hex
        task = "\n".join(q["text"] for q in self.manifest["query"])
        if not task or not system.rstrip("\n").endswith("当前用户交给你的任务：\n" + task):
            raise ValueError("Missing frozen task in canonical system message")
        async def begin():
            state = self.official.get_session(self.session_id)
            state.session_out_dir = str(output_dir / "official")
            state.session_light_out_dir = str(output_dir / "official-light")
            state.debug_input_dir = output_dir / "official-inputs"
            state.output_path = self.official._session_output_path(state, light=False)
            state.light_output_path = self.official._session_output_path(state, light=True)
        asyncio.run_coroutine_threadsafe(begin(), self.loop).result()

    def process_frames(self, prepared_dir, frames, prompt, log):
        # prompt is exclusively the original, causally available source ASR text.
        content = [{"type": "text", "text": prompt}] if prompt else []
        content += [{"type": "image_url", "image_url": {"url": (prepared_dir / f["file"]).as_uri()}} for f in frames]
        payload = {"model": self.config.get("served_model_name", "joyai-vl-interaction"),
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": content}],
            "frame_time_ranges": [f"{f['pts_sec']:.3f}s-{f['pts_sec']:.3f}s" for f in frames]}
        log("input.jsonl", {"kind": "official_application_request", "payload": payload,
                            "session_id": self.session_id})
        result = asyncio.run_coroutine_threadsafe(self.official.handle_chat_completions(
                        Request(payload, self.session_id)), self.loop).result(timeout=600)
        raw = json.loads(result.body)
        receive = log("raw-output.jsonl", {"kind": "official_application_response", "raw": raw,
                                          "http_status": result.status})
        if result.status != 200 or "error" in raw:
            raise RuntimeError(json.dumps(raw, ensure_ascii=False))
        text = raw["choices"][0]["message"]["content"] or ""
        # Interpret only native control markers. Keep benchmark [action] tags.
        if text.strip() == "</silence>":
            return {"responses": []}
        if text.startswith("</response>"):
            text = text[len("</response>"):].strip()
        responses = []
        if text:
            responses.append({"response_id": raw["id"], "text": text,
                "start_media_s": receive["media_wall_s"], "end_media_s": receive["media_wall_s"],
                "onset_source": "first_text_receive", "first_receive_utc": receive["utc"],
                "first_receive_monotonic_s": receive["monotonic_s"]})
        return {"responses": responses}

    def finish_session(self, log):
        async def drain_started_summaries():
            state = self.official.sessions.get(self.session_id)
            tasks = [job["task"] for job in state.async_pending_summary_jobs] if state else []
            if tasks:
                log("runlog.jsonl", {"event": "await_already_started_summary_outputs", "jobs": len(tasks),
                                    "source_audio_added": False, "further_main_generation": False})
                await asyncio.gather(*tasks)
        asyncio.run_coroutine_threadsafe(drain_started_summaries(), self.loop).result(timeout=320)
        reset = asyncio.run_coroutine_threadsafe(self.official.handle_reset(
                        Request({}, self.session_id)), self.loop).result(timeout=120)
        log("runlog.jsonl", {"event": "official_session_reset", "raw": json.loads(reset.body)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    adapter = Adapter.load(json.loads(Path(args.config).read_text()))
    result = adapter.run_case(Path(args.prepared), Path(args.output))
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)
