"""AURA's upstream SessionHistory + AsyncLLM application, causal ASR input.

Preserves native video/control/decoding code. Only transport, the benchmark
system prompt and source-only media scheduling are replaced; no TTS is loaded.
"""
import argparse
import asyncio
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

try:
    from .asr_pipeline_v3 import SourceAlignedASRAdapter
except ImportError:
    from asr_pipeline_v3 import SourceAlignedASRAdapter

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class Adapter(SourceAlignedASRAdapter):
    minimum_frames = 2

    @classmethod
    def load(cls, config):
        self = cls()
        self.config = dict(config)
        repo = Path(config["repo_path"])
        sys.path.insert(0, str(repo))
        spec = importlib.util.spec_from_file_location("omnilife_official_aura", repo / "Qwen3_VL_online_streaming_v2_ContextManaged.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.module = module
        previous = sys.argv
        sys.argv = [str(repo / "Qwen3_VL_online_streaming_v2_ContextManaged.py"),
                    "--model", config["model_path"], "--max-model-len", "262144",
                    "--max-seq-len", "262144", "--gpu-memory-utilization", str(config.get("memory_fraction", 0.65)),
                    "--kv-offloading-size", "10", "--disable-hybrid-kv-cache-manager",
                    "--block-size", "16", "--prefix-caching-hash-algo", "xxhash",
                    "--mm-encoder-attn-backend", "FLASH_ATTN", "--mm-encoder-tp-mode", "data",
                    "--max-num-batched-tokens", "15360", "--temperature", "0.5", "--max-tokens", "128",
                    "--cross-turn-penalty", "1", "--cross-turn-lookback", "10",
                    "--cross-turn-ngram-sizes", "--enable-pruning", "--max-rounds", "45",
                    "--num-rounds-keep", "30", "--max-context-qas", "10", "--target-fps", "1"]
        try:
            self.args = module.parse_args()
        finally:
            sys.argv = previous
        module.setup_silent_token_id(config["model_path"])
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.gpu_before_load = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], text=True)
        self.loop.run_until_complete(module.init_async_engine(self.args))
        self.original_engine = module.async_engine
        self.settings = {"official_repo_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                         "native_args": vars(self.args), "gpu_before_load": self.gpu_before_load,
                         "vllm_version": "0.17.1", "seed": int(config.get("seed", 0)),
                         "helper_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path(__file__), Path(__file__).with_name("asr_pipeline_v3.py"),
                               repo / "Qwen3_VL_online_streaming_v2_ContextManaged.py", repo / "context_manage.py"]}}
        return self

    def start_session(self, system, output_dir, log):
        module = self.module
        history = module.SessionHistory(max_rounds=45, num_rounds_keep=30,
                    pruning_enabled=True, max_context_qas=10,
                    debug_context_file=str(output_dir / "official-context.jsonl"))
        history.system_prompt = system
        history._system_msg = {"role": "system", "content": system}
        history._reset()
        self.session = module.StreamingSession(session_id=output_dir.name, history=history,
                                              input_queue=asyncio.Queue(), output_queue=asyncio.Queue(),
                                              cross_turn_penalty=module.CrossTurnPenalty(module.model_tokenizer,
                                                    window=10, logit_penalty=1.0, ngram_sizes=[]))
        self.active_log = log
        self.current_responses = {}
        self.stream_error = None
        self.stream_results = 0
        log("runlog.jsonl", {"event": "official_session_reset", "new_history": True,
                            "new_cross_turn_penalty": True, "previous_case_kv_context_reused": False})
        owner = self
        class EngineTap:
            def __getattr__(self, name):
                return getattr(owner.original_engine, name)

            async def generate(self, *args, **kwargs):
                try:
                    async for result in owner.original_engine.generate(*args, **kwargs):
                        owner.stream_results += 1
                        outputs = [{"index": out.index, "text": out.text,
                                    "token_ids": list(out.token_ids), "finish_reason": out.finish_reason,
                                    "stop_reason": out.stop_reason} for out in result.outputs]
                        log("raw-output.jsonl", {"kind": "vllm_stream", "request_id": result.request_id,
                            "prompt_token_ids": list(result.prompt_token_ids or []), "outputs": outputs,
                            "finished": result.finished})
                        if any(out.finish_reason == "error" for out in result.outputs):
                            raise RuntimeError("Official vLLM returned finish_reason=error")
                        if result.finished:
                            owner.case_usage["model_calls"] += 1
                            owner.case_usage["prompt_tokens"] += len(result.prompt_token_ids or [])
                            owner.case_usage["completion_tokens"] += sum(len(out.token_ids) for out in result.outputs)
                        yield result
                except Exception:
                    owner.stream_error = traceback.format_exc()
                    raise
        module.async_engine = EngineTap()
        def output(token, response_id, **flags):
            received = log("raw-output.jsonl", {"kind": "official_text_delivery", "text": token,
                                              "response_id": response_id, "flags": flags})
            if flags.get("is_silent"):
                return
            if token:
                response = self.current_responses.setdefault(response_id, {"response_id": response_id,
                    "text": "", "start_media_s": received["media_wall_s"], "end_media_s": received["media_wall_s"],
                    "onset_source": "first_text_receive", "first_receive_utc": received["utc"],
                    "first_receive_monotonic_s": received["monotonic_s"]})
                response["text"] += token
                response["end_media_s"] = received["media_wall_s"]
        module.send_streaming_token_to_client = output

    def process_frames(self, prepared_dir, frames, prompt, log):
        import numpy as np
        from PIL import Image
        module = self.module
        arrays = []
        for frame in frames:
            with Image.open(prepared_dir / frame["file"]) as image:
                arrays.append(np.asarray(image.convert("RGB")))
        array = np.stack(arrays)
        metadata = {"fps": 1.0, "duration": float(self.manifest["duration_sec"]),
                    "total_num_frames": len(self.manifest["frames"]),
                    "frames_indices": [f["pts_sec"] for f in frames], "video_backend": "source_jpeg",
                    "do_sample_frames": False}
        if len(frames) == 1:
            array = np.concatenate([array, array], axis=0)
            metadata["frames_indices"] = [frames[0]["pts_sec"], frames[0]["pts_sec"]]
            log("input.jsonl", {"kind": "official_temporal_factor_repeat", "source_frame": frames[0],
                               "native_occurrences": 2, "future_frame_added": False})
        log("input.jsonl", {"kind": "native_video_metadata", "metadata": metadata,
                            "frames": frames, "timestamp_origin": "actual_source_PTS_not_per_turn_reset"})
        self.current_responses = {}
        self.stream_results = 0
        sampling = module.SamplingParams(temperature=0.5, max_tokens=128, seed=self.case_seed + self.turn_index,
                     **self.session.cross_turn_penalty.build_sampling_kwargs())
        self.loop.run_until_complete(module.generate_response_with_video(
                                    self.session, (array, metadata), prompt, sampling, self.args))
        if self.stream_error:
            raise RuntimeError(self.stream_error)
        if self.stream_results == 0:
            raise RuntimeError("Official AURA helper returned without a model result")
        return {"responses": list(self.current_responses.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("config", "prepared", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    model = Adapter.load(json.loads(Path(args.config).read_text()))
    result = model.run_case(Path(args.prepared), Path(args.output))
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)
