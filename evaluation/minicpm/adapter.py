"""Released MiniCPM-o 4.5 duplex, using the shared causal AV recording harness.

No external ASR, synthesized task audio, forced questions, or forced listening. Native generate uses the official README greedy/20 example.
"""
from pathlib import Path
import json

from .causal_av import Adapter as RecordedAVAdapter


class OfficialTextDuplex:
    def __init__(self, duplex):
        self.official_duplex = duplex

    def __getattr__(self, name):
        return getattr(self.official_duplex, name)

    def streaming_generate(self, **_recording_harness_options):
        return self.official_duplex.streaming_generate(
            prompt_wav_path=None, decode_mode="greedy",
            max_new_speak_tokens_per_chunk=20)


class Adapter(RecordedAVAdapter):
    @classmethod
    def load(cls, config):
        import numpy as np
        import torch
        from transformers import AutoModel
        self = cls()
        self.config = dict(config)
        self.np, self.torch = np, torch
        torch.manual_seed(config["seed"])
        self.model = AutoModel.from_pretrained(
            config["model_path"], trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            init_vision=True, init_audio=True, init_tts=True).eval().cuda()
        native = self.model.as_duplex(generate_audio=False, force_listen_count=0)
        self.duplex = OfficialTextDuplex(native)
        self.tokenizer = native.tokenizer
        self.configure_length_penalty = lambda *_: ({}, "not_applied_official_minicpm")
        self.session_count = 0
        return self

    def run_case(self, prepared_dir, output_dir):
        result = super().run_case(prepared_dir, output_dir)
        settings = result["settings"]
        for name in ("length_penalty", "memory_minutes"):
            settings.pop(name, None)
        settings.update(model_id="minicpm-o-4.5", model_path=self.config["model_path"],
                        decode_mode="greedy", max_new_speak_tokens_per_chunk=20,
                        force_listen_count=0, audio_route="native_audio_encoder",
                        init_tts=True, tts_output_enabled=False, official_checkpoint_revision=self.config["checkpoint_revision"],
                        official_defaults=self.duplex.official_duplex._default_duplex_params,
                        native_api="MiniCPMO.as_duplex(generate_audio=False)")
        return result
