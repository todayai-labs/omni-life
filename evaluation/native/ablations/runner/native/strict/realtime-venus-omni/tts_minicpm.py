"""Task-speech input probe; imports the ORIGINAL frozen MiniCPM loader."""
from tts_native import NativeTTSAdapter


class Adapter(NativeTTSAdapter):
    module_name = "adapters.minicpm_native"
    model_id = "minicpm-o-4.5-local"
