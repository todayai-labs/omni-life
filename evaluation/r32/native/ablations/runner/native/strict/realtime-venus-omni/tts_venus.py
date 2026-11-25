"""Task-speech input probe; imports the ORIGINAL frozen Venus loader."""
from tts_native import NativeTTSAdapter


class Adapter(NativeTTSAdapter):
    module_name = "adapters.venus_v2"
    model_id = "realtime-venus-omni"
