import importlib

# Registry maps: model_name -> (module_relative_path, class_name)
# This defines the available models without actually importing their code yet.
MODEL_REGISTRY = {
    "xtts": (".xtts_v2", "XTTSv2Model"),
    "llasa": (".llasa_1b", "Llasa1BModel"),
    "llasa_1b": (".llasa_1b", "Llasa1BModel"),
    "llasa_3b": (".llasa_1b", "Llasa3BModel"),
    "llasa_8b": (".llasa_1b", "Llasa8BModel"),
    "llasa_1b_multilingual": (".llasa_1b_multilingual", "Llasa1BMultilingualModel"),
    "fireredtts": (".fireredtts", "FireRedTTSModel"),
    "cosyvoice2": (".cosyvoice2", "CosyVoice2Model"),
    "sparktts": (".sparktts", "SparkTTSModel"),
    "valle-x": (".vallex", "ValleXModel"),
    "chatterbox": (".chatterbox", "ChatterboxTTSModel"),
    "chatterbox-mtl": (".chatterbox_mtl", "ChatterboxMultilingualTTSModel"),
    "indextts": (".indextts", "IndexTTSModel"),
    "orpheus-tts": (".orpheustts", "OrpheusTTSModel"),
    # "higgs": (".higgs_audio", "HiggsAudioModel"),
    "tortoise": (".tortoise_tts", "TortoiseTTSModel"),
    "openaudio": (".openaudio_s1", "OpenAudioS1Model"),
    "fishaudio-s2-pro": (".fishaudio_s2_pro", "FishAudioS2ProModel"),
    "t5gemma-tts": (".t5gemma_tts", "T5GemmaTTSModel"),
    "openvoice_v2": (".openvoice_v2", "OpenVoiceV2TTSModel"),
    "maskgct": (".maskgct", "MaskGCTModel"),
    "vevo": (".vevotts", "VevoTTSModel"),
    "voicestar": (".voicestar", "VoiceStarModel"),
    "neutts_air": (".neutts_air", "NeuTTSAirModel"),
    "qwen3_tts_0_6b": (".qwen3_tts", "Qwen3_0_6B"),
    "qwen3_tts_1_7b": (".qwen3_tts", "Qwen3_1_7B"),
}

# We expose the keys so `run.py` can still use `AVAILABLE_MODELS.keys()` for argparse choices
AVAILABLE_MODELS = MODEL_REGISTRY

def get_model(model_name, device=None):
    """Factory function to lazily instantiate a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}")
    
    module_path, class_name = MODEL_REGISTRY[model_name]
    
    print(f"[Lazy Import] Importing {class_name} from {module_path}...")
    
    # Dynamically import the module using importlib
    # package=__package__ ensures relative imports (like .maskgct) work correctly
    module = importlib.import_module(module_path, package=__package__)
    
    # Get the class from the module
    model_class = getattr(module, class_name)
    
    return model_class(device=device)