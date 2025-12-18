from .xtts_v2 import XTTSv2Model
from .llasa_1b import Llasa1BModel
from .fireredtts import FireRedTTSModel
from .cosyvoice2 import CosyVoice2Model
from .sparktts import SparkTTSModel
from .vallex import ValleXModel
from .chatterbox import ChatterboxTTSModel
from .chatterbox_mtl import ChatterboxMultilingualTTSModel
from .indextts import IndexTTSModel
from .orpheustts import OrpheusTTSModel
# from .higgs_audio import HiggsAudioModel 
from .tortoise_tts import TortoiseTTSModel
from .openaudio_s1 import OpenAudioS1Model

# This registry maps command-line arguments to Model Classes
AVAILABLE_MODELS = {
    "xtts": XTTSv2Model,
    "llasa": Llasa1BModel,
    "fireredtts": FireRedTTSModel,
    "cosyvoice2": CosyVoice2Model,
    "sparktts": SparkTTSModel,
    "valle-x": ValleXModel,
    "chatterbox": ChatterboxTTSModel,
    "chatterbox-mtl": ChatterboxMultilingualTTSModel,
    "indextts": IndexTTSModel,
    "orpheus-tts": OrpheusTTSModel,
    # "higgs": HiggsAudioModel,
    "tortoise": TortoiseTTSModel,
    "openaudio": OpenAudioS1Model,
}

def get_model(model_name, device=None):
    """Factory function to instantiate a model by name."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_class = AVAILABLE_MODELS[model_name]
    return model_class(device=device)