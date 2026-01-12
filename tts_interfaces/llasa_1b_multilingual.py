import torch
import soundfile as sf
import librosa
import os

from .llasa_1b import Llasa1BModel

class Llasa1BMultilingualModel(Llasa1BModel):
    def load_model(self):
        print("Loading Llasa-1B-Multilingual and XCodec2...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Ensure xcodec2 is installed or in python path
        from xcodec2.modeling_xcodec2 import XCodec2Model

        llasa_1b = 'HKUSTAudio/Llasa-1B-Multilingual'
        self.tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
        self.model = AutoModelForCausalLM.from_pretrained(llasa_1b)
        self.model.eval()
        self.model.to(self.device)

        model_path = "HKUSTAudio/xcodec2"
        self.codec_model = XCodec2Model.from_pretrained(model_path)
        self.codec_model.eval().cuda() # Codec usually requires CUDA based on demo