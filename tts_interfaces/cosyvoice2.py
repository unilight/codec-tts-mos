import sys
import os
import torch
import torchaudio

# Relative import for base class
from .base import BaseTTSModel

class CosyVoice2Model(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned FireRedTTS repository
        self.repo_path = "/data/group1/z44476r/Experiments/CosyVoice" 
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading CosyVoice2 0.5B from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        # This allows 'from fireredtts.fireredtts import ...' to work
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)
            sys.path.append(os.path.join(self.repo_path, 'third_party/Matcha-TTS'))

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        except ImportError:
            raise ImportError(f"Could not import CosyVoice. Check if '{self.repo_path}' is correct.")

        # 3. INITIALIZE
        self.model = CosyVoice2("/data/group1/z44476r/Experiments/CosyVoice/pretrained_models/hf-CosyVoice2-0.5B", load_jit=False, load_trt=False, fp16=False)

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        from cosyvoice.utils.file_utils import load_wav

        # prompt_speech_16k = load_wav(ref_audio_path, 16000)[:16000*30]
        # audios = list(self.model.inference_zero_shot(text, ref_text, prompt_speech_16k, stream=False))

        audios = list(self.model.inference_zero_shot(text, ref_text, ref_audio_path, stream=False))

        assert len(audios) == 1
        audio = audios[0]["tts_speech"]

        # Save
        torchaudio.save(output_path, audio.detach().cpu(), self.model.sample_rate)