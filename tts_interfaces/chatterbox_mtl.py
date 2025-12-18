import sys
import os
import torch
import torchaudio

# Relative import for base class
from .base import BaseTTSModel

class ChatterboxMultilingualTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Path to your Chatterbox repository/experiment folder
        self.repo_path = "/data/group1/z44476r/Experiments/chatterbox"
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading Chatterbox from {self.repo_path}...")

        # 1. ADD REPO TO PYTHONPATH
        # This ensures we can import chatterbox if it's not installed globally
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
        # 2. LOAD MODEL
        # The example uses .from_pretrained(device="cuda")
        # We map self.device (which might be 'cuda' or 'cpu') to that argument.
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            
    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        wav = self.model.generate(text, audio_prompt_path=ref_audio_path, language_id=language)
            
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu()
        
        torchaudio.save(output_path, wav, self.model.sr)