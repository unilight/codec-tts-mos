import sys
import os
import torch
import torchaudio

# Relative import for base class
from .base import BaseTTSModel

class FireRedTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned FireRedTTS repository
        self.repo_path = "/data/group1/z44476r/Experiments/FireRedTTS" 
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading FireRedTTS from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        # This allows 'from fireredtts.fireredtts import ...' to work
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from fireredtts.fireredtts import FireRedTTS
        except ImportError:
            raise ImportError(f"Could not import FireRedTTS. Check if '{self.repo_path}' is correct.")

        # 2. CONSTRUCT ABSOLUTE PATHS
        # The model needs to find its configs relative to its own folder, 
        # or we explicitly give it absolute paths.
        config_path = os.path.join(self.repo_path, "configs/config_24k.json")
        
        # You might need to change 'pretrained_models' to the actual folder name inside the repo
        pretrained_path = os.path.join(self.repo_path, "pretrained_models/fireredtts1") 

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at: {config_path}")

        # 3. INITIALIZE
        self.model = FireRedTTS(
            config_path=config_path,
            pretrained_path=pretrained_path
        )

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        # Map common language codes to what FireRedTTS expects
        # The demo uses "zh", assuming it supports "en" as well.
        lang_map = {
            "en": "en",
            "zh": "zh",
            "cn": "zh"
        }
        target_lang = lang_map.get(language.lower(), "en")

        # Synthesize
        # The demo returns a tensor
        try:
            rec_wavs = self.model.synthesize(
                prompt_wav=ref_audio_path,
                text=text,
                lang=target_lang,
            )
            
            # Post-process
            rec_wavs = rec_wavs.detach().cpu()
            
            # Save (Demo uses 24k)
            torchaudio.save(output_path, rec_wavs, 24000)
            
        except Exception as e:
            print(f"FireRedTTS Generation Error: {e}")
            raise e