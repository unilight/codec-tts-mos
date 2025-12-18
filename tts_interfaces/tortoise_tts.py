import sys
import os
import soundfile as sf
import torch

# Relative import for base class
from .base import BaseTTSModel

class TortoiseTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Path to your Tortoise-TTS repository/experiment folder
        self.repo_path = "/data/group1/z44476r/Experiments/tortoise-tts"
        
        self.tts = None
        self.utils_audio = None
        self.load_model()

    def load_model(self):
        print(f"Loading Tortoise-TTS from {self.repo_path}...")

        # 1. ADD REPO TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from tortoise import api, utils
            
            # Store utils for loading audio later
            self.utils_audio = utils.audio
            
            # 2. INITIALIZE
            # We initialize with defaults as per your snippet. 
            # Note: TextToSpeech() automatically handles device placement (cuda if available)
            # You can pass use_deepspeed=True or kv_cache=True here if your setup supports it.
            self.tts = api.TextToSpeech()
            
        except ImportError as e:
            print(f"\n[!] Failed to import Tortoise-TTS from {self.repo_path}")
            print("Ensure you are running in the correct environment.")
            raise e

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        # Snippet usage:
        # reference_clips = [tortoise.utils.audio.load_audio(path, 22050)]
        # pcm_audio = tts.tts_with_preset(text, voice_samples=reference_clips, preset='fast')
        
        try:
            # 1. Load Reference Audio
            # Tortoise expects reference audio loaded at 22050Hz
            ref_clip = self.utils_audio.load_audio(ref_audio_path, 22050)
            reference_clips = [ref_clip]
            
            # 2. Generate
            # preset='fast' is used as per your snippet. 
            # Other options: 'ultra_fast', 'standard', 'high_quality'
            pcm_audio = self.tts.tts_with_preset(
                text, 
                voice_samples=reference_clips, 
                preset='fast'
            )
            
            # 3. Save
            # Tortoise outputs a tensor of shape (1, T). Squeeze to (T,).
            # Output sample rate is 24000Hz.
            if isinstance(pcm_audio, torch.Tensor):
                pcm_audio = pcm_audio.cpu().numpy()
            
            sf.write(output_path, pcm_audio.squeeze(), 24000)
            
        except Exception as e:
            print(f"Tortoise-TTS Generation Error: {e}")
            raise e