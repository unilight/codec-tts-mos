import sys
import os
import torch
import soundfile as sf

# Relative import for base class
from .base import BaseTTSModel

class SparkTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned SparkTTS repository
        self.repo_path = "/data/group1/z44476r/Experiments/Spark-TTS" 
        
        # Point this to the specific model folder (e.g., Spark-TTS-0.5B)
        self.model_dir_name = "pretrained_models/Spark-TTS-0.5B"
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading SparkTTS from {self.repo_path}...")

        # 1. ADD REPO TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        # 2. IMPORT THE MODEL
        from cli.SparkTTS import SparkTTS
            
        # Construct absolute path to model weights
        model_path = os.path.join(self.repo_path, self.model_dir_name)
            
        # Initialize
        self.model = SparkTTS(model_path, self.device)
            
    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        # SparkTTS requires the prompt transcript
        if not ref_text:
            print(f"Warning: SparkTTS requires 'ref_text'. Using empty string.")
            ref_text = ""

        wav = self.model.inference(
            text,
            ref_audio_path,
            ref_text,
        )
        sf.write(output_path, wav, 16000)
