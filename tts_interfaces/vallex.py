import sys
import os
import soundfile as sf
from pathlib import Path
import torch

# Relative import for base class
from .base import BaseTTSModel

class ValleXModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Path to your VALL-E-X repository
        self.repo_path = "/data/group1/z44476r/Experiments/VALL-E-X"
        
        self.generation_utils = None
        self.prompt_utils = None
        self.SAMPLE_RATE = 24000 # Fallback
        
        self.load_model()

    def load_model(self):
        print(f"Loading VALL-E-X from {self.repo_path}...")

        # 1. ADD REPO TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            # 2. IMPORT UTILS
            # VALL-E-X depends heavily on these modules being available
            import utils.generation as generation_utils
            import utils.prompt_making as prompt_utils
            
            self.generation_utils = generation_utils
            self.prompt_utils = prompt_utils
            
            # Get sample rate if available
            self.SAMPLE_RATE = getattr(generation_utils, "SAMPLE_RATE", 24000)
            
            # 3. PRELOAD MODELS
            # This downloads/loads checkpoints. 
            # Note: VALL-E-X looks for checkpoints in the current working directory or specific paths.
            # If it fails to find checkpoints, you might need to symlink the 'checkpoints' folder 
            # from the repo to your current running directory.
            generation_utils.preload_models()
            
        except ImportError as e:
            print(f"\n[!] Failed to import VALL-E-X modules from {self.repo_path}")
            print("Ensure you are running in the 'vall-e-x' conda environment.")
            raise e

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        # VALL-E-X prompt making works best with transcript
        if ref_text is None:
            # It can work without transcript (whisper backend), but provided text is faster/better
            ref_text = "" 

        # 1. PREPARE PROMPT
        # We use the filename as the prompt name key
        ref_wav_id = Path(ref_audio_path).stem
        
        # make_prompt generates specific files (npz) in a 'customs' folder in CWD
        self.prompt_utils.make_prompt(
            name=ref_wav_id, 
            audio_prompt_path=ref_audio_path, 
            transcript=ref_text
        )

        # 2. GENERATE AUDIO
        # We pass the ID of the prompt we just created
        # language='auto' lets VALL-E-X detect language from text
        audio_array = self.generation_utils.generate_audio(
            text, 
            prompt=ref_wav_id,
            language='auto' 
        )

        # 3. SAVE
        # VALL-E-X returns a numpy array (usually float32)
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
            
        sf.write(output_path, audio_array, self.SAMPLE_RATE)