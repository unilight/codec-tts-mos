import sys
import os
import torch
import soundfile as sf

# Relative import for base class
from .base import BaseTTSModel

class Qwen3Wrapper(BaseTTSModel):
    def __init__(self, device=None, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the specific model path or HuggingFace repo ID
        self.model_name_or_path = model_name
        
        # Optional: If you have the qwen_tts library locally
        # self.repo_path = "/path/to/Qwen3-TTS-Repo" 
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading Qwen3-TTS from {self.model_name_or_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH (Optional if installed via pip)
        # if hasattr(self, 'repo_path') and self.repo_path not in sys.path:
        #     sys.path.append(self.repo_path)

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError("Could not import qwen_tts. Ensure it is installed or the path is added.")

        # 2. DETERMINE DEVICE MAP
        # If self.device is specific (e.g., 'cuda:0'), use it. Otherwise allow 'auto'.
        device_map = self.device if self.device else "auto"

        # 3. INITIALIZE
        self.model = Qwen3TTSModel.from_pretrained(
            self.model_name_or_path,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def generate(self, text, ref_audio_path, output_path, language="English", ref_text=None):
        """
        Generates audio using Qwen3-TTS voice cloning.
        
        Args:
            text (str): The target text to synthesize.
            ref_audio_path (str): Path to the reference audio file.
            output_path (str): Path where the output wav will be saved.
            language (str): Target language (default "English").
            ref_text (str): The transcript of the reference audio.
        """
        if ref_text is None:
            print("Warning: Qwen3-TTS typically requires `ref_text` for accurate voice cloning.")

        # The model returns (wavs, sr)
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )

        # Qwen3 returns a list of waveforms, we take the first one
        audio_data = wavs[0]

        # Save using soundfile (as per Qwen3 docs)
        sf.write(output_path, audio_data, sr)

class Qwen3_1_7B(Qwen3Wrapper):
    """Wrapper specifically for the 1.7B Base model."""
    def __init__(self, device=None):
        super().__init__(device, model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")

class Qwen3_0_6B(Qwen3Wrapper):
    """Wrapper specifically for the 0.6B Base model."""
    def __init__(self, device=None):
        super().__init__(device, model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base")