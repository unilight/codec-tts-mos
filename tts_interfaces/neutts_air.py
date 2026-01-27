import sys
import os
import soundfile as sf
import torch

# Relative import for base class
from .base import BaseTTSModel

class NeuTTSAirModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned NeuTTS-Air repository
        self.repo_path = "/data/group1/z44476r/Experiments/neutts-air"
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading NeuTTS-Air from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from neuttsair.neutts import NeuTTSAir
        except ImportError as e:
            raise ImportError(f"Could not import neuttsair. Check if '{self.repo_path}' is correct. Error: {e}")

        # Determine device string
        # self.device might be a torch.device object or None
        if self.device is None:
            dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev_str = str(self.device).split(":")[0] # Simplify 'cuda:0' to 'cuda' if needed, though usually explicit is fine. 
            # Keeping it simple based on example which used "cpu" string.
            if "cuda" in str(self.device):
                dev_str = "cuda"
            else:
                dev_str = "cpu"

        # Initialize as per example
        self.model = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air-q4-gguf",
            backbone_device=dev_str,
            codec_repo="neuphonic/neucodec",
            codec_device=dev_str
        )

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None, **kwargs):
        """
        Args:
            text (str): Target text.
            ref_audio_path (str): Path to reference audio.
            output_path (str): Path to save result.
            ref_text (str): Transcript of the reference audio.
        """
        if self.model is None:
            raise RuntimeError("NeuTTS-Air model is not initialized.")

        if ref_text is None:
             print(f"[NeuTTS-Air Warning] No ref_text provided for {output_path}. Using empty string.")
             ref_text = ""

        # 1. Encode Reference
        ref_codes = self.model.encode_reference(ref_audio_path)

        # 2. Run Inference
        # infer(input_text, ref_codes, ref_text)
        wav = self.model.infer(text, ref_codes, ref_text)

        # 3. Save Output
        # Example hardcodes 24000 Hz
        sf.write(output_path, wav, 24000)