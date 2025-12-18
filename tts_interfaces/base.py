import os
from abc import ABC, abstractmethod
import torch
import soundfile as sf
import numpy as np
import librosa

class BaseTTSModel(ABC):
    """
    Abstract base class for TTS models.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    @abstractmethod
    def load_model(self):
        """Load model weights/configs here."""
        pass

    @abstractmethod
    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        """
        text: str
            The text to synthesize.
        ref_audio_path: str
            Path to the reference audio for zero-shot cloning.
        output_path: str
            Where to save the result.
        language: str
            Language code.
        ref_text: str (Optional)
            The transcript of the reference audio. Required for some models (e.g., Llasa).
        """
        pass

class MockTTSModel(BaseTTSModel):
    def load_model(self):
        print("Mock model loaded.")

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        import shutil
        print(f"Mock Generating: '{text[:20]}...' (Ref Text: {ref_text[:10] if ref_text else 'None'}) -> {output_path}")
        if os.path.exists(ref_audio_path):
            shutil.copy(ref_audio_path, output_path)
        else:
            with open(output_path, 'w') as f:
                f.write("dummy wav content")