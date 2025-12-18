import sys
import os

from .base import BaseTTSModel

class IndexTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        self.repo_path = "/data/group1/z44476r/Experiments/index-tts"
        
        self.model = None
        self.load_model()

    def load_model(self):
        print(f"Loading IndexTTS from {self.repo_path}...")

        # 1. ADD REPO TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        from indextts.infer_v2 import IndexTTS2
        
        # 2. CONSTRUCT ABSOLUTE PATHS
        # The example uses "checkpoints/config.yaml" and "checkpoints"
        # We assume these are relative to the repo root.
        cfg_path = os.path.join(self.repo_path, "checkpoints", "config.yaml")
        model_dir = os.path.join(self.repo_path, "checkpoints")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found at: {cfg_path}")

        # 3. INITIALIZE
        # We stick to the flags provided in your example. 
        # You can dynamically set use_fp16 based on self.device if needed, 
        # but usually it's safer to stick to the working defaults.
        self.model = IndexTTS2(
            cfg_path=cfg_path, 
            model_dir=model_dir, 
            use_fp16=False, 
            use_cuda_kernel=False, 
            use_deepspeed=False
        )
            

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        # IndexTTS infer method signature:
        # infer(self, spk_audio_prompt, text, output_path, verbose=False)
        
        self.model.infer(
            spk_audio_prompt=ref_audio_path, 
            text=text, 
            output_path=output_path, 
            verbose=False
        )