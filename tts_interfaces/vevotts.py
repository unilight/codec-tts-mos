import sys
import os
import torch
from huggingface_hub import snapshot_download

# Relative import for base class
from .base import BaseTTSModel

class VevoTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned Amphion repository
        self.repo_path = "/data/group1/z44476r/Experiments/Amphion"
        
        # Set espeak library path specifically for Vevo (Amphion)
        # Required before phonemizer is initialized in imports
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/data/group1/z44476r/espeak-ng/.local/lib/libespeak-ng.so"
        
        self.pipeline = None
        self.load_model()

    def load_model(self):
        print(f"Loading VevoTTS from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        # This allows 'from models.vc.vevo...' to work
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from models.vc.vevo.vevo_utils import VevoInferencePipeline, save_audio
        except ImportError as e:
            raise ImportError(f"Could not import VevoTTS modules. Check if '{self.repo_path}' is correct. Error: {e}")

        # 2. DOWNLOAD/LOCATE CHECKPOINTS (HuggingFace)
        hf_repo = "amphion/Vevo"
        print(f"Downloading/Loading checkpoints from HF: {hf_repo}...")
        
        def get_ckpt_path(pattern, subpath):
            local_dir = snapshot_download(
                repo_id=hf_repo,
                repo_type="model",
                allow_patterns=[pattern],
            )
            return os.path.join(local_dir, subpath)

        # Download components
        # Tokenizer
        content_style_tokenizer_ckpt_path = get_ckpt_path("tokenizer/vq8192/*", "tokenizer/vq8192")
        # AR Model
        ar_ckpt_path = get_ckpt_path("contentstyle_modeling/PhoneToVq8192/*", "contentstyle_modeling/PhoneToVq8192")
        # Flow Matching
        fmt_ckpt_path = get_ckpt_path("acoustic_modeling/Vq8192ToMels/*", "acoustic_modeling/Vq8192ToMels")
        # Vocoder
        vocoder_ckpt_path = get_ckpt_path("acoustic_modeling/Vocoder/*", "acoustic_modeling/Vocoder")

        # 3. CONFIG PATHS
        # These are expected to be in the local repo clone
        # Construct absolute paths based on self.repo_path
        ar_cfg_path = os.path.join(self.repo_path, "models/vc/vevo/config/PhoneToVq8192.json")
        fmt_cfg_path = os.path.join(self.repo_path, "models/vc/vevo/config/Vq8192ToMels.json")
        vocoder_cfg_path = os.path.join(self.repo_path, "models/vc/vevo/config/Vocoder.json")

        # Check configs exist
        for p in [ar_cfg_path, fmt_cfg_path, vocoder_cfg_path]:
            if not os.path.exists(p):
                 raise FileNotFoundError(f"VevoTTS config file not found: {p}")

        # 4. INITIALIZE PIPELINE
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pipeline = VevoInferencePipeline(
            content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
            ar_cfg_path=ar_cfg_path,
            ar_ckpt_path=ar_ckpt_path,
            fmt_cfg_path=fmt_cfg_path,
            fmt_ckpt_path=fmt_ckpt_path,
            vocoder_cfg_path=vocoder_cfg_path,
            vocoder_ckpt_path=vocoder_ckpt_path,
            device=self.device,
        )

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None, **kwargs):
        from models.vc.vevo.vevo_utils import save_audio
        
        if self.pipeline is None:
            raise RuntimeError("VevoTTS pipeline is not initialized.")

        # VevoTTS relies heavily on the reference text for alignment/style
        if not ref_text:
             print(f"[VevoTTS Warning] ref_text is None. Quality might be suboptimal.")
        
        # In the example, timbre_ref_wav_path defaults to ref_wav_path if not provided
        timbre_ref_wav_path = ref_audio_path

        # Assume source text language and reference text language are the same 
        # unless logic changes (passed via kwargs etc)
        src_language = language
        ref_language = language

        gen_audio = self.pipeline.inference_ar_and_fm(
            src_wav_path=None,
            src_text=text,
            style_ref_wav_path=ref_audio_path,
            timbre_ref_wav_path=timbre_ref_wav_path,
            style_ref_wav_text=ref_text,
            src_text_language=src_language,
            style_ref_wav_text_language=ref_language,
        )

        save_audio(gen_audio, output_path=output_path)