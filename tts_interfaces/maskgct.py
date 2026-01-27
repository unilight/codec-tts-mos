import sys
import os
import torch
import soundfile as sf
import safetensors
from huggingface_hub import hf_hub_download

# Relative import for base class
from .base import BaseTTSModel

class MaskGCTModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned Amphion repository
        self.repo_path = "/data/group1/z44476r/Experiments/Amphion"
        
        self.pipeline = None
        self.load_model()

    def load_model(self):
        print(f"Loading MaskGCT from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        # This allows 'from models.tts.maskgct...' to work
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from models.tts.maskgct.maskgct_utils import (
                load_config,
                build_semantic_model,
                build_semantic_codec,
                build_acoustic_codec,
                build_t2s_model,
                build_s2a_model,
                MaskGCT_Inference_Pipeline
            )
        except ImportError as e:
            raise ImportError(f"Could not import MaskGCT modules. Check if '{self.repo_path}' is correct and points to the Amphion root. Error: {e}")

        # 2. LOAD CONFIG
        # Assuming standard Amphion structure
        cfg_path = os.path.join(self.repo_path, "models/tts/maskgct/config/maskgct.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found at {cfg_path}")
            
        cfg = load_config(cfg_path)

        # 3. BUILD MODEL COMPONENTS
        # Ensure we use the device passed to __init__
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Building Semantic Model...")
        semantic_model, semantic_mean, semantic_std = build_semantic_model(self.device)
        
        print("Building Semantic Codec...")
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, self.device)
        
        print("Building Acoustic Codec...")
        codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, self.device)
        
        print("Building T2S Model...")
        t2s_model = build_t2s_model(cfg.model.t2s_model, self.device)
        
        print("Building S2A Models...")
        s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, self.device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, self.device)

        # 4. DOWNLOAD CHECKPOINTS (HuggingFace)
        # This will cache models locally
        hf_repo = "amphion/MaskGCT"
        print(f"Downloading/Loading checkpoints from HF: {hf_repo}...")
        
        try:
            semantic_code_ckpt = hf_hub_download(hf_repo, "semantic_codec/model.safetensors")
            codec_encoder_ckpt = hf_hub_download(hf_repo, "acoustic_codec/model.safetensors")
            codec_decoder_ckpt = hf_hub_download(hf_repo, "acoustic_codec/model_1.safetensors")
            t2s_model_ckpt = hf_hub_download(hf_repo, "t2s_model/model.safetensors")
            s2a_1layer_ckpt = hf_hub_download(hf_repo, "s2a_model/s2a_model_1layer/model.safetensors")
            s2a_full_ckpt = hf_hub_download(hf_repo, "s2a_model/s2a_model_full/model.safetensors")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoints from HuggingFace. Network error? Detail: {e}")

        # 5. LOAD CHECKPOINTS
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

        # 6. INITIALIZE PIPELINE
        self.pipeline = MaskGCT_Inference_Pipeline(
            semantic_model,
            semantic_codec,
            codec_encoder,
            codec_decoder,
            t2s_model,
            s2a_model_1layer,
            s2a_model_full,
            semantic_mean,
            semantic_std,
            self.device,
        )

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None, **kwargs):
        """
        Args:
            text (str): The target text to synthesize.
            ref_audio_path (str): Path to the reference audio (prompt).
            output_path (str): Path to save the output wav.
            language (str): Target language code (e.g., 'en', 'zh').
            ref_text (str): Text content of the reference audio. Highly recommended for MaskGCT.
        """
        if self.pipeline is None:
            raise RuntimeError("MaskGCT pipeline is not initialized.")

        if not ref_text:
            print(f"[MaskGCT Warning] No 'ref_text' provided for {output_path}. Quality might degrade.")
            ref_text = ""

        # MaskGCT supports 'en' and 'zh' primarily currently, but we pass through the lang arg
        # Assuming prompt language matches target language for zero-shot TTS tasks usually
        prompt_language = language 
        target_language = language

        # Run Inference
        # target_len=None lets the model predict the duration
        recovered_audio = self.pipeline.maskgct_inference(
            ref_audio_path,
            ref_text,
            text,
            prompt_language,
            target_language,
            target_len=None 
        )

        # Save output
        # MaskGCT output is typically 24kHz
        sf.write(output_path, recovered_audio, 24000)