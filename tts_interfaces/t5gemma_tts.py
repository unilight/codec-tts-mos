import sys
import os
import torch
import soundfile as sf
import numpy as np

# Relative import for base class
from .base import BaseTTSModel

class T5GemmaTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Path to your T5Gemma-TTS repository
        self.repo_path = "/data/group1/z44476r/Experiments/T5Gemma-TTS"
        
        # Path to the model weights directory (HF format)
        # Default in snippet was "./t5gemma_voice_hf"
        self.model_dir = "Aratako/T5Gemma-TTS-2b-2b"
        
        self.model = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.model_args = None
        self.modules = {}
        
        self.load_model()

    def load_model(self):
        print(f"Loading T5Gemma-TTS from {self.repo_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            # Import dependencies from the repo
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from data.tokenizer import AudioTokenizer
            from duration_estimator import estimate_duration
            from inference_tts_utils import (
                inference_one_sample,
                normalize_text_with_lang,
                get_audio_info,
                get_sample_rate,
            )
            
            # Store helper functions
            self.modules = {
                "estimate_duration": estimate_duration,
                "inference_one_sample": inference_one_sample,
                "normalize_text_with_lang": normalize_text_with_lang,
                "get_audio_info": get_audio_info,
                "get_sample_rate": get_sample_rate,
            }

            # 1. Load Model
            print("Loading Model Weights...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto" if self.device == 'cuda' else None,
            )
            
            # Manual move to device if not quantized/auto-mapped
            is_quantized = getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False)
            if self.device == 'cuda' and not is_quantized:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.model_args = self.model.config

            # 2. Load Text Tokenizer
            tokenizer_name = getattr(self.model_args, "text_tokenizer_name", None) or getattr(self.model_args, "t5gemma_model_name", None)
            if not tokenizer_name:
                 # Fallback if config doesn't specify, maybe try model_dir or standard gemma/t5 name
                 tokenizer_name = self.model_dir 
            self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            # 3. Load Audio Tokenizer
            # The snippet uses AudioTokenizer(backend="xcodec2", ...)
            self.audio_tokenizer = AudioTokenizer(
                backend="xcodec2",
                model_name=getattr(self.model_args, "xcodec2_model_name", None),
            )
            
        except ImportError as e:
            print(f"\n[!] Failed to import T5Gemma-TTS modules from {self.repo_path}")
            print("Ensure you are running with the repo in PYTHONPATH or correct environment.")
            raise e

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        if not ref_text:
            print("Warning: T5Gemma-TTS requires reference text. Using empty string (might cause issues).")
            ref_text = ""
            
        # Unpack modules
        inference_one_sample = self.modules["inference_one_sample"]
        normalize_text_with_lang = self.modules["normalize_text_with_lang"]
        estimate_duration = self.modules["estimate_duration"]
        get_audio_info = self.modules["get_audio_info"]
        get_sample_rate = self.modules["get_sample_rate"]
        
        try:
            # 1. Normalize Text
            # We assume 'en' maps to what the normalizer expects. 
            # The normalizer returns (text, lang_code).
            target_text, lang_code = normalize_text_with_lang(text, language)
            
            # Normalize Reference Text
            prefix_transcript, _ = normalize_text_with_lang(ref_text, lang_code)
            
            # 2. Estimate Duration
            # T5Gemma needs an estimated duration for generation
            target_generation_length = estimate_duration(
                target_text=target_text,
                reference_speech=ref_audio_path,
                reference_transcript=prefix_transcript,
                target_lang=lang_code,
                reference_lang=lang_code,
            )
            
            # 3. Audio Info for Prompt cutoff
            info = get_audio_info(ref_audio_path)
            cut_off_sec = 100 # Default from snippet
            prompt_end_frame = int(cut_off_sec * get_sample_rate(info))
            
            # 4. Decode Config
            # Using defaults from the snippet
            decode_config = {
                "top_k": 30,
                "top_p": 0.9,
                "min_p": 0,
                "temperature": 0.8,
                "stop_repetition": 3,
                "codec_audio_sr": self.audio_tokenizer.sample_rate,
                "codec_sr": getattr(self.model_args, "encodec_sr", 50),
                "silence_tokens": [],
                "sample_batch_size": 1,
            }

            # 5. Inference
            res = inference_one_sample(
                model=self.model,
                model_args=self.model_args,
                text_tokenizer=self.text_tokenizer,
                audio_tokenizer=self.audio_tokenizer,
                audio_fn=ref_audio_path,
                target_text=target_text,
                lang=lang_code,
                device=self.device,
                decode_config=decode_config,
                prompt_end_frame=prompt_end_frame,
                target_generation_length=target_generation_length,
                prefix_transcript=prefix_transcript,
                multi_trial=[],
                repeat_prompt=0,
                return_frames=False,
            )
            
            # 6. Save Output
            # res is (concat_audio, gen_audio)
            concat_audio, gen_audio = res
            
            # Move to CPU and numpy
            gen_audio_np = gen_audio[0].cpu().float().numpy().squeeze()
            
            # Save
            sf.write(output_path, gen_audio_np, self.audio_tokenizer.sample_rate)
            
        except Exception as e:
            print(f"T5Gemma-TTS Generation Error: {e}")
            raise e