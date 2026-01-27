import sys
import os
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace

# Relative import for base class
from .base import BaseTTSModel

class VoiceStarModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION: EDIT THIS PATH ---
        # Point this to the root of the cloned VoiceStar repository
        self.repo_path = "/data/group1/z44476r/Experiments/VoiceStar"
        self.model_name = "VoiceStar_840M_30s"
        
        self.model = None
        self.args = None
        self.phn2num = None
        self.audio_tokenizer = None
        self.text_tokenizer = None
        
        self.load_model()

    def load_model(self):
        print(f"Loading VoiceStar from {self.repo_path}...")

        # 1. DYNAMICALLY ADD TO PYTHONPATH
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from models import voice_star
            from data.tokenizer import AudioTokenizer, TextTokenizer
            # Storing module reference for inference
            self.inference_utils = __import__('inference_tts_utils') 
        except ImportError as e:
            raise ImportError(f"Could not import VoiceStar modules. Check if '{self.repo_path}' is correct. Error: {e}")

        # 2. DOWNLOAD/LOAD CHECKPOINT
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_dir = os.path.join(self.repo_path, "pretrained")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_fn = os.path.join(ckpt_dir, f"{self.model_name}.pth")

        if not os.path.exists(ckpt_fn):
            print(f"[Info] Downloading {self.model_name} checkpoint...")
            url = f"https://huggingface.co/pyp1/VoiceStar/resolve/main/{self.model_name}.pth?download=true"
            # Using wget via os.system as in the example, or use torch.hub.download_url_to_file
            os.system(f"wget '{url}' -O {ckpt_fn}")

        print(f"[Info] Loading checkpoint {ckpt_fn}...")
        # Add safe globals for Namespace if using newer torch versions
        try:
            torch.serialization.add_safe_globals([Namespace])
        except AttributeError:
            pass # Old torch versions don't need this

        bundle = torch.load(ckpt_fn, map_location=self.device, weights_only=True)
        self.args = bundle["args"]
        self.phn2num = bundle["phn2num"]
        
        # 3. INITIALIZE MODEL
        self.model = voice_star.VoiceStar(self.args)
        self.model.load_state_dict(bundle["model"])
        self.model.to(self.device)
        self.model.eval()

        # 4. INITIALIZE TOKENIZERS
        # Determine signature path based on config
        if self.args.n_codebooks == 4:
            sig_name = "encodec_6f79c6a8.th"
        elif self.args.n_codebooks == 8:
            sig_name = "encodec_8cb1024_giga.th"
        else:
            sig_name = "encodec_6f79c6a8.th"
            
        signature_path = os.path.join(self.repo_path, "pretrained", sig_name)
        if not os.path.exists(signature_path):
             print(f"[Warning] Encodec signature not found at {signature_path}. AudioTokenizer might fail.")

        self.audio_tokenizer = AudioTokenizer(signature=signature_path)
        self.text_tokenizer = TextTokenizer(backend="espeak")

    def estimate_duration(self, ref_audio_path, text):
        """Estimate duration based on seconds per character from reference."""
        info = torchaudio.info(ref_audio_path)
        audio_duration = info.num_frames / info.sample_rate
        length_text = max(len(text), 1)
        spc = audio_duration / length_text  
        return len(text) * spc

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None, **kwargs):
        """
        Runs VoiceStar inference.
        """
        inference_one_sample = self.inference_utils.inference_one_sample

        # 1. Setup Reference Text
        # The pipeline usually requires reference text. If missing, the original code uses Whisper.
        # Here we assume ref_text is provided by the benchmark runner. 
        # If absolutely missing, we'll pass empty string or handle it, but VoiceStar usually needs it.
        prefix_transcript = ref_text if ref_text else ""
        if not prefix_transcript:
            print(f"[VoiceStar Warning] No ref_text provided for {output_path}. Quality might suffer.")

        # 2. Duration Estimation
        # In the benchmark, we might not have explicit target duration, so we estimate.
        target_generation_length = self.estimate_duration(ref_audio_path, text)

        # 3. Configuration (Defaults from example)
        # Note: 'cut_off_sec' in example defaults to 100s, effectively using full ref audio.
        cut_off_sec = 100 
        info = torchaudio.info(ref_audio_path)
        prompt_end_frame = int(cut_off_sec * info.sample_rate)
        
        delay_pattern_increment = self.args.n_codebooks + 1

        # Configuration dictionary
        decode_config = {
            'top_k': kwargs.get('top_k', 10),
            'top_p': kwargs.get('top_p', 1),
            'min_p': kwargs.get('min_p', 1),
            'temperature': kwargs.get('temperature', 1),
            'stop_repetition': 3,
            'kvcache': 1,
            'codec_audio_sr': 16000,
            'codec_sr': 50,
            'silence_tokens': [],
            'sample_batch_size': 1
        }

        # 4. Inference
        try:
            concated_audio, gen_audio = inference_one_sample(
                self.model, 
                self.args, 
                self.phn2num, 
                self.text_tokenizer, 
                self.audio_tokenizer,
                ref_audio_path, 
                text,
                self.device, 
                decode_config,
                prompt_end_frame=prompt_end_frame,
                target_generation_length=target_generation_length,
                delay_pattern_increment=delay_pattern_increment,
                prefix_transcript=prefix_transcript,
                multi_trial=[],
                repeat_prompt=kwargs.get('repeat_prompt', 1),
            )

            # 5. Save
            # result is [1, T], take [0]
            gen_audio_tensor = gen_audio[0].cpu()
            
            # Ensure dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            torchaudio.save(output_path, gen_audio_tensor, 16000)
            
        except Exception as e:
            print(f"Error during VoiceStar generation: {e}")
            raise e