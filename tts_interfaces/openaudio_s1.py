import sys
import os
import torch
import torchaudio
import soundfile as sf
import numpy as np

# Relative import for base class
from .base import BaseTTSModel

class OpenAudioS1Model(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Path to the Fish Speech repository
        self.repo_path = "/data/group1/z44476r/Experiments/fish-speech"
        
        # Checkpoint paths (Relative to repo_path)
        self.vq_checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
        self.vq_config_name = "modded_dac_vq"
        self.llm_checkpoint_path = "checkpoints/openaudio-s1-mini"
        
        self.vq_model = None
        self.llm_model = None
        self.decode_one_token = None
        self.modules = {} # To hold dynamically imported modules
        
        self.load_model()

    def load_model(self):
        print(f"Loading OpenAudio-S1 (Fish Speech) from {self.repo_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            # 1. IMPORT VQ MODULES
            # We import the load_model function from the DAC inference script
            from fish_speech.models.dac.inference import load_model as load_vq_model
            
            # 2. IMPORT LLM MODULES
            from fish_speech.models.text2semantic.inference import (
                init_model as init_llm_model, 
                generate_long
            )
            
            self.modules['load_vq_model'] = load_vq_model
            self.modules['init_llm_model'] = init_llm_model
            self.modules['generate_long'] = generate_long
            
            # 3. LOAD VQ MODEL
            # Hydras initialization inside load_model might need CWD to be repo root or config paths carefully handled.
            # We try calling it. If it fails due to config paths, we might need to chdir temporarily.
            cwd = os.getcwd()
            try:
                os.chdir(self.repo_path)
                print("Loading VQ Codec...")
                self.vq_model = load_vq_model(
                    config_name=self.vq_config_name,
                    checkpoint_path=self.vq_checkpoint_path,
                    device=self.device
                )
            finally:
                os.chdir(cwd)

            # 4. LOAD LLM MODEL
            print("Loading Text2Semantic LLM...")
            # Precision: torch.half or torch.bfloat16
            precision = torch.half if self.device == 'cuda' else torch.float32
            
            # init_model returns (model, decode_fn)
            self.llm_model, self.decode_one_token = init_llm_model(
                checkpoint_path=os.path.join(self.repo_path, self.llm_checkpoint_path),
                device=self.device,
                precision=precision,
                compile=False # Compilation can take time, disabled for benchmark speed
            )
            
            # Setup caches
            with torch.device(self.device):
                self.llm_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.llm_model.config.max_seq_len,
                    dtype=next(self.llm_model.parameters()).dtype,
                )
                
        except ImportError as e:
            print(f"\n[!] Failed to import Fish Speech modules from {self.repo_path}")
            print("Ensure dependencies (hydra, omegaconf, etc.) are installed.")
            raise e
        except Exception as e:
            print(f"Error loading OpenAudio-S1: {e}")
            raise e

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        if not ref_text:
            print("Warning: OpenAudio-S1 requires reference text. Using empty string.")
            ref_text = ""
            
        generate_long = self.modules['generate_long']
        
        try:
            # --- STEP 1: ENCODE REFERENCE AUDIO (VQ) ---
            # Load and resample
            audio, sr = torchaudio.load(ref_audio_path)
            
            # Mix to mono if necessary
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
                
            # Resample to model rate (usually 44100 for DAC, but checked dynamically)
            if sr != self.vq_model.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.vq_model.sample_rate)
            
            audios = audio[None].to(self.device)
            audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
            
            # Encode
            with torch.no_grad():
                indices, _ = self.vq_model.encode(audios, audio_lengths)
            
            if indices.ndim == 3:
                indices = indices[0] # (num_codebooks, T)
                
            # Prompt tokens expects the tensor directly
            # The generate_long function moves them to CPU internally if needed for lists
            prompt_tokens = indices
            
            # --- STEP 2: GENERATE SEMANTIC TOKENS (LLM) ---
            
            # Run generation
            generator = generate_long(
                model=self.llm_model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=0, # 0 means auto based on length? Or default
                top_p=0.8,
                repetition_penalty=1.1,
                temperature=0.8,
                compile=False,
                iterative_prompt=True,
                chunk_length=300,
                prompt_text=[ref_text],
                prompt_tokens=[prompt_tokens]
            )
            
            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break
            
            if not codes:
                print("Error: No codes generated.")
                return

            # Concatenate generated codes
            # response.codes is (num_codebooks, T_chunk)
            final_codes = torch.cat(codes, dim=1).to(self.device)
            
            # --- STEP 3: DECODE TO AUDIO (VQ) ---
            
            # Decode expects (B, num_codebooks, T)
            # final_codes is (num_codebooks, T) -> Unsqueeze to (1, NC, T)
            features = final_codes.unsqueeze(0)
            feature_lens = torch.tensor([features.shape[2]], device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                fake_audios, _ = self.vq_model.decode(features, feature_lens)
                
            # Save
            fake_audio = fake_audios[0, 0].float().cpu().numpy()
            sf.write(output_path, fake_audio, self.vq_model.sample_rate)
            
        except Exception as e:
            print(f"OpenAudio-S1 Generation Error: {e}")
            raise e