import os
import sys

import soundfile as sf
import torch

from .base import BaseTTSModel


class FishAudioS2ProModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)

        self.repo_path = "/mrnas04/internal/wenchin-h/Experiments/fish-speech"
        self.checkpoint_path = os.path.join(self.repo_path, "checkpoints/s2-pro")
        self.codec_checkpoint_path = os.path.join(self.checkpoint_path, "codec.pth")

        self.llm_model = None
        self.codec_model = None
        self.decode_one_token = None
        self.modules = {}

        self.load_model()

    def load_model(self):
        print(f"Loading Fish Audio S2 Pro from {self.repo_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from fish_speech.models.text2semantic.inference import (
                decode_to_audio,
                encode_audio,
                generate_long,
                init_model,
                load_codec_model,
            )

            self.modules["decode_to_audio"] = decode_to_audio
            self.modules["encode_audio"] = encode_audio
            self.modules["generate_long"] = generate_long
            self.modules["load_codec_model"] = load_codec_model

            precision = torch.bfloat16 if self.device == "cuda" else torch.float32

            print("Loading Text2Semantic Dual-AR model...")
            self.llm_model, self.decode_one_token = init_model(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                precision=precision,
                compile=False,
            )

            print("Loading S2 Pro codec...")
            self.codec_model = load_codec_model(
                self.codec_checkpoint_path,
                self.device,
                precision=precision,
            )

        except ImportError as e:
            print(f"\n[!] Failed to import Fish Speech modules from {self.repo_path}")
            print("Ensure the Fish Speech environment and dependencies are active.")
            raise e
        except Exception as e:
            print(f"Error loading Fish Audio S2 Pro: {e}")
            raise e

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        if not ref_text:
            print(
                "Warning: Fish Audio S2 Pro voice cloning works best with reference text. "
                "Using empty string."
            )
            ref_text = ""

        encode_audio = self.modules["encode_audio"]
        decode_to_audio = self.modules["decode_to_audio"]
        generate_long = self.modules["generate_long"]

        try:
            prompt_tokens = encode_audio(
                ref_audio_path,
                self.codec_model,
                self.device,
            ).cpu()

            generator = generate_long(
                model=self.llm_model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                text=text,
                num_samples=1,
                max_new_tokens=0,
                top_p=0.9,
                top_k=30,
                temperature=1.0,
                compile=False,
                iterative_prompt=True,
                chunk_length=300,
                prompt_text=[ref_text],
                prompt_tokens=[prompt_tokens],
            )

            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes)
                elif response.action == "next":
                    break

            if not codes:
                raise RuntimeError("Fish Audio S2 Pro did not generate semantic tokens.")

            final_codes = torch.cat(codes, dim=1).to(self.device)
            audio = decode_to_audio(final_codes, self.codec_model)

            sf.write(
                output_path,
                audio.float().cpu().numpy(),
                self.codec_model.sample_rate,
            )

        except Exception as e:
            print(f"Fish Audio S2 Pro generation error: {e}")
            raise e
