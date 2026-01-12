import sys
import os
import torch
import shutil

# Relative import for base class
from .base import BaseTTSModel

class OpenVoiceV2TTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        self.repo_path = "/data/group1/z44476r/Experiments/OpenVoice"
        self.ckpt_converter = os.path.join(self.repo_path, "checkpoints_v2/converter")
        self.ckpt_base_speakers = os.path.join(self.repo_path, "checkpoints_v2/base_speakers/ses")
        
        self.tone_color_converter = None
        self.se_extractor = None
        self.melo_TTS = None
        self.melo_models = {} # Cache for different languages
        self.current_lang = None
        
        self.load_model()

    def load_model(self):
        print(f"Loading OpenVoice v2 from {self.repo_path}...")

        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            from melo.api import TTS
            
            self.se_extractor = se_extractor
            self.melo_TTS = TTS # Class reference
            
            # 1. Load Tone Color Converter
            print("Loading Tone Color Converter...")
            config_path = os.path.join(self.ckpt_converter, 'config.json')
            checkpoint_path = os.path.join(self.ckpt_converter, 'checkpoint.pth')
            
            self.tone_color_converter = ToneColorConverter(config_path, device=self.device)
            self.tone_color_converter.load_ckpt(checkpoint_path)
            
        except ImportError as e:
            print(f"\n[!] Failed to import OpenVoice modules from {self.repo_path}")
            print("Ensure dependencies (openvoice, melo, etc.) are installed.")
            raise e

    def get_melo_model(self, language):
        # Map common language codes to Melo/OpenVoice codes
        lang_map = {
            "en": "EN",
            "zh": "ZH",
            "es": "ES",
            "fr": "FR",
            "ja": "JP",
            "kr": "KR"
        }
        target_lang = lang_map.get(language.lower(), "EN")
        
        # OpenVoice v2 recommends 'EN_NEWEST' for English if available, but standard 'EN' is safer default
        if target_lang == "EN":
            target_lang = "EN_NEWEST" # Trying newest as per example hints

        if target_lang in self.melo_models:
            return self.melo_models[target_lang], target_lang
            
        print(f"Loading MeloTTS base model for language: {target_lang}...")
        try:
            model = self.melo_TTS(language=target_lang, device=self.device)
            self.melo_models[target_lang] = model
            return model, target_lang
        except Exception:
            # Fallback to standard EN if NEWEST fails
            if target_lang == "EN_NEWEST":
                print("Fallback to standard EN...")
                model = self.melo_TTS(language="EN", device=self.device)
                self.melo_models["EN"] = model
                return model, "EN"
            raise

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        try:
            # 1. Prepare Base Model
            model, lang_code = self.get_melo_model(language)
            speaker_ids = model.hps.data.spk2id
            
            # Select a default base speaker (usually the first one or a specific default)
            # For EN_NEWEST, keys might look different.
            # We iterate keys to find a valid source_se file.
            source_se = None
            speaker_id = None
            
            for speaker_key in speaker_ids.keys():
                speaker_id = speaker_ids[speaker_key]
                formatted_key = speaker_key.lower().replace('_', '-')
                
                # Check if we have the SE for this speaker
                se_path = os.path.join(self.ckpt_base_speakers, f'{formatted_key}.pth')
                if os.path.exists(se_path):
                    source_se = torch.load(se_path, map_location=self.device)
                    break
            
            if source_se is None:
                raise FileNotFoundError(f"Could not find a base speaker SE in {self.ckpt_base_speakers}")

            # 2. Extract Target SE
            target_se, _ = self.se_extractor.get_se(
                ref_audio_path, 
                self.tone_color_converter, 
                vad=True
            )

            # 3. Generate Base Audio (TTS)
            # Create a temp path for the intermediate TTS result
            temp_dir = os.path.dirname(output_path)
            temp_src_path = os.path.join(temp_dir, f"tmp_melo_{os.path.basename(output_path)}")
            
            # MPS workaround from example
            if self.device == 'cpu' and torch.backends.mps.is_available():
                 # This is a weird hack in the example, mirroring it just in case
                 # torch.backends.mps.is_available = lambda: False
                 pass

            model.tts_to_file(text, speaker_id, temp_src_path, speed=1.0)

            # 4. Tone Color Conversion
            encode_message = "@MyShell"
            self.tone_color_converter.convert(
                audio_src_path=temp_src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=output_path,
                message=encode_message
            )
            
            # Cleanup
            if os.path.exists(temp_src_path):
                os.remove(temp_src_path)
                
        except Exception as e:
            print(f"OpenVoice Generation Error: {e}")
            # We do not raise e here so the pipeline continues
            return