import torch
import librosa
import soundfile as sf
import os

from .base import BaseTTSModel

class OrpheusTTSModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        self.model = None
        self.tokenizer = None
        self.snac_model = None
        self.load_model()

    def load_model(self):
        print("Loading Orpheus-3B and SNAC...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from snac import SNAC
        except ImportError as e:
            print("Failed to import dependencies. Ensure 'transformers' and 'snac' are installed.")
            raise e

        # 1. Load SNAC (Codec)
        # Using the hubertsiuzdak/snac_24khz model as per the snippet
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self.device)

        # 2. Load Orpheus (LLM)
        model_name = "canopylabs/orpheus-3b-0.1-pretrained"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)

    def _tokenise_audio(self, waveform):
        # Ensure waveform is (1, T) and float32
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.to(self.device)
        else:
            waveform = torch.tensor(waveform).to(self.device)
            
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        waveform = waveform.to(dtype=torch.float32)
        # SNAC expects (B, 1, T)
        waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        # Interleave codes logic from the snippet
        all_codes = []
        # codes structure: List of tensors. 
        # codes[0]: (B, T/Hop1), codes[1]: (B, T/Hop2), codes[2]: (B, T/Hop3)
        # The loop assumes batch size 1 -> codes[i][0]
        
        length = codes[0].shape[1]
        c0 = codes[0][0]
        c1 = codes[1][0]
        c2 = codes[2][0]

        for i in range(length):
            all_codes.append(c0[i].item() + 128266)
            all_codes.append(c1[2*i].item() + 128266 + 4096)
            all_codes.append(c2[4*i].item() + 128266 + (2*4096))
            all_codes.append(c2[(4*i)+1].item() + 128266 + (3*4096))
            all_codes.append(c1[(2*i)+1].item() + 128266 + (4*4096))
            all_codes.append(c2[(4*i)+2].item() + 128266 + (5*4096))
            all_codes.append(c2[(4*i)+3].item() + 128266 + (6*4096))

        return all_codes

    def _redistribute_codes(self, code_list):
        # Reverse the interleaving
        layer_1 = []
        layer_2 = []
        layer_3 = []
        
        # Each "frame" is 7 tokens
        num_frames = (len(code_list) + 1) // 7
        
        for i in range(num_frames):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1] - 4096)
            layer_3.append(code_list[7*i+2] - (2*4096))
            layer_3.append(code_list[7*i+3] - (3*4096))
            layer_2.append(code_list[7*i+4] - (4*4096))
            layer_3.append(code_list[7*i+5] - (5*4096))
            layer_3.append(code_list[7*i+6] - (6*4096))
            
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self.device),
            torch.tensor(layer_2).unsqueeze(0).to(self.device),
            torch.tensor(layer_3).unsqueeze(0).to(self.device)
        ]
        
        with torch.inference_mode():
            audio_hat = self.snac_model.decode(codes)
            
        return audio_hat

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        if not ref_text:
            print("Warning: Orpheus TTS requires reference text (transcript). Using empty string.")
            ref_text = " " # Space to avoid empty tokenization issues

        # 1. Load and Tokenize Reference Audio
        # Orpheus uses 24k sample rate
        audio_array, _ = librosa.load(ref_audio_path, sr=24000)
        audio_tokens_list = self._tokenise_audio(audio_array)

        # 2. Prepare Special Tokens
        start_tokens = torch.tensor([[128259]], dtype=torch.int64).to(self.device) # SOH
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64).to(self.device) # EOT + Audio Start
        final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64).to(self.device) # Audio End + EOH

        # 3. Prepare Reference Text Tokens
        prompt_tokked = self.tokenizer(ref_text, return_tensors="pt")
        ref_text_ids = prompt_tokked["input_ids"].to(self.device)
        
        # Audio tokens tensor
        audio_tokens_tensor = torch.tensor([audio_tokens_list], dtype=torch.int64).to(self.device)

        # 4. Construct Zero-Shot Prompt (Context)
        # SOH + SOT + RefText + EOT/AudioStart + RefAudioTokens + AudioEnd/EOH
        # Note: In your snippet, 'start_tokens' is used before 'input_ids' in the prompt part, 
        # but for the zeroprompt part, it matches: start_tokens (SOH) + input_ids (RefText).
        # Wait, the snippet logic is:
        # zeroprompt = start_tokens + input_ids (RefText) + end_tokens + audio_tokens + final_tokens
        
        zeroprompt_input_ids = torch.cat([
            start_tokens, 
            ref_text_ids, 
            end_tokens, 
            audio_tokens_tensor, 
            final_tokens
        ], dim=1)

        # 5. Prepare Target Text Input
        target_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # Full Input: ZeroPrompt + SOH + TargetText + EOT/AudioStart
        # Note: The snippet does: zeroprompt + start_tokens + input_ids + end_tokens
        full_input_ids = torch.cat([
            zeroprompt_input_ids, 
            start_tokens, 
            target_ids, 
            end_tokens
        ], dim=1)

        # 6. Generate
        # We don't need batch padding here since we process one sample at a time
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=full_input_ids,
                max_new_tokens=2048, # Increased to be safe
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=128258, # Audio End Token
            )

        # 7. Post-Process Output
        token_to_find = 128257 # Audio Start
        token_to_remove = 128258 # Audio End

        # We want the NEW tokens. 
        # The snippet logic finds the *last* occurrence of 'token_to_find' (Audio Start)
        # which corresponds to the start of the *target* audio generation.
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            # Fallback
            cropped_tensor = generated_ids

        # Remove EOS tokens if present in the cropped tensor
        # Flatten to list for processing
        raw_codes = cropped_tensor[0].cpu().numpy().tolist()
        clean_codes = [c for c in raw_codes if c != token_to_remove]

        # Truncate to multiple of 7
        new_length = (len(clean_codes) // 7) * 7
        clean_codes = clean_codes[:new_length]

        # Shift back by offset (128266)
        shifted_codes = [t - 128266 for t in clean_codes]

        if not shifted_codes:
            print("Warning: Orpheus generated empty audio codes.")
            return

        # 8. Decode to Audio
        audio_hat = self._redistribute_codes(shifted_codes)
        
        # 9. Save
        # audio_hat is (1, 1, T)
        audio_out = audio_hat.squeeze().cpu().numpy()
        sf.write(output_path, audio_out, 24000)