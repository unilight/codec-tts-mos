import torch
import soundfile as sf
import librosa
import os

from .base import BaseTTSModel

class Llasa1BModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        self.tokenizer = None
        self.model = None
        self.codec_model = None
        self.load_model()

    def load_model(self):
        print("Loading Llasa-1B and XCodec2...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Ensure xcodec2 is installed or in python path
        from xcodec2.modeling_xcodec2 import XCodec2Model

        llasa_1b = 'HKUSTAudio/Llasa-1b'
        self.tokenizer = AutoTokenizer.from_pretrained(llasa_1b)
        self.model = AutoModelForCausalLM.from_pretrained(llasa_1b)
        self.model.eval()
        self.model.to(self.device)

        model_path = "HKUSTAudio/xcodec2"
        self.codec_model = XCodec2Model.from_pretrained(model_path)
        self.codec_model.eval().cuda() # Codec usually requires CUDA based on demo

    @staticmethod
    def ids_to_speech_tokens(speech_ids):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    @staticmethod
    def extract_speech_ids(speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith('<|s_') and token_str.endswith('|>'):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids

    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        if ref_text is None:
            print(f"Warning: Llasa-1B works best with reference text. Using empty string for ref_text.")
            ref_text = ""

        # 1. Process Audio Prompt
        prompt_wav, sr = librosa.load(ref_audio_path, sr=16000)
        prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)
        if self.device == 'cuda':
            prompt_wav = prompt_wav.cuda()

        # 2. Encode Prompt
        with torch.no_grad():
            vq_code_prompt = self.codec_model.encode_code(input_waveform=prompt_wav)
            vq_code_prompt = vq_code_prompt[0,0,:] # Flatten
            
            speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)

            # 3. Construct Prompt Text
            # Llasa format: prompt_text + ' ' + target_text
            input_text = f"{ref_text} {text}"
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

            # 4. Tokenize
            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                chat, 
                tokenize=True, 
                return_tensors='pt', 
                continue_final_message=True
            )
            input_ids = input_ids.to(self.device)
            speech_end_id = self.tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            # 5. Generate
            outputs = self.model.generate(
                input_ids,
                max_length=4096, # Increased slightly to be safe
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=1,           
                temperature=0.8,
            )

            # 6. Extract and Decode
            # Slice off the prompt tokens
            generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]
            
            speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            speech_tokens = self.extract_speech_ids(speech_tokens)
            
            if len(speech_tokens) == 0:
                print("Warning: No speech tokens generated.")
                return

            speech_tokens = torch.tensor(speech_tokens).to(self.device).unsqueeze(0).unsqueeze(0)
            
            # Decode to waveform
            gen_wav = self.codec_model.decode_code(speech_tokens)
            
            # Depending on implementation, you might want to strip the prompt audio from the result
            # The demo comments say: "if only need the generated part: gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]"
            # Since the model is autoregressive and we fed the prompt prefix, the output usually includes the prompt + new audio.
            # However, the `generated_ids` slicing above: `[input_ids.shape[1]-len(speech_ids_prefix):-1]`
            # keeps the prefix tokens in the list we decode. So we likely decode Prompt + Target.
            # We should slice the audio to remove the prompt duration.
            
            prompt_len = prompt_wav.shape[1]
            # Check if generation is longer than prompt
            if gen_wav.shape[2] > prompt_len:
                final_wav = gen_wav[0, 0, prompt_len:].cpu().numpy()
            else:
                # Fallback if something weird happened
                final_wav = gen_wav[0, 0, :].cpu().numpy()

            sf.write(output_path, final_wav, 16000)