import sys
import os
import torch
import soundfile as sf
import re
import copy
import yaml
import langid
import jieba
from typing import List, Optional
from dataclasses import asdict
import tqdm
from loguru import logger

from .base import BaseTTSModel

# --- Constants copied from generation.py ---
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"
MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

# --- Helper Functions copied from generation.py ---
def normalize_chinese_punctuation(text):
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        "“": '"', "”": '"', "‘": "'", "’": "'", "、": ",", "—": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def prepare_chunk_text(text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1):
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        try:
            language = langid.classify(text)[0]
        except:
            language = "en"
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            if chunks:
                chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")

def _build_system_message_with_audio_prompt(system_message, TextContent, AudioContent, Message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(role="system", content=contents)
    return ret


class HiggsAudioModel(BaseTTSModel):
    def __init__(self, device=None):
        super().__init__(device)
        
        # --- CONFIGURATION ---
        # Update this path to where your higgs-audio repo/files are located
        self.repo_path = "/data/group1/z44476r/Experiments/higgs-audio"
        
        # Default model settings
        self.model_path = "bosonai/higgs-audio-v2-generation-3B-base"
        self.audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None
        self.collator = None
        self.config = None
        self.kv_caches = None
        
        # Imported modules placeholders
        self.modules = {}
        
        self.load_model()

    def load_model(self):
        print(f"Loading HiggsAudio from {self.repo_path}...")
        
        if self.repo_path not in sys.path:
            sys.path.append(self.repo_path)

        try:
            # Import modules dynamically
            from boson_multimodal.model.higgs_audio import HiggsAudioModel as HAM, HiggsAudioConfig
            from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
            from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
            from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
            from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
            from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
            from transformers import AutoConfig, AutoTokenizer
            from transformers.cache_utils import StaticCache

            # Store modules for use in methods
            self.modules = {
                "HiggsAudioModel": HAM,
                "HiggsAudioSampleCollator": HiggsAudioSampleCollator,
                "ChatMLDatasetSample": ChatMLDatasetSample,
                "prepare_chatml_sample": prepare_chatml_sample,
                "revert_delay_pattern": revert_delay_pattern,
                "Message": Message,
                "ChatMLSample": ChatMLSample,
                "AudioContent": AudioContent,
                "TextContent": TextContent,
                "StaticCache": StaticCache
            }

            # Device setup
            if self.device == 'cuda' and torch.cuda.is_available():
                self.device_str = "cuda:0"
            elif self.device == 'mps':
                self.device_str = "mps"
            else:
                self.device_str = "cpu"

            # Load Audio Tokenizer
            # For MPS, use CPU for audio tokenizer due to embedding operation limitations
            audio_tok_dev = "cpu" if self.device_str == "mps" else self.device_str
            self.audio_tokenizer = load_higgs_audio_tokenizer(self.audio_tokenizer_path, device=audio_tok_dev)

            # Load Model
            print("Loading Model Weights...")
            self.model = HAM.from_pretrained(
                self.model_path,
                device_map=self.device_str,
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()

            # Load Config & Text Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.config = AutoConfig.from_pretrained(self.model_path)
            
            # Setup Collator
            self.collator = HiggsAudioSampleCollator(
                whisper_processor=None,
                audio_in_token_id=self.config.audio_in_token_idx,
                audio_out_token_id=self.config.audio_out_token_idx,
                audio_stream_bos_id=self.config.audio_stream_bos_id,
                audio_stream_eos_id=self.config.audio_stream_eos_id,
                encode_whisper_embed=self.config.encode_whisper_embed,
                pad_token_id=self.config.pad_token_id,
                return_audio_in_tokens=self.config.encode_audio_in_tokens,
                use_delay_pattern=self.config.use_delay_pattern,
                round_to=1,
                audio_num_codebooks=self.config.audio_num_codebooks,
            )

            # Static KV Cache (Optional optimization)
            self.use_static_kv_cache = True if "cuda" in self.device_str else False
            self.kv_cache_lengths = [1024, 4096, 8192]
            
            if self.use_static_kv_cache:
                self._init_static_kv_cache()

        except ImportError as e:
            print(f"Failed to import HiggsAudio modules. Check repo path: {self.repo_path}")
            raise e

    def _init_static_kv_cache(self):
        StaticCache = self.modules["StaticCache"]
        cache_config = copy.deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(self.kv_cache_lengths)
        }
        # Capture CUDA graphs
        if "cuda" in self.device_str:
            self.model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def _prepare_generation_context(self, text, ref_audio_path, ref_text):
        Message = self.modules["Message"]
        AudioContent = self.modules["AudioContent"]
        
        messages = []
        audio_ids = []
        
        # We assume single speaker prompt for this benchmark
        if ref_audio_path and os.path.exists(ref_audio_path):
            # 1. Encode Audio Prompt
            audio_tokens = self.audio_tokenizer.encode(ref_audio_path)
            audio_ids.append(audio_tokens)
            
            # 2. Build User/Assistant Prompt Context
            # User: [SPEAKER0] {transcript}
            # Assistant: {audio_content}
            
            prompt_text = ref_text if ref_text else " "
            
            messages.append(Message(
                role="user",
                content=f"[SPEAKER0] {prompt_text}"
            ))
            
            messages.append(Message(
                role="assistant",
                content=AudioContent(audio_url=ref_audio_path)
            ))
            
            # 3. System Message
            # Simple default system message
            system_message = Message(
                role="system",
                content="Generate audio following instruction.\n\n"
                        f"<|scene_desc_start|>\nSPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}\n<|scene_desc_end|>"
            )
            
            # Use helper to inject audio placeholder logic
            system_message = _build_system_message_with_audio_prompt(
                system_message.content if isinstance(system_message.content, str) else "", # Logic inside helper expects str initially? 
                # Wait, the helper _build_system_message takes a string and returns a Message object.
                # Let's reconstruct the call properly.
                self.modules["TextContent"],
                self.modules["AudioContent"],
                self.modules["Message"]
            )
            
            # Actually, let's just use the string passed to the helper
            sys_msg_str = ("Generate audio following instruction.\n\n"
                           f"<|scene_desc_start|>\nSPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}\n<|scene_desc_end|>")
            
            system_message = _build_system_message_with_audio_prompt(
                sys_msg_str,
                self.modules["TextContent"],
                self.modules["AudioContent"],
                self.modules["Message"]
            )
            
            messages.insert(0, system_message)
            
        else:
            # No reference audio (Zero-shot / Random voice)
            messages.append(Message(
                role="system",
                content="Generate audio following instruction."
            ))

        return messages, audio_ids

    @torch.inference_mode()
    def generate(self, text, ref_audio_path, output_path, language="en", ref_text=None):
        Message = self.modules["Message"]
        ChatMLSample = self.modules["ChatMLSample"]
        ChatMLDatasetSample = self.modules["ChatMLDatasetSample"]
        AudioContent = self.modules["AudioContent"]
        
        # 1. Prepare Context
        messages, audio_ids = self._prepare_generation_context(text, ref_audio_path, ref_text)
        
        # 2. Chunk Text (if needed, defaults to no chunking or simple)
        chunked_text = prepare_chunk_text(text, chunk_method=None)
        
        # 3. Generation Loop
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        
        # Setup KV Cache
        if self.use_static_kv_cache:
            self._prepare_kv_caches()

        for chunk_text in chunked_text:
            generation_messages.append(Message(role="user", content=chunk_text))
            
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            
            # Prepare inputs
            input_tokens, _, _, _ = self.modules["prepare_chatml_sample"](chatml_sample, self.tokenizer)
            postfix = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)
            
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
                audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self.collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self.device_str)

            # Generate
            outputs = self.model.generate(
                **batch,
                max_new_tokens=2048,
                use_cache=True,
                do_sample=True,
                temperature=0.3, # From your example
                top_k=50,
                top_p=0.95,
                past_key_values_buckets=self.kv_caches if self.use_static_kv_cache else None,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self.tokenizer,
            )

            # Process Output
            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self.config.use_delay_pattern:
                    audio_out_ids = self.modules["revert_delay_pattern"](audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self.audio_tokenizer.codebook_size - 1)[:, 1:-1])
            
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            # Update history for next chunk
            generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

        # 4. Final Decode and Save
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        if concat_audio_out_ids.device.type == "mps":
            concat_audio_out_ids = concat_audio_out_ids.detach().cpu()

        concat_wv = self.audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]
        
        sf.write(output_path, concat_wv, sr)