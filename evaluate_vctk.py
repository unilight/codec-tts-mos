import argparse
import os
import sys
import re
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import jiwer
from tqdm import tqdm
import torch.nn.functional as F

# Transformers & SpeechBrain
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    pipeline,
    AutoFeatureExtractor, 
    AutoModelForAudioClassification
)
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.classifiers import EncoderClassifier as AccentClassifier

from normalizers import EnglishTextNormalizer

# --- Constants & Configuration ---
DEFAULT_WHISPER = "openai/whisper-large-v3"
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_EMOTION_MODEL = "expresso_emotion_classifier"
DEFAULT_ACCENT_MODEL = "Jzuluaga/accent-id-commonaccent_ecapa"

class AudioEvaluator:
    def __init__(self, emotion_model_path):
        # Device Setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"🚀 Initializing models on device: {self.device} (dtype: {self.torch_dtype})")

        # 1. ASR (Whisper via Pipeline)
        print(f"   - Loading Whisper Pipeline ({DEFAULT_WHISPER})...")
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                DEFAULT_WHISPER, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            model.to(self.device)
            processor = AutoProcessor.from_pretrained(DEFAULT_WHISPER)

            self.asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        except Exception as e:
            print(f"❌ Error loading Whisper: {e}")
            sys.exit(1)

        self.normalizer = EnglishTextNormalizer()

        # 2. Speaker Embedding (SpeechBrain)
        print(f"   - Loading Speaker Encoder ({DEFAULT_SPEAKER_MODEL})...")
        # SpeechBrain expects "cuda" not "cuda:0" usually, handling cleanly:
        sb_device = "cuda" if "cuda" in self.device else "cpu"
        self.spk_encoder = EncoderClassifier.from_hparams(
            source=DEFAULT_SPEAKER_MODEL, 
            run_opts={"device": sb_device}
        )

        # 3. Accent Embedding (SpeechBrain)
        print(f"   - Loading Accent Encoder ({DEFAULT_ACCENT_MODEL})...")
        # SpeechBrain expects "cuda" not "cuda:0" usually, handling cleanly:
        sb_device = "cuda" if "cuda" in self.device else "cpu"
        self.accent_model = classifier = EncoderClassifier.from_hparams(
            source=DEFAULT_ACCENT_MODEL,
            savedir="pretrained_models/accent-id-commonaccent_ecapa"
        )

        # 4. UTMOS
        print(f"   - Loading UTMOS...")
        try:
            self.utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to("cuda")
        except Exception as e:
            print(f"⚠️ Error loading UTMOS: {e}")
            sys.exit(1)

    def load_audio_tensor(self, path, target_sr=16000):
        """Helper to load audio as tensor for SpeechBrain/Emotion models."""
        try:
            sig, sr = torchaudio.load(path)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr).to(sig.device)
                sig = resampler(sig)
            return sig.to(self.device)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def get_duration(self, path):
        """Returns duration in seconds using torchaudio.info (fast)."""
        try:
            metadata = torchaudio.info(path)
            return metadata.num_frames / metadata.sample_rate
        except Exception:
            return 0.0

    def compute_wer(self, audio_path, ref_text):
        """Transcribes audio using Pipeline and computes WER."""
        if not ref_text or str(ref_text) == "nan":
            return None, ""

        try:
            # Pipeline handles loading and resampling internally
            result = self.asr_pipe(audio_path)
            transcription = result["text"]
            
            hyp_norm = self.normalizer(transcription)
            ref_norm = self.normalizer(ref_text)
            
            if not ref_norm: # Handle empty reference
                return 0.0 if not hyp_norm else 1.0, transcription

            error = jiwer.wer(ref_norm, hyp_norm)
            return error, transcription
        except Exception as e:
            print(f"WER Error on {audio_path}: {e}")
            return None, ""

    def compute_spk_sim(self, gen_tensor, gt_path):
        """Computes Cosine Similarity between generated (tensor) and ground truth (path)."""
        if not gt_path or str(gt_path) == "nan" or not os.path.exists(gt_path):
            return None
        
        try:
            # Load GT here
            sig_gt = self.load_audio_tensor(gt_path)
            if sig_gt is None: return None

            with torch.no_grad():
                emb_gen = self.spk_encoder.encode_batch(gen_tensor)
                emb_gt = self.spk_encoder.encode_batch(sig_gt)
            
            sim = F.cosine_similarity(emb_gen.squeeze(), emb_gt.squeeze(), dim=0)
            return sim.item()
        except Exception as e:
            print(f"SpkSim Error: {e}")
            return None

    def compute_utmos_score(self, audio_tensor, input_sr=16000):
        """
        Computes the UTMOS quality score for the input audio.
        Model: tarepan/SpeechMOS:v1.2.0 (utmos22_strong)
        """
        if self.utmos_model is None:
            # You can uncomment the line below to load on demand, 
            # but it is better to pass it in __init__
            # print("Warning: UTMOS model not loaded.")
            return None

        if audio_tensor is None: return None

        try:
            # Ensure tensor is [Batch, Time] for UTMOS
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() > 2:
                 audio_tensor = audio_tensor.squeeze()
                 if audio_tensor.dim() == 1:
                     audio_tensor = audio_tensor.unsqueeze(0)

            # Ensure input is on the same device as the model
            if audio_tensor.device != self.device:
                audio_tensor = audio_tensor.to(self.device)

            with torch.no_grad():
                # UTMOS forward pass takes (tensor, sample_rate)
                score = self.utmos_model(audio_tensor, input_sr)
            
            # Returns a tensor of shape [Batch, 1] or [Batch]
            return score.mean().item()

        except Exception as e:
            print(f"UTMOS Error: {e}")
            return None

    def compute_accent_sim(self, gen_tensor, gt_path):
        """Computes Cosine Similarity between accent embeddings."""
        if self.accent_model is None:
            return None

        try:
            # 1. Load GT
            sig_gt = self.load_audio_tensor(gt_path)
            if sig_gt is None: return None

            with torch.no_grad():
                # SpeechBrain EncoderClassifier.encode_batch returns [Batch, 1, EmbDim]
                emb_gen = self.accent_model.encode_batch(gen_tensor)
                emb_gt = self.accent_model.encode_batch(sig_gt)

            # Squeeze to [EmbDim] (assuming batch size 1) for cosine similarity
            sim = F.cosine_similarity(emb_gen.squeeze(), emb_gt.squeeze(), dim=0)
            return sim.item()

        except Exception as e:
            print(f"AccentSim Error: {e}")
            return None

def parse_filename(stem):
    """Parses filename ex01_confused_00366 -> speaker, emotion, id."""
    parts = stem.split('_')
    return parts[0], parts[1]
    # if len(parts) >= 3:
        # return parts[0], parts[1], '_'.join(parts[2:])
    # return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesized speech (WER, Speaker Sim, Emotion).")
    parser.add_argument("--results_dir", type=str, default="./results", help="Root directory containing model subfolders")
    parser.add_argument("--metadata_csv", type=str, default="./expresso_test_tts.csv", help="Path to the metadata CSV file")
    parser.add_argument("--emotion_model", type=str, default=DEFAULT_EMOTION_MODEL, help="Path or HF ID for emotion classifier")
    parser.add_argument("--output_csv", type=str, default="results/evaluation_metrics.csv", help="Path to save results CSV")
    
    args = parser.parse_args()

    # 1. Load Metadata
    print(f"Loading metadata from {args.metadata_csv}...")
    df_meta = pd.read_csv(args.metadata_csv)
    # Create key for matching: filename stem
    df_meta['key'] = df_meta['ground_truth_path'].apply(lambda x: Path(x).stem)
    df_meta.set_index('key', inplace=True)
    print(f"   - Loaded {len(df_meta)} rows.")

    # 2. Scan Files
    root_dir = Path(args.results_dir)
    tasks = []
    
    assert root_dir.exists(), "Results directory does not exist."
    print(f"Scanning {root_dir}...")
    for model_dir in root_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            wav_files = list(model_dir.glob('*.wav'))
            
            for wav_path in wav_files:
                stem = wav_path.stem
                # spk, emo, uid = parse_filename(stem)
                spk, uid = parse_filename(stem)
                if spk:
                    tasks.append({
                        'model': model_name,
                        'speaker': spk,
                        # 'target_emotion': emo,
                        'key': stem,
                        'path': str(wav_path)
                    })

    if not tasks:
        print("No WAV files found to process.")
        sys.exit(1)

    print(f"   - Found {len(tasks)} files across {len(set(t['model'] for t in tasks))} models.")

    # 3. Initialize Evaluator
    evaluator = AudioEvaluator(args.emotion_model)

    # 4. Processing Loop
    results = []
    skipped_count = 0
    
    print("\n🚀 Starting Evaluation Loop...")
    for item in tqdm(tasks, desc="Evaluating"):
        gen_path = item['path']
        key = item['key']
        
        # Retrieve GT info
        if key in df_meta.index:
            meta_row = df_meta.loc[key]
            if isinstance(meta_row, pd.DataFrame): meta_row = meta_row.iloc[0]
            ref_text = meta_row.get('text', "") 
            gt_path = meta_row.get('ground_truth_path', None)
        else:
            ref_text = ""
            gt_path = None

        # --- LENGTH CHECKS (Before loading heavy models) ---
        dur_gen = evaluator.get_duration(gen_path)
        
        # Check 1: Too short (< 1s)
        if dur_gen < 1:
            skipped_count += 1
            # Optional: Log this as a failure in CSV without metrics
            continue 

        # # Check 2: Hallucination relative to GT (Difference > 1.0s)
        # if gt_path and os.path.exists(gt_path):
        #     dur_gt = evaluator.get_duration(gt_path)
        #     if abs(dur_gen - dur_gt) > 1.0:
        #         skipped_count += 1
        #         continue
        
        # --- Metrics Calculation ---
        
        # A. WER
        wer, transcript = evaluator.compute_wer(gen_path, ref_text)
        if wer is None:
            skipped_count += 1
            continue

        # Load Tensor for Spk/Emotion
        audio_tensor = evaluator.load_audio_tensor(gen_path)

        # B. Speaker Sim
        # (Passes tensor for gen, path for gt to avoid re-loading gt unnecessarily in main loop)
        spk_sim = evaluator.compute_spk_sim(audio_tensor, gt_path)

        # C. Accent Sim
        accent_sim = evaluator.compute_accent_sim(audio_tensor, gt_path)
        
        # D. UTMOS
        utmos = evaluator.compute_utmos_score(audio_tensor)

        results.append({
            'model': item['model'],
            'filename': key,
            'duration_gen': round(dur_gen, 2),
            'wer': wer * 100,
            'spk_sim': spk_sim,
            'accent_sim': accent_sim,
            'transcript': transcript,
            'utmos': utmos,
        })

    # 5. Save & Summarize
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)
    
    print(f"\n✅ Processing complete. {skipped_count} samples skipped due to length violations.")
    print(f"✅ Results saved to {args.output_csv}")

    if not df_res.empty:
        print("\n📊 --- Evaluation Summary ---")
        summary = df_res.groupby('model').agg({
            'wer': 'mean',
            'spk_sim': 'mean',
            'accent_sim': 'mean',
            'utmos': 'mean',
        }).reset_index()
        
        summary['wer'] = summary['wer'].round(3)
        summary['spk_sim'] = summary['spk_sim'].round(3)
        summary['accent_sim'] = summary['accent_sim'].round(3)
        summary['utmos'] = summary['utmos'].round(3)
        
        # print(summary[['model', 'wer', 'spk_sim', 'emotion_acc', 'emo_sim']].to_markdown(index=False))
        print(summary[['model', 'wer', 'spk_sim', 'accent_sim', 'utmos']].to_markdown(index=False))

if __name__ == "__main__":
    main()