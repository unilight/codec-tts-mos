import argparse
import os
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import pyopenjtalk
import torch
import torch.nn.functional as F
import torchaudio
from funasr import AutoModel
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

# Transformers & SpeechBrain
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

# --- Constants & Configuration ---
DEFAULT_WHISPER = "openai/whisper-large-v3"
DEFAULT_SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_EMOTION_MODEL = "iic/emotion2vec_plus_large"


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    sentence = pyopenjtalk.g2p(sentence, kana=True)

    return sentence


def cosine_similarity_numpy(vec1, vec2):
    """
    Calculates the cosine similarity between two numpy vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors, ranging from -1 to 1.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Avoid division by zero for zero vectors
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


class AudioEvaluator:
    def __init__(self, emotion_model_path):
        # Device Setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(
            f"🚀 Initializing models on device: {self.device} (dtype: {self.torch_dtype})"
        )

        # 1. ASR (Whisper via Pipeline)
        print(f"   - Loading Whisper Pipeline ({DEFAULT_WHISPER})...")
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                DEFAULT_WHISPER,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
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

        # 2. Speaker Embedding (SpeechBrain)
        print(f"   - Loading Speaker Encoder ({DEFAULT_SPEAKER_MODEL})...")
        # SpeechBrain expects "cuda" not "cuda:0" usually, handling cleanly:
        sb_device = "cuda" if "cuda" in self.device else "cpu"
        self.spk_encoder = EncoderClassifier.from_hparams(
            source=DEFAULT_SPEAKER_MODEL, run_opts={"device": sb_device}
        )

        # 3. Emotion2Vec
        print(f"   - Loading Emotion2Vec ({DEFAULT_EMOTION_MODEL})...")
        # emotion2vec uses funasr; allow it to handle device mapping or pass explicitly if needed
        self.emotion2vec = AutoModel(model=DEFAULT_EMOTION_MODEL, hub="hf", device=self.device)

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

    def compute_cer(self, audio_path, ref_text):
        """Transcribes audio using Pipeline and computes CER."""
        if not ref_text or str(ref_text) == "nan":
            return None, ""

        try:
            # Pipeline handles loading and resampling internally
            result = self.asr_pipe(audio_path)
            transcription = result["text"]

            hyp_norm = normalize_sentence(transcription)
            ref_norm = normalize_sentence(ref_text)

            if not ref_norm:  # Handle empty reference
                return 0.0 if not hyp_norm else 1.0, transcription

            error = jiwer.cer(ref_norm, hyp_norm)
            # jiwer.cer(groundtruth, transcription, return_dict=True)
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
            if sig_gt is None:
                return None

            with torch.no_grad():
                emb_gen = self.spk_encoder.encode_batch(gen_tensor)
                emb_gt = self.spk_encoder.encode_batch(sig_gt)

            sim = F.cosine_similarity(emb_gen.squeeze(), emb_gt.squeeze(), dim=0)
            return sim.item()
        except Exception as e:
            print(f"SpkSim Error: {e}")
            return None

    def extract_emotion_embeddings_batch(self, paths_list):
        """
        Extracts embeddings for a list of paths using emotion2vec batch inference.
        Returns a list of embeddings (numpy arrays) corresponding to the input paths.
        """
        if not paths_list:
            return []

        try:
            # funasr generate takes a list of strings
            # disable_pbar=True to avoid spamming progress bars during the main loop
            results = self.emotion2vec.generate(
                paths_list, 
                granularity="utterance", 
                disable_pbar=True
            )
            
            # Results structure: [{'key': 'filename', 'feats': [...]}, ...]
            # We extract just the 'feats'
            embeddings = [res['feats'] for res in results]
            return embeddings
        except Exception as e:
            print(f"❌ Batch Emotion Inference Error: {e}")
            # Return None for all if batch fails, to allow individual retry or skip
            return [None] * len(paths_list)


def parse_filename(stem):
    """Parses filename ex01_confused_00366 -> speaker, emotion, id."""
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], "_".join(parts[2:])
    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthesized speech (WER, Speaker Sim, Emotion)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Root directory containing model subfolders",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="./expresso_test_tts.csv",
        help="Path to the metadata CSV file",
    )
    parser.add_argument(
        "--emotion_model",
        type=str,
        default=DEFAULT_EMOTION_MODEL,
        help="Path or HF ID for emotion classifier",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/evaluation_metrics.csv",
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing files (affects emotion2vec batching)",
    )

    args = parser.parse_args()

    # 1. Load Metadata
    print(f"Loading metadata from {args.metadata_csv}...")
    df_meta = pd.read_csv(args.metadata_csv)
    # Create key for matching: filename stem
    df_meta["key"] = df_meta["ground_truth_path"].apply(lambda x: Path(x).stem)
    df_meta.set_index("key", inplace=True)
    print(f"   - Loaded {len(df_meta)} rows.")

    # 2. Scan Files
    root_dir = Path(args.results_dir)
    tasks = []

    if not root_dir.exists():
        print(f"Error: Results directory '{root_dir}' does not exist.")
        sys.exit(1)

    print(f"Scanning {root_dir}...")
    for model_dir in root_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            wav_files = list(model_dir.glob("*.wav"))

            for wav_path in wav_files:
                stem = wav_path.stem
                spk, emo, uid = parse_filename(stem)
                if spk:
                    tasks.append(
                        {
                            "model": model_name,
                            "speaker": spk,
                            "target_emotion": emo,
                            "key": stem,
                            "path": str(wav_path),
                        }
                    )

    if not tasks:
        print("No WAV files found to process.")
        sys.exit(1)

    print(
        f"   - Found {len(tasks)} files across {len(set(t['model'] for t in tasks))} models."
    )

    # 3. Initialize Evaluator
    evaluator = AudioEvaluator(args.emotion_model)

    # 4. Processing Loop
    results = []
    skipped_count = 0
    batch_size = args.batch_size

    print(f"\n🚀 Starting Evaluation Loop (Batch Size: {batch_size})...")

    # Iterate in batches
    for i in tqdm(range(0, len(tasks), batch_size), desc="Evaluating Batches"):
        batch = tasks[i : i + batch_size]
        
        valid_batch_items = []
        gen_paths_for_emo = []
        gt_paths_for_emo = []

        # Step 4a: Pre-filter batch for valid duration/files
        for item in batch:
            gen_path = item["path"]
            key = item["key"]

            # Retrieve GT info
            if key in df_meta.index:
                meta_row = df_meta.loc[key]
                if isinstance(meta_row, pd.DataFrame):
                    meta_row = meta_row.iloc[0]
                ref_text = meta_row.get("text", "")
                gt_path = meta_row.get("ground_truth_path", None)
            else:
                ref_text = ""
                gt_path = None
            
            # Store info in item for later steps
            item['ref_text'] = ref_text
            item['gt_path'] = gt_path

            # --- LENGTH CHECKS ---
            dur_gen = evaluator.get_duration(gen_path)
            item['duration_gen'] = dur_gen

            # Check 1: Too short (< 0.5s)
            if dur_gen < 0.5:
                skipped_count += 1
                continue

            # Check 2: Hallucination relative to GT (Difference > 10.0s)
            if gt_path and os.path.exists(gt_path):
                dur_gt = evaluator.get_duration(gt_path)
                if abs(dur_gen - dur_gt) > 10.0:
                    skipped_count += 1
                    continue
            
            # If passed checks
            valid_batch_items.append(item)
            gen_paths_for_emo.append(gen_path)
            
            # If GT exists, we will process it, otherwise map to None
            if gt_path and os.path.exists(gt_path):
                gt_paths_for_emo.append(gt_path)
            else:
                gt_paths_for_emo.append(None)

        if not valid_batch_items:
            continue

        # Step 4b: Batch Inference for Emotion
        # 1. Generate Embeddings
        gen_embs = evaluator.extract_emotion_embeddings_batch(gen_paths_for_emo)
        
        # 2. GT Embeddings
        # Optimization: Only infer unique valid GT paths to save time
        unique_valid_gt = list(set([p for p in gt_paths_for_emo if p is not None]))
        if unique_valid_gt:
            unique_gt_embs = evaluator.extract_emotion_embeddings_batch(unique_valid_gt)
            gt_path_to_emb = dict(zip(unique_valid_gt, unique_gt_embs))
        else:
            gt_path_to_emb = {}

        # Step 4c: Calculate metrics for each item in the batch
        for idx, item in enumerate(valid_batch_items):
            gen_path = item['path']
            gt_path = item['gt_path']
            ref_text = item['ref_text']
            
            # A. Emotion Sim (using pre-computed batch embeddings)
            emb_gen = gen_embs[idx]
            emb_gt = gt_path_to_emb.get(gt_path) # Returns None if gt_path was None or invalid
            
            emo_sim = None
            if emb_gen is not None and emb_gt is not None:
                emo_sim = cosine_similarity_numpy(emb_gen, emb_gt)
                
            # B. CER (Individual)
            cer, transcript = evaluator.compute_cer(gen_path, ref_text)

            # C. Speaker Sim (Individual - requires loading tensor)
            # Note: SpeechBrain encode_batch could also be batched, but keeping logic simpler for now
            audio_tensor = evaluator.load_audio_tensor(gen_path)
            spk_sim = evaluator.compute_spk_sim(audio_tensor, gt_path)

            results.append(
                {
                    "model": item["model"],
                    "filename": item["key"],
                    "duration_gen": round(item['duration_gen'], 2),
                    "cer": cer * 100 if cer is not None else None,
                    "spk_sim": spk_sim,
                    "emo_sim": emo_sim,
                    "transcript": transcript,
                }
            )

    # 5. Save & Summarize
    df_res = pd.DataFrame(results)
    df_res.to_csv(args.output_csv, index=False)

    print(
        f"\n✅ Processing complete. {skipped_count} samples skipped due to length violations."
    )
    print(f"✅ Results saved to {args.output_csv}")

    if not df_res.empty:
        print("\n📊 --- Evaluation Summary ---")
        summary = (
            df_res.groupby("model")
            .agg(
                {
                    "cer": "mean",
                    "spk_sim": "mean",
                    "emo_sim": "mean",
                    # 'emotion_match': 'mean'
                }
            )
            .reset_index()
        )

        # summary['emotion_acc'] = (summary['emotion_match'] * 100).round(2)
        summary["cer"] = summary["cer"].round(3)
        summary["spk_sim"] = summary["spk_sim"].round(3)
        summary["emo_sim"] = summary["emo_sim"].round(3)

        # print(summary[['model', 'cer', 'spk_sim', 'emotion_acc']].to_markdown(index=False))
        print(summary[["model", "cer", "spk_sim", "emo_sim"]].to_markdown(index=False))


if __name__ == "__main__":
    main()