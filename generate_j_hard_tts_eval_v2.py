import argparse
import csv
import glob
import os
from pathlib import Path

import torch
from tqdm import tqdm

from tts_interfaces import AVAILABLE_MODELS, get_model


DEFAULT_CORPORA_DIR = Path("j_hard_tts_eval_v2/data")
DEFAULT_AUDIO_DIR = Path("/mrnas04/internal/wenchin-h/Corpora/common-voice/cv-corpus-21.0-2025-03-14/ja/clips")
DEFAULT_METADATA_CSV = Path("jhard_eval_v2.csv")
DEFAULT_OUTPUT_ROOT = Path("results_j_hard_tts_eval_v2")


def process_corpus_file(txt_path: str, audio_dir: Path):
    samples = []

    if not os.path.exists(txt_path):
        print(f"Warning: Corpus file not found: {txt_path}")
        return samples

    subset = Path(txt_path).stem
    with open(txt_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 4:
                print(f"Skipping malformed line in {txt_path}:{line_number}: {line}")
                continue

            ref_filename = parts[0].strip()
            ref_text = parts[1].strip()
            sample_id = parts[2].strip()
            target_text = parts[3].strip()

            ref_path = Path(ref_filename)
            if not ref_path.is_absolute():
                ref_path = audio_dir / ref_filename

            if not ref_path.exists():
                print(f"Warning: Audio file missing: {ref_path}")
                continue

            samples.append({
                "ground_truth_path": sample_id,
                "reference_path": str(ref_path),
                "text": target_text,
                "ref_text": ref_text,
                "subset": subset,
            })

    return samples


def build_metadata(corpora_dir: Path, audio_dir: Path, metadata_csv: Path):
    txt_files = sorted(glob.glob(str(corpora_dir / "*.txt")))
    if not txt_files:
        raise ValueError(f"No text files found in {corpora_dir}")

    all_samples = []
    for txt_file in tqdm(txt_files, desc="Preparing metadata"):
        all_samples.extend(process_corpus_file(txt_file, audio_dir))

    if not all_samples:
        raise ValueError("No valid samples found while preparing metadata.")

    with open(metadata_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ground_truth_path", "reference_path", "text", "ref_text", "subset"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_samples)

    return all_samples


def get_filename_from_path(path: str) -> str:
    return Path(path).stem


def generate_samples(rows, model_name: str, output_root: Path, lang: str, num_samples: int):
    # os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
    #     "/data/group1/z44476r/espeak-ng/.local/lib/libespeak-ng.so"
    # )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tts_engine = get_model(model_name, device=device)

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} sample(s) per row for {len(rows)} rows.")
    for row in tqdm(rows, desc="Generating audio"):
        text = row["text"]
        ref_path = row["reference_path"]
        gt_path = row["ground_truth_path"]
        ref_text = row.get("ref_text", "")
        subset = row.get("subset")

        file_id = get_filename_from_path(gt_path)
        target_dir = model_dir / subset if subset else model_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            for i in range(num_samples):
                output_path = target_dir / f"{file_id}-{i}.wav"
                if output_path.exists():
                    continue

                tts_engine.generate(
                    text=text,
                    ref_audio_path=ref_path,
                    output_path=str(output_path),
                    language=lang,
                    ref_text=ref_text,
                )
        except Exception as exc:
            print(f"Error processing {file_id}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"Generation complete. Results saved to {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare J-HARD TTS Eval v2 metadata and generate TTS samples.")
    parser.add_argument("--corpora_dir", type=Path, default=DEFAULT_CORPORA_DIR)
    parser.add_argument("--audio_dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata_csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lang", type=str, default="ja")
    parser.add_argument("--model", type=str, required=True, choices=list(AVAILABLE_MODELS.keys()))
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--metadata_only", action="store_true")
    args = parser.parse_args()

    rows = build_metadata(args.corpora_dir, args.audio_dir, args.metadata_csv)
    print(f"Wrote metadata CSV to {args.metadata_csv}")

    if args.metadata_only:
        return

    generate_samples(
        rows=rows,
        model_name=args.model,
        output_root=args.output_root,
        lang=args.lang,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
