import os
import csv
import argparse
from tqdm import tqdm
from pathlib import Path
import torch

from tts_interfaces import AVAILABLE_MODELS, get_model


def get_filename_from_path(path):
    """Extracts filename without extension from a full path."""
    return Path(path).stem


def main():
    parser = argparse.ArgumentParser(description="Run TTS generation benchmark.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument(
        "--lang", type=str, default="en", help="Language ID (for some models)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Which model wrapper to use.",
    )
    args = parser.parse_args()

    # Set espeak library path
    # This must be set before phonemizer is initialized in the subsequent imports
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
        "/data/group1/z44476r/espeak-ng/.local/lib/libespeak-ng.so"
    )

    # 1. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Model dynamically
    # The factory function handles the lookup and instantiation
    tts_engine = get_model(args.model, device=device)

    # 2. Setup Output Directory
    model_dir = os.path.join(args.output_root, args.model)
    os.makedirs(model_dir, exist_ok=True)

    # 3. Read CSV
    print(f"Reading data from {args.input_csv}...")
    rows = []
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} samples.")

    # 4. Generation Loop
    for row in tqdm(rows):
        text = row["text"]
        ref_path = row["reference_path"]
        gt_path = row["ground_truth_path"]

        # New column extraction (might be None if using old csv)
        ref_text = row.get("ref_text", "")

        file_id = get_filename_from_path(gt_path)
        output_filename = f"{file_id}.wav"
        output_path = os.path.join(model_dir, output_filename)

        try:
            # certain models are weak when there are start/end silences (like VCTK)
            # Provide start and end when possible
            if "llasa" in args.model and "ref_start" in row and "ref_end" in row:
                ref_start = int(row["ref_start"]) / 1e7
                ref_end = int(row["ref_end"]) / 1e7
                print(ref_start, ref_end)

                tts_engine.generate(
                    text=text,
                    ref_audio_path=ref_path,
                    output_path=output_path,
                    language=args.lang,
                    ref_text=ref_text,  # Passing the new parameter
                    ref_start=ref_start,
                    ref_end=ref_end,
                )
            else:
                tts_engine.generate(
                    text=text,
                    ref_audio_path=ref_path,
                    output_path=output_path,
                    language=args.lang,
                    ref_text=ref_text,  # Passing the new parameter
                )
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            import traceback

            traceback.print_exc()

    print(f"Generation complete. Results saved to {model_dir}")


if __name__ == "__main__":
    main()
