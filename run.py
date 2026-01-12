import os
import csv
import argparse
from tqdm import tqdm
from pathlib import Path
import torch

from tts_interfaces import AVAILABLE_MODELS, get_model

# --- CONFIGURATION ---
# INPUT_CSV = 'expresso_test_tts.csv'
# OUTPUT_ROOT = './results'

def get_filename_from_path(path):
    """Extracts filename without extension from a full path."""
    return Path(path).stem

def main():
    parser = argparse.ArgumentParser(description="Run TTS generation benchmark.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--lang", type=str, default="en", help="Language ID (for some models)")
    parser.add_argument("--model", type=str, default="mock", choices=list(AVAILABLE_MODELS.keys()),
                        help="Which model wrapper to use.")
    args = parser.parse_args()

    # 1. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Initialize Model dynamically
    # The factory function handles the lookup and instantiation
    # try:
        # tts_engine = get_model(args.model)
    # except Exception as e:
        # print(f"Failed to load model {args.model}: {e}")
        # return
    tts_engine = get_model(args.model, device=device)

    # 2. Setup Output Directory
    model_dir = os.path.join(args.output_root, args.model)
    os.makedirs(model_dir, exist_ok=True)

    # 3. Read CSV
    print(f"Reading data from {args.input_csv}...")
    rows = []
    with open(args.input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} samples.")

    # 4. Generation Loop
    for row in tqdm(rows):
        text = row['text']
        ref_path = row['reference_path']
        gt_path = row['ground_truth_path']
        
        # New column extraction (might be None if using old csv)
        ref_text = row.get('ref_text', "") 
        
        file_id = get_filename_from_path(gt_path)
        output_filename = f"{file_id}.wav"
        output_path = os.path.join(model_dir, output_filename)

        try:
            tts_engine.generate(
                text=text, 
                ref_audio_path=ref_path, 
                output_path=output_path,
                language=args.lang,
                ref_text=ref_text # Passing the new parameter
            )
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Generation complete. Results saved to {model_dir}")

if __name__ == "__main__":
    main()