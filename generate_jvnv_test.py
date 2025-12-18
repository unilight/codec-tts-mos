import csv
import os
import glob
from collections import defaultdict

# --- CONFIGURATION ---
# Update these paths to match your actual environment
ROOT_DIR = '/data/group1/z44476r/Corpora/jvnv/jvnv_v1'
TRANSCRIPTS_FILE = '/data/group1/z44476r/Corpora/jvnv/jvnv_v1/transcription.csv' # Replace with actual path if available
OUTPUT_CSV = 'jvnv_test_tts.csv'

def parse_filename(filename):
    """
    Parses the JVNV filename to extract metadata.
    Expected format: Speaker_Style_CorpusType_ID.wav
    Example: F1_anger_free_01.wav
    """
    # Remove extension
    stem = os.path.splitext(filename)[0]
    
    parts = stem.split('_')
    
    # We expect at least 4 parts: Speaker, Style, Type, ID
    # e.g. F1, anger, free, 01
    if len(parts) < 4:
        return None
        
    speaker = parts[0]
    file_id = parts[-1]
    
    # Handle styles/types that might have underscores (just in case)
    # Based on your ls: F1_anger_free_01 -> Speaker=F1, Style=anger, Type=free
    # We assume the last part is ID, second to last is Type, rest is Style/Speaker
    
    # Specific logic for JVNV based on provided example:
    corpus_type = parts[-2] # 'free'
    style = "_".join(parts[1:-2]) # 'anger' (handles multi-word styles if they exist)
    
    return {
        'original': stem,
        'speaker': speaker,
        'style': style,
        'corpus': corpus_type, # 'free', etc.
        'id': file_id,
        'filename': filename
    }

def load_transcriptions(filepath):
    """
    Loads transcriptions into a dictionary {filename_stem: text}.
    """
    transcripts = {}
    if not os.path.exists(filepath):
        print(f"Warning: Transcript file not found at {filepath}. Text will be empty.")
        return transcripts

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                key = parts[0]
                text = parts[2]
                transcripts[key] = text
    return transcripts

def crawl_directory(root_path):
    """
    Walks through the directory to find all valid WAV files.
    Structure: root/Speaker/Style/Type/*.wav
    """
    valid_files = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.wav'):
                parsed = parse_filename(file)
                if parsed:
                    valid_files.append(parsed)
    
    return valid_files

def construct_path(file_info):
    """
    Constructs the full filepath.
    JVNV Structure: ROOT_DIR/Speaker/Style/Type/Filename
    Example: .../F1/anger/free/F1_anger_free_01.wav
    """
    return os.path.join(
        ROOT_DIR,
        file_info['speaker'],
        file_info['style'],
        file_info['corpus'], # 'free'
        file_info['filename']
    )

def main():
    # 1. Load Data
    print("Loading transcripts...")
    transcripts = load_transcriptions(TRANSCRIPTS_FILE)
    print(len(transcripts))
    
    print(f"Scanning directory {ROOT_DIR} for files...")
    file_objects = crawl_directory(ROOT_DIR)
    
    if not file_objects:
        print("No files found! Check ROOT_DIR path.")
        return

    # 2. Group by (Speaker, Style, CorpusType)
    # We include CorpusType (free) in the grouping key to avoid mixing different types
    grouped_files = defaultdict(list)
    
    for f_info in file_objects:
        key = (f_info['speaker'], f_info['style'], f_info['corpus'])
        grouped_files[key].append(f_info)
        
    # 3. Pair files and Write CSV
    print(f"Processing {len(grouped_files)} groups...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ground_truth_path', 'reference_path', 'text', 'ref_text'])
        
        pairs_count = 0
        
        for (speaker, style, corpus), files in grouped_files.items():
            # Sort by original filename to ensure 01, 02, 03 order
            files.sort(key=lambda x: x['original'])
            
            # Pair adjacent files (01 with 02, 03 with 04, etc.)
            for i in range(0, len(files) - 1, 2):
                ground_truth = files[i]
                reference = files[i+1]
                
                # Get Text
                gt_stem = ground_truth['original'][3:]
                ref_stem = reference['original'][3:]
                
                target_text = transcripts[gt_stem]
                ref_text = transcripts[ref_stem]
                
                # Construct absolute paths
                gt_path = construct_path(ground_truth)
                ref_path = construct_path(reference)
                
                # Verification (Optional: comment out if slow)
                if not os.path.exists(gt_path):
                    print(f"Error: Constructed path not found: {gt_path}")
                    continue

                writer.writerow([gt_path, ref_path, target_text, ref_text])
                pairs_count += 1

    print(f"Done! Generated {pairs_count} pairs in '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()