


import csv
import os
import re
from collections import defaultdict

# --- CONFIGURATION ---
SEGMENTS_FILE = "/data/group1/z44476r/Corpora/expresso/splits/test.txt"
TRANSCRIPTS_FILE = '/data/group1/z44476r/Corpora/expresso/read_transcriptions.txt'
OUTPUT_CSV = 'expresso_test_tts.csv'

# You can change this to your actual root directory
ROOT_DIR = '/data/group1/z44476r/Corpora/expresso/audio_48khz'

def parse_filename(filename):
    """
    Parses the filename to extract speaker, style, corpus, and id.
    Assumes format: <speaker>_<style>_<number>
    
    Returns:
        dict: {'speaker': str, 'style': str, 'corpus': str, 'id': str}
        or None if invalid/multi-speaker
    """
    # 1. Filter out multi-speaker files (containing hyphens) and unwanted keywords
    if '-' in filename or 'emphasis' in filename or 'essentials' in filename:
        return None
        
    parts = filename.split('_')
    
    if len(parts) < 3:
        # Invalid format (need at least speaker, style, id)
        return None
        
    speaker = parts[0]
    file_id = parts[-1]
    
    # Handle styles that might contain underscores (e.g., default_longform)
    # Everything between Speaker and ID is the Style.
    style = "_".join(parts[1:-1])
    corpus = "base"
    
    return {
        'original': filename,
        'speaker': speaker,
        'style': style,
        'corpus': corpus,
        'id': file_id
    }

def load_transcriptions(filepath):
    """Loads transcriptions into a dictionary {filename: text}."""
    transcripts = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by first whitespace (tab or space)
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                filename, text = parts
                transcripts[filename] = text
    return transcripts

def load_segments(filepath):
    """Loads valid filenames from segments file."""
    valid_files = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # The line might look like: "filename (start,end)" or just "filename"
            # We assume the filename ends before the first '(' or whitespace
            
            # Split by whitespace first
            token = line.split()[0]
            
            # Just in case there is weird formatting, remove potential parenthesis content manually
            if '(' in token:
                token = token.split('(')[0]
            
            filename = token.strip()
            
            parsed = parse_filename(filename)
            if parsed:
                valid_files.append(parsed)
                
    return valid_files

def construct_path(file_info):
    """Constructs the full filepath based on the required template."""
    # Template: <root_dir>/read/{speaker}/{style}/{corpus}/{speaker}_{style}_{id}.wav
    return f"{ROOT_DIR}/read/{file_info['speaker']}/{file_info['style']}/{file_info['corpus']}/{file_info['original']}.wav"

def main():
    # 1. Load Data
    print("Loading transcripts...")
    transcripts = load_transcriptions(TRANSCRIPTS_FILE)
    
    print("Loading segments...")
    file_objects = load_segments(SEGMENTS_FILE)
    
    # 2. Group by (Speaker, Style)
    # Key: (speaker, style), Value: List of file_info dicts
    grouped_files = defaultdict(list)
    
    for f_info in file_objects:
        key = (f_info['speaker'], f_info['style'])
        grouped_files[key].append(f_info)
        
    # 3. Pair files and Write CSV
    print(f"Processing {len(grouped_files)} groups...")
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # UPDATED HEADER: Added ref_text
        writer.writerow(['ground_truth_path', 'reference_path', 'text', 'ref_text'])
        
        pairs_count = 0
        
        for (speaker, style), files in grouped_files.items():
            files.sort(key=lambda x: x['original'])
            
            for i in range(0, len(files) - 1, 2):
                ground_truth = files[i]
                reference = files[i+1]
                
                # Get text for Ground Truth
                gt_filename = ground_truth['original']
                target_text = transcripts.get(gt_filename, "")

                # Get text for Reference (New)
                ref_filename = reference['original']
                ref_text = transcripts.get(ref_filename, "")
                
                if not target_text:
                    print(f"Warning: No transcript found for target {gt_filename}")
                
                # Construct paths
                gt_path = construct_path(ground_truth)
                ref_path = construct_path(reference)
                assert os.path.exists(gt_path), f"{gt_path} does not exist!"
                assert os.path.exists(ref_path), f"{ref_path} does not exist!"
                
                writer.writerow([gt_path, ref_path, target_text, ref_text])
                pairs_count += 1

    print(f"Done! Generated {pairs_count} pairs in '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()