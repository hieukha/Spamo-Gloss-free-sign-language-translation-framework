"""
Script to create annotation .npy files from CSV metadata for ViSL dataset.
Usage:
    python scripts/visl_create_annotation.py --mode train val test

This creates files compatible with SpaMo's data loading format:
    - preprocess/ViSL/train_info.npy
    - preprocess/ViSL/train_info_ml.npy (with multilingual translations)
    - preprocess/ViSL/dev_info.npy
    - preprocess/ViSL/dev_info_ml.npy
    - preprocess/ViSL/test_info.npy
    - preprocess/ViSL/test_info_ml.npy
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_num_frames(frames_dir: Path, video_name: str) -> int:
    """Get the number of frames for a video."""
    frame_folder = frames_dir / video_name
    if frame_folder.exists():
        return len(list(frame_folder.glob('*.png')))
    return 0


def create_annotation(dataset_root: str, mode: str, output_root: str):
    """
    Create annotation .npy files for a given mode.
    
    Args:
        dataset_root: Root directory of ViSL dataset
        mode: Dataset mode (train, val, test)
        output_root: Output directory for annotation files
    """
    mode_dir = Path(dataset_root) / mode
    frames_dir = mode_dir / 'frames'
    
    # Check for translated metadata first, fall back to original
    metadata_path = mode_dir / 'sentence_clips_metadata_translated.csv'
    if not metadata_path.exists():
        metadata_path = mode_dir / 'sentence_clips_metadata.csv'
        print(f"Warning: Using non-translated metadata for {mode}")
    
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Map mode names (val -> dev for SpaMo compatibility)
    output_mode = 'dev' if mode == 'val' else mode
    
    print(f"\n{'='*50}")
    print(f"Creating annotation for {mode} ({len(df)} samples)")
    print(f"{'='*50}")
    
    # Create annotation dictionaries
    data = {'prefix': f'./dataset_ViSL/{mode}/frames'}
    data_ml = {'prefix': f'./dataset_ViSL/{mode}/frames'}
    
    valid_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {mode}"):
        video_name = row['name']
        
        # Get number of frames
        num_frames = get_num_frames(frames_dir, video_name)
        
        if num_frames == 0:
            skipped_count += 1
            continue
        
        # Create basic annotation entry
        entry = {
            'fileid': video_name,
            'folder': f'{mode}/{video_name}/*.png',
            'signer': f"Signer{row.get('signer_id', 0):02d}",
            'gloss': '',  # ViSL doesn't have gloss annotations
            'text': str(row['text']).strip(),
            'num_frames': num_frames,
            'original_info': f"{video_name}|{row.get('video_source', '')}|{row.get('duration', 0)}",
            'tag': 'visl'
        }
        
        data[valid_count] = entry.copy()
        
        # Create multilingual annotation entry
        entry_ml = entry.copy()
        
        # Add translations if available
        entry_ml['en_text'] = str(row.get('en_text', row['text'])).strip()
        entry_ml['fr_text'] = str(row.get('fr_text', row['text'])).strip()
        entry_ml['es_text'] = str(row.get('es_text', row['text'])).strip()
        
        # Handle NaN values
        for lang in ['en_text', 'fr_text', 'es_text']:
            if entry_ml[lang] == 'nan' or pd.isna(entry_ml.get(lang)):
                entry_ml[lang] = entry_ml['text']
        
        data_ml[valid_count] = entry_ml
        valid_count += 1
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Save annotation files
    output_path = Path(output_root) / f'{output_mode}_info.npy'
    output_path_ml = Path(output_root) / f'{output_mode}_info_ml.npy'
    
    np.save(output_path, data, allow_pickle=True)
    np.save(output_path_ml, data_ml, allow_pickle=True)
    
    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path_ml}")
    print(f"Valid samples: {valid_count}")
    print(f"Skipped (no frames): {skipped_count}")
    
    # Show sample entry
    if valid_count > 0:
        print("\nSample entry:")
        sample = data_ml[0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + '...'
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Create ViSL annotation files')
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_ViSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--output_root', type=str,
                        default='/workspace/khanh/SpaMo/preprocess/ViSL',
                        help='Output directory for annotation files')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    
    args = parser.parse_args()
    
    for mode in args.mode:
        create_annotation(args.dataset_root, mode, args.output_root)
    
    print("\n" + "="*50)
    print("Annotation creation completed!")
    print("="*50)
    print(f"\nAnnotation files saved to: {args.output_root}")
    print("\nFiles created:")
    for f in sorted(Path(args.output_root).glob('*.npy')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()





