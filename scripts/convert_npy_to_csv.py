import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def convert_npy_to_csv(npy_path: str, output_csv: str):
    """Convert Phoenix14T annotation .npy file to CSV format."""
    print(f"Loading annotation file: {npy_path}")
    data = np.load(npy_path, allow_pickle=True).item()
    
    # Get prefix if exists
    prefix = data.get('prefix', '')
    print(f"Prefix: {prefix}")
    
    # Collect all samples (exclude 'prefix' key)
    samples = []
    sample_keys = [k for k in data.keys() if k != 'prefix']
    sample_keys.sort()
    
    print(f"Found {len(sample_keys)} samples")
    
    for key in sample_keys:
        sample = data[key]
        row = {
            'index': key,
            'fileid': sample.get('fileid', ''),
            'folder': sample.get('folder', ''),
            'text': sample.get('text', ''),
            'gloss': sample.get('gloss', ''),
            'num_frames': sample.get('num_frames', 0),
            'en_text': sample.get('en_text', ''),
            'fr_text': sample.get('fr_text', ''),
            'es_text': sample.get('es_text', ''),
            'signer': sample.get('signer', ''),
            'original_info': sample.get('original_info', ''),
            'tag': sample.get('tag', ''),
        }
        samples.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(samples)
    
    # Reorder columns
    column_order = [
        'index', 'fileid', 'folder', 'signer', 'num_frames',
        'text', 'gloss', 'en_text', 'fr_text', 'es_text',
        'original_info', 'tag'
    ]
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV file: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Phoenix14T .npy annotation to CSV')
    parser.add_argument('--input', type=str, 
                       default='/workspace/khanh/SpaMo/preprocess/Phoenix14T/train_info_ml.npy',
                       help='Path to input .npy file')
    parser.add_argument('--output', type=str,
                       default='/workspace/khanh/SpaMo/preprocess/Phoenix14T/train_info_ml.csv',
                       help='Path to output CSV file')
    
    args = parser.parse_args()
    convert_npy_to_csv(args.input, args.output)