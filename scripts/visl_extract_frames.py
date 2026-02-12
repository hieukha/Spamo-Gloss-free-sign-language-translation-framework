"""
Script to extract frames from video files for ViSL dataset.
Usage:
    python scripts/visl_extract_frames.py --mode train val test --fps 25
"""

import os
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_frames(video_path: str, output_dir: str, fps: int = 25) -> int:
    """
    Extract frames from a video file using ffmpeg.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract
        
    Returns:
        Number of extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use ffmpeg to extract frames
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality
        '-hide_banner', '-loglevel', 'error',
        os.path.join(output_dir, '%06d.png')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Count extracted frames
        num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        return num_frames
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}: {e.stderr.decode()}")
        return 0


def process_mode(dataset_root: str, mode: str, fps: int, num_workers: int = 4):
    """
    Process all videos for a given mode (train/val/test).
    
    Args:
        dataset_root: Root directory of ViSL dataset
        mode: Dataset mode (train, val, test)
        fps: Frames per second to extract
        num_workers: Number of parallel workers
    """
    mode_dir = Path(dataset_root) / mode
    video_dir = mode_dir / 'sentence_clips'
    frames_dir = mode_dir / 'frames'
    metadata_path = mode_dir / 'sentence_clips_metadata.csv'
    
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    print(f"\n{'='*50}")
    print(f"Processing {mode} set: {len(df)} videos")
    print(f"{'='*50}")
    
    # Track frame counts for updating metadata
    frame_counts = {}
    
    def process_video(row):
        video_name = row['name']
        video_path = video_dir / f"{video_name}.mp4"
        output_dir = frames_dir / video_name
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return video_name, 0
        
        # Skip if frames already exist
        if output_dir.exists() and len(list(output_dir.glob('*.png'))) > 0:
            num_frames = len(list(output_dir.glob('*.png')))
            return video_name, num_frames
        
        num_frames = extract_frames(str(video_path), str(output_dir), fps)
        return video_name, num_frames
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, row): row['name'] 
                   for _, row in df.iterrows()}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {mode}"):
            video_name, num_frames = future.result()
            frame_counts[video_name] = num_frames
    
    # Save frame counts
    frame_counts_path = mode_dir / 'frame_counts.csv'
    pd.DataFrame([
        {'name': name, 'num_frames': count} 
        for name, count in frame_counts.items()
    ]).to_csv(frame_counts_path, index=False)
    
    print(f"Frame counts saved to: {frame_counts_path}")
    print(f"Total videos processed: {len(frame_counts)}")
    print(f"Videos with frames: {sum(1 for c in frame_counts.values() if c > 0)}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from ViSL videos')
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_ViSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second to extract')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    for mode in args.mode:
        process_mode(args.dataset_root, mode, args.fps, args.num_workers)
    
    print("\n" + "="*50)
    print("Frame extraction completed!")
    print("="*50)


if __name__ == '__main__':
    main()





