"""
Extract frames from ViSL videos in batches, tar+pigz, upload to Google Drive, 
then delete local frames to save disk space.

Pipeline per batch:
  1. Extract frames for N videos (ffmpeg)
  2. Tar+pigz the frames into a .tar.gz
  3. Upload to Google Drive (rclone)
  4. Delete local frames
  5. Repeat

Usage:
    conda activate spamo
    python scripts/visl_extract_frames_upload.py \
        --dataset_root /dataset/khanh/dataset_VSL \
        --mode train \
        --batch_size 300 \
        --gdrive_root gdrive:SpaMo/dataset_VSL_frames \
        --fps 25 --num_workers 4
"""

import os
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Frame extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, fps: int = 25) -> int:
    """Extract frames from a video file using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',
        '-hide_banner', '-loglevel', 'error',
        os.path.join(output_dir, '%06d.png')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        return num_frames
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_path}: {e.stderr.decode()}")
        return 0


def extract_batch(video_list, video_dir, frames_dir, fps, num_workers):
    """Extract frames for a batch of videos. Returns dict of {name: num_frames}."""
    frame_counts = {}
    
    def process_video(video_name):
        video_path = video_dir / f"{video_name}.mp4"
        output_dir = frames_dir / video_name
        
        if not video_path.exists():
            return video_name, 0
        
        # Skip if frames already exist locally
        if output_dir.exists() and len(list(output_dir.glob('*.png'))) > 0:
            num_frames = len(list(output_dir.glob('*.png')))
            return video_name, num_frames
        
        num_frames = extract_frames(str(video_path), str(output_dir), fps)
        return video_name, num_frames
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, name): name for name in video_list}
        for future in tqdm(futures, total=len(futures), desc="  Extracting"):
            name, count = future.result()
            frame_counts[name] = count
    
    return frame_counts


# ── Tar + Upload + Delete ───────────────────────────────────────────────────

def tar_and_upload(
    frames_dir: Path, 
    video_names: list, 
    batch_idx: int, 
    mode: str,
    gdrive_root: str,
    pigz_threads: int = 32
):
    """Tar+pigz batch frames, upload to Drive, delete local."""
    
    tar_name = f"frames_{mode}_batch_{batch_idx:04d}.tar.gz"
    tar_path = frames_dir.parent / tar_name
    
    # 1. Create tar.gz with only this batch's frame dirs
    print(f"  📦 Creating {tar_name}...")
    
    # Create a file list for tar
    file_list = frames_dir.parent / f".tar_list_{batch_idx}.txt"
    with open(file_list, 'w') as f:
        for name in video_names:
            frame_dir = frames_dir / name
            if frame_dir.exists():
                # Relative path from frames_dir parent
                f.write(f"frames/{name}\n")
    
    cmd_tar = (
        f"tar -c -C {frames_dir.parent} -T {file_list} "
        f"| pigz -p {pigz_threads} -0 > {tar_path}"
    )
    
    result = subprocess.run(cmd_tar, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Tar failed: {result.stderr}")
        file_list.unlink(missing_ok=True)
        return False
    
    file_list.unlink(missing_ok=True)
    
    tar_size = tar_path.stat().st_size / (1024**3)
    print(f"  📦 {tar_name}: {tar_size:.2f} GB")
    
    # 2. Upload to Google Drive
    print(f"  ☁️  Uploading to {gdrive_root}/{mode}/...")
    cmd_upload = [
        'rclone', 'copy', str(tar_path),
        f"{gdrive_root}/{mode}/",
        '--progress', '--drive-chunk-size', '256M'
    ]
    
    result = subprocess.run(cmd_upload, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Upload failed: {result.stderr}")
        return False
    
    print(f"  ✅ Upload complete!")
    
    # 3. Delete local tar and frame directories
    tar_path.unlink(missing_ok=True)
    
    deleted = 0
    for name in video_names:
        frame_dir = frames_dir / name
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
            deleted += 1
    
    print(f"  🗑️  Deleted {deleted} local frame directories + tar file")
    return True


# ── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"completed_batches": [], "frame_counts": {}}


def save_checkpoint(checkpoint_path: Path, data: dict):
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Main ────────────────────────────────────────────────────────────────────

def process_mode(
    dataset_root: str, 
    mode: str, 
    fps: int, 
    num_workers: int,
    batch_size: int,
    gdrive_root: str,
    pigz_threads: int
):
    mode_dir = Path(dataset_root) / mode
    video_dir = mode_dir / 'sentence_clips'
    frames_dir = mode_dir / 'frames'
    metadata_path = mode_dir / 'sentence_clips_metadata.csv'
    checkpoint_path = mode_dir / 'extract_frames_checkpoint.json'
    
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    import pandas as pd
    df = pd.read_csv(metadata_path)
    all_videos = df['name'].tolist()
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    completed_batches = set(checkpoint.get("completed_batches", []))
    all_frame_counts = checkpoint.get("frame_counts", {})
    
    # Split into batches
    batches = []
    for i in range(0, len(all_videos), batch_size):
        batches.append(all_videos[i:i + batch_size])
    
    total_batches = len(batches)
    already_done = sum(1 for i in range(total_batches) if i in completed_batches)
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode} | Videos: {len(all_videos)} | Batches: {total_batches}")
    print(f"Batch size: {batch_size} | Already done: {already_done}")
    print(f"Upload to: {gdrive_root}/{mode}/")
    print(f"{'='*60}")
    
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, batch_videos in enumerate(batches):
        if batch_idx in completed_batches:
            continue
        
        print(f"\n--- Batch {batch_idx + 1}/{total_batches} ({len(batch_videos)} videos) ---")
        
        # Step 1: Extract frames
        frame_counts = extract_batch(
            batch_videos, video_dir, frames_dir, fps, num_workers
        )
        all_frame_counts.update({k: v for k, v in frame_counts.items()})
        
        # Check if any frames were actually extracted
        extracted = [name for name, count in frame_counts.items() if count > 0]
        if not extracted:
            print(f"  ⚠️  No frames extracted, skipping upload")
            completed_batches.add(batch_idx)
            checkpoint["completed_batches"] = sorted(completed_batches)
            checkpoint["frame_counts"] = all_frame_counts
            save_checkpoint(checkpoint_path, checkpoint)
            continue
        
        # Step 2: Tar + Upload + Delete
        success = tar_and_upload(
            frames_dir, extracted, batch_idx, mode,
            gdrive_root, pigz_threads
        )
        
        if success:
            completed_batches.add(batch_idx)
        else:
            print(f"  ⚠️  Batch {batch_idx} upload failed, will retry on next run")
        
        # Save checkpoint
        checkpoint["completed_batches"] = sorted(completed_batches)
        checkpoint["frame_counts"] = all_frame_counts
        save_checkpoint(checkpoint_path, checkpoint)
    
    # Save final frame_counts.csv
    frame_counts_path = mode_dir / 'frame_counts.csv'
    pd.DataFrame([
        {'name': name, 'num_frames': count} 
        for name, count in all_frame_counts.items()
    ]).to_csv(frame_counts_path, index=False)
    
    print(f"\n✅ Mode {mode} complete!")
    print(f"  Total videos: {len(all_frame_counts)}")
    print(f"  Videos with frames: {sum(1 for c in all_frame_counts.values() if c > 0)}")
    print(f"  Frame counts saved to: {frame_counts_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames + upload to Google Drive in batches'
    )
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_VSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second to extract')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel ffmpeg workers')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Number of videos per batch before upload (default: 2000)')
    parser.add_argument('--gdrive_root', type=str,
                        default='gdrive:SpaMo/dataset_VSL_frames',
                        help='Google Drive path for uploads')
    parser.add_argument('--pigz_threads', type=int, default=32,
                        help='Number of pigz compression threads')
    
    args = parser.parse_args()
    
    # Verify rclone is available
    try:
        subprocess.run(['rclone', 'version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ rclone not found! Install with: conda install -c conda-forge rclone")
        return
    
    for mode in args.mode:
        process_mode(
            args.dataset_root, mode, args.fps, args.num_workers,
            args.batch_size, args.gdrive_root, args.pigz_threads
        )
    
    print("\n" + "="*60)
    print("🎉 All done!")
    print("="*60)


if __name__ == '__main__':
    main()
