"""
Script to extract motion features (VideoMAE) for ViSL dataset.
Usage:
    python scripts/visl_mae_extract_feature.py \
        --mode train val test \
        --device cuda:0 \
        --batch_size 64
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import VideoMAEModel, VideoMAEImageProcessor

sys.path.append('./')
from utils.helpers import sliding_window_for_list


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class VideoMAEFeatureReader:
    """VideoMAE feature extractor for motion features."""
    
    def __init__(
        self, 
        model_name: str = 'MCG-NJU/videomae-large', 
        cache_dir: str = None,
        device: str = 'cuda:0',
        overlap_size: int = 8,
        nth_layer: int = -1
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        print(f"Loading VideoMAE model: {model_name}")
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(self.device).eval()
        print("VideoMAE model loaded successfully!")
        
    @torch.no_grad()
    def get_feats(self, video_clips: list) -> np.ndarray:
        """
        Extract features from video clips.
        
        Args:
            video_clips: List of video clips, each clip is a list of 16 PIL Images
            
        Returns:
            Feature array of shape (num_clips, feature_dim)
        """
        inputs = self.image_processor(images=video_clips, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, output_hidden_states=True).hidden_states
        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]  # CLS token
        
        return outputs.cpu().numpy()


def extract_features_for_video(
    reader: VideoMAEFeatureReader,
    frames_dir: Path,
    batch_size: int = 64,
    overlap_size: int = 8
) -> np.ndarray:
    """
    Extract VideoMAE features for all frames of a video.
    
    Args:
        reader: VideoMAE feature reader
        frames_dir: Directory containing frame images
        batch_size: Batch size for processing
        overlap_size: Overlap size for sliding window
        
    Returns:
        Feature array of shape (num_clips, feature_dim)
    """
    # Get sorted frame files
    frame_files = sorted(frames_dir.glob('*.png'))
    
    if len(frame_files) == 0:
        return None
    
    # VideoMAE requires exactly 16 frames per clip
    # Pad if too short
    if len(frame_files) < 16:
        frame_files = frame_files + [frame_files[-1]] * (16 - len(frame_files))
    
    # Create sliding windows
    frame_chunks = sliding_window_for_list(
        list(frame_files), 
        window_size=16,  # VideoMAE requires exactly 16 frames per clip
        overlap_size=overlap_size
    )
    
    # Load images for each chunk
    video_clips = []
    for chunk in frame_chunks:
        clip = [Image.open(f).convert('RGB') for f in chunk]
        video_clips.append(clip)
    
    # Process in batches
    all_feats = []
    for i in range(0, len(video_clips), batch_size):
        batch = video_clips[i:min(i + batch_size, len(video_clips))]
        feats = reader.get_feats(batch)
        all_feats.append(feats)
    
    return np.vstack(all_feats)


def process_mode(
    dataset_root: str,
    mode: str,
    output_root: str,
    reader: VideoMAEFeatureReader,
    batch_size: int = 64,
    overlap_size: int = 8
):
    """
    Process all videos for a given mode.
    
    Args:
        dataset_root: Root directory of ViSL dataset
        mode: Dataset mode (train, val, test)
        output_root: Output directory for features
        reader: VideoMAE feature reader
        batch_size: Batch size for processing
        overlap_size: Overlap size for sliding window
    """
    mode_dir = Path(dataset_root) / mode
    frames_root = mode_dir / 'frames'
    
    # Map mode names for output (val -> dev)
    output_mode = 'dev' if mode == 'val' else mode
    output_dir = Path(output_root) / output_mode
    os.makedirs(output_dir, exist_ok=True)
    
    if not frames_root.exists():
        print(f"Frames directory not found: {frames_root}")
        return
    
    # Get all video directories
    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    
    print(f"\n{'='*50}")
    print(f"Extracting VideoMAE features for {mode}: {len(video_dirs)} videos")
    print(f"{'='*50}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for video_dir in tqdm(video_dirs, desc=f"Processing {mode}"):
        video_name = video_dir.name
        output_path = output_dir / f"{video_name}.npy"
        
        # Skip if already processed
        if output_path.exists():
            skip_count += 1
            continue
        
        try:
            features = extract_features_for_video(
                reader, video_dir, batch_size, overlap_size
            )
            
            if features is not None:
                np.save(output_path, features)
                success_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            error_count += 1
    
    print(f"\nResults for {mode}:")
    print(f"  Processed: {success_count}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  Errors: {error_count}")


def main():
    parser = argparse.ArgumentParser(description='Extract VideoMAE features for ViSL')
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_ViSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--output_root', type=str,
                        default='/workspace/khanh/SpaMo/features/mae_feat_ViSL',
                        help='Output directory for features')
    parser.add_argument('--model_name', type=str,
                        default='MCG-NJU/videomae-large',
                        help='VideoMAE model name')
    parser.add_argument('--cache_dir', type=str,
                        default='/workspace/khanh/SpaMo/cache/models',
                        help='Cache directory for model')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--overlap_size', type=int, default=8,
                        help='Overlap size for sliding window')
    
    args = parser.parse_args()
    
    # Initialize feature reader
    reader = VideoMAEFeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        overlap_size=args.overlap_size
    )
    
    # Process each mode
    for mode in args.mode:
        process_mode(
            args.dataset_root,
            mode,
            args.output_root,
            reader,
            args.batch_size,
            args.overlap_size
        )
    
    print("\n" + "="*50)
    print("VideoMAE feature extraction completed!")
    print("="*50)
    print(f"Features saved to: {args.output_root}")


if __name__ == '__main__':
    main()





