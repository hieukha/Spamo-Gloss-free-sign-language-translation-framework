"""
Script to extract spatial features (CLIP ViT) for ViSL dataset.
Extracts per-frame CLS token features using CLIP ViT-L/14.

Usage:
    python scripts/visl_vit_extract_feature.py \
        --mode train val test \
        --device cuda:0 \
        --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, CLIPVisionModel

sys.path.append('./')
from utils.s2wrapper import forward as multiscale_forward


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class ViTFeatureReader:
    """CLIP ViT feature extractor for spatial (per-frame) features."""
    
    def __init__(
        self, 
        model_name: str = 'openai/clip-vit-large-patch14', 
        cache_dir: str = None,
        device: str = 'cuda:0',
        s2_mode: str = '',
        scales: list = None,
        nth_layer: int = -1
    ):
        self.device = device
        self.s2_mode = s2_mode
        self.scales = scales or []
        self.nth_layer = nth_layer
        
        print(f"Loading CLIP ViT model: {model_name}")
        
        # Try best available attention: flash_attention_2 → sdpa → eager
        for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
            try:
                self.model = CLIPVisionModel.from_pretrained(
                    model_name, 
                    output_hidden_states=True, 
                    cache_dir=cache_dir,
                    attn_implementation=attn_impl,
                ).to(device).eval()
                print(f"✅ Using attention: {attn_impl}")
                break
            except (ImportError, ValueError) as e:
                print(f"⚠️  {attn_impl} not available: {e}")
                continue
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        print("CLIP ViT model loaded successfully!")

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()
    def get_feats(self, frames: list) -> np.ndarray:
        """
        Extract features from a list of PIL Images (frames).
        
        Args:
            frames: List of PIL Images
            
        Returns:
            Feature array of shape (num_frames, feature_dim)
        """
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device).pixel_values
        
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(
                self.forward_features, inputs, 
                scales=self.scales, num_prefix_token=1
            )
        else:
            outputs = self.forward_features(inputs)
        
        return outputs[:, 0].cpu().numpy()  # CLS token


def extract_features_for_video(
    reader: ViTFeatureReader,
    frames_dir: Path,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract ViT spatial features for all frames of a video.
    
    Args:
        reader: ViT feature reader
        frames_dir: Directory containing frame images
        batch_size: Batch size for processing
        
    Returns:
        Feature array of shape (num_frames, feature_dim)
    """
    # Get sorted frame files
    frame_files = sorted(frames_dir.glob('*.png'))
    
    if len(frame_files) == 0:
        return None
    
    # Load all frames
    frames = [Image.open(f).convert('RGB') for f in frame_files]
    
    # Process in batches
    all_feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:min(i + batch_size, len(frames))]
        feats = reader.get_feats(batch)
        all_feats.append(feats)
    
    return np.concatenate(all_feats, axis=0)


def process_mode(
    dataset_root: str,
    mode: str,
    output_root: str,
    reader: ViTFeatureReader,
    batch_size: int = 32,
):
    """
    Process all videos for a given mode.
    
    Args:
        dataset_root: Root directory of ViSL dataset
        mode: Dataset mode (train, val, test)
        output_root: Output directory for features
        reader: ViT feature reader
        batch_size: Batch size for processing
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
    print(f"Extracting ViT spatial features for {mode}: {len(video_dirs)} videos")
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
                reader, video_dir, batch_size
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
    parser = argparse.ArgumentParser(description='Extract ViT spatial features for ViSL')
    parser.add_argument('--dataset_root', type=str, 
                        default='/workspace/khanh/SpaMo/dataset_ViSL',
                        help='Root directory of ViSL dataset')
    parser.add_argument('--output_root', type=str,
                        default='/workspace/khanh/SpaMo/features/vit_feat_ViSL',
                        help='Output directory for features')
    parser.add_argument('--model_name', type=str,
                        default='openai/clip-vit-large-patch14',
                        help='CLIP ViT model name')
    parser.add_argument('--cache_dir', type=str,
                        default='/workspace/khanh/SpaMo/cache/models',
                        help='Cache directory for model')
    parser.add_argument('--mode', nargs='+', type=str, 
                        default=['train', 'val', 'test'],
                        help='Dataset modes to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--s2_mode', type=str, default='',
                        help='S2 wrapping mode (e.g. "s2wrapping")')
    parser.add_argument('--scales', nargs='+', type=int, default=[],
                        help='List of scales for S2 wrapping')
    parser.add_argument('--nth_layer', type=int, default=-1,
                        help='Which hidden layer to use (-1 = last)')
    
    args = parser.parse_args()
    
    # Initialize feature reader
    reader = ViTFeatureReader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer
    )
    
    # Process each mode
    for mode in args.mode:
        process_mode(
            args.dataset_root,
            mode,
            args.output_root,
            reader,
            args.batch_size,
        )
    
    print("\n" + "="*50)
    print("ViT spatial feature extraction completed!")
    print("="*50)
    print(f"Features saved to: {args.output_root}")


if __name__ == '__main__':
    main()
