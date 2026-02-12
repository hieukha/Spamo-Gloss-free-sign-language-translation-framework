"""
Dataset class for Vietnamese Sign Language (ViSL) dataset.
"""

import torch
import os
import numpy as np
import random
import math
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


class ViSL(torch.utils.data.Dataset):
    """
    Dataset class for the Vietnamese Sign Language (ViSL) dataset.
    
    This class handles loading video features and annotations for sign language translation,
    supporting both spatial and spatiotemporal feature types.
    """
    
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = 'dev',
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = '',
        spatiotemporal_postfix: Union[str, List[str]] = ''
    ):
        """
        Initialize the ViSL dataset.
        
        Args:
            anno_root: Root directory for annotation files
            vid_root: Root directory for video/frame files
            feat_root: Root directory for spatial features
            mae_feat_root: Root directory for spatiotemporal features
            mode: Dataset split ('train', 'dev', or 'test')
            spatial: Whether to load spatial features
            spatiotemporal: Whether to load spatiotemporal features
            spatial_postfix: Filename postfix for spatial features
            spatiotemporal_postfix: Filename postfix for spatiotemporal features
        """
        super().__init__()
        
        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix
        
        # Validate inputs
        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")
        
        # Load annotations
        anno_path = self.anno_root / f'{mode}_info_ml.npy'
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        self.data = np.load(anno_path, allow_pickle=True).item()
        
        # Set up directory paths
        self.spatial_dir = self.feat_root / self.mode
        self.spatiotemporal_dir = self.mae_feat_root / self.mode
        
        # Validate directories
        self._validate_directories()
        
        print(f"Loaded ViSL {mode} dataset: {len(self)} samples")

    def _validate_directories(self) -> None:
        """Validate that all necessary directories exist."""
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        """Load spatial features for a given file ID."""
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Load spatiotemporal features for a given file ID."""
        if isinstance(self.spatiotemporal_postfix, str):
            glor_path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not glor_path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {glor_path}")
            return torch.tensor(np.load(glor_path))
        else:
            features = []
            for postfix in self.spatiotemporal_postfix:
                path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
                features.append(torch.tensor(np.load(path)))
            return features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a dataset item by index."""
        data = self.data[index]
        file_id = data['fileid']
        pixel_value = None
        glor_value = None
        
        # Load spatial features if enabled
        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])
        
        # Load spatiotemporal features if enabled
        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]
        
        # Create result dictionary
        result = {
            'pixel_value': pixel_value,
            'glor_value': glor_value,
            'bool_mask_pos': None,
            'text': self._normalize_text(data['text']),
            'gloss': data.get('gloss', ''),
            'id': file_id,
            'num_frames': len(pixel_value) if pixel_value is not None else 0,
            'vid_path': str(self.vid_root / data.get('folder', '')),
            'lang': 'Vietnamese'  # Important: Set language to Vietnamese
        }
        
        # Add language texts if available (for in-context learning)
        for lang in ['en', 'es', 'fr']:
            if f'{lang}_text' in data:
                result[f'{lang}_text'] = self._normalize_text(str(data[f'{lang}_text']))
            else:
                result[f'{lang}_text'] = result['text']
        
        # Store original data for reference
        result['original_info'] = data
        
        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text by ensuring it ends with a period."""
        text = str(text).strip()
        if not text.endswith('.'):
            text = f"{text}."
        return text

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.data) - 1  # -1 for 'prefix' key

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch





