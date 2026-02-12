#!/bin/bash
# Master script to prepare ViSL dataset for SpaMo finetuning
# Usage: bash scripts/visl_prepare_all.sh

set -e

echo "=============================================="
echo "ViSL Dataset Preparation for SpaMo"
echo "=============================================="

# Configuration
DATASET_ROOT="/workspace/khanh/SpaMo/dataset_ViSL"
DEVICE="cuda:0"

# Step 1: Extract frames from videos
echo ""
echo ">>> Step 1: Extracting frames from videos..."
echo "=============================================="
python scripts/visl_extract_frames.py \
    --dataset_root $DATASET_ROOT \
    --mode train val test \
    --fps 25 \
    --num_workers 4

# Step 2: Translate texts
echo ""
echo ">>> Step 2: Translating texts to EN/FR/ES..."
echo "=============================================="
echo "Note: This step requires internet connection for Google Translate API"
python scripts/visl_translate_text.py \
    --dataset_root $DATASET_ROOT \
    --mode train val test

# Step 3: Create annotation files
echo ""
echo ">>> Step 3: Creating annotation .npy files..."
echo "=============================================="
python scripts/visl_create_annotation.py \
    --dataset_root $DATASET_ROOT \
    --output_root /workspace/khanh/SpaMo/preprocess/ViSL \
    --mode train val test

# Step 4: Extract ViT (spatial) features
echo ""
echo ">>> Step 4: Extracting ViT spatial features..."
echo "=============================================="
python scripts/visl_vit_extract_feature.py \
    --dataset_root $DATASET_ROOT \
    --output_root /workspace/khanh/SpaMo/features/vit_feat_ViSL \
    --device $DEVICE \
    --batch_size 32 \
    --mode train val test

# Step 5: Extract VideoMAE (motion) features
echo ""
echo ">>> Step 5: Extracting VideoMAE motion features..."
echo "=============================================="
python scripts/visl_mae_extract_feature.py \
    --dataset_root $DATASET_ROOT \
    --output_root /workspace/khanh/SpaMo/features/mae_feat_ViSL \
    --device $DEVICE \
    --batch_size 16 \
    --mode train val test

echo ""
echo "=============================================="
echo "ViSL dataset preparation completed!"
echo "=============================================="
echo ""
echo "Files created:"
echo "  - preprocess/ViSL/*.npy (annotations)"
echo "  - features/vit_feat_ViSL/ (spatial features)"
echo "  - features/mae_feat_ViSL/ (motion features)"
echo ""
echo "To start finetuning, run:"
echo "  python main.py -c configs/finetune_visl.yaml -e bleu"





