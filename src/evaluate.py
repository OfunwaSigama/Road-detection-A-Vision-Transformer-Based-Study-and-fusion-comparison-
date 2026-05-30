"""
Evaluate a trained model on a given test sequence.
Usage: python evaluate.py --model_type rgb --checkpoint path.pth --test_seq 0002
"""
import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets import Kitti360RGBDataset, Kitti360FusionDataset
from src.metrics import RoadMetrics
from src.models import RGBBaseline, EarlyFusion, LateFusion, CrossAttentionFusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=['rgb', 'early', 'late', 'cross'], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_seq", default="0002")
    args = parser.parse_args()

    data_root = os.environ.get("KITTI360_ROOT")
    if not data_root:
        raise RuntimeError("Set KITTI360_ROOT environment variable")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_type == 'rgb':
        model = RGBBaseline().to(device)
        test_ds = Kitti360RGBDataset(data_root, [args.test_seq], 'val', image_size=(512,192))
    elif args.model_type == 'early':
        model = EarlyFusion().to(device)
        test_ds = Kitti360FusionDataset(data_root, [args.test_seq], 'val', 'early', depth_dir="depth_maps/numpy")
    elif args.model_type == 'late':
        model = LateFusion().to(device)
        test_ds = Kitti360FusionDataset(data_root, [args.test_seq], 'val', 'late', depth_dir="depth_maps/numpy")
    else:  # cross
        model = CrossAttentionFusion().to(device)
        test_ds = Kitti360FusionDataset(data_root, [args.test_seq], 'val', 'cross_attention', depth_dir="depth_maps/numpy")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    metrics = RoadMetrics()
    with torch.no_grad():
        for batch in loader:
            if args.model_type == 'rgb':
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
            elif args.model_type == 'early':
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
            else:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                masks = batch['mask'].to(device)
                logits = model(rgb, depth)

            logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:],
                                                     mode='bilinear', align_corners=False)
            pred = logits.argmax(1)
            metrics.update(pred, masks)

    res = metrics.compute_all()
    print(f"\nResults on sequence {args.test_seq}:")
    for k, v in res.items():
        print(f"  {k}: {v:.6f}")

if __name__ == "__main__":
    main()
