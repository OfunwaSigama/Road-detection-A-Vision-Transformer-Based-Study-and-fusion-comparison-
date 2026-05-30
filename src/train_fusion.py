#train_fusion.py
"""
Early and Late Fusion training with 8-fold cross-validation
"""
import os
import sys
import json
import time
import math
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets import Kitti360FusionDataset
from src.metrics import RoadMetrics
from src.models import EarlyFusion, LateFusion

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_fold(config, train_ds, val_ds, test_ds, fold_id, test_seq, fusion_type, device):
    start_time = time.time()
    print(f"\n{'='*80}\nFOLD {fold_id} | {fusion_type} | Test={test_seq}\n{'='*80}")

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['training']['num_workers'],
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['training']['num_workers'],
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'],
                             shuffle=False, num_workers=config['training']['num_workers'],
                             pin_memory=True)

    if fusion_type == 'early':
        model = EarlyFusion().to(device)
    else:  
        model = LateFusion().to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['training']['learning_rate'],
                                  weight_decay=config['training']['weight_decay'])

    def get_lr(epoch):
        warmup = config['training']['warmup_epochs']
        total = config['training']['epochs']
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_iou = 0.0
    best_state = None
    nan_batches = 0

    for epoch in range(config['training']['epochs']):
        model.train()
        train_m = RoadMetrics()
        pbar = tqdm(train_loader, desc=f'E{epoch+1:02d}', leave=False)

        for batch in pbar:
            if fusion_type == 'early':
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
            else:  # late
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                masks = batch['mask'].to(device)
                logits = model(rgb, depth)

            logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:],
                                                     mode='bilinear', align_corners=False)
            logits = torch.clamp(logits, -config['training']['logit_clip'],
                                 config['training']['logit_clip'])
            if torch.isnan(logits).any():
                nan_batches += 1
                continue
            loss = criterion(logits, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            loss = torch.clamp(loss, max=config['training']['loss_clip'])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['training']['gradient_clip'])
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(1)
                train_m.update(pred, masks)
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        scheduler.step()

        # Validation
        model.eval()
        val_m = RoadMetrics()
        with torch.no_grad():
            for batch in val_loader:
                if fusion_type == 'early':
                    images, masks = batch
                    images, masks = images.to(device), masks.to(device)
                    logits = model(images)
                else:
                    rgb, depth, masks = batch['rgb'].to(device), batch['depth'].to(device), batch['mask'].to(device)
                    logits = model(rgb, depth)

                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:],
                                                         mode='bilinear', align_corners=False)
                pred = logits.argmax(1)
                val_m.update(pred, masks)

        train_metrics = train_m.compute_all()
        val_metrics = val_m.compute_all()
        print(f"  E{epoch+1:02d} | Train IoU:{train_metrics['road_iou']:.4f} | Val IoU:{val_metrics['road_iou']:.4f}")

        if val_metrics['road_iou'] > best_iou:
            best_iou = val_metrics['road_iou']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_m = RoadMetrics()
    with torch.no_grad():
        for batch in test_loader:
            if fusion_type == 'early':
                images, masks = batch
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
            else:
                rgb, depth, masks = batch['rgb'].to(device), batch['depth'].to(device), batch['mask'].to(device)
                logits = model(rgb, depth)

            logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:],
                                                     mode='bilinear', align_corners=False)
            pred = logits.argmax(1)
            test_m.update(pred, masks)

    results = test_m.compute_all()
    elapsed = time.time() - start_time

    print(f"\nTest results for {test_seq} ({fusion_type}):")
    print(f"  Road IoU: {results['road_iou']:.6f}")
    print(f"  F1-score: {results['f1_score']:.6f}")
    print(f"  Time: {elapsed/60:.1f} min | NaN batches: {nan_batches}")

    del model, optimizer
    torch.cuda.empty_cache()
    return results

def main():
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_root = os.environ.get("KITTI360_ROOT")
    if data_root is None:
        raise RuntimeError("Please set KITTI360_ROOT environment variable.")
    config['data']['root'] = data_root

    set_seed(config['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    sequences = config['cross_validation']['sequences']
    fusion_types = ['early', 'late']  

    for fusion_type in fusion_types:
        print(f"\n{'#'*80}\n# TRAINING: {fusion_type.upper()} FUSION\n{'#'*80}")

        all_results = {}
        progress_file = Path(f"results/{fusion_type}_progress.json")
        if progress_file.exists():
            with open(progress_file) as f:
                all_results = json.load(f)

        for i, test_seq in enumerate(sequences):
            if test_seq in all_results:
                print(f"⏭ Skipping {test_seq} (already completed)")
                continue

            others = [s for s in sequences if s != test_seq]
            val_seq, train_seqs = others[0], others[1:]

            train_ds = Kitti360FusionDataset(config['data']['root'], train_seqs, 'train', fusion_type,
                                             config['data']['image_size'], config['data']['depth_dir'])
            val_ds   = Kitti360FusionDataset(config['data']['root'], [val_seq], 'val', fusion_type,
                                             config['data']['image_size'], config['data']['depth_dir'])
            test_ds  = Kitti360FusionDataset(config['data']['root'], [test_seq], 'val', fusion_type,
                                             config['data']['image_size'], config['data']['depth_dir'])

            result = train_fold(config, train_ds, val_ds, test_ds, i+1, test_seq, fusion_type, device)
            all_results[test_seq] = result

            with open(progress_file, 'w') as f:
                json.dump(all_results, f, indent=2)

        # Summary
        if all_results:
            ious = [all_results[s]['road_iou'] for s in sequences if s in all_results]
            print(f"\n{fusion_type.upper()} SUMMARY:")
            print(f"Mean Road IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
            with open(f"results/{fusion_type}_final.json", 'w') as f:
                json.dump({'mean_iou': float(np.mean(ious)), 'std_iou': float(np.std(ious)),
                           'per_sequence': all_results}, f, indent=2)

if __name__ == "__main__":
    main()
