import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from albumentations import Compose, Resize, HorizontalFlip, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensorV2


class Kitti360RGBDataset(Dataset):
    """RGB-only dataset for KITTI-360 road segmentation"""
    def __init__(self, root_dir, sequences, split='train', image_size=(512, 192)):
        self.root = Path(root_dir)
        self.sequences = sequences
        self.split = split
        self.image_size = image_size
        self.samples = self._load_samples()
        self._setup_transforms()

    def _load_samples(self):
        samples = []
        for seq in self.sequences:
            seq_name = f"2013_05_28_drive_{int(seq):04d}_sync"
            img_dir = self.root / "Camera" / "download_2d_perspective" / "KITTI-360" / "data_2d_raw" / seq_name / "image_00" / "data_rect"
            lbl_dir = self.root / "Camera" / "data_2d_semantics" / "train" / seq_name / "image_00" / "semantic"
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / f"{img_path.stem}.png"
                if lbl_path.exists():
                    samples.append((img_path, lbl_path))
        return samples

    def _setup_transforms(self):
        if self.split == 'train':
            self.transform = Compose([
                Resize(self.image_size[1], self.image_size[0]),
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(self.image_size[1], self.image_size[0]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        mask = (lbl == 7).astype(np.uint8)
        aug = self.transform(image=img, mask=mask)
        return aug['image'], aug['mask'].long()


class Kitti360FusionDataset(Dataset):
    """Dataset for early/late/cross-attention fusion (RGB + depth)"""
    def __init__(self, root_dir, sequences, split='train', fusion_type='early',
                 image_size=(512, 192), depth_dir="depth_maps/numpy"):
        self.root = Path(root_dir)
        self.sequences = sequences
        self.split = split
        self.fusion_type = fusion_type
        self.image_size = image_size
        self.depth_dir = Path(depth_dir)
        self.samples = self._load_samples()
        self._setup_transforms()

    def _load_samples(self):
        samples = []
        for seq in self.sequences:
            seq_name = f"2013_05_28_drive_{int(seq):04d}_sync"
            img_dir = self.root / "Camera" / "download_2d_perspective" / "KITTI-360" / "data_2d_raw" / seq_name / "image_00" / "data_rect"
            lbl_dir = self.root / "Camera" / "data_2d_semantics" / "train" / seq_name / "image_00" / "semantic"
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                frame_id = img_path.stem
                lbl_path = lbl_dir / f"{frame_id}.png"
                depth_path = self.depth_dir / f"{seq}_{frame_id}.npy"
                if lbl_path.exists() and depth_path.exists():
                    samples.append((img_path, lbl_path, depth_path))
        return samples

    def _setup_transforms(self):
        if self.split == 'train':
            self.rgb_transform = Compose([
                Resize(self.image_size[1], self.image_size[0]),
                HorizontalFlip(p=0.5),
                RandomBrightnessContrast(p=0.2),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.rgb_transform = Compose([
                Resize(self.image_size[1], self.image_size[0]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        self.depth_transform = Compose([
            Resize(self.image_size[1], self.image_size[0]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, depth_path = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        mask = (lbl == 7).astype(np.uint8)
        depth = np.load(depth_path).astype(np.float32)
        depth = np.clip(depth, 0, 80.0) / 80.0

        rgb_aug = self.rgb_transform(image=img)
        depth_uint8 = (depth * 255).astype(np.uint8)
        depth_aug = self.depth_transform(image=depth_uint8)
        depth_tensor = depth_aug['image'].float() / 255.0

        mask_resized = cv2.resize(mask, (self.image_size[0], self.image_size[1]),
                                  interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).long()

        if self.fusion_type == 'early':
            rgbd = torch.cat([rgb_aug['image'], depth_tensor], dim=0)
            return rgbd, mask_tensor
        else: 
            return {'rgb': rgb_aug['image'], 'depth': depth_tensor, 'mask': mask_tensor}
