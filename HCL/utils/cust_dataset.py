import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from utils.logger import get_logger
from utils.dist import is_dist_avail_and_initialized
import glob

def load_custom_dataset(cfg):
    # This version does NOT hardcode the corrupted_dir name
    corrupted_dir = cfg.data.dataroot  # You must now provide this in cfg
    gt_dir = cfg.data.get("gt_dir", None) if cfg.data.get("gt_available", False) else None
    mask_dir = cfg.data.get("mask_dir", None) if cfg.data.get("masks_available", False) else None

    img_size = cfg.data.img_size
    return RealCorruptedDataset(
        corrupted_dir=corrupted_dir,
        gt_dir=gt_dir,
        mask_dir=mask_dir,
        img_size=img_size
    )


class RealCorruptedDataset(Dataset):
    def __init__(self, corrupted_dir, gt_dir=None, mask_dir=None, img_size=256):
        self.corrupted_paths = sorted(
            glob.glob(os.path.join(corrupted_dir, '**', '*.[jpJP][pnPNgG]*'), recursive=True)
        )

        # self.gt_paths = sorted(
        #     glob.glob(os.path.join(gt_dir, '**', '*'))  # Modify if gt also nested
        # ) if gt_dir else None
        self.gt_paths = sorted([
            f for f in glob.glob(os.path.join(gt_dir, '**', '*'), recursive=True)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]) if gt_dir else None

        self.mask_paths = sorted(
            glob.glob(os.path.join(mask_dir, '**', '*'))  # Modify if masks also nested
        ) if mask_dir else None

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        print(corrupted_dir)

    def __len__(self):
        return len(self.corrupted_paths)

    def __getitem__(self, idx):
        corrupted = self.img_transform(Image.open(self.corrupted_paths[idx]).convert('RGB'))
        gt = self.img_transform(Image.open(self.gt_paths[idx]).convert('RGB')) if self.gt_paths else corrupted.clone()
        noise = torch.zeros_like(corrupted)
        mask = self.mask_transform(Image.open(self.mask_paths[idx]).convert('L')) if self.mask_paths else torch.zeros_like(corrupted[0:1])
        return corrupted, gt, noise, mask
