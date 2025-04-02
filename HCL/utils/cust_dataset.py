import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from utils.logger import get_logger
from utils.dist import is_dist_avail_and_initialized

def load_custom_dataset(cfg):
    corrupted_dir = os.path.join(cfg.data.dataroot, "corrupted")
    gt_dir = os.path.join(cfg.data.dataroot, "gt") if cfg.data.get("gt_available", False) else None
    mask_dir = os.path.join(cfg.data.dataroot, "masks") if cfg.data.get("masks_available", False) else None

    img_size = cfg.data.img_size
    return RealCorruptedDataset(
        corrupted_dir=corrupted_dir,
        gt_dir=gt_dir,
        mask_dir=mask_dir,
        img_size=img_size
    )


class RealCorruptedDataset(Dataset):
    def __init__(self, corrupted_dir, gt_dir=None, mask_dir=None, img_size=256):
        self.corrupted_paths = sorted([os.path.join(corrupted_dir, f) for f in os.listdir(corrupted_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]) if gt_dir else None
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]) if mask_dir else None

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.corrupted_paths)

    def __getitem__(self, idx):
        corrupted = self.img_transform(Image.open(self.corrupted_paths[idx]).convert('RGB'))
        gt = self.img_transform(Image.open(self.gt_paths[idx]).convert('RGB')) if self.gt_paths else corrupted.clone()
        mask = self.mask_transform(Image.open(self.mask_paths[idx]).convert('L')) if self.mask_paths else torch.zeros_like(corrupted[0:1])

        return corrupted, gt, corrupted, mask
