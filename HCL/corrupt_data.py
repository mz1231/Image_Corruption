import argparse
from yacs.config import CfgNode as CN
from utils.data import get_dataset, get_dataloader
from utils.mask_blind import DatasetWithMaskBlind
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm  # optional, for a progress bar
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    return parser

def create_data(cfg):
    test_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='test',
    )
    real_dataset_test = None
    if cfg.mask.noise_type == 'real':
        real_dataset_test = ConcatDataset([
            get_dataset(
                name=d['name'],
                dataroot=d['dataroot'],
                img_size=d['img_size'],
                split='test',
            )
            for d in cfg.mask.real_dataset
        ])
    test_set = DatasetWithMaskBlind(
        dataset=test_set,
        mask_type=cfg.mask.mask_type,
        dir_path=getattr(cfg.mask, 'dir_path', None),
        dir_invert_color=getattr(cfg.mask, 'dir_invert_color', False),
        rect_num=getattr(cfg.mask, 'rect_num', (0, 4)),
        rect_length_ratio=getattr(cfg.mask, 'rect_length_ratio', (0.2, 0.8)),
        brush_num=getattr(cfg.mask, 'brush_num', (1, 9)),
        brush_turns=getattr(cfg.mask, 'brush_turns', (4, 18)),
        brush_width_ratio=getattr(cfg.mask, 'brush_width_ratio', (0.02, 0.1)),
        brush_length_ratio=getattr(cfg.mask, 'brush_length_ratio', (0.1, 0.25)),
        noise_type=getattr(cfg.mask, 'noise_type', 'constant'),
        constant_value=getattr(cfg.mask, 'constant_value', (0, 0, 0)),
        real_dataset=real_dataset_test,
        smooth_radius=cfg.mask.smooth_radius,
        is_train=False,
    )
    return test_set


if __name__ == '__main__':
    print('hello')
    args, unknown_args = get_parser().parse_known_args()

    # Load config from file
    cfg = CN(new_allowed=True)                     # allow new keys during merge
    cfg.merge_from_file(args.config)               # load from YAML
    cfg.set_new_allowed(False)                     # disallow new keys afterward

    # Optional: handle CLI overrides like --train.n_steps 1000
    cfg.merge_from_list(unknown_args)

    cfg.freeze()                                   # lock the config

    test_set = create_data(cfg)

    output_dir = './data/classifier/train/corrupted'
    
    # Loop through the entire dataset and save each corrupted image
    for idx in tqdm(range(len(test_set)), desc="Saving corrupted images"):
        corrupted_image, original_image, noise, mask = test_set[idx]
        img = corrupted_image

        # Convert to PIL and save
        img_pil = to_pil_image(img)
        img_pil.save(os.path.join(output_dir, f"places_corrupted_{idx:04d}.png"))
        
