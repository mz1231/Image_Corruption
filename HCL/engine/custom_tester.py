import os
import tqdm
from argparse import Namespace
from yacs.config import CfgNode as CN

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import FrechetInceptionDistance, LearnedPerceptualImagePatchSimilarity

from metrics import BinaryCrossEntropy, IntersectionOverUnion
from models import InpaintNet, RefineNet, Classifier
from utils.logger import get_logger
from utils.misc import init_seeds, get_bare_model
from utils.data import get_dataloader
from utils.cust_dataset import load_custom_dataset
from utils.dist import get_rank, get_world_size, get_local_rank, init_distributed_mode
from utils.dist import main_process_only, is_dist_avail_and_initialized, is_main_process


class CustomTester:
    def __init__(self, args: Namespace, cfg: CN):
        self.args, self.cfg = args, cfg

        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(self.cfg.seed + get_rank())

        self.logger = get_logger()
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f"Number of devices: {get_world_size()}")

        self.test_set = load_custom_dataset(self.cfg)
        self.logger.info(f'Size of test set: {len(self.test_set)}')

        self.inpaintnet = InpaintNet(**self.cfg.inpaintnet).to(self.device).eval()
        self.refinenet = RefineNet(**self.cfg.refinenet).to(self.device).eval()
        self.classifier = Classifier(dim=self.cfg.inpaintnet.proj_dim).to(self.device).eval()
        self.n_stages = len(self.cfg.inpaintnet.dim_mults)

        if self.cfg.test.pretrained:
            self.load_pretrained(self.cfg.test.pretrained)

        if is_dist_avail_and_initialized():
            self.inpaintnet = DDP(self.inpaintnet, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=True)
            self.refinenet = DDP(self.refinenet, device_ids=[get_local_rank()], output_device=get_local_rank(), find_unused_parameters=True)
            self.classifier = DDP(self.classifier, device_ids=[get_local_rank()], output_device=get_local_rank())

    def load_pretrained(self, model_path: str):
        ckpt = torch.load(model_path, map_location='cpu')
        if 'inpaintnet' in ckpt:
            self.inpaintnet.load_state_dict(ckpt['inpaintnet'])
            self.logger.info(f"Loaded inpaintnet from {model_path}")
        if 'refinenet' in ckpt:
            self.refinenet.load_state_dict(ckpt['refinenet'])
            self.logger.info(f"Loaded refinenet from {model_path}")
        if 'classifier' in ckpt:
            self.classifier.load_state_dict(ckpt['classifier'])
            self.logger.info(f"Loaded classifier from {model_path}")

    @torch.no_grad()
    def evaluate(self):
        self.logger.info('Start evaluating...')
        cfg = self.cfg.test

        test_set = self.test_set
        if cfg.n_eval is not None:
            if cfg.n_eval < len(self.test_set):
                test_set = Subset(self.test_set, torch.arange(cfg.n_eval))
                self.logger.info(f"Use a subset of test set, {cfg.n_eval}/{len(self.test_set)}")
            else:
                self.logger.warning(f'Size of test set <= n_eval, ignore n_eval')

        micro_batch = self.cfg.dataloader.micro_batch
        if micro_batch == 0:
            micro_batch = self.cfg.dataloader.batch_size
        self.logger.info(f'Batch size per device: {micro_batch}')
        self.logger.info(f'Effective batch size: {micro_batch * get_world_size()}')
        test_loader = get_dataloader(
            dataset=test_set,
            shuffle=False,
            drop_last=False,
            batch_size=micro_batch,
            num_workers=self.cfg.dataloader.num_workers,
            pin_memory=self.cfg.dataloader.pin_memory,
            prefetch_factor=self.cfg.dataloader.prefetch_factor,
        )

        metric_mask = [MetricCollection(
            dict(bce=BinaryCrossEntropy(),
                 acc=BinaryAccuracy(multidim_average='samplewise'),
                 f1=BinaryF1Score(multidim_average='samplewise'),
                 iou=IntersectionOverUnion())
        ).to(self.device) for _ in range(self.n_stages)]
        # These metrics expect images to be in [0, 1]
        metric_image = MetricCollection(
            dict(psnr=PeakSignalNoiseRatio(data_range=1, dim=(1, 2, 3)),
                 ssim=StructuralSimilarityIndexMeasure(data_range=1),
                 lpips=LearnedPerceptualImagePatchSimilarity(normalize=True))
        ).to(self.device)
        # FID metric expect images to be in [0, 255] and type uint8
        metric_fid = FrechetInceptionDistance().to(self.device)

        pbar = tqdm.tqdm(test_loader, desc='Evaluating', ncols=120, disable=not is_main_process())
        for X, gt_img, noise, mask in pbar:
            X = X.to(device=self.device, dtype=torch.float32)
            gt_img = gt_img.to(device=self.device, dtype=torch.float32)
            mask = mask.to(device=self.device, dtype=torch.float32)
            recX, projs, conf_mask_hier = self.inpaintnet(X, classifier=self.classifier)
            pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
            refX = self.refinenet(recX, pred_mask)
            for st in range(self.n_stages):
                acc_conf = F.interpolate(conf_mask_hier['acc_confs'][st], size=mask.shape[-2:])
                metric_mask[st].update(acc_conf, mask.long())
            metric_image.update((refX + 1) / 2, (gt_img + 1) / 2)
            metric_fid.update(((refX + 1) / 2 * 255).to(dtype=torch.uint8), real=False)
            metric_fid.update(((gt_img + 1) / 2 * 255).to(dtype=torch.uint8), real=True)
        pbar.close()

        for k, v in metric_image.compute().items():
            self.logger.info(f'{k}: {v.mean()}')
        self.logger.info(f'fid: {metric_fid.compute()}')
        for st in range(self.n_stages):
            for k, v in metric_mask[st].compute().items():
                self.logger.info(f'stage{st}-{k}: {v.mean().item()}')
        self.logger.info('End of evaluation')
        
    @main_process_only
    @torch.no_grad()
    def sample(self):
        self.logger.info('Start sampling...')
        inpaintnet = get_bare_model(self.inpaintnet)
        refinenet = get_bare_model(self.refinenet)
        classifier = get_bare_model(self.classifier)

        cfg = self.cfg.test
        os.makedirs(cfg.save_dir, exist_ok=True)

        ids = torch.randperm(len(self.test_set))[:cfg.n_samples] if cfg.random else torch.arange(cfg.n_samples)

        for i in tqdm.tqdm(ids, desc='Sampling', ncols=120):
            data = self.test_set[i]
            X = data[0].unsqueeze(0).to(self.device)
            gt_img = data[1] if len(data) > 1 else X.squeeze(0).cpu()
            noise = data[2] if len(data) > 2 else gt_img
            mask = data[3] if len(data) > 3 else torch.zeros_like(X.squeeze(0)[0:1])

            recX, _, conf_mask_hier = inpaintnet(X, classifier=classifier)
            pred_mask = F.interpolate(conf_mask_hier['pred_masks'][0].float(), X.shape[-2:])
            refX = refinenet(recX, pred_mask)

            gt_img = (gt_img + 1) / 2
            noise = (noise + 1) / 2
            mask = mask.repeat(3, 1, 1).cpu()
            X = (X.squeeze(0).cpu() + 1) / 2
            pred_masks = [F.interpolate(m.float(), X.shape[-2:]).squeeze(0).repeat(3, 1, 1).cpu()
                           for m in conf_mask_hier['pred_masks']]
            refX = (refX.squeeze(0).cpu() + 1) / 2

            save_image([gt_img, noise, mask, X, *reversed(pred_masks), refX],
                       os.path.join(cfg.save_dir, f'{i.item()}.png'),
                       nrow=8, normalize=True, value_range=(0, 1))

        self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
        self.logger.info('End of sampling')
