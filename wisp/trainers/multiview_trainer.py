# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging as log
import git
import random
import torch
import pandas as pd
from typing import Optional
from tqdm import tqdm
import wisp
from wisp.config import configure, autoconfig, instantiate
from wisp.trainers import BaseTrainer, ConfigBaseTrainer
from wisp.trainers.tracker import Tracker
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.datasets import MultiviewDataset
from wisp.core import Rays, RenderBuffer


@configure
class ConfigMultiviewTrainer(ConfigBaseTrainer):
    prune_every: int = 100
    """ Invokes nef.prune() logic every "prune_every" iterations. """

    random_lod: bool = False
    """ If True, random LODs in the occupancy structure will be picked per step, during tracing.
        If False, will always used the highest level of the occupancy structure.
    """

    rgb_lambda: float = 1.0
    """ Loss weight for rgb_loss component. """


class MultiviewTrainer(BaseTrainer):

    def __init__(self,
                 cfg: ConfigMultiviewTrainer,
                 pipeline: wisp.models.Pipeline,
                 train_dataset: wisp.datasets.WispDataset,
                 validation_dataset: wisp.datasets.WispDataset,
                 tracker: Tracker,
                 device: torch.device = 'cuda',
                 scene_state: Optional[wisp.framework.WispState] = None):
        super().__init__(cfg=cfg, pipeline=pipeline, train_dataset=train_dataset, tracker=tracker, device=device,
                         scene_state=scene_state)
        self.validation_dataset = validation_dataset

    def populate_scenegraph(self):
        """ Updates the scenegraph with information about available objects.
        Doing so exposes these objects to other components, like visualizers and loggers.
        """
        super().populate_scenegraph()
        self.scene_state.graph.cameras = self.train_dataset.cameras

    def pre_step(self):
        """Override pre_step to support pruning.
        """
        super().pre_step()
        
        if self.cfg.prune_every > -1 and \
           self.total_iterations > 1 and \
           self.total_iterations % self.cfg.prune_every == 0:
            self.pipeline.nef.prune()

    @torch.cuda.nvtx.range("MultiviewTrainer.step")
    def step(self, data):
        """Implement the optimization over image-space loss.
        """
        # Map to device
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['rgb'].to(self.device).squeeze(0)

        self.optimizer.zero_grad(set_to_none=True)
        loss = 0
        
        if self.cfg.random_lod:
            # Sample from a geometric distribution
            population = [i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [2**i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        rb = self.pipeline(rays=rays, lod_idx=lod_idx, channels=["rgb"])

        # RGB Loss
        #rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
        rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])

        rgb_loss = rgb_loss.mean()
        loss += self.cfg.rgb_lambda * rgb_loss

        # Aggregate losses
        self.tracker.metrics.total_loss += loss.item()
        self.tracker.metrics.rgb_loss += rgb_loss.item()
        self.tracker.metrics.num_samples += 1   # Multiview batches are assumed as single image per batch
        
        with torch.cuda.nvtx.range("MultiviewTrainer.backward"):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
    def log_console(self):
        total_loss = self.tracker.metrics.average_metric('total_loss')
        rgb_loss = self.tracker.metrics.average_metric('rgb_loss')
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log_text += ' | rgb loss: {:>.3E}'.format(rgb_loss)
        
        log.info(log_text)

    def evaluate_metrics(self, dataset: MultiviewDataset, valid_log_dir:str, lod_idx, name=None, lpips_model=None):

        img_count = len(dataset)
        img_shape = dataset.img_shape

        psnr_total = 0.0
        lpips_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, full_batch in tqdm(enumerate(dataset)):
                gts = full_batch['rgb'].to('cuda')
                rays = full_batch['rays'].to('cuda')
                rb = self.tracker.visualizer.render(self.pipeline, rays, lod_idx=lod_idx)

                gts = gts.reshape(*img_shape, -1)
                rb = rb.reshape(*img_shape, -1)

                psnr_total += psnr(rb.rgb[...,:3], gts[...,:3])
                if lpips_model:
                    lpips_total += lpips(rb.rgb[...,:3], gts[...,:3], lpips_model)
                ssim_total += ssim(rb.rgb[...,:3], gts[...,:3])
                
                out_rb = RenderBuffer(rgb=rb.rgb, depth=rb.depth, alpha=rb.alpha,
                                      gts=gts, err=(gts[..., :3] - rb.rgb[..., :3])**2)
                exrdict = out_rb.reshape(*img_shape, -1).cpu().exr_dict()
                
                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                try:
                    write_exr(os.path.join(valid_log_dir, out_name + ".exr"), exrdict)
                except:
                    if hasattr(self, "exr_exception"):
                        pass
                    else:
                        self.exr_exception = True
                        log.info("Skipping EXR logging since pyexr is not found.")
                write_png(os.path.join(valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb)

        psnr_total /= img_count
        lpips_total /= img_count
        ssim_total /= img_count

        metrics_dict = {"psnr": psnr_total, "ssim": ssim_total}

        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)

        if lpips_model:
            log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
            metrics_dict["lpips"] = lpips_total
        log.info(log_text)
 
        return metrics_dict
    
    def validate(self):
        self.pipeline.eval()

        record_dict = self.tracker.get_app_config(as_dict=True)
        if record_dict is None:
            log.info("app_config not supplied to Tracker, config won't be logged in the pandas log")
            record_dict = {}
        else:
            record_dict = pd.json_normalize(record_dict, sep='.').to_dict(orient='records')[0]

        # record_dict contains config args, but omits torch.Tensor fields which were not explicitly converted to
        # numpy or some other format. This is required as parquet doesn't support torch.Tensors
        # (and also for output size considerations)
        record_dict = {k: v for k, v in record_dict.items() if not isinstance(v, torch.Tensor)}

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[:8]
            record_dict["git_sha"] = sha
        except:
            log.info("Validation script not running within a git directory, so skipping SHA logging.")

        self.tracker.log_table("args", record_dict, self.epoch)

        log_dir = self.tracker.log_dir
        log_fname = self.tracker.log_fname
        dataset_name = os.path.splitext(os.path.basename(self.validation_dataset.dataset_path))[0]
        model_fname = os.path.abspath(os.path.join(log_dir, f'model.pth'))
        record_dict.update({"dataset_name" : dataset_name, "epoch": self.epoch, 
                            "log_fname" : log_fname, "model_fname": model_fname})
        parent_log_dir = os.path.dirname(log_dir)

        log.info("Beginning validation...")
        img_shape = self.validation_dataset.img_shape
        log.info(f"Running validation on dataset with {len(self.validation_dataset)} images "
                 f"at resolution {img_shape[0]}x{img_shape[1]}")

        valid_log_dir = os.path.join(log_dir, "val")
        log.info(f"Saving validation result to {valid_log_dir}")
        if not os.path.exists(valid_log_dir):
            os.makedirs(valid_log_dir)

        lods = list(range(self.pipeline.nef.grid.num_lods))
        try:
            from lpips import LPIPS
            lpips_model = LPIPS(net='vgg').cuda()
        except:
            lpips_model = None
            if hasattr(self, "lpips_exception"):
                pass
            else:
                self.lpips_exception = True
                log.info("Skipping LPIPS since lpips is not found.")
        evaluation_results = self.evaluate_metrics(self.validation_dataset, valid_log_dir, lods[-1],
                                                   f"lod{lods[-1]}", lpips_model=lpips_model)
        record_dict.update(evaluation_results)
        for key in evaluation_results:
            self.tracker.log_metric(f"Validation/{key}", evaluation_results[key], self.epoch)
        
        df = pd.DataFrame.from_records([record_dict])
        df['lod'] = lods[-1]
        fname = os.path.join(parent_log_dir, f"logs.parquet")
        if os.path.exists(fname):
            df_ = pd.read_parquet(fname)
            df = pd.concat([df_, df])
        df.to_parquet(fname, index=False)

    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        super().pre_training()
        self.tracker.metrics.define_metric('rgb_loss', aggregation_type=float)

    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """
        self.tracker.log_360_orbit(pipeline=self.pipeline)
        super().post_training()
