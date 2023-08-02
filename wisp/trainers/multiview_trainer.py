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
from wisp.datasets.transforms import SampleRays
from wisp.core import Rays, RenderBuffer

import torch.nn.functional as F
from typing import Tuple
import math
import git

@configure
class ConfigMultiviewTrainer(ConfigBaseTrainer):
    start_prune : int = 1000
    """ The iteration to start the pruning process. """

    prune_every: int = 100
    """ Invokes nef.prune() logic every "prune_every" iterations. """

    random_lod: bool = False
    """ If True, random LODs in the occupancy structure will be picked per step, during tracing.
        If False, will always used the highest level of the occupancy structure.
    """

    rgb_lambda: float = 1.0
    """ Loss weight for rgb_loss component. """
 
    opacity_loss : float = 0.0
    """ Loss weight for the opacity loss. Set this to a non-zero value if the object tends to disappear. """

    rgb_loss_type : str = 'l2'  # Options: 'l2', 'l1', 'huber'
    """ The loss to use for the rgb_loss components. """

    rgb_loss_denom : str = 'rays' # Options: 'rays' or 'samples']
    """ Whether to average the loss over rays, or samples along rays. """

    target_sample_size : int = 2**18
    """ The target total sample count for adaptive ray batching. """

    save_valid_imgs : bool = False
    """ Whether to save images when running validation. """

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
    
    def calc_adaptive_rays(self, rays, warmup=False):
        if warmup:
            raymarch_results = self.pipeline.nef.grid.raymarch(rays,
                                             level=self.pipeline.nef.grid.active_lods[-1],
                                             num_samples=self.pipeline.tracer.num_steps,
                                             raymarch_type=self.pipeline.tracer.raymarch_type)
            self.pipeline.tracer.prev_num_samples = raymarch_results.samples.shape[0]

        samples_per_ray = self.pipeline.tracer.get_prev_num_samples() / rays.shape[0]
        num_rays = self.cfg.target_sample_size / max(samples_per_ray, 1)
        num_rays = int(math.floor(min(num_rays, 2**18)))
        if isinstance(self.train_dataset.transform, SampleRays):
            self.train_dataset.transform.set_num_samples(num_rays)
        else:
            raise Exception("SampleRays should be used as the transform for the dataset")

    @torch.cuda.nvtx.range("MultiviewTrainer.step")
    def step(self, data):
        """Implement the optimization over image-space loss.
        """
        # Map to device
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['rgb'].to(self.device).squeeze(0)

        if self.pipeline.tracer.get_prev_num_samples() is None:
            # Warm up run
            self.calc_adaptive_rays(rays, warmup=True)
            return

        self.optimizer.zero_grad()
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
        if self.cfg.rgb_loss_type == 'l2':
            rgb_loss = torch.nn.functional.mse_loss(rb.rgb, img_gts, reduction='none')
        elif self.cfg.rgb_loss_type == 'l1':
            rgb_loss = torch.abs(rb.rgb - img_gts)
        elif self.cfg.rgb_loss_type == 'huber':
            rgb_loss = torch.nn.functional.smooth_l1_loss(rb.rgb, img_gts, reduction='none')
        else:
            raise NotImplementedError

        #self.dataset.data["err"][idx.item()][ray_idx] = (rgb_loss.detach().mean(-1) > 1e-5).cpu()

        if self.cfg.rgb_loss_denom == 'samples':
            rgb_loss = rgb_loss.sum() / self.pipeline.tracer.prev_num_samples
        elif self.cfg.rgb_loss_denom == 'rays':
            rgb_loss = rgb_loss.mean()
            #rgb_loss = rgb_loss.sum() / rb.hit.sum()
        else:
            raise NotImplementedError
        loss += rgb_loss

        if self.cfg.opacity_loss > 0.0 and self.total_iterations < 1000:
            opacity_loss = ((1.0 - rb.alpha)**2).mean()
            loss += self.cfg.opacity_loss * opacity_loss
        
        self.tracker.metrics.total_loss += loss.item()
        self.tracker.metrics.rgb_loss += rgb_loss.item()
        self.tracker.metrics.num_samples += 1   # Multiview batches are assumed as single image per batch

        
        with torch.cuda.nvtx.range("MultiviewTrainer.backward"):
            if self.cfg.enable_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
        self.calc_adaptive_rays(rays, warmup=False)
        
        if self.cfg.scheduler:
            self.scheduler.step()

    def log_console(self):
        total_loss = self.tracker.metrics.average_metric('total_loss')
        rgb_loss = self.tracker.metrics.average_metric('rgb_loss')
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log_text += ' | rgb loss: {:>.3E}'.format(rgb_loss)
        
        log.info(log_text)
        
    def evaluate_metrics(self, validation_dataset, lod_idx, name=None):
        
        valid_log_dir = os.path.join(self.tracker.log_dir, "val")
        img_shape = validation_dataset.img_shape
        
        imgs = list(self.validation_dataset.data["rgb"])
        rays = self.validation_dataset.data["rays"]

        ray_os = list(rays.origins.cuda())
        ray_ds = list(rays.dirs.cuda())
        imgs = [img.cuda() for img in imgs]
        if 'lpips' in self.cfg.valid_metrics:
            lpips_model = LPIPS(net='vgg').cuda()

        metrics_dict = {}
        for key in self.cfg.valid_metrics:
            metrics_dict[key] = 0.0
    
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for idx, (img, ray_o, ray_d) in tqdm(enumerate(zip(imgs, ray_os, ray_ds))):
                    
                    rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                    rays = rays.reshape(-1, 3)
                    rb = self.tracker.visualizer.render(self.pipeline, rays, lod_idx=lod_idx)
                    rb = rb.reshape(*img_shape[:2], -1)
                    gts = img.reshape(*img_shape[:2], -1)
                    if 'psnr' in self.cfg.valid_metrics:
                        metrics_dict['psnr'] += psnr(rb.rgb[...,:3], gts[...,:3])
                    if 'lpips' in self.cfg.valid_metrics:
                        metrics_dict['lpips'] += lpips(rb.rgb[...,:3], gts[...,:3], lpips_model)
                    if 'ssim' in self.cfg.valid_metrics:
                        metrics_dict['ssim'] += ssim(rb.rgb[...,:3], gts[...,:3])
                    
                    out_rb = RenderBuffer(rgb=rb.rgb, depth=rb.depth, alpha=rb.alpha,
                                          gts=gts, err=(gts[..., :3] - rb.rgb[..., :3])**2)
                    out_name = f"{idx}"
                    if name is not None:
                        out_name += "-" + name

                    if self.cfg.save_valid_imgs:
                        try:
                            exrdict = out_rb.reshape(*img_shape[:2], -1).cpu().exr_dict()
                            write_exr(os.path.join(valid_log_dir, out_name + ".exr"), exrdict)
                        except:
                            if hasattr(self, "exr_exception"):
                                pass
                            else:
                                self.exr_exception = True
                                log.info("Skipping EXR logging since pyexr is not found.")
                        write_png(os.path.join(valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb)
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        for key in metrics_dict:
            metrics_dict[key] /= len(imgs)
            if not key in self.return_dict:
                self.return_dict[key] = metrics_dict[key]
            else:
                self.return_dict[key] = max(self.return_dict[key], metrics_dict[key])
            if key == 'psnr':
                log_text += ' | {}: {:.2f}'.format(f"{name} {key}", metrics_dict[key])
            else:
                log_text += ' | {}: {:.6f}'.format(f"{name} {key}", metrics_dict[key])
        log.info(log_text)
        
        return metrics_dict

    def validate(self):

        self.pipeline.eval()
        
        record_dict = self.tracker.get_record_dict()
        if record_dict is None:
            log.info("app_config not supplied to Tracker, config won't be logged in the pandas log")
            record_dict = {}

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha[:8]
            record_dict["git_sha"] = sha
        except:
            log.info("Validation script not running within a git directory, so skipping SHA logging.")

        dataset_name = os.path.splitext(os.path.basename(self.validation_dataset.dataset_path))[0]
        model_fname = os.path.abspath(os.path.join(self.tracker.log_dir, f'model.pth'))
        record_dict.update({"dataset_name" : dataset_name, "epoch": self.epoch, 
                            "log_fname" : self.tracker.log_fname, "model_fname": model_fname})

        log.info("Beginning validation...")
        img_shape = self.validation_dataset.img_shape
        log.info(f"Running validation on dataset with {len(self.validation_dataset)} images "
                 f"at resolution {img_shape[0]}x{img_shape[1]}")


        valid_log_dir = os.path.join(self.tracker.log_dir, "val")
        log.info(f"Saving validation result to {valid_log_dir}")
        if not os.path.exists(valid_log_dir):
            os.makedirs(valid_log_dir)
        lods = list(range(self.pipeline.nef.grid.num_lods))
        evaluation_results = self.evaluate_metrics(self.validation_dataset, lods[-1], f"lod{lods[-1]}")
        record_dict.update(evaluation_results)
        for key in evaluation_results:
            self.tracker.log_metric(f"validation/{key}", evaluation_results[key], self.epoch)

        self.tracker.log_table("args", record_dict, self.epoch)

        df = pd.DataFrame.from_records([record_dict])
        df['lod'] = lods[-1]
        parent_log_dir = os.path.dirname(self.tracker.log_dir)
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
        # self.tracker.log_360_orbit(pipeline=self.pipeline)
        super().post_training()
