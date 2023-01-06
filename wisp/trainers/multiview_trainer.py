# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging as log
from tqdm import tqdm
import random
import pandas as pd
import torch
from lpips import LPIPS
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays, RenderBuffer

import wandb
import numpy as np
from PIL import Image

class MultiviewTrainer(BaseTrainer):

    def pre_step(self):
        """Override pre_step to support pruning.
        """
        super().pre_step()
        
        if self.extra_args["prune_every"] > -1 and self.iteration > 0 and self.iteration % self.extra_args["prune_every"] == 0:
            self.pipeline.nef.prune()

    def init_log_dict(self):
        """Custom log dict.
        """
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0.0

    def step(self, data):
        """Implement the optimization over image-space loss.
        """

        # Map to device
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        self.optimizer.zero_grad()
            
        loss = 0
        
        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [2**i for i in range(self.pipeline.nef.grid.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays, lod_idx=lod_idx, channels=["rgb"])

            # RGB Loss
            #rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            
            rgb_loss = rgb_loss.mean()
            loss += self.extra_args["rgb_loss"] * rgb_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()

        self.log_dict['total_loss'] += loss.item()
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
    def log_cli(self):
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'] / len(self.train_data_loader))
        
        log.info(log_text)

    def evaluate_metrics(self, rays, imgs, lod_idx, name=None):
        
        ray_os = list(rays.origins)
        ray_ds = list(rays.dirs)
        lpips_model = LPIPS(net='vgg').cuda()

        psnr_total = 0.0
        lpips_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, (img, ray_o, ray_d) in tqdm(enumerate(zip(imgs, ray_os, ray_ds))):
                
                rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                rays = rays.reshape(-1, 3)
                rays = rays.to('cuda')
                rb = self.renderer.render(self.pipeline, rays, lod_idx=lod_idx)
                rb = rb.reshape(*img.shape[:2], -1)
                
                gts = img.cuda()
                psnr_total += psnr(rb.rgb[...,:3], gts[...,:3])
                lpips_total += lpips(rb.rgb[...,:3], gts[...,:3], lpips_model)
                ssim_total += ssim(rb.rgb[...,:3], gts[...,:3])
                
                out_rb = RenderBuffer(rgb=rb.rgb, depth=rb.depth, alpha=rb.alpha,
                                      gts=gts, err=(gts[..., :3] - rb.rgb[..., :3])**2)
                exrdict = out_rb.reshape(*img.shape[:2], -1).cpu().exr_dict()
                
                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                write_exr(os.path.join(self.valid_log_dir, out_name + ".exr"), exrdict)
                write_png(os.path.join(self.valid_log_dir, out_name + ".png"), rb.cpu().image().byte().rgb.numpy())

        psnr_total /= len(imgs)
        lpips_total /= len(imgs)  
        ssim_total /= len(imgs)
                
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
        log.info(log_text)
 
        return {"psnr" : psnr_total, "lpips": lpips_total, "ssim": ssim_total}

    def render_final_view(self, num_angles, camera_distance):
        angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
        x = -camera_distance * np.sin(angles)
        y = self.extra_args["camera_origin"][1]
        z = -camera_distance * np.cos(angles)
        for d in range(self.extra_args["num_lods"]):
            out_rgb = []
            for idx in tqdm(range(num_angles + 1), desc=f"Generating 360 Degree of View for LOD {d}"):
                log_metric_to_wandb(f"LOD-{d}-360-Degree-Scene/step", idx, step=idx)
                out = self.renderer.shade_images(
                    self.pipeline,
                    f=[x[idx], y, z[idx]],
                    t=self.extra_args["camera_lookat"],
                    fov=self.extra_args["camera_fov"],
                    lod_idx=d,
                    camera_clamp=self.extra_args["camera_clamp"]
                )
                out = out.image().byte().numpy_dict()
                if out.get('rgb') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGB", out['rgb'].T, idx)
                    out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
                if out.get('rgba') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/RGBA", out['rgba'].T, idx)
                if out.get('depth') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Depth", out['depth'].T, idx)
                if out.get('normal') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Normal", out['normal'].T, idx)
                if out.get('alpha') is not None:
                    log_images_to_wandb(f"LOD-{d}-360-Degree-Scene/Alpha", out['alpha'].T, idx)
                wandb.log({})
        
            rgb_gif = out_rgb[0]
            gif_path = os.path.join(self.log_dir, "rgb.gif")
            rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)
            wandb.log({f"360-Degree-Scene/RGB-Rendering/LOD-{d}": wandb.Video(gif_path)})
    
    def validate(self):
        self.pipeline.eval()

        # record_dict contains trainer args, but omits torch.Tensor fields which were not explicitly converted to
        # numpy or some other format. This is required as parquet doesn't support torch.Tensors
        # (and also for output size considerations)
        record_dict = {k: v for k, v in self.extra_args.items() if not isinstance(v, torch.Tensor)}
        dataset_name = os.path.splitext(os.path.basename(self.extra_args['dataset_path']))[0]
        model_fname = os.path.abspath(os.path.join(self.log_dir, f'model.pth'))
        record_dict.update({"dataset_name" : dataset_name, "epoch": self.epoch, 
                            "log_fname" : self.log_fname, "model_fname": model_fname})
        parent_log_dir = os.path.dirname(self.log_dir)

        log.info("Beginning validation...")

        validation_split = self.extra_args.get('valid_split', 'val')
        data = self.dataset.get_images(split=validation_split, mip=self.extra_args['mip'])
        imgs = list(data["imgs"])

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        lods = list(range(self.pipeline.nef.grid.num_lods))
        evaluation_results = self.evaluate_metrics(data["rays"], imgs, lods[-1], f"lod{lods[-1]}")
        record_dict.update(evaluation_results)
        if self.using_wandb:
            log_metric_to_wandb("Validation/psnr", evaluation_results['psnr'], self.epoch)
            log_metric_to_wandb("Validation/lpips", evaluation_results['lpips'], self.epoch)
            log_metric_to_wandb("Validation/ssim", evaluation_results['ssim'], self.epoch)
        
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
        if self.using_wandb:
            for d in range(self.extra_args["num_lods"]):
                wandb.define_metric(f"LOD-{d}-360-Degree-Scene")
                wandb.define_metric(
                    f"LOD-{d}-360-Degree-Scene",
                    step_metric=f"LOD-{d}-360-Degree-Scene/step"
                )

    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """
        wandb_viz_nerf_angles = self.extra_args.get("wandb_viz_nerf_angles", 0)
        wandb_viz_nerf_distance = self.extra_args.get("wandb_viz_nerf_distance")
        if self.using_wandb and wandb_viz_nerf_angles != 0:
            self.render_final_view(
                num_angles=wandb_viz_nerf_angles,
                camera_distance=wandb_viz_nerf_distance
            )
        super().post_training()
