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
from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays, RenderBuffer

class MultiviewTrainer(BaseTrainer):

    def pre_epoch(self, epoch):
        """Override pre_epoch to support pruning.
        """
        super().pre_epoch(epoch)   
        
        if self.extra_args["prune_every"] > -1 and epoch > 0 and epoch % self.extra_args["prune_every"] == 0:
            self.pipeline.nef.prune()
            self.init_optimizer()

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['image_count'] = 0

    def step(self, epoch, n_iter, data):
        """Implement the optimization over image-space loss.
        """
        self.scene_state.optimization.iteration = n_iter

        timer = PerfTimer(activate=False, show_memory=False)

        # Map to device
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        timer.check("map to device")

        self.optimizer.zero_grad(set_to_none=True)
        
        timer.check("zero grad")
            
        loss = 0
        
        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.pipeline.nef.num_lods)]
            weights = [2**i for i in range(self.pipeline.nef.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays, lod_idx=lod_idx, channels=["rgb"])
            timer.check("inference")

            # RGB Loss
            #rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            
            rgb_loss = rgb_loss.mean()
            loss += self.extra_args["rgb_loss"] * rgb_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()
            timer.check("loss")

        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += 1
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        timer.check("backward and step")
        
    def log_tb(self, epoch):
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])
        
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'Loss/{key}', self.log_dict[key], epoch)

        log.info(log_text)

        self.pipeline.eval()

    def evaluate_metrics(self, epoch, rays, imgs, lod_idx, name=None):
        
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
                
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
        log.info(log_text)
 
        return {"psnr" : psnr_total, "lpips": lpips_total, "ssim": ssim_total}

    def validate(self, epoch=0):
        self.pipeline.eval()

        record_dict = self.extra_args
        dataset_name = os.path.splitext(os.path.basename(self.extra_args['dataset_path']))[0]
        model_fname = os.path.abspath(os.path.join(self.log_dir, f'model.pth'))
        record_dict.update({"dataset_name" : dataset_name, "epoch": epoch, 
                            "log_fname" : self.log_fname, "model_fname": model_fname})
        parent_log_dir = os.path.dirname(self.log_dir)

        log.info("Beginning validation...")
        
        data = self.dataset.get_images(split=self.extra_args['valid_split'], mip=self.extra_args['mip'])
        imgs = list(data["imgs"])

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")
        log.info(f"Saving validation result to {self.valid_log_dir}")
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)

        lods = list(range(self.pipeline.nef.num_lods))
        record_dict.update(self.evaluate_metrics(epoch, data["rays"], imgs, lods[-1], f"lod{lods[-1]}"))
        
        df = pd.DataFrame.from_records([record_dict])
        df['lod'] = lods[-1]
        fname = os.path.join(parent_log_dir, f"logs.parquet")
        if os.path.exists(fname):
            df_ = pd.read_parquet(fname)
            df = pd.concat([df_, df])
        df.to_parquet(fname, index=False)
