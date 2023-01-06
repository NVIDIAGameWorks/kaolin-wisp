# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys
import numpy as np
import torch

import argparse
import glob
import logging as log


from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from torch.utils.data import DataLoader
from wisp.utils import PerfTimer
from wisp.datasets import SDFDataset
from wisp.ops.sdf import compute_sdf_iou
from wisp.ops.image import hwc_to_chw

import wandb


class SDFTrainer(BaseTrainer):

    def init_log_dict(self):
        """Custom logging dictionary.
        """
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0
        self.log_dict['l2_loss'] = 0

    def step(self, data):
        """Implement training from ground truth TSDF.
        """
        # Map to device
        pts = data[0].to(self.device)
        gts = data[1].to(self.device)
        if self.extra_args["sample_tex"]:
            rgb = data[2].to(self.device) 

        # Prepare for inference
        batch_size = pts.shape[0]
        self.pipeline.zero_grad()

        # Calculate loss
        loss = 0

        l2_loss = 0.0
        _l2_loss = 0.0
        rgb_loss = 0.0

        preds = []

        if self.extra_args["sample_tex"]:
            for lod_idx in self.loss_lods:
                preds.append(*self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels=["rgb", "sdf"]))
            for i, pred in zip(self.loss_lods, preds):
                _rgb_loss = ((pred[0] - rgb[...,:3])**2).sum()

                rgb_loss += _rgb_loss     

                res = 1.0
                _l2_loss = ((pred[1] - res * gts)**2).sum()
                l2_loss += _l2_loss
            
                loss += rgb_loss
                self.log_dict['rgb_loss'] += rgb_loss.item()
        else:
            for lod_idx in self.loss_lods:
                preds.append(*self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels=["sdf"]))
            for i, pred in zip(self.loss_lods, preds):
                res = 1.0
                _l2_loss = ((pred - res * gts)**2).sum()
                l2_loss += _l2_loss
        
        loss += l2_loss

        loss /= batch_size

        # Update logs
        self.log_dict['l2_loss'] += _l2_loss.item()
        self.log_dict['total_loss'] += loss.item() 

        # Backpropagate
        loss.backward()
        self.optimizer.step()

    def log_cli(self):
        """Override logging.
        """
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | l2 loss: {:>.3E}'.format(self.log_dict['l2_loss'] / len(self.train_data_loader))
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'] / len(self.train_data_loader))
        log.info(log_text)

    def render_tb(self):
        super().render_tb()

        self.pipeline.eval()
        for d in [self.pipeline.nef.grid.num_lods - 1]:
            if self.extra_args["log_2d"]:
                out_x = self.renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=0)
                out_y = self.renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=1)
                out_z = self.renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=2)
                self.writer.add_image(f'Cross-section/X/{d}', hwc_to_chw(out_x), self.epoch)
                self.writer.add_image(f'Cross-section/Y/{d}', hwc_to_chw(out_y), self.epoch)
                self.writer.add_image(f'Cross-section/Z/{d}', hwc_to_chw(out_z), self.epoch)
                if self.using_wandb:
                    log_images_to_wandb(f'Cross-section/X/{d}', hwc_to_chw(out_x), self.epoch)
                    log_images_to_wandb(f'Cross-section/Y/{d}', hwc_to_chw(out_y), self.epoch)
                    log_images_to_wandb(f'Cross-section/Z/{d}', hwc_to_chw(out_z), self.epoch)

    def validate(self):
        """Implement validation. Just computes IOU.
        """
            
        # Same as training since we're overfitting
        metric_name = None
        if self.dataset.initialization_mode == "mesh":
            metric_name = "volumetric_iou"
        elif self.dataset.initialization_mode == "grid":
            metric_name = "narrowband_iou"
        else:
            raise NotImplementedError
        
        val_dict = {}
        val_dict[metric_name] = []
    
        # Uniform points metrics
        for n_iter, data in enumerate(self.train_data_loader):

            pts = data[0].to(self.device)
            gts = data[1].to(self.device)
            nrm = data[2].to(self.device) if self.extra_args["get_normals"] else None
                

            for lod_idx in self.loss_lods:
                # TODO(ttakkawa): Currently the SDF metrics computed for sparse grid-based SDFs are not entirely proper
                pred = self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels="sdf")
                val_dict[metric_name] += [float(compute_sdf_iou(pred, gts))]

        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)

        for k, v in val_dict.items():
            score_total = 0.0
            for lod, score in zip(self.loss_lods, v):
                self.writer.add_scalar(f'Validation/{k}/{lod}', score, self.epoch)
                score_total += score
            log_text += ' | {}: {:.4f}'.format(k, score_total / len(v))
        log.info(log_text)
