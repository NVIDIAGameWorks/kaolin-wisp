# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging as log
import torch
import wisp
from typing import Optional
from wisp.trainers import BaseTrainer, ConfigBaseTrainer
from wisp.trainers.tracker import Tracker
from wisp.datasets import MeshSampledSDFDataset, OctreeSampledSDFDataset, SDFBatch
from wisp.ops.sdf import compute_sdf_iou
from wisp.ops.image import hwc_to_chw
from wisp.config import configure


@configure
class ConfigSDFTrainer(ConfigBaseTrainer):
    log_2d : bool = False
    """ Log 2D slices of the SDF to the loggers. """

    only_last : bool = True
    """ Train only the last LOD. """

    resample: bool = False
    """ Should resample dataset at the end of each epoch. May add diversity but slow down training. """


class SDFTrainer(BaseTrainer):

    def __init__(self,
                 cfg: ConfigSDFTrainer,
                 pipeline: wisp.models.Pipeline,
                 train_dataset: wisp.datasets.WispDataset,
                 tracker: Tracker,
                 device: torch.device = 'cuda',
                 scene_state: Optional[wisp.framework.WispState] = None):
        super().__init__(cfg=cfg, pipeline=pipeline, train_dataset=train_dataset, tracker=tracker, device=device,
                         scene_state=scene_state)

    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        super().pre_training()
        self.tracker.metrics.define_metric('rgb_loss', aggregation_type=float)
        self.tracker.metrics.define_metric('l2_loss', aggregation_type=float)

    def pre_epoch(self):
        super().pre_epoch()
        self.loss_lods = [i for i in range(self.pipeline.nef.grid.num_lods)]
        if self.cfg.only_last:
            self.loss_lods = self.loss_lods[-1:]

    def post_epoch(self):
        super().post_epoch()
        if self.cfg.resample:
            self.resample_dataset()

    def step(self, data):
        """Implement training from ground truth TSDF.
        """
        # Map to device
        pts = data['coords'].to(self.device)
        gts = data['sdf'].to(self.device)
        sample_tex = hasattr(self.train_dataset, 'sample_tex') and self.train_dataset.sample_tex
        if sample_tex:
            rgb = data['rgb'].to(self.device)

        # Prepare for inference
        batch_size = pts.shape[0]
        self.pipeline.zero_grad()

        # Calculate loss
        loss = 0

        l2_loss = 0.0
        _l2_loss = 0.0
        rgb_loss = 0.0
        _rgb_loss = None

        preds = []

        if sample_tex:
            for lod_idx in self.loss_lods:
                preds.append(*self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels=["rgb", "sdf"]))
            for i, pred in zip(self.loss_lods, preds):
                _rgb_loss = ((pred[0] - rgb[...,:3])**2).sum()

                rgb_loss += _rgb_loss     

                res = 1.0
                _l2_loss = ((pred[1] - res * gts)**2).sum()
                l2_loss += _l2_loss
            
                loss += rgb_loss
        else:
            for lod_idx in self.loss_lods:
                preds.append(*self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels=["sdf"]))
            for i, pred in zip(self.loss_lods, preds):
                res = 1.0
                _l2_loss = ((pred - res * gts)**2).sum()
                l2_loss += _l2_loss
        
        loss += l2_loss


        # Aggregate losses
        self.tracker.metrics.total_loss += loss.item()
        self.tracker.metrics.l2_loss += _l2_loss.item()
        if _rgb_loss is not None:
            self.tracker.metrics.rgb_loss += _rgb_loss.item()
        self.tracker.metrics.num_samples += batch_size

        # Backpropagate
        with torch.cuda.nvtx.range("SDFTrainer.backward"):
            loss /= batch_size
            loss.backward()
            self.optimizer.step()

    def log_console(self):
        """Override logging.
        """
        total_loss = self.tracker.metrics.average_metric('total_loss')
        l2_loss = self.tracker.metrics.average_metric('l2_loss')
        rgb_loss = self.tracker.metrics.average_metric('rgb_loss')
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log_text += ' | l2 loss: {:>.3E}'.format(l2_loss)
        log_text += ' | rgb loss: {:>.3E}'.format(rgb_loss)
        log.info(log_text)

    def render_snapshot(self):
        super().render_snapshot()

        self.pipeline.eval()
        for d in [self.pipeline.nef.grid.num_lods - 1]:
            if self.cfg.log_2d:
                renderer = self.tracker.visualizer
                out_x = renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=0)
                out_y = renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=1)
                out_z = renderer.sdf_slice(self.pipeline.nef.get_forward_function("sdf"), dim=2)
                out_x = torch.FloatTensor(out_x)
                out_y = torch.FloatTensor(out_y)
                out_z = torch.FloatTensor(out_z)

                self.tracker.log_image(f'Cross-section/X/{d}', hwc_to_chw(out_x), self.epoch)
                self.tracker.log_image(f'Cross-section/Y/{d}', hwc_to_chw(out_y), self.epoch)
                self.tracker.log_image(f'Cross-section/Z/{d}', hwc_to_chw(out_z), self.epoch)

    def validate(self):
        """Implement validation. Just computes IOU.
        """
        # Same as training since we're overfitting
        if isinstance(self.train_dataset, MeshSampledSDFDataset):
            metric_name = "volumetric_iou"
        elif isinstance(self.train_dataset, OctreeSampledSDFDataset):
            metric_name = "narrowband_iou"
        else:
            raise NotImplementedError
        
        val_dict = {}
        val_dict[metric_name] = []
    
        # Uniform points metrics
        for n_iter, data in enumerate(self.train_data_loader):

            pts = data['coords'].to(self.device)
            gts = data['sdf'].to(self.device)
            nrm = data['normals'].to(self.device) if data.get('normals') is not None else None

            for lod_idx in self.loss_lods:
                # TODO(ttakkawa): Currently the SDF metrics computed for sparse grid-based SDFs are not entirely proper
                pred = self.pipeline.nef(coords=pts, lod_idx=lod_idx, channels="sdf")
                val_dict[metric_name] += [float(compute_sdf_iou(pred, gts))]

        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)

        for k, v in val_dict.items():
            score_total = 0.0
            for lod, score in zip(self.loss_lods, v):
                self.tracker.log_metric(f'Validation/{k}/{lod}', score, self.epoch)
                score_total += score
            log_text += ' | {}: {:.4f}'.format(k, score_total / len(v))
        log.info(log_text)
