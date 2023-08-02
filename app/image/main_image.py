# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import logging
import dataclasses
import torch
from typing import Optional
from wisp.app_utils import default_log_setup
from wisp.config import parse_config, configure, autoconfig, instantiate, print_config
from wisp.models.pipeline import Pipeline
from wisp.models.grids import HashGrid
from wisp.trainers import ImageTrainer, ConfigBaseTrainer
from wisp.models.nefs import ImageNeuralField
from wisp.datasets import ImageDataset
from wisp.trainers.tracker import Tracker, ConfigTracker

@configure
class ImageAppConfig:
    
    nef: autoconfig(ImageNeuralField)
    """ Neural field configuration, including the feature grid, decoders and optional embedders. """
    grid: autoconfig(HashGrid.from_geometric)
    """ Feature grid used by the neural field. Grids are located in `wisp.models.grids` """
    dataset: autoconfig(ImageDataset)
    """ Image dataset in use. """
    trainer: ConfigBaseTrainer
    """ Trainer config. """
    tracker: ConfigTracker
    """ Experiments tracker for reporting to tensorboard & wandb, creating visualizations and aggregating metrics. """
    scaling_factor : float = 2.0
    """ The max resolution of the grid will be set to the max resolution of the image 
    divided by the scaling factor. 
    """
    valid_only : bool = False
    """ Only run validation. """
    log_level : int = logging.INFO
    """ Sets the global output log level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL """
    profile : bool = False
    """ Enable profiling if True. """
    detect_anomaly : bool =  False
    """ Enable anomaly detection if True. """
    device: str = 'cuda'
    """ Device used to run the optimization """
    pretrained: Optional[str] = None
    """ If specified, a pretrained model will be loaded from this path. None will create a new model. """

cfg = parse_config(ImageAppConfig, yaml_arg='--config')  # Obtain args by priority: cli args > config yaml > config defaults
device = torch.device(cfg.device)
default_log_setup(cfg.log_level)

train_dataset = instantiate(cfg.dataset)

# Set the max resolution of the grid to the max of the image resolution divided by scaling factor
# (follows the procedure from Instant NGP, check the paper for details)
res = int(max(train_dataset.h, train_dataset.w) // cfg.scaling_factor)
setattr(cfg.grid, 'max_grid_res', res)
grid = instantiate(cfg.grid, blas=None)

if cfg.pretrained and cfg.trainer.model_format == "full":
    pipeline = torch.load(cfg.pretrained)   # Load a full pretrained pipeline: model + weights
else:
    nef = instantiate(cfg.nef, grid=grid)
    pipeline = Pipeline(nef=nef)
    if cfg.pretrained and cfg.trainer.model_format == "state_dict":
        pipeline.load_state_dict(torch.load(cfg.pretrained))

print_config(cfg)

exp_name: str = cfg.trainer.exp_name
tracker = Tracker(cfg=cfg.tracker, exp_name=exp_name)
tracker.save_app_config(cfg)
trainer = ImageTrainer(cfg=cfg.trainer,
                       pipeline=pipeline,
                       train_dataset=train_dataset,
                       tracker=tracker,
                       device=device)

if cfg.valid_only:
    trainer.validate()
else:
    if cfg.profile:
        import torch
        with torch.autograd.profiler.emit_nvtx():
            trainer.train()
    else:
        trainer.train()
