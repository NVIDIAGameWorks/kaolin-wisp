# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import logging
import torch
from typing import Optional
from wisp.app_utils import default_log_setup
from wisp.config import parse_config, configure, autoconfig, instantiate, print_config
from wisp.framework import WispState
from wisp.accelstructs import OctreeAS, AxisAlignedBBoxAS
from wisp.models.grids import OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.models.nefs import NeuralRadianceField
from wisp.models.pipeline import Pipeline
from wisp.tracers import PackedRFTracer
from wisp.datasets import NeRFSyntheticDataset, RTMVDataset, SampleRays
from wisp.trainers import MultiviewTrainer, ConfigMultiviewTrainer
from wisp.trainers.tracker import Tracker, ConfigTracker


@configure
class NeRFAppConfig:
    """ A script for training simple NeRF variants with grid backbones."""

    blas: autoconfig(OctreeAS.make_dense, OctreeAS.from_pointcloud, AxisAlignedBBoxAS)
    """ Bottom Level Acceleration structure used by the neural field grid to track occupancy, accelerate raymarch. """
    grid: autoconfig(OctreeGrid, HashGrid.from_geometric, TriplanarGrid, CodebookOctreeGrid)
    """ Feature grid used by the neural field. Grids are located in `wisp.models.grids` """
    nef: autoconfig(NeuralRadianceField)
    """ Neural field configuration, including the feature grid, decoders and optional embedders.
    NeuralRadianceField maps 3D coordinates (+ 2D view direction) -> RGB + density.
    Uses spatial feature grids internally for faster feature interpolation and raymarching.
    """
    tracer: autoconfig(PackedRFTracer)
    """ Tracers are responsible for taking input rays, marching them through the neural field to render 
    an output RenderBuffer.
    """
    dataset: autoconfig(NeRFSyntheticDataset, RTMVDataset)
    """ Multiview dataset used by the trainer. """
    dataset_transform: autoconfig(SampleRays)
    """ Composition of dataset transforms used online by the dataset to process batches. """
    trainer: ConfigMultiviewTrainer
    """ Configuration for trainer used to optimize the neural field. """
    tracker: ConfigTracker
    """ Experiments tracker for reporting to tensorboard & wandb, creating visualizations and aggregating metrics. """
    log_level: int = logging.INFO
    """ Sets the global output log level: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL """
    pretrained: Optional[str] = None
    """ If specified, a pretrained model will be loaded from this path. None will create a new model. """
    device: str = 'cuda'
    """ Device used to run the optimization """
    interactive: bool = os.environ.get('WISP_HEADLESS') != '1'
    """ Set to --interactive=True for interactive mode which uses the GUI.
    The default value is set according to the env variable WISP_HEADLESS, if available. 
    Otherwise, interactive mode is on by default. """


cfg = parse_config(NeRFAppConfig, yaml_arg='--config')  # Obtain args by priority: cli args > config yaml > config defaults
device = torch.device(cfg.device)
default_log_setup(cfg.log_level)
if cfg.interactive:
    cfg.tracer.bg_color = (0.0, 0.0, 0.0)
    cfg.trainer.render_every = -1
    cfg.trainer.save_every = -1
    cfg.trainer.valid_every = -1
print_config(cfg)

# Loads a multiview dataset comprising of pairs of images and calibrated cameras:
# NeRFSyntheticDataset - refers to the standard NeRF format popularized by Mildenhall et al. 2020,
#   including additions to the metadata format added by Muller et al. 2022.
# 'rtmv' - refers to the dataset published by Tremblay et. al 2022,
# RTMVDataset - A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis",
#   this dataset includes depth information which allows for performance improving optimizations in some cases.
dataset_transform = instantiate(cfg.dataset_transform)  # SampleRays creates batches of rays from the dataset
train_dataset = instantiate(cfg.dataset, transform=dataset_transform)  # A Multiview dataset
validation_dataset = None
if cfg.trainer.valid_every > -1 or cfg.trainer.mode == 'validate':
    validation_dataset = train_dataset.create_split(split=cfg.trainer.valid_split, transform=None)

if cfg.pretrained and cfg.trainer.model_format == "full":
    pipeline = torch.load(cfg.pretrained)   # Load a full pretrained pipeline: model + weights
else:  # Create model from scratch
    # Optimization: if dataset contains depth info, initialize only cells known to be occupied
    pointcloud = train_dataset.as_pointcloud() if train_dataset.supports_depth() else None
    # blas is the occupancy acceleration structure, for speeding up ray tracing
    blas = instantiate(cfg.blas, pointcloud=pointcloud)
    grid = instantiate(cfg.grid, blas=blas)  # A grid keeps track of both features and occupancy
    nef = instantiate(cfg.nef, grid=grid)    # NeRF uses a grid as the backbone
    # tracer is used to generate and integrate samples for the neural field.
    # Wisp's implementation of NeRF uses the PackedRFTracer to trace the neural field:
    # - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
    #    see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    # - RF: Radiance Field
    tracer = instantiate(cfg.tracer)
    pipeline = Pipeline(nef=nef, tracer=tracer)  # Binds neural field and tracer together to a single NeRF callable
    if cfg.pretrained and cfg.trainer.model_format == "state_dict":
        pipeline.load_state_dict(torch.load(cfg.pretrained))

# Joint trainer / app state - scene_state contains various global definitions
exp_name: str = cfg.trainer.exp_name
scene_state: WispState = WispState()
tracker = Tracker(cfg=cfg.tracker, exp_name=exp_name)
tracker.save_app_config(cfg)
trainer = MultiviewTrainer(cfg=cfg.trainer,
                           pipeline=pipeline,
                           train_dataset=train_dataset,
                           validation_dataset=validation_dataset,
                           tracker=tracker,
                           device=device,
                           scene_state=scene_state)

# The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
# - Headless, which will run the train() function until all training steps are exhausted.
# - Interactive mode, which uses the gui. In this case, an OptimizationApp uses events to prompt the trainer to
#   take training steps, while also taking care to render output to users (see: iterate()).
#   In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
if not cfg.interactive:
    logging.info("Running headless. For the app, set --interactive=True or $WISP_HEADLESS=0.")
    if cfg.trainer.mode == 'validate':
        trainer.validate()
    elif cfg.trainer.mode == 'train':
        trainer.train()  # Run in headless mode
else:
    from wisp.renderer.app.optimization_app import OptimizationApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
    app = OptimizationApp(wisp_state=scene_state, trainer_step_func=trainer.iterate, experiment_name=exp_name)
    app.run()  # Run in interactive mode
