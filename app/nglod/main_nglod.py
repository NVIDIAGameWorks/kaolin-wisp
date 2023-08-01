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
from wisp.config import parse_config, configure, autoconfig, instantiate, print_config, get_config_target
from wisp.framework import WispState
from wisp.datasets import SDFDataset, MeshSampledSDFDataset, OctreeSampledSDFDataset
from wisp.accelstructs import OctreeAS, AxisAlignedBBoxAS
from wisp.models.grids import OctreeGrid, TriplanarGrid, HashGrid
from wisp.models.nefs import NeuralSDF
from wisp.models.pipeline import Pipeline
from wisp.tracers import PackedSDFTracer
from wisp.trainers import SDFTrainer, ConfigSDFTrainer
from wisp.trainers.tracker import Tracker, ConfigTracker


@configure
class SDFAppConfig:
    """ A script for training neural SDF variants with grid backbones.
    See: Takikawa et al. 2021 - "Neural Geometric Level of Detail: Real-time Rendering with Implicit 3D Shapes".
    """
    blas: autoconfig(OctreeAS.from_mesh, AxisAlignedBBoxAS)
    """ Bottom Level Acceleration structure used by the neural field grid to track occupancy, accelerate queries. """
    grid: autoconfig(OctreeGrid, HashGrid.from_geometric, HashGrid.from_octree, TriplanarGrid)
    """ Feature grid used by the neural field. Grids are located in `wisp.models.grids` """
    nef: autoconfig(NeuralSDF)
    """ Signed distance field configuration, including the feature grid, a decoder and optional embedder.
    NeuralSDF maps 3D coordinates -> SDF values. Uses spatial feature grids internally for faster feature interpolation.
    """
    tracer: autoconfig(PackedSDFTracer)
    """ Tracers are responsible for taking input rays, marching them through the neural field to render 
    an output RenderBuffer. In this app, the tracer is only used for rendering during test time.
    """
    dataset: autoconfig(MeshSampledSDFDataset, OctreeSampledSDFDataset)
    """ SDF dataset used by the trainer. """
    trainer: ConfigSDFTrainer
    """ Configuration for trainer used to optimize the neural sdf. """
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


cfg = parse_config(SDFAppConfig, yaml_arg='--config')  # Obtain args by priority: cli args > config yaml > config defaults
device = torch.device(cfg.device)
default_log_setup(cfg.log_level)
if cfg.interactive:
    cfg.tracer.bg_color = (0.0, 0.0, 0.0)
    cfg.trainer.render_every = -1
    cfg.trainer.save_every = -1
    cfg.trainer.valid_every = -1
print_config(cfg)

if cfg.pretrained and cfg.trainer.model_format == "full":
    pipeline = torch.load(cfg.pretrained)   # Load a full pretrained pipeline: model + weights
else:  # Create model from scratch
    # blas is the occupancy acceleration structure, possibly initialized sparsely from a mesh
    blas = instantiate(cfg.blas)
    grid = instantiate(cfg.grid, blas=blas)  # A grid keeps track of both features and occupancy
    nef = instantiate(cfg.nef, grid=grid)    # nef here is a SDF which uses a grid as the backbone
    # tracer here is used to efficiently render the SDF at test time (note: not used during optimization).
    # Wisp's implementation of Neural Geometric LOD uses PackedSDFTracer to trace the neural field:
    # - Packed: each ray yields a custom number of sphere tracing steps,
    #   which are therefore packed in a flat form within a tensor,
    #   see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    # - SDF: Signed Distance Function
    tracer = instantiate(cfg.tracer)
    pipeline = Pipeline(nef=nef, tracer=tracer)  # Binds neural field and tracer together to a single SDF callable
    if cfg.pretrained and cfg.trainer.model_format == "state_dict":
        pipeline.load_state_dict(torch.load(cfg.pretrained))

# Loads a SDF dataset comprising of sdf samples generated from an existing mesh.
# MeshSampledSDFDataset - refers to samples generated directly from the mesh surface. This dataset is decoupled from
#   the optimized model and it's blas (occupancy structure).
# OctreeSampledSDFDataset - refers to samples generated from an octree, initialized from a mesh.
#   This dataset has the benefit of limiting the sampling region to areas which are actually occupied by the mesh.
#   It also allows for equal distribution of samples per the octree cells.
if get_config_target(cfg.dataset) is OctreeSampledSDFDataset:
    assert OctreeSampledSDFDataset.supports_blas(pipeline.nef.grid.blas)
train_dataset: SDFDataset = instantiate(cfg.dataset, occupancy_struct=blas)

# Joint trainer / app state - scene_state contains various global definitions
exp_name: str = cfg.trainer.exp_name
scene_state: WispState = WispState()
tracker = Tracker(cfg=cfg.tracker, exp_name=exp_name)
trainer = SDFTrainer(cfg=cfg.trainer,
                     pipeline=pipeline,
                     train_dataset=train_dataset,
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
