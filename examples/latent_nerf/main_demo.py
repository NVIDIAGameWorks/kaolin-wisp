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
from wisp.app_utils import default_log_setup
from wisp.config import parse_config, configure, autoconfig, instantiate, print_config
from wisp.framework import WispState
from wisp.datasets import NeRFSyntheticDataset, SampleRays
from wisp.trainers import MultiviewTrainer, ConfigMultiviewTrainer
from wisp.trainers.tracker import Tracker, ConfigTracker
from wisp.accelstructs import OctreeAS
from wisp.models.grids import OctreeGrid, HashGrid
from wisp.models.pipeline import Pipeline
from wisp.tracers import PackedRFTracer
from funny_neural_field import FunnyNeuralField


@configure  # configure with no params is a dataclass that holds various config groups
class DemoAppConfig:
    """ An example app over kaolin-wisp to visualize internal layers of the NeRF decoder."""

    # autoconfig() scans the constructor or class and automatically generates a configuration dataclass!
    # then the config options match the function or constructor args
    blas: autoconfig(OctreeAS.make_dense)
    """ Bottom Level Acceleration structure used by the neural field grid to track occupancy, accelerate raymarch.
        This demo always uses initializes a fully occupied octree (the hashgrid may further prune it). 
    """
    grid: autoconfig(OctreeGrid, HashGrid.from_geometric)
    """ Feature grid used by the neural field. Grids are located in `wisp.models.grids` """
    nef: autoconfig(FunnyNeuralField)
    """ Our custom neural field configuration, including the feature grid, decoders and optional embedders.
    FunnyNeuralField maps 3D coordinates (+ 2D view direction) -> RGB + density.
    Uses spatial feature grids internally for faster feature interpolation and raymarching.
    It also exposes the latent information from the decoder.
    """
    tracer: autoconfig(PackedRFTracer)
    """ Tracers are responsible for taking input rays, marching them through the neural field to render 
    an output RenderBuffer. PackedRFTracer is differentiable and used for optimizing and tracing neural fields. 
    """
    dataset: autoconfig(NeRFSyntheticDataset)
    """ Multiview dataset used by the trainer. Here we assume the nerf synthetic dataset format. """
    dataset_transform: autoconfig(SampleRays)
    """ Composition of dataset transforms used online by the dataset to process batches. """
    trainer: ConfigMultiviewTrainer  # we manually define the trainer config, rather than auto generating it.
    """ Configuration for trainer used to optimize the neural field.
    If you create your own trainer, make sure to change the configuration accordingly!
    """
    tracker: ConfigTracker
    """ Experiments tracker for reporting to tensorboard & wandb, creating visualizations and aggregating metrics. """
    log_level: int = logging.INFO
    """ Sets the global output log level: 
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL 
    """
    device: str = 'cuda'
    """ Device used to run the optimization """
    interactive: bool = os.environ.get('WISP_HEADLESS') != '1'
    """ Set to --interactive=True for interactive mode which uses the GUI.
    The default value is set according to the env variable WISP_HEADLESS, if available. 
    Otherwise, interactive mode is on by default. """


# Obtain args by priority: cli args > config yaml > config defaults
cfg = parse_config(DemoAppConfig, yaml_arg='--config')
device = torch.device(cfg.device)
default_log_setup(cfg.log_level)
if cfg.interactive:
    cfg.tracer.bg_color = (0.0, 0.0, 0.0)
    cfg.trainer.render_every = -1
    cfg.trainer.save_every = -1
    cfg.trainer.valid_every = -1
print_config(cfg)

# Here, NeRF is trained with a multiview NeRFSyntheticDataset,
# which knows how to generate RGB rays from a set of images + cameras
# The dataset uses a single transform: SampleRays, which creates batches of rays from the dataset
train_dataset = instantiate(cfg.dataset, transform=instantiate(cfg.dataset_transform))  # A Multiview dataset
validation_dataset = None
if cfg.trainer.valid_every > -1 or cfg.trainer.mode == 'validate':
    validation_dataset = train_dataset.create_split(split='val', transform=None)

# To build our neural field (FunnyNeuralField), we'll need a feature grid and an occupancy structure.
# For grid, a fully occupied octree isn't the fastest option in wisp, but it's a straightforward example for this demo.
# The hash is a faster alternative, try to experiment with both in the config above!
blas = instantiate(cfg.blas)             # blas is the occupancy acceleration structure, for speeding up ray tracing
grid = instantiate(cfg.grid, blas=blas)  # A grid keeps track of both features and occupancy
nerf = instantiate(cfg.nef, grid=grid)    # NeRF uses a grid as the backbone
# tracer is used to generate and integrate samples for the neural field.
# Wisp's implementation of NeRF uses the PackedRFTracer to trace the neural field:
# - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
#    see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
# - RF: Radiance Field
tracer = instantiate(cfg.tracer)
pipeline = Pipeline(nef=nerf, tracer=tracer)  # Binds neural field and tracer together to a single NeRF callable
pipeline = pipeline.to(device)

# Joint trainer / app state, to allow the trainer and gui app to share updates
exp_name: str = cfg.trainer.exp_name
scene_state: WispState = WispState()
tracker = Tracker(cfg=cfg.tracker, exp_name=exp_name)
trainer = MultiviewTrainer(cfg=cfg.trainer,
                           pipeline=pipeline,
                           train_dataset=train_dataset,
                           validation_dataset=validation_dataset,
                           tracker=tracker,
                           device=device,
                           scene_state=scene_state)


# The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
# - Headless, which will run the train() function until all training steps are exhausted.
# - Interactive mode, which uses the gui. In this case, an DemoApp uses events to prompt the trainer to
#   take training steps, while also taking care to render output to users (see: iterate()).
#   In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
if not cfg.interactive:
    logging.info("Running headless. For the app, set --interactive=True or $WISP_HEADLESS=0.")
    if cfg.trainer.mode == 'validate':
        trainer.validate()
    elif cfg.trainer.mode == 'train':
        trainer.train()  # Run in headless mode
else:
    from demo_app import DemoApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
    app = DemoApp(wisp_state=scene_state, background_task=trainer.iterate, window_name="SIGGRAPH 2022 Demo")
    app.run()  # Run in interactive mode
