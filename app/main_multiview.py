# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from cuda_guard import setup_cuda_context
setup_cuda_context()  # Must be called before any torch operations take place

import argparse
import logging
import os
import torch

import app_utils

from wisp.models.grids import OctreeGrid, HashGrid
from wisp.trainers import MultiviewTrainer
from wisp.datasets import MultiviewDataset
from wisp.datasets.transforms import SampleRays
from wisp.config_parser import *
from wisp.framework import WispState

def parse_multiview_options():
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train a NeRF.')
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')
    trainer_group = add_trainer_argument_group(parser)
    trainer_group.add_argument('--prune-every', type=int, default=-1,
                               help='Prune every N epochs')
    trainer_group.add_argument('--random-lod', action='store_true',
                               help='Use random lods to train.')
    add_grid_argument_group(parser)
    embedder_group = add_embedder_argument_group(parser)
    embedder_group.add_argument('--view-multires', type=int, default=4,
                                help='log2 of max freq')
    add_net_argument_group(parser)
    data_group = add_dataset_argument_group(parser)
    data_group.add_argument('--multiview-dataset-format', default='standard',
                            choices=['standard', 'rtmv'],
                            help='Data format for the transforms')
    data_group.add_argument('--num-rays-sampled-per-img', type=int, default='4096',
                            help='Number of rays to sample per image')
    data_group.add_argument('--mip', type=int, default=None, 
                            help='MIP level of ground truth image')
    add_logging_argument_group(parser)
    add_optimizer_argument_group(parser)
    add_renderer_argument_group(parser)

    return parser

# Usual boilerplate
parser = parse_multiview_options()
app_utils.add_log_level_flag(parser)
app_group = parser.add_argument_group('app')
# Add custom args if needed for app
args, args_str = argparse_to_str(parser)


app_utils.default_log_setup(args.log_level)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = SampleRays(args.num_rays_sampled_per_img)
train_dataset = MultiviewDataset(args.dataset_path, args.multiview_dataset_format,
                                 args.mip, args.bg_color, args.dataset_num_workers,
                                 transform=transform)
train_dataset.init()

pipeline = get_pipeline_from_config(
    'NeuralRadianceField',
    # TODO: condense this into args dict
    dict(
        grid_type=args.grid_type,
        interpolation_type=args.interpolation_type,
        multiscale_type=args.multiscale_type,
        embedder_type=args.embedder_type,
        activation_type=args.activation_type,
        layer_type=args.layer_type,
        base_lod=args.base_lod,
        num_lods=args.num_lods,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        pos_multires=args.pos_multires,
        view_multires=args.view_multires, # nerf specific
        num_layers=args.num_layers,
        position_input=args.position_input,
        codebook_bitwidth=args.codebook_bitwidth
    ),
    'PackedRFTracer',
    dict(
        raymarch_type=args.raymarch_type,
        num_steps=args.num_steps,
        step_size=args.step_size,
        bg_color=args.bg_color
    ), args.pretrained, args.model_format)

# Initialization of the grid using the dataset
if pipeline.nef.grid is not None:
    if isinstance(pipeline.nef.grid, OctreeGrid):
        if not args.valid_only and not pipeline.nef.grid.blas_initialized():
            if args.multiview_dataset_format in ['rtmv']: # TODO(ttakikawa): the semantic is not good shouldn't be linked to rtmv specifically
                pipeline.nef.grid.init_from_pointcloud(train_dataset.coords)
            else:
                pipeline.nef.grid.init_dense()
            pipeline.to(device)
    if isinstance(pipeline.nef.grid, HashGrid):
        if not args.valid_only:
            if args.tree_type == 'quad':
                # TODO(operel): why not simply do that at ctor?
                pipeline.nef.grid.init_from_octree(args.base_lod, args.num_lods)
            elif args.tree_type == 'geometric':
                # TODO(operel): why not simply do that at ctor?
                pipeline.nef.grid.init_from_geometric(16, args.max_grid_res, args.num_lods)
            else:
                raise NotImplementedError
            pipeline.to(device)

optim_cls, optim_params = get_optimizer_from_config(args.optimizer_type)

scene_state = WispState()
trainer = MultiviewTrainer(pipeline, train_dataset, args.epochs, args.batch_size,
                           optim_cls, args.lr, args.weight_decay, args.grid_lr_weight,
                           optim_params, args.log_dir, device, exp_name=args.exp_name,
                           info=args_str, extra_args=vars(args),
                           render_every=args.render_every,
                           save_every=args.save_every, scene_state=scene_state)
        
if os.environ.get('WISP_HEADLESS') == '1':
    logging.info("Running headless. For the app, set WISP_HEADLESS=0")
    if args.valid_only:
        trainer.validate()
    else:
        trainer.train()
else:
    from wisp.renderer.app.optimization_app import OptimizationApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
    renderer = OptimizationApp(wisp_state=scene_state,
                               trainer_step_func=trainer.iterate,
                               experiment_name="wisp trainer")
    renderer.run()
