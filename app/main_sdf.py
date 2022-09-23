# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from cuda_guard import setup_cuda_context
setup_cuda_context()     # Must be called before any torch operations take place

import argparse
import logging
import os
import torch

import app_utils

from wisp.models.grids import OctreeGrid
from wisp.trainers import SDFTrainer
from wisp.datasets import SDFDataset
from wisp.datasets.transforms import SampleRays
from wisp.config_parser import *
from wisp.framework import WispState

def parse_sdf_options():
    parser = argparse.ArgumentParser(description='Train an SDF.')
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')
    trainer_group = add_trainer_argument_group(parser)
    trainer_group.add_argument('--log-2d', action='store_true',
                               help='Log cutting plane renders to TensorBoard.')
    add_grid_argument_group(parser)
    add_embedder_argument_group(parser)
    net_group = add_net_argument_group(parser)
    net_group.add_argument('--nef-type', type=str, default='NeuralSDF',
                           choices=list(str2nef.keys()),
                           help='The neural field class to be used.')
    data_group = add_dataset_argument_group(parser)
    data_group.add_argument('--sample-mode', type=str, nargs='*', 
                            default=['rand', 'near', 'near', 'trace', 'trace'],
                            help='The sampling scheme to be used.')
    data_group.add_argument('--get-normals', action='store_true',
                            help='Sample the normals.')
    data_group.add_argument('--num-samples', type=int, default=100000,
                            help='Number of samples per mode (or per epoch for SPC)')
    data_group.add_argument('--num-samples-on-mesh', type=int, default=100000000,
                            help='Number of samples generated on mesh surface to initialize occupancy structures')
    data_group.add_argument('--sample-tex', action='store_true',
                            help='Sample textures')
    data_group.add_argument('--mode-mesh-norm', type=str, default='sphere',
                            choices=['sphere', 'aabb', 'planar', 'none'],
                            help='Normalize the mesh')
    data_group.add_argument('--samples-per-voxel', type=int, default=256,
                            help='Number of samples per voxel (for SDF initialization from grid)')
    add_logging_argument_group(parser)
    add_optimizer_argument_group(parser)
    renderer_group = add_renderer_argument_group(parser)
    renderer_group.add_argument('--tracer-type', type=str, default='PackedSDFTracer',
                                choices=list(str2tracer.keys()),
                                help='The tracer to be used.')
    return parser


# Usual boilerplate
parser = parse_sdf_options()
app_utils.add_log_level_flag(parser)
app_group = parser.add_argument_group('app')
# Add custom args if needed for app
args, args_str = argparse_to_str(parser)


app_utils.default_log_setup(args.log_level)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = SDFDataset(args.sample_mode, args.num_samples, args.get_normals,
                           args.sample_tex)

pipeline = get_pipeline_from_config(args.nef_type, dict(
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
    num_layers=args.num_layers,
    position_input=args.position_input,
    codebook_bitwidth=args.codebook_bitwidth
), args.tracer_type, dict(
    raymarch_type=args.raymarch_type,
    num_steps=args.num_steps,
    step_size=args.step_size,
    min_dis=args.min_dis,
), args.pretrained, args.model_format)

if pipeline.nef.grid is not None:
    if isinstance(pipeline.nef.grid, OctreeGrid):
        if not args.valid_only and not pipeline.nef.grid.blas_initialized():
            pipeline.nef.grid.init_from_mesh(
                args.dataset_path, sample_tex=args.sample_tex,
		num_samples=args.num_samples_on_mesh)
            pipeline.to(device)
        train_dataset.init_from_grid(pipeline.nef.grid, args.samples_per_voxel)
    else:
        train_dataset.init_from_mesh(args.dataset_path, args.mode_mesh_norm)

optim_cls, optim_params = get_optimizer_from_config(args.optimizer_type)

scene_state = WispState()
trainer = SDFTrainer(pipeline, train_dataset, args.epochs, args.batch_size,
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
