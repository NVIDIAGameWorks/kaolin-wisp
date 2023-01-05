# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from wisp.cuda_guard import setup_cuda_context
setup_cuda_context()  # Must be called before any torch operations take place

import os
import argparse
import logging
import torch
from wisp.app_utils import default_log_setup, args_to_log_format
import wisp.config_parser as config_parser
from wisp.framework import WispState
from wisp.datasets import SDFDataset
from wisp.models.grids import BLASGrid, OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.tracers import BaseTracer, PackedSDFTracer
from wisp.models.nefs import BaseNeuralField, NeuralSDF
from wisp.models.pipeline import Pipeline
from wisp.trainers import BaseTrainer, SDFTrainer

def parse_args():
    """Wisp mains define args per app.
    Args are collected by priority: cli args > config yaml > argparse defaults
    For convenience, args are divided into groups.
    """
    parser = argparse.ArgumentParser(description='A script for training Signed Distance Fields with various backbones,'
                                                 'following the structure of the paper '
                                                 '"Neural Geometric Level of Detail: Real-time Rendering with '
                                                 'Implicit 3D Shapes".')
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')

    log_group = parser.add_argument_group('logging')
    log_group.add_argument('--exp-name', type=str,
                           help='Experiment name, unique id for trainers, logs.')
    log_group.add_argument('--log_level', action='store', type=int, default=logging.INFO,
                           help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
    log_group.add_argument('--perf', action='store_true', default=False,
                           help='Use high-level profiling for the trainer.')

    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--dataset-path', type=str,
                            help='Path to the dataset')
    data_group.add_argument('--dataset-num-workers', type=int, default=-1,
                            help='Number of workers for dataset preprocessing, '
                                 'if it supports multiprocessing. -1 indicates no multiprocessing.')
    data_group.add_argument('--dataloader-num-workers', type=int, default=0,
                            help='Number of workers for dataloader.')
    data_group.add_argument('--bg-color', default='white', choices=['white', 'black'],
                            help='Background color') # TODO (operel): do we need that?
    data_group.add_argument('--sample-mode', type=str, nargs='*',
                            default=['rand', 'near', 'near', 'trace', 'trace'],
                            help='The sampling scheme to be used for generating points in space, near and on surface.')
    data_group.add_argument('--get-normals', action='store_true',
                            help='If specified, sample normals as well.')
    data_group.add_argument('--num-samples', type=int, default=100000,
                            help='Number of samples per mode (or per epoch for SPC)')
    data_group.add_argument('--sample-tex', action='store_true',
                            help='If true, samples textures to initialize grid')
    data_group.add_argument('--num-samples-on-mesh', type=int, default=100000000,
                            help='Grid initialization from mesh: number of sdf samples generated on mesh faces. '
                                 'Higher amount takes more time but reduces the probability of holes.')
    data_group.add_argument('--mode-mesh-norm', type=str, default='sphere',
                            choices=['sphere', 'aabb', 'planar', 'none'],
                            help='Normalize the mesh')
    data_group.add_argument('--samples-per-voxel', type=int, default=256,
                            help='Number of samples per voxel (for SDF initialization from grid)')

    grid_group = parser.add_argument_group('grid')
    grid_group.add_argument('--grid-type', type=str, default='OctreeGrid',
                            choices=config_parser.list_modules('grid'),
                            help='Type of to use, i.e.:'
                                 '"OctreeGrid", "CodebookOctreeGrid", "TriplanarGrid", "HashGrid".'
                                 'Grids are located in `wisp.models.grids`')
    grid_group.add_argument('--interpolation-type', type=str, default='linear', choices=['linear', 'closest'],
                            help='Interpolation type to use for samples within grids.'
                                 'For a 3D grid structure, linear uses trilinear interpolation of 8 cell nodes,'
                                 'closest uses the nearest neighbor.')
    grid_group.add_argument('--blas-type', type=str, default='octree',  # TODO(operel)
                            choices=['octree',],
                            help='Type of acceleration structure to use for fast occupancy queries.')
    grid_group.add_argument('--multiscale-type', type=str, default='sum', choices=['sum', 'cat'],
                            help='Aggregation of choice for multi-level grids, for features from different LODs.')
    grid_group.add_argument('--feature-dim', type=int, default=5,
                            help='Dimensionality for features stored within the grid nodes.')
    grid_group.add_argument('--feature-std', type=float, default=0.01,
                            help='Grid initialization: standard deviation used for randomly sampling initial features.')
    grid_group.add_argument('--feature-bias', type=float, default=0.0,
                            help='Grid initialization: bias used for randomly sampling initial features.')
    grid_group.add_argument('--base-lod', type=int, default=5,
                            help='Number of levels in grid, which book-keep occupancy but not features.'
                                 'The total number of levels in a grid is `base_lod + num_lod - 1`')
    grid_group.add_argument('--num-lods', type=int, default=4,
                            help='Number of levels in grid, which store concrete features.')
    grid_group.add_argument('--codebook-bitwidth', type=int, default=19,
                            help='For Codebook and HashGrids only: determines the table size as 2**(bitwidth).')
    grid_group.add_argument('--tree-type', type=str, default='geometric', choices=['geometric', 'quad'],
                            help='For HashGrids only: how the resolution of the grid is determined. '
                                 '"geometric" uses the geometric sequence initialization from InstantNGP,'
                                 'where "quad" uses an octree sampling pattern.')
    grid_group.add_argument('--min-grid-res', type=int, default=16,
                            help='For HashGrids only: min grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--max-grid-res', type=int, default=2048,
                            help='For HashGrids only: max grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--blas-level', type=float, default=7,
                            help='For HashGrids only: Determines the number of levels in the acceleration structure '
                                 'used to track the occupancy status (bottom level acceleration structure).')

    nef_group = parser.add_argument_group('nef')
    nef_group.add_argument('--pos-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode input coordinates'
                                'or view directions.')
    nef_group.add_argument('--pos-multires', type=int, default=4,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of input coordinates')
    nef_group.add_argument('--position-input', action='store_true', default=True,
                           help='If True, positional embedding will be concatenated to grid features.')
    nef_group.add_argument('--layer-type', type=str, default='none',
                           choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    nef_group.add_argument('--activation-type', type=str, default='relu',
                           choices=['relu', 'sin'])
    nef_group.add_argument('--hidden-dim', type=int,
                           help='MLP Decoder of neural field: width of hidden layer.')
    nef_group.add_argument('--num-layers', type=int,
                           help='MLP Decoder of neural field: number of hidden layers.')

    tracer_group = parser.add_argument_group('tracer')
    tracer_group.add_argument('--num-steps', type=int, default=1024,
                              help='Inference only: max number of steps used by Sphere Trace if query did not converge')
    tracer_group.add_argument('--step-size', type=str, default=0.8,
                              help='Inference only: scale factor for step size used to advance the Sphere Tracer.')

    trainer_group = parser.add_argument_group('trainer')
    trainer_group.add_argument('--epochs', type=int, default=250,
                               help='Number of epochs to run the training.')
    trainer_group.add_argument('--batch-size', type=int, default=512,
                               help='Batch size for the training.')
    trainer_group.add_argument('--resample', action='store_true',
                               help='Resample the dataset after every epoch.')
    trainer_group.add_argument('--only-last', action='store_true',
                               help='Train only last LOD.')
    trainer_group.add_argument('--resample-every', type=int, default=1,
                               help='Resample every N epochs')
    trainer_group.add_argument('--model-format', type=str, default='full', choices=['full', 'state_dict'],
                               help='Format in which to save models.')
    trainer_group.add_argument('--pretrained', type=str,
                               help='Path to pretrained model weights.')
    trainer_group.add_argument('--save-as-new', action='store_true',
                               help='Save the model at every epoch (no overwrite).')
    trainer_group.add_argument('--save-every', type=int, default=5,
                               help='Save the model at every N epoch.')
    trainer_group.add_argument('--render-tb-every', type=int, default=5,
                               help='Render every N epochs')
    trainer_group.add_argument('--log-tb-every', type=int, default=5, # TODO (operel): move to logging
                               help='Render to tensorboard every N epochs')
    trainer_group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                               help='Log file directory for checkpoints.')
    trainer_group.add_argument('--grow-every', type=int, default=-1,
                               help='Grow network every X epochs')
    trainer_group.add_argument('--growth-strategy', type=str, default='increase',
                               choices=['onebyone',      # One by one trains one level at a time.
                                        'increase',      # Increase starts from [0] and ends up at [0,...,N]
                                        'shrink',        # Shrink strats from [0,...,N] and ends up at [N]
                                        'finetocoarse',  # Fine to coarse starts from [N] and ends up at [0,...,N]
                                        'onlylast'],     # Only last starts and ends at [N]
                               help='Strategy for coarse-to-fine training')
    trainer_group.add_argument('--valid-only', action='store_true',
                               help='Run validation only (and do not run training).')
    trainer_group.add_argument('--valid-every', type=int, default=-1,
                               help='Frequency of running validation.')
    trainer_group.add_argument('--log-2d', action='store_true', help='Log cutting plane renders to TensorBoard.')
    trainer_group.add_argument('--wandb-project', type=str, default=None,
                               help='Weights & Biases Project')
    trainer_group.add_argument('--wandb-run-name', type=str, default=None,
                               help='Weights & Biases Run Name')
    trainer_group.add_argument('--wandb-entity', type=str, default=None,
                               help='Weights & Biases Entity')

    optimizer_group = parser.add_argument_group('optimizer')
    optimizer_group.add_argument('--optimizer-type', type=str, default='adam',
                                 choices=config_parser.list_modules('optim'),
                                 help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                                      'and fused optimizers from `apex`, if apex is installed.')
    optimizer_group.add_argument('--lr', type=float, default=0.001,
                                 help='Base optimizer learning rate.')
    optimizer_group.add_argument('--eps', type=float, default=1e-8, help='Eps value for numerical stability.')
    optimizer_group.add_argument('--weight-decay', type=float, default=0,
                                 help='Weight decay, applied only to decoder weights.')
    optimizer_group.add_argument('--grid-lr-weight', type=float, default=100.0,
                                 help='Relative learning rate weighting applied only for the grid parameters'
                                      '(e.g. parameters which contain "grid" in their name)')

    # Evaluation renderer (definitions do not affect interactive renderer)
    offline_renderer_group = parser.add_argument_group('renderer')
    offline_renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512],
                                        help='Width/height to render at.')
    offline_renderer_group.add_argument('--render-batch', type=int, default=0,
                                        help='Batch size (in number of rays) for batched rendering.')
    offline_renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8],
                                        help='Camera origin.')
    offline_renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0],
                                        help='Camera look-at/target point.')
    offline_renderer_group.add_argument('--camera-fov', type=float, default=30,
                                        help='Camera field of view (FOV).')
    offline_renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp',
                                        help='Camera projection.')
    offline_renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10],
                                        help='Camera clipping bounds.')
    offline_renderer_group.add_argument('--matcap-path', type=str,
                                        default='data/matcaps/matcap_plastic_yellow.jpg',
                                        help='Path to the matcap texture to render with.')
    offline_renderer_group.add_argument('--ao', action='store_true',
                                        help='Use ambient occlusion.')
    offline_renderer_group.add_argument('--shadow', action='store_true',
                                        help='Use shadowing.')
    offline_renderer_group.add_argument('--shading-mode', type=str, default='normal',
                                        choices=['matcap', 'rb', 'normal'],
                                        help='Shading mode.')

    # Parse CLI args & config files
    args = config_parser.parse_args(parser)

    # Override some definitions for interactive app, such as validation logic and default data background color
    if is_interactive():
        args.bg_color = 'black'
        args.save_every = -1
        args.render_tb_every = -1
        args.valid_every = -1

    # Also obtain args as grouped hierarchy, useful for, i.e., logging
    args_dict = config_parser.get_grouped_args(parser, args)
    return args, args_dict


def load_dataset(args, pipeline: Pipeline) -> torch.utils.data.Dataset:
    """ Loads a dataset of SDF samples generated over the surface of a mesh. """
    if isinstance(pipeline.nef.grid, OctreeGrid):
        train_dataset = SDFDataset.from_grid(
            sample_mode=args.sample_mode,
            num_samples=args.num_samples,
            get_normals=args.get_normals,
            sample_tex=args.sample_tex,
            grid=pipeline.nef.grid,
            samples_per_voxel=args.samples_per_voxel)
    else:
        train_dataset = SDFDataset.from_mesh(
            sample_mode=args.sample_mode,
            num_samples=args.num_samples,
            get_normals=args.get_normals,
            sample_tex=args.sample_tex,
            dataset_path=args.dataset_path,
            mode_norm=args.mode_mesh_norm)
    return train_dataset


def load_grid(args) -> BLASGrid:
    """Loads the hierarchical feature grid to use within the neural sdf pipeline.
    Grid choices are interesting to explore, so we leave the exact backbone type configurable,
    and show how grid instances may be explicitly constructed.
    Grids choices, for example, are: OctreeGrid, TriplanarGrid, HashGrid
    See corresponding grid constructors for each of their arg details.
    """
    grid = None
    if args.grid_type == "OctreeGrid":
        # For SDF pipelines case, the grid may be initialized from the mesh to speed up the optimization.
        grid = OctreeGrid.from_mesh(
            mesh_path=args.dataset_path,
            num_samples_on_mesh=args.num_samples_on_mesh,
            feature_dim=args.feature_dim,
            base_lod=args.base_lod,
            num_lods=args.num_lods,
            interpolation_type=args.interpolation_type,
            multiscale_type=args.multiscale_type,
            feature_std=args.feature_std,
            feature_bias=args.feature_bias,
        )
    elif args.grid_type == "TriplanarGrid":
        grid = TriplanarGrid(
            feature_dim=args.feature_dim,
            base_lod=args.base_lod,
            num_lods=args.num_lods,
            interpolation_type=args.interpolation_type,
            multiscale_type=args.multiscale_type,
            feature_std=args.feature_std,
            feature_bias=args.feature_bias,
        )
    elif args.grid_type == "HashGrid":
        # "geometric" - determines the resolution of the grid using geometric sequence initialization from InstantNGP,
        if args.tree_type == "geometric":
            grid = HashGrid.from_geometric(
                feature_dim=args.feature_dim,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                min_grid_res=args.min_grid_res,
                max_grid_res=args.max_grid_res,
                blas_level=args.blas_level
            )
        # "quad" - determines the resolution of the grid using an octree sampling pattern.
        elif args.tree_type == "octree":
            grid = HashGrid.from_octree(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                blas_level=args.blas_level
            )
    else:
        raise ValueError(f"Unknown grid_type argument: {args.grid_type}")
    return grid


def load_tracer(args) -> BaseTracer:
    """ Tracer to be used in inference time for rendering the SDF """
    # TODO (operel): SDFTracer with tex
    tracer = PackedSDFTracer(
        num_steps=args.num_steps,
        step_size=args.step_size
    )
    return tracer


def load_neural_field(args) -> BaseNeuralField:
    """ Creates a "Neural Field" instance which converts input coordinates to some output signal.
    Here a NeuralSDF is created, which maps 3D coordinates -> SDF values.
    NeuralSDF is a combo of a spatial feature grid and a single decoder.
    The NeuralSDF uses the grid internally for faster feature interpolation and raytracing.
    """
    grid = load_grid(args=args)
    nef = NeuralSDF(
        grid=grid,
        pos_embedder=args.pos_embedder,
        pos_multires=args.pos_multires,
        position_input=args.position_input,
        activation_type=args.activation_type,
        layer_type=args.layer_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    return nef


def load_neural_pipeline(args, device) -> Pipeline:
    """ In Wisp, a Pipeline comprises of a neural field + a tracer (the latter is optional in some cases).
    Together, they form the complete pipeline required to render a neural primitive from input rays / coordinates.
    """
    nef = load_neural_field(args=args)
    tracer = load_tracer(args=args)
    pipeline = Pipeline(nef=nef, tracer=tracer)
    if args.pretrained:
        if args.model_format == "full":
            pipeline = torch.load(args.pretrained)
        else:
            pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)
    return pipeline


def load_trainer(pipeline, train_dataset, device, scene_state, args, args_dict) -> BaseTrainer:
    """ Loads the SDF trainer.
    The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
    - Headless, which will run the train() function until all training steps are exhausted.
    - Interactive mode, which uses the gui. In this case, an OptimizationApp uses events to prompt the trainer to
      take training steps, while also taking care to render output to users (see: iterate()).
      In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
    """
    # args.optimizer_type is the name of some optimizer class (from torch.optim or apex),
    # Wisp's config_parser is able to pick this app's args with corresponding names to the optimizer constructor args.
    # The actual construction of the optimizer instance happens within the trainer.
    optimizer_cls = config_parser.get_module(name=args.optimizer_type)
    optimizer_params = config_parser.get_args_for_function(args, optimizer_cls)
    trainer = SDFTrainer(pipeline=pipeline,
                         dataset=train_dataset,
                         num_epochs=args.epochs,
                         batch_size=args.batch_size,
                         optim_cls=optimizer_cls,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         grid_lr_weight=args.grid_lr_weight,
                         optim_params=optimizer_params,
                         log_dir=args.log_dir,
                         device=device,
                         exp_name=args.exp_name,
                         info=args_to_log_format(args_dict),
                         extra_args=vars(args),
                         render_tb_every=args.render_tb_every,
                         save_every=args.save_every,
                         scene_state=scene_state,
                         trainer_mode='validate' if args.valid_only else 'train',
                         using_wandb=args.wandb_project is not None)
    return trainer


def load_app(args, scene_state, trainer):
    """ Used only in interactive mode. Creates an interactive app, which employs a renderer which displays
    the latest information from the trainer (see: OptimizationApp).
    The OptimizationApp can be customized or further extend to support even more functionality.
    """
    if not is_interactive():
        logging.info("Running headless. For the app, set $WISP_HEADLESS=0.")
        return None  # Interactive mode is disabled
    else:
        from wisp.renderer.app.optimization_app import OptimizationApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
        app = OptimizationApp(wisp_state=scene_state,
                              trainer_step_func=trainer.iterate,
                              experiment_name="wisp trainer")
        return app


def is_interactive() -> bool:
    """ Returns True if interactive mode with gui is on, False is HEADLESS mode is forced or no-interactive requested"""
    return os.environ.get('WISP_HEADLESS') != '1'


args, args_dict = parse_args()  # Obtain args by priority: cli args > config yaml > argparse defaults
default_log_setup(args.log_level)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipeline = load_neural_pipeline(args=args, device=device)
train_dataset = load_dataset(args=args, pipeline=pipeline)
scene_state = WispState()  # Joint trainer / app state
trainer = load_trainer(pipeline=pipeline, train_dataset=train_dataset, device=device, scene_state=scene_state,
                       args=args, args_dict=args_dict)
app = load_app(args=args, scene_state=scene_state, trainer=trainer)

if app is not None:
    app.run()  # Interactive Mode
else:
    if args.valid_only:
        trainer.validate()
    else:
        trainer.train()
