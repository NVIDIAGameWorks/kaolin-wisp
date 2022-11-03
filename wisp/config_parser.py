# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys
import argparse
import pprint
import yaml
import torch
from wisp.datasets import *
from wisp.models import Pipeline
from wisp.models.nefs import *
from wisp.models.grids import *
from wisp.tracers import *
from wisp.datasets.transforms import *

str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

def register_class(cls, name):
    globals()[name] = cls

def parse_options(return_parser=False):
    """Function used to parse options.
    
    Apps should use these CLI options, and then extend using parser.add_argument_group('app')
    
    Args:
        return_parser : If true, will return the parser object instead of the parsed arguments.
                        This is useful if you want to keep the parser around to add special argument
                        groups through app.
    """
    
    # New CLI parser
    parser = argparse.ArgumentParser(description='ArgumentParser for kaolin-wisp.')
    
    ###################
    # Global arguments
    ###################
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--trainer-type', type=str,
                              help='Trainer class to use')
    global_group.add_argument('--exp-name', type=str,
                              help='Experiment name.')
    global_group.add_argument('--perf', action='store_true',
                              help='Use high-level profiling for the trainer.')
    global_group.add_argument('--detect-anomaly', action='store_true',
                              help='Turn on anomaly detection.')
    global_group.add_argument('--config', type=str,
                              help='Path to config file to replace defaults.')

    ###################
    # Grid arguments
    ###################
    grid_group = parser.add_argument_group('grid')
    
    grid_group.add_argument('--grid-type', type=str, default='OctreeGrid',
                            choices=['None', 'OctreeGrid', 'CodebookOctreeGrid', 'TriplanarGrid', 'HashGrid'],
                            help='Type of grid to use.')
    grid_group.add_argument('--interpolation-type', type=str, default='linear',
                            choices=['linear', 'closest'],
                            help='SPC interpolation mode.')
    grid_group.add_argument('--as-type', type=str, default='none', 
                            choices=['none', 'octree'],
                            help='Type of accelstruct to use.')
    grid_group.add_argument('--raymarch-type', type=str, default='voxel',
                            choices=['voxel', 'ray'],
                            help='Method of raymarching. `voxel` samples within each primitive, \
                                  `ray` samples within rays and then filters them with the primitives. \
                                  See the accelstruct for details.')
    grid_group.add_argument('--multiscale-type', type=str, default='sum',
                            choices=['cat', 'sum'],
                            help='Type of multiscale aggregation function to use.')
    grid_group.add_argument('--feature-dim', type=int, default=32,
                            help='Feature map dimension')
    grid_group.add_argument('--feature-std', type=float, default=0.0,
                            help='Feature map std')
    grid_group.add_argument('--feature-bias', type=float, default=0.0,
                            help='Feature map bias')
    grid_group.add_argument('--noise-std', type=float, default=0.0,
                            help='Added noise to features in training.')
    grid_group.add_argument('--num-lods', type=int, default=1,
                            help='Number of LODs')
    grid_group.add_argument('--base-lod', type=int, default=2,
                            help='Base level LOD')
    grid_group.add_argument('--max-grid-res', type=int, default=2048,
                            help='The maximum grid resolution. Used only in geometric initialization.')
    grid_group.add_argument('--tree-type', type=str, default='quad', 
                            choices=['quad', 'geometric'],
                            help='What type of tree to use. `quad` is a quadtree or octree-like growing \
                                  scheme, whereas geometric is the Instant-NGP growing scheme.')
    grid_group.add_argument('--codebook-bitwidth', type=int, default=8, 
                            help='Bitwidth to use for the codebook. The number of vectors will be 2^bitwidth.')

    ###################
    # Embedder arguments
    ###################
    embedder_group = parser.add_argument_group('embedder')
    embedder_group.add_argument('--embedder-type', type=str, default='none',
                                choices=['none', 'positional', 'fourier'])
    embedder_group.add_argument('--pos-multires', type=int, default=10,
                                help='log2 of max freq')
    embedder_group.add_argument('--view-multires', type=int, default=4,
                                help='log2 of max freq')

    ###################
    # Decoder arguments (and general global network things)
    ###################
    net_group = parser.add_argument_group('net')
    
    net_group.add_argument('--nef-type', type=str,
                          help='The neural field class to be used.')
    net_group.add_argument('--layer-type', type=str, default='none',
                            choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    net_group.add_argument('--activation-type', type=str, default='relu',
                            choices=['relu', 'sin'])
    net_group.add_argument('--decoder-type', type=str, default='basic',
                            choices=['none', 'basic'])

    net_group.add_argument('--num-layers', type=int, default=1,
                          help='Number of layers for the decoder')
    net_group.add_argument('--hidden-dim', type=int, default=128,
                          help='Network width')
    net_group.add_argument('--out-dim', type=int, default=1,
                          help='output dimension')
    net_group.add_argument('--skip', type=int, default=None,
                          help='Layer to have skip connection.')
    net_group.add_argument('--pretrained', type=str,
                          help='Path to pretrained model weights.')
    net_group.add_argument('--position-input', action='store_true',
                          help='Use position as input.')

    ###################
    # Arguments for dataset
    ###################
    data_group = parser.add_argument_group('dataset')

    data_group.add_argument('--dataset-type', type=str, default=None,
                            choices=['sdf', 'multiview'],
                            help='Dataset class to use')
    data_group.add_argument('--dataset-path', type=str,
                            help='Path to the dataset')
    data_group.add_argument('--dataset-num-workers', type=int, default=-1, 
                            help='Number of workers for dataset preprocessing, if it supports multiprocessing. \
                                 -1 indicates no multiprocessing.')

    # SDF Dataset
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

    # Multiview Dataset
    data_group.add_argument('--multiview-dataset-format', default='standard',
                            choices=['standard', 'rtmv'],
                            help='Data format for the transforms')
    data_group.add_argument('--num-rays-sampled-per-img', type=int, default='4096',
                            help='Number of rays to sample per image')
    data_group.add_argument('--bg-color', default='white',
                            choices=['white', 'black'],
                            help='Background color')
    data_group.add_argument('--mip', type=int, default=None, 
                            help='MIP level of ground truth image')

    ###################
    # Arguments for optimizer
    ###################
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--optimizer-type', type=str, default='adam', choices=list(str2optim.keys()), 
                             help='Optimizer to be used.')
    optim_group.add_argument('--lr', type=float, default=0.001, 
                             help='Learning rate.')
    optim_group.add_argument('--weight-decay', type=float, default=0, 
                             help='Weight decay.')
    optim_group.add_argument('--grid-lr-weight', type=float, default=100.0,
                             help='Relative LR weighting for the grid')
    optim_group.add_argument('--rgb-loss', type=float, default=1.0, 
                            help='Weight of rgb loss')

    ###################
    # Arguments for training
    ###################
    train_group = parser.add_argument_group('trainer')
    train_group.add_argument('--epochs', type=int, default=250, 
                             help='Number of epochs to run the training.')
    train_group.add_argument('--batch-size', type=int, default=512, 
                             help='Batch size for the training.')
    train_group.add_argument('--resample', action='store_true', 
                             help='Resample the dataset after every epoch.')
    train_group.add_argument('--only-last', action='store_true', 
                             help='Train only last LOD.')
    train_group.add_argument('--resample-every', type=int, default=1,
                             help='Resample every N epochs')
    train_group.add_argument('--model-format', type=str, default='full',
                             choices=['full', 'state_dict'],
                             help='Format in which to save models.')
    train_group.add_argument('--save-as-new', action='store_true', 
                             help='Save the model at every epoch (no overwrite).')
    train_group.add_argument('--save-every', type=int, default=5, 
                             help='Save the model at every N epoch.')
    train_group.add_argument('--render-every', type=int, default=5,
                                help='Render every N epochs')
    # TODO (ttakikawa): Only used for SDFs, but also should support RGB etc
    train_group.add_argument('--log-2d', action='store_true', 
                             help='Log cutting plane renders to TensorBoard.')
    train_group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                             help='Log file directory for checkpoints.')
    # TODO (ttakikawa): This is only really used in the SDF training but it should be useful for multiview too
    train_group.add_argument('--grow-every', type=int, default=-1,
                             help='Grow network every X epochs')
    train_group.add_argument('--prune-every', type=int, default=-1,
                             help='Prune every N epochs')
    # TODO (ttakikawa): Only used in multiview training, combine with the SDF growing schemes.
    train_group.add_argument('--random-lod', action='store_true',
                             help='Use random lods to train.')
    # One by one trains one level at a time. 
    # Increase starts from [0] and ends up at [0,...,N]
    # Shrink strats from [0,...,N] and ends up at [N]
    # Fine to coarse starts from [N] and ends up at [0,...,N]
    # Only last starts and ends at [N]
    train_group.add_argument('--growth-strategy', type=str, default='increase',
                             choices=['onebyone','increase','shrink', 'finetocoarse', 'onlylast'],
                             help='Strategy for coarse-to-fine training')
    
    ###################
    # Arguments for training
    ###################
    valid_group = parser.add_argument_group('validation')
    
    valid_group.add_argument('--valid-only', action='store_true',
                             help='Run validation only (and do not run training).')
    valid_group.add_argument('--valid-every', type=int, default=-1,
                             help='Frequency of running validation.')
    valid_group.add_argument('--valid-split', type=str, default='val',
                             help='Split to use for validation.')

    ###################
    # Arguments for renderer
    ###################
    renderer_group = parser.add_argument_group('renderer')
    renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512], 
                                help='Width/height to render at.')
    renderer_group.add_argument('--render-batch', type=int, default=0, 
                                help='Batch size (in number of rays) for batched rendering.')
    renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8], 
                                help='Camera origin.')
    renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0], 
                                help='Camera look-at/target point.')
    renderer_group.add_argument('--camera-fov', type=float, default=30, 
                                help='Camera field of view (FOV).')
    renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp', 
                                help='Camera projection.')
    renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10], 
                                help='Camera clipping bounds.')
    renderer_group.add_argument('--tracer-type', type=str, default='PackedRFTracer', 
                                help='The tracer to be used.')
    
    # TODO(ttakikawa): In the future the interface will be such that you either select an absolute step size or 
    #                  you select the number of steps to take. Sphere tracing will take step-scales.
    renderer_group.add_argument('--num-steps', type=int, default=128,
                                help='Number of steps for raymarching / spheretracing / etc')
    renderer_group.add_argument('--step-size', type=float, default=1.0,
                                help='Scale of step size')
    
    # Sphere tracing stuff
    renderer_group.add_argument('--min-dis', type=float, default=0.0003,
                                help='Minimum distance away from surface for spheretracing')
    
    # TODO(ttakikawa): Shader stuff... will be more modular in future
    renderer_group.add_argument('--matcap-path', type=str, 
                                default='data/matcaps/matcap_plastic_yellow.jpg', 
                                help='Path to the matcap texture to render with.')
    renderer_group.add_argument('--ao', action='store_true',
                                help='Use ambient occlusion.')
    renderer_group.add_argument('--shadow', action='store_true',
                                help='Use shadowing.')
    renderer_group.add_argument('--shading-mode', type=str, default='normal',
                                choices=['matcap', 'rb', 'normal'],
                                help='Shading mode.')

    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)

def parse_yaml_config(config_path, parser):
    """Parses and sets the parser defaults with a yaml config file.

    Args:
        config_path : path to the yaml config file.
        parser : The parser for which the defaults will be set.
        parent : True if parsing the parent yaml. Should never be set to True by the user.
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    list_of_valid_fields = []
    for group in parser._action_groups:
        group_dict = {list_of_valid_fields.append(a.dest) for a in group._group_actions}
    list_of_valid_fields = set(list_of_valid_fields)
    
    defaults_dict = {}
    
    # Load the parent config if it exists
    parent_config_path = config_dict.pop("parent", None)

    if parent_config_path is not None:
        if not os.path.isabs(parent_config_path):
            parent_config_path = os.path.join(os.path.split(config_path)[0], parent_config_path)
        with open(parent_config_path) as f:
            parent_config_dict = yaml.safe_load(f)
        if "parent" in parent_config_dict.keys():
            raise Exception("Hierarchical configs of more than 1 level deep are not allowed.")
        for key in parent_config_dict:
            for field in parent_config_dict[key]:
                if field not in list_of_valid_fields:
                    raise ValueError(
                        f"ERROR: {field} is not a valid option. Check for typos in the config."
                    )
                defaults_dict[field] = parent_config_dict[key][field]
        
    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            defaults_dict[field] = config_dict[key][field]

    parser.set_defaults(**defaults_dict)

def argparse_to_str(parser, args=None):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.
    
    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    
    if args is None:
        args = parser.parse_args()

    if args.config is not None:
        parse_yaml_config(args.config, parser)

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

def get_optimizer_from_config(args):
    """Utility function to get the optimizer from the parsed config.
    """
    optim_cls = str2optim[args.optimizer_type]
    if args.optimizer_type == 'adam':
        optim_params = {'eps': 1e-15}
    elif args.optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}
    return optim_cls, optim_params

def get_modules_from_config(args):
    """Utility function to get the modules for training from the parsed config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nef = globals()[args.nef_type](**vars(args))
    tracer = globals()[args.tracer_type](**vars(args))
    pipeline = Pipeline(nef, tracer)

    if args.pretrained:
        if args.model_format == "full":
            pipeline = torch.load(args.pretrained)
        else:
            pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)

    if args.dataset_type == "multiview":
        transform = SampleRays(args.num_rays_sampled_per_img)
        train_dataset = MultiviewDataset(**vars(args), transform=transform)
        train_dataset.init()
        
        if pipeline.nef.grid is not None:
            if isinstance(pipeline.nef.grid, OctreeGrid):
                if not args.valid_only and not pipeline.nef.grid.blas_initialized():
                    if args.multiview_dataset_format in ['rtmv']:
                        pipeline.nef.grid.init_from_pointcloud(train_dataset.coords)
                    else:
                        pipeline.nef.grid.init_dense()
                    pipeline.to(device)
            if isinstance(pipeline.nef.grid, HashGrid):
                if not args.valid_only:
                    if args.tree_type == 'quad':
                        pipeline.nef.grid.init_from_octree(args.base_lod, args.num_lods)
                    elif args.tree_type == 'geometric':
                        pipeline.nef.grid.init_from_geometric(16, args.max_grid_res, args.num_lods)
                    else:
                        raise NotImplementedError
                    pipeline.to(device)

    elif args.dataset_type == "sdf":
        train_dataset = SDFDataset(args.sample_mode, args.num_samples,
                                   args.get_normals, args.sample_tex)
        
        if pipeline.nef.grid is not None:
            if isinstance(pipeline.nef.grid, OctreeGrid):
                
                if not args.valid_only and not pipeline.nef.grid.blas_initialized():
                    pipeline.nef.grid.init_from_mesh(
                        args.dataset_path, sample_tex=args.sample_tex, num_samples=args.num_samples_on_mesh)
                    pipeline.to(device)
                
                train_dataset.init_from_grid(pipeline.nef.grid, args.samples_per_voxel)
            else:
                train_dataset.init_from_mesh(args.dataset_path, args.mode_mesh_norm)
    else:
        raise ValueError(f'"{args.dataset_type}" unrecognized dataset_type')
    return pipeline, train_dataset, device
