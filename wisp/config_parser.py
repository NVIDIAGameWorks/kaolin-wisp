# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import logging
import os
import pprint
import yaml
import torch

import wisp
from wisp.models import Pipeline
from wisp.models import nefs
from wisp.models import grids
from wisp import tracers
from wisp import datasets

# This file contains all the configuration and command-line parsing of the app

__all__ = [
    'ParseKwargs',
    'str2dataset',
    'str2optim',
    'str2nef',
    'str2grid',
    'str2tracer',
    'add_logging_argument_group',
    'add_grid_argument_group',
    'add_embedder_argument_group',
    'add_net_argument_group',
    'add_dataset_argument_group',
    'add_optimizer_argument_group',
    'add_trainer_argument_group',
    'add_renderer_argument_group',
    'argparse_to_str',
    'get_optimizer_from_config',
    'get_pipeline_from_config'
]

# This is to use a dictionary as input
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        args_dict = dict()
        for value in values:
            key, value = value.split('=')
            args_dict[key] = value
        setattr(namespace, self.dest, args_dict)

# all the str2X are registered options as dictionary for parsing
# You can add an option by adding it to the dictionary
# Currently we have: str2optim, str2dataset, str2nef, str2grid, str2tracer
str2optim = {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()}

try:
    import apex
    for m in dir(apex.optimizers):
        if m[0].isupper():
            str2optim[m.lower()] = getattr(apex.optimizers, m)
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Cannot import apex for fused optimizers")

str2dataset = {}
for name in dir(datasets):
    mod = getattr(datasets, name)
    if isinstance(mod, type) and \
            issubclass(mod, torch.utils.data.Dataset) and \
            mod != torch.utils.data.Dataset:
        str2dataset[name] = mod

str2nef = {}
for name in dir(nefs):
    mod = getattr(nefs, name)
    if isinstance(mod, type) and \
            issubclass(mod, nefs.BaseNeuralField) and \
            mod != nefs.BaseNeuralField:
        str2nef[name] = mod

str2grid = {}
for name in dir(grids):
    mod = getattr(grids, name)
    if isinstance(mod, type) and \
            issubclass(mod, grids.BLASGrid) and \
            mod != grids.BLASGrid:
        str2grid[name] = mod

str2tracer = {}
for name in dir(tracers):
    mod = getattr(tracers, name)
    if isinstance(mod, type) and \
            issubclass(mod, tracers.BaseTracer) and \
            mod != tracers.BaseTracer:
        str2tracer[name] = mod

# For building parser those are general parser arguments that are expected to be used across most applications
def add_logging_argument_group(parser):
    group = parser.add_argument_group('logging')
    group.add_argument('--exp-name', type=str,
                       help='Experiment name.')
    group.add_argument('--perf', action='store_true',
                       help='Use high-level profiling for the trainer.')
    group.add_argument('--detect-anomaly', action='store_true',
                       help='Turn on anomaly detection.')
    return group

def add_grid_argument_group(parser):
    group = parser.add_argument_group('grid')
    group.add_argument('--grid-type', type=str, default='OctreeGrid',
                       choices=list(str2grid.keys()),
                       help='Type of grid to use.')
    group.add_argument('--interpolation-type', type=str, default='linear',
                       choices=['linear', 'closest'],
                       help='SPC interpolation mode.')
    # TODO(ttakikawa): shouldn't raymarch with sdf, but pipeline force to
    group.add_argument('--raymarch-type', type=str, default='voxel',
                       choices=['voxel', 'ray'],
                       help='Method of raymarching. `voxel` samples within each primitive, \
                             `ray` samples within rays and then filters them with the primitives. \
                             See the accelstruct for details.')
    group.add_argument('--multiscale-type', type=str, default='sum',
                       choices=['cat', 'sum'],
                       help='Type of multiscale aggregation function to use.')
    group.add_argument('--feature-dim', type=int, default=32,
                       help='Feature map dimension')
    group.add_argument('--feature-std', type=float, default=0.0,
                       help='Feature map std')
    group.add_argument('--feature-bias', type=float, default=0.0,
                       help='Feature map bias')
    group.add_argument('--noise-std', type=float, default=0.0,
                       help='Added noise to features in training.')
    group.add_argument('--num-lods', type=int, default=1,
                       help='Number of LODs')
    group.add_argument('--base-lod', type=int, default=2,
                       help='Base level LOD')
    group.add_argument('--max-grid-res', type=int, default=2048,
                       help='The maximum grid resolution. Used only in geometric initialization.')
    group.add_argument('--tree-type', type=str, default='quad', 
                       choices=['quad', 'geometric'],
                       help='What type of tree to use. `quad` is a quadtree or octree-like growing \
                             scheme, whereas geometric is the Instant-NGP growing scheme.')
    group.add_argument('--codebook-bitwidth', type=int, default=8, 
                       help='Bitwidth to use for the codebook. '
                            'The number of vectors will be 2^bitwidth.')
    return group

def add_embedder_argument_group(parser):
    group = parser.add_argument_group('embedder')
    group.add_argument('--embedder-type', type=str, default='none',
                       choices=['none', 'positional', 'fourier'])
    group.add_argument('--pos-multires', type=int, default=10,
                       help='log2 of max freq')
    return group

def add_net_argument_group(parser):
    group = parser.add_argument_group('net')
    group.add_argument('--layer-type', type=str, default='none',
                       choices=['none', 'spectral_norm', 'frobenius_norm',
                                'l_1_norm', 'l_inf_norm'])
    group.add_argument('--activation-type', type=str, default='relu',
                       choices=['relu', 'sin'])
    group.add_argument('--num-layers', type=int, default=1,
                      help='Number of layers for the decoder')
    group.add_argument('--hidden-dim', type=int, default=128,
                      help='Network width')
    group.add_argument('--out-dim', type=int, default=1,
                      help='output dimension')
    group.add_argument('--skip', type=int, default=None,
                      help='Layer to have skip connection.')
    group.add_argument('--pretrained', type=str,
                      help='Path to pretrained model weights.')
    group.add_argument('--position-input', action='store_true',
                      help='Use position as input.')
    return group

def add_dataset_argument_group(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--dataset-path', type=str, help='Path to the dataset')
    group.add_argument('--dataset-num-workers', type=int, default=-1, 
                       help='Number of workers for dataset preprocessing, '
                            'if it supports multiprocessing. '
                            '-1 indicates no multiprocessing.')
    group.add_argument('--bg-color', default='white',
                       choices=['white', 'black'],
                       help='Background color')
    return group

def add_optimizer_argument_group(parser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--optimizer-type', type=str, default='adam',
                       choices=list(str2optim.keys()), 
                       help='Optimizer to be used.')
    group.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate.')
    group.add_argument('--weight-decay', type=float, default=0, 
                       help='Weight decay.')
    group.add_argument('--grid-lr-weight', type=float, default=100.0,
                       help='Relative LR weighting for the grid')
    group.add_argument('--rgb-loss', type=float, default=1.0, 
                      help='Weight of rgb loss')
    return group

def add_trainer_argument_group(parser):
    group = parser.add_argument_group('trainer')
    group.add_argument('--epochs', type=int, default=250, 
                       help='Number of epochs to run the training.')
    group.add_argument('--batch-size', type=int, default=512, 
                       help='Batch size for the training.')
    group.add_argument('--resample', action='store_true', 
                       help='Resample the dataset after every epoch.')
    group.add_argument('--only-last', action='store_true', 
                       help='Train only last LOD.')
    group.add_argument('--resample-every', type=int, default=1,
                       help='Resample every N epochs')
    group.add_argument('--model-format', type=str, default='full',
                       choices=['full', 'state_dict'],
                       help='Format in which to save models.')
    group.add_argument('--save-as-new', action='store_true', 
                       help='Save the model at every epoch (no overwrite).')
    group.add_argument('--save-every', type=int, default=5, 
                       help='Save the model at every N epoch.')
    group.add_argument('--render-every', type=int, default=5,
                          help='Render every N epochs')
    group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                       help='Log file directory for checkpoints.')
    # TODO (ttakikawa): This is only really used in the SDF training but
    #                   it should be useful for multiview too
    group.add_argument('--grow-every', type=int, default=-1,
                       help='Grow network every X epochs')
    # One by one trains one level at a time. 
    # Increase starts from [0] and ends up at [0,...,N]
    # Shrink strats from [0,...,N] and ends up at [N]
    # Fine to coarse starts from [N] and ends up at [0,...,N]
    # Only last starts and ends at [N]
    group.add_argument('--growth-strategy', type=str, default='increase',
                       choices=['onebyone','increase','shrink', 'finetocoarse', 'onlylast'],
                       help='Strategy for coarse-to-fine training')
    group.add_argument('--valid-only', action='store_true',
                       help='Run validation only (and do not run training).')
    group.add_argument('--valid-every', type=int, default=-1,
                       help='Frequency of running validation.')
    return group

def add_renderer_argument_group(parser):
    group = parser.add_argument_group('renderer')
    group.add_argument('--render-res', type=int, nargs=2, default=[512, 512], 
                       help='Width/height to render at.')
    group.add_argument('--render-batch', type=int, default=0, 
                       help='Batch size (in number of rays) for batched rendering.')
    group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8], 
                       help='Camera origin.')
    group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0], 
                       help='Camera look-at/target point.')
    group.add_argument('--camera-fov', type=float, default=30, 
                       help='Camera field of view (FOV).')
    group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp', 
                       help='Camera projection.')
    group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10], 
                       help='Camera clipping bounds.')
    # TODO(ttakikawa): In the future the interface will be such that
    #                  you either select an absolute step size or 
    #                  you select the number of steps to take.
    #                  Sphere tracing will take step-scales.
    group.add_argument('--num-steps', type=int, default=128,
                       help='Number of steps for raymarching / spheretracing / etc')
    group.add_argument('--step-size', type=float, default=1.0,
                       help='Scale of step size')
    # Sphere tracing stuff
    group.add_argument('--min-dis', type=float, default=0.0003,
                       help='Minimum distance away from surface for spheretracing')
    # TODO(ttakikawa): Shader stuff... will be more modular in future
    group.add_argument('--matcap-path', type=str, 
                       default='data/matcaps/matcap_plastic_yellow.jpg', 
                       help='Path to the matcap texture to render with.')
    group.add_argument('--ao', action='store_true',
                       help='Use ambient occlusion.')
    group.add_argument('--shadow', action='store_true',
                       help='Use shadowing.')
    group.add_argument('--shading-mode', type=str, default='normal',
                       choices=['matcap', 'rb', 'normal'],
                       help='Shading mode.')
    return group

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
    defaults_dict_args = {}
    
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
                if isinstance(parent_config_dict[key][field], dict):
                    defaults_dict_args[field] = parent_config_dict[key][field]
                    defaults_dict[field] = None
                else:
                    defaults_dict[field] = parent_config_dict[key][field]
 
    # Loads child parent and overwrite the parent configs
    # The yaml files assumes the argument groups, which aren't actually nested.
    for key in config_dict:
        for field in config_dict[key]:
            if field not in list_of_valid_fields:
                raise ValueError(
                    f"ERROR: {field} is not a valid option. Check for typos in the config."
                )
            if isinstance(config_dict[key][field], dict):
                defaults_dict_args[field] = config_dict[key][field]
                defaults_dict[field] = None
            else:
                defaults_dict[field] = config_dict[key][field]

    parser.set_defaults(**defaults_dict)
    return defaults_dict_args

def parse_default_dict(parser):
    default_dict_args = {}
    for group in parser._action_groups:
        for a in group._group_actions:
            if isinstance(a, ParseKwargs):
                default_dict_args[a.dest] = a.default
    return default_dict_args

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

    defaults_dict = parse_default_dict(parser)

    if args.config is not None:
        config_defaults_dict = parse_yaml_config(args.config, parser)
        for key, val in config_defaults_dict.items():
            if key in defaults_dict:
                defaults_dict[key].update(val)
            else:
                defaults_dict[key] = val
        args = parser.parse_args()

    for key, val in defaults_dict.items():
        cmd_line_val = getattr(args, key)
        if cmd_line_val is not None:
            val.update(cmd_line_val)
        setattr(args, key, val)
 
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str

def get_optimizer_from_config(optimizer_type):
    """Utility function to get the optimizer from the parsed config.
    """
    if optimizer_type not in str2optim:
        raise KeyError(f"{optimizer_type} is not a registered optimizer. "
                       f"Optimizers registered are: {list(str2.optim.keys())}.")
    optim_cls = str2optim[optimizer_type]
    if optimizer_type == 'adam':
        optim_params = {'eps': 1e-15}
    elif optimizer_type == 'sgd':
        optim_params = {'momentum': 0.8}
    else:
        optim_params = {}
    return optim_cls, optim_params

def get_pipeline_from_config(nef_type, nef_args, tracer_type, tracer_args,
                             pretrained, model_format):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nef = str2nef[nef_type](**nef_args)
    tracer = str2tracer[tracer_type](**tracer_args)
    pipeline = Pipeline(nef, tracer)

    if pretrained:
        if model_format == "full":
            pipeline = torch.load(pretrained)
        else:
            pipeline.load_state_dict(torch.load(pretrained))
    pipeline.to(device)
    return pipeline
