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
import yaml
import inspect
import torch
from typing import Dict, List, Any

from wisp.models import nefs
from wisp.models import grids
from wisp import tracers
from wisp import datasets

# This file contains all the configuration and command-line parsing general to all app

__all__ = [
    'list_modules',
    'register_module',
    'get_module',
    'get_args_for_function',
    'get_grouped_args',
    'parse_args'
]

# str2mod ("str to module") are registered wisp blocks the config parser is aware of, and is able to dynamically load.
# You may register additional options by adding them to the dictionary here.
# ConfigParser expects the following default module categories:
str2mod = {
    'optim': {m.lower(): getattr(torch.optim, m) for m in dir(torch.optim) if m[0].isupper()},
    'dataset': {},
    'nef': {},
    'grid': {},
    'tracer': {}
}


def list_modules(module_type) -> List[str]:
    """ Returns a list of all available modules from a certain category.
    Args:
        type: a str from the following categories: ['nef', 'grid', 'tracer', 'dataset', 'optim']
    """
    return list(str2mod[module_type].keys())


def register_module(module_type, name, mod):
    """Register module to be used with config parser.
    Users should use this class to load their classes by name.
    """
    if module_type not in str2mod:
        raise ValueError(f"'{module_type}' is an unknown type")

    if name in str2mod[module_type]:
        raise KeyError(f"'{name}' already exist in type '{module_type}'")
    str2mod[module_type][name] = mod


def get_module(name, module_type=None):
    """Get module class by name, assuming it was registered with `register_module`.'"""
    types_to_check = []
    if module_type is None:
        types_to_check = str2mod
    else:
        if module_type not in str2mod:
            raise ValueError(f"'{module_type}' is an unknown type")
        types_to_check.append(module_type)

    for t in types_to_check:
        if name in str2mod[t]:
            return str2mod[t][name]

    raise ValueError(f"'{name}' is not a known module for any of the types '{types_to_check}'. "
                     f"registered modules are '{str2mod[module_type].keys()}'")


def get_args_for_function(args, func):
    """ Given a func (for example an __init__(..) function or from_X(..)), and also the parsed args,
    return the subset of args that func expects and args contains. """
    has_kwargs = inspect.getfullargspec(func).varkw != None
    if has_kwargs:
        collected_args = vars(args)
    else:
        parameters = dict(inspect.signature(func).parameters)
        collected_args = {a: getattr(args, a) for a in parameters if hasattr(args, a)}
    return collected_args


def get_grouped_args(parser, args) -> Dict[str, Any]:
    """Group args to a grouped hierarchy.

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.
        args : The parsed arguments. Will compute from the parser if None.

    Returns:
        args    : The parsed arguments.
        arg_str : The string to be printed.
    """
    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))
    return args_dict


# -- Register all wisp library modules here -- this makes them loadable by specifying the class name to get_module() --
for name in dir(datasets):
    mod = getattr(datasets, name)
    if isinstance(mod, type) and \
            issubclass(mod, torch.utils.data.Dataset) and \
            mod != torch.utils.data.Dataset:
        register_module('dataset', name, mod)

for name in dir(nefs):
    mod = getattr(nefs, name)
    if isinstance(mod, type) and \
            issubclass(mod, nefs.BaseNeuralField) and \
            mod != nefs.BaseNeuralField:
        register_module('nef', name, mod)

for name in dir(grids):
    mod = getattr(grids, name)
    if isinstance(mod, type) and \
            issubclass(mod, grids.BLASGrid) and \
            mod != grids.BLASGrid:
        register_module('grid', name, mod)

for name in dir(tracers):
    mod = getattr(tracers, name)
    if isinstance(mod, type) and \
            issubclass(mod, tracers.BaseTracer) and \
            mod != tracers.BaseTracer:
        register_module('tracer', name, mod)

try:
    import apex

    for m in dir(apex.optimizers):
        if m[0].isupper():
            register_module('optim', m.lower(), getattr(apex.optimizers, m))
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Cannot import apex for fused optimizers")


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


def parse_args(parser) -> argparse.Namespace:
    """Parses args by priority into a flat configuration.
    The various options take the following precedence:
    1. CLI args, explicitly specified
    2. YAML configuration, defined with `--config <PATH>.yaml`
    3. argparse defaults

    Args:
        parser (argparse.parser): Parser object. Needed for the argument groups.

    Returns:
        args    : The parsed arguments, as a flat configuration.
    """

    args = parser.parse_args()
    defaults_dict = dict()

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

    return args
