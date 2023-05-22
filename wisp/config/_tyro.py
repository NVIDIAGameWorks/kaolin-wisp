# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os, sys
import re
import yaml
import itertools
from typing_extensions import Annotated
from typing import List, Set, Dict, Optional
from collections import defaultdict
import dataclasses
import argparse
import tyro
from ._exceptions import handle_custom_errors, AmbiguousArgument



def parse_args_tyro(config_type, yaml_arg: Optional[str]='--config'):
    """Parse args from a config dataclass.

    args = parse_args_tyro(AppConfig)

    Args:
        config_type (type): The type for the config object.
        yaml_arg (str): Name of config path arg. By default this is --config. Expected to start with --.
            If None, no configuration yaml is expected.

    Returns:
        (dataclass): Parsed config dataclass object.
    """
    # Configure tyro's options.
    # ----------------------------
    # tyro.conf.AvoidSubcommands:
    #   Avoid creating subcommands when a default is provided for unions over nested types.
    # tyro.conf.ConsolidateSubcommandArgs:
    #   More robust to reordering of options, ensuring that any new options can simply be placed at the end
    #   of the command. i.e. allows the latter instead of the former:
    #   `main.py grid:hash --grid.arg1 foo --grid.arg2 bar dataset:nerf --dataset.root baz
    #   `main.py grid:hash dataset:nerf --grid.arg1 foo --dataset.root baz --grid.arg2 bar
    # tyro.conf.FlagConversionOff
    #   Required to support both optional and non-optional boolean args
    # tyro.conf.SuppressFixed
    #   Hides fields which are marked as fixed (i.e. predetermined value in dataclass).
    #   Useful for hiding away zen_meta fields from hydra-zen.
    tyro_markers = [
        tyro.conf.AvoidSubcommands,
        tyro.conf.ConsolidateSubcommandArgs,
        tyro.conf.FlagConversionOff,
        tyro.conf.SuppressFixed,
    ]
    apply_tyro_markers = tyro.conf.configure(*tyro_markers)
    decorated_config_type = apply_tyro_markers(config_type)

    # If a config file has been specified, pop it from sys.argv and return the path.
    path = None
    if yaml_arg is not None:
        path = find_config_file(sys.argv, yaml_arg)

    cli_subcommand_pos, cli_args = _collect_cli_args_and_subcommands()
    cli_mapping = _resolve_shortened_arg_names(config_type, cli_args)

    config_subcommands, config_args = None, None
    # If a config file is passed in, use as default.
    if path is not None:
        assert os.path.exists(path), f'Invalid configuration file path: "{path}". Please review your {yaml_arg} arg.'
        config_subcommands, config_args = load_config(path)

    # Rebuild sys.argv here in a format that satisfies tyro's conditions for subcommands and full arg names
    _reform_sys_argv(config_type, config_subcommands, config_args, cli_subcommand_pos, cli_args, cli_mapping)

    # If printing help, remove the ConsolidateSubcommandArgs marker which disorients the custom formatter
    additional_info = None
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        if tyro.conf.ConsolidateSubcommandArgs in tyro_markers:
            del tyro_markers[tyro_markers.index(tyro.conf.ConsolidateSubcommandArgs)]
            apply_tyro_markers = tyro.conf.configure(*tyro_markers)
            decorated_config_type = apply_tyro_markers(config_type)
    else:
        # Use tyro's underlying parser to handle any cases we have custom help messages for.
        # If an error with a custom message occurs, this lines exits the program.
        # Otherwise, this line returns gracefully and we let tyro take care of the rest.
        additional_info = handle_custom_errors(decorated_config_type)

    try:
        args = tyro.cli(decorated_config_type)
    except SystemExit as e:
        if additional_info:
            sys.stderr.write(additional_info)
        raise e

    return args


def find_config_file(argv, yaml_arg='--config'):
    """ Helper function to find the config path from argv.
    This allows to extract a yaml file path from the CLI args, for additional configuration to the CLI.

    This function also removes the --config flag from argv.

    Args:
        argv (List[str]): The `sys.argv` list.
        yaml_arg (str): The name of the config path field. Expected to start with --.

    Returns:
        (str): Path to the config file. Returns None if no config is specified.
    """
    for i, arg in enumerate(argv):
        arg = arg.split('=')
        if arg[0] == yaml_arg:
            if len(arg) == 1:
                if i < len(argv) - 1 and not argv[i + 1].startswith("--"):
                    path = argv[i + 1]
                    argv.pop(i)
                    argv.pop(i)
                else:
                    raise Exception(f"No config file passed in after {yaml_arg}.")
            else:
                path = arg[1]
                argv.pop(i)
            return path
    return None


def load_config(path):
    """Loads a yaml config file, and returns the content in the format of CLI args.
    This is useful to enhance CLI args with additional config from the yaml.

    This function also preprocesses the args to add "--" prefix and replace the "_" by "-".

    Args:
        path (str): path to the config file

    Returns:
        (dict, dict):

            - the subcommands as mapping of "group" to "subcommand"
            - the args as a mapping of "full key" (group.arg) to "value"
    """
    with open(path, 'r') as f:
        config_d = yaml.safe_load(f)
    # Collect subcommands and args from config
    config_args = {}
    config_subcommands = {}

    def _add_to_config_args(k, v):
        """ This function reads a single entry and adds it to the config_args dict.
        k is the field name, including all parent prefixes.
        v is the yaml value of the field.
        """
        if isinstance(v, list):
            v = [str(i) for i in v]  # Lists are handled as separate args so don't squash to a single arg
        else:
            v = str(v)
        config_args[f'--{k}'] = v  # Otherwise map k:v

    def _recursive_arggroups(_group, _group_name):
        """ Recursively crawl the yaml.
        The current recursion is looking at the parameters of a given group inside a yaml.
        _group_name:
            ... -
            ...  |
            ...  }-- _group
            ...  |
            ... -
        """
        if isinstance(_group, dict):
            # New hierarchy, iterate of of the items and crawl further if needed
            for k, v in _group.items():
                # constructors are handled differently: we keep them aside to convert them to subcommands.
                # This tells tyro which of the Config options it should pick when parsing.
                if k == "constructor":
                    config_subcommands[_group_name] = v.replace("_", "-")
                else:
                    # Stick to argparse conventions: arg_name with arg-name
                    k = k.replace('_', '-')
                    if isinstance(v, dict):
                        _recursive_arggroups(_group=v, _group_name=f'{_group_name}.{k}')
                    else:
                        _add_to_config_args(k=_group_name + "." + k, v=v)
        else:
            # group is a single entry of k: v, simply add it to config_args
            _add_to_config_args(k=_group_name, v=group)

    # Go over each of the entries in the yaml and crawl them recursively with `_recursive_arggroups`
    for group_name, group in config_d.items():
        group_name = group_name.replace('_', '-')
        _recursive_arggroups(group, group_name)
    return config_subcommands, config_args


def write_config_to_yaml(config, path: str):
    """Writes config to path in yaml format.

    Usage:
        write_config_to_path(config_object, "config.yaml")

    Args:
        config (dataclass): Dataclass config.
        path (str): Path to write the config file to.
    """
    config_dict = dataclasses.asdict(config)

    # This is needed since tyro will by default use tuples instead of lists,
    # but yaml writers expect lists for proper behaviour.
    def _replace_tuple(_dict):
        for key in _dict:
            item = _dict[key]
            if isinstance(item, dict):
                _replace_tuple(item)
            elif isinstance(item, tuple):
                _dict[key] = list(item)

    _replace_tuple(config_dict)

    with open(path, 'w') as outfile:
        yaml.dump(config_dict, outfile)


def _collect_cli_args_and_subcommands():
    """ Collects the subcommand and arg names from the CLI, this mapping is later used to restructure sys.argv in
        a format that satisfies tyro & argparse.
    """
    # Load the script arguments
    cli_args = set()
    cli_subcommand_pos = {}
    # Start by collecting the real CLI arguments first, these take precedence over everything else
    for i, a in enumerate(sys.argv[1:], 1):
        if a.startswith('--'):
            # args
            t = a.split('=')
            k = t[0]    # whether this is --arg=value or --arg value, t[0] should give us "arg"
            key = k.replace('_', '-')
            cli_args.add(key)
        elif ':' in a and \
                '\'' not in a and \
                '"' not in a:
            # subcommand
            t = a.split(':')
            key = t[0].replace('_', '-')
            cli_subcommand_pos[key] = i
        else:
            # expected to be an arg value
            continue
    return cli_subcommand_pos, cli_args


def _reform_sys_argv(config_type,
                     config_subcommands, config_args, cli_subcommand_pos, cli_args, cli_mapping,
                     subcommands_override_args=False):
    """ Restructures sys.argv in a format valid for argparse & tyro.
    This function delicately takes the existing CLI args and subcommands, and the flattened yaml args and subcommands,
    and resolves both together to construct a final sys.argv which allows a more flexible format than the default
    argparse.
    This function will:
    1. Iterate all CLI args, and replace them with their full, unambiguous names.
        i.e: --num-workers   --->   --dataset.num-workers
        i.e. --lr            --->   --trainer.optimizer.lr
      This works only if the full names are not ambiguous (i.e. if num-workers matches more than 1 prefix, we fail here)
    2. Pops all existing CLI subcommands, and reinjects them at the right order.
    3. Appends the yaml subcommands to sys.argv
    4. Appends the yaml args to sys.argv in a flat manner, this allows wisp to support special fields also, like
       the "constructor" field.
    """
    # --- (1) ---
    # Starting reforming sys.argv for tyro.
    # First we map all args from the CLI to their full name, including prefixes
    for idx, arg in enumerate(sys.argv):
        arg_name = arg.split('=')[0]
        if arg_name in cli_mapping and arg_name in cli_args:
            sys.argv[idx] = arg.replace(arg_name, cli_mapping[arg_name])
    cli_args = set([cli_mapping[arg] if arg in cli_mapping else arg for arg in cli_args])

    # --- (2) ---
    # Iterate existing CLI subcommands - possibly users have listed them in a wrong order that tyro and argparse
    # wouldn't like.
    # subcommand_list is the list of all subcommand names this program expects
    subcommand_list = list_subcommands(config_type, include_options=False)
    # subcommands_order maps subcommand name to its expected order in the CLI
    subcommands_order = {k: i for i, k in enumerate(subcommand_list)}
    # Sort the existing CLI subcommands according to the priority above
    sorted_subcommands = sorted(cli_subcommand_pos, key=lambda x: subcommands_order[x])
    # Since we were dealing with subcommand names, and we want their chosen values as well, fetch them from argv here
    full_subcommands = {k: sys.argv[v] for k, v in cli_subcommand_pos.items()}
    full_sorted_subcommands = [full_subcommands[key] for key in sorted_subcommands]
    # Pop all existing subcommands from sys.argv
    for subcmd_idx in sorted(cli_subcommand_pos.values(), reverse=True):
        sys.argv.pop(subcmd_idx)
    # And now reinject those full subcommands in the right order
    for idx in range(len(full_sorted_subcommands)):
        sys.argv.insert(1 + idx, full_sorted_subcommands[idx])

    # If no config file loaded - quit here
    if config_subcommands is None or config_args is None:
        return

    # --- (3) ---
    # Next we list the subcommands to resolve the autoconfig notation - this tells tyro which Config class
    # to load.
    next_insertion_idx = 1
    ignored_group = set()
    cli_subcommand_pos = {k: idx for idx, k in enumerate(sorted_subcommands)}   # Use updated sorted mapping
    for i, group in enumerate(subcommand_list):
        next_insertion_idx = i + 1
        if group in cli_subcommand_pos:
            if subcommands_override_args:
                ignored_group.add(group)
        elif group in config_subcommands:
            sys.argv.insert(next_insertion_idx, group + ":" + config_subcommands[group])

    # --- (4) ---
    # Finally we take all arguments collected from the yaml and append them to sys.argv
    # By doing so we feed the yaml into tyro indirectly, with better control over it's parsing
    # (i.e. we can support the special "constructor" field)
    for key, arg in config_args.items():
        if (key not in cli_args) and (key.split('.')[0][2:] not in ignored_group):
            # List args are added with spaces
            if isinstance(arg, list):
                sys.argv.append(key)
                for a in arg:
                    sys.argv.append(a)
            else:
                # Most args are added to argv as "--arg=val"
                sys.argv.append(key + "=" + arg)


def annotate_subcommand(config_type, use_ctor_name=True):
    """ Annotates Union configs with tyro markers that make them "named" subcomands.
        That is - dataclasses with a field of Union[conf1, conf2, ...] can select a conf from the CLI or yaml config
        by annotating them as subcommands here.
    """
    if config_type.__name__.startswith('Config'):
        # config_type.__name__ is expected to be in the form "ConfigThisSpecificClassName".
        # We want to convert it into a form of: this-specific-class-name
        shorter_name = config_type.__name__[len('Config'):]
        if use_ctor_name:
            # Use __ctor_name__ field to extract classname or classname + function
            shorter_name = config_type.__ctor_name__
        else:
            # Split into tokens of:
            # - A capital letter followed by one or more lower-case letters.
            # - One or more capital letters not followed by a lower-case letter.
            # - One or more lower-case letters.
            subcommand_tokens = re.findall('[A-Z][a-z]+|[A-Z]+(?![a-z])|[a-z]+', shorter_name)
            # Convert all to lowercase and join with a dash
            shorter_name = '-'.join([w.lower() for w in subcommand_tokens])
        return Annotated[config_type, tyro.conf.subcommand(name=shorter_name, prefix_name=True)]
    else:
        # config_type.__name__ is NOT of the form "ConfigThisSpecificClassName", so just pass it as it is
        return config_type


def _resolve_shortened_arg_names(config_type, cli_args: Set[str]) -> Dict[str, str]:
    """Since configs are nested, tyro expect args to be specified with a full prefix of all nested config classes.
    For brevity, wisp support short arg names in the CLI if the arg name is unique and matches a single config
    prefix.
    This function collects all legal argnames and safely maps the shortened CLI arg names to their full nested name.
    If the arg is ambiguous, an error is raised to alert users with options to choose from.

    For example:
        @dataclass
        class Config:
            foo: ConfigFoo

        @dataclass
        class ConfigFoo:
            bar: ConfigBar
            baz: ConfigBaz

        @dataclass
        class ConfigBar:
            arg1: int
            arg2: int

        @dataclass
        class ConfigBaz:
            arg2: str

        `--arg1` will be mapped to foo.bar.arg1
        `--arg2` will prompt an error to choose between --foo.bar.arg2, --foo.baz.arg2

    Args:
        config_type: A config dataclass.
        cli_args: List of CLI arguments, obtained from sys.argv in a clean manner,
        usually using `_collect_cli_args_and_subcommands()`.
    Returns:
        A mapping of args short names to their nested subcommand prefixes.
    """
    arg_mapping = list_args(config_type)
    legal_args = set(itertools.chain(*arg_mapping.values()))
    args_with_prefixes = dict()
    for arg in cli_args:
        stripped_arg = arg[2:] if arg.startswith('--') else arg
        if stripped_arg in legal_args:
            args_with_prefixes[arg] = arg
        else:
            possible_mappings = arg_mapping[stripped_arg]
            if len(possible_mappings) == 0:
                # This is either a reserved argument (i.e. --help), or a bad arg unsupported by this configuration.
                # Either way, don't do anything here, let tyro handle that eventually
                pass
            elif len(possible_mappings) > 1:
                raise AmbiguousArgument(f'The configuration parser received the shortened arg name: {arg}, '
                                        f'but this arg matches more than one config class: {possible_mappings}.\n'
                                        f'Please resolve the ambiguity by passing a full arg prefix which matches '
                                        f'one of these options.')
            else:
                args_with_prefixes[arg] = f'--{possible_mappings[0]}' if stripped_arg != arg else possible_mappings[0]
    return args_with_prefixes


def list_subcommands(config_type, include_options: bool = True) -> List[str]:
    """Recursively traverses through tyro's parser -> nested subparsers for the given config_type schema.
    This function collects all subcommands it finds, in the order they're encountered.

    Args:
        config_type: A config dataclass, fields may contain unions of types,
            which will trigger tyro to define subcommands.
        include_options:
            If True, subcommands and choices will be returned, i.e. ['foo:fromA', 'foo:fromB', 'bar:fromC']
            If False, only the subcommands are returned, i.e ['foo', 'bar']

    Returns:
        A list of subcommands (+ choices, optionally) in the order they're first encountered. Duplicates are eradicated.
    """
    subcommands = list()
    parsers = [tyro.extras.get_parser(config_type)]

    while parsers:
        parser = parsers.pop(0)
        action_choices = [action.choices for action in parser._actions
                          if isinstance(action, argparse._SubParsersAction)]
        for choices in action_choices:
            for subcommand, subcommand_parser in choices.items():
                subcommands.append(subcommand)
                parsers.append(subcommand_parser)

    # Remove choices from subcommands if include_options=False
    if not include_options:
        subcommands = [subc.split(':')[0] for subc in subcommands]

    # Eliminate duplicates and preserve order (dicts are guaranteed to preserve insertion order from python >=3.7)
    subcommands = list(dict.fromkeys(subcommands))
    return subcommands


def list_args(config_type) -> Dict[str, List[str]]:
    """Recursively traverses through tyro's parser -> nested subparsers for the given config_type schema.
    This function collects all argnames it finds, with their subcommand prefixes, in the order they're encountered.

    For example:
        @dataclass
        class Config:
            foo: ConfigFoo

        @dataclass
        class ConfigFoo:
            bar: ConfigBar
            baz: ConfigBaz

        @dataclass
        class ConfigBar:
            arg1: int
            arg2: int

        @dataclass
        class ConfigBaz:
            arg2: str

        will return:
            arg1: [foo.bar.arg1]                 <-- unique arg name.
            arg2: [foo.bar.arg2, foo.baz.arg2]   <-- conflict resolved by prefixes!

    Args:
        config_type: A config dataclass, fields may contain unions of types,
            which will trigger tyro to define subcommands.
    Returns:
        A mapping of args short names to their nested subcommand prefixes,
        in the order they're first encountered. Duplicates are eradicated.
    """
    parsers = [tyro.extras.get_parser(config_type)]
    args = []

    while parsers:
        parser = parsers.pop(0)
        args.extend([action.dest for action in parser._actions
                     if not isinstance(action, argparse._SubParsersAction) and
                     not isinstance(action, argparse._HelpAction)])
        action_choices = [action.choices for action in parser._actions
                          if isinstance(action, argparse._SubParsersAction)]
        for choices in action_choices:
            for subcommand, subcommand_parser in choices.items():
                parsers.append(subcommand_parser)

    # Eliminate duplicates and preserve order (dicts are guaranteed to preserve insertion order from python >=3.7)
    args = list(dict.fromkeys(args))

    # Create a mapping of short arg name to full prefix paths:
    #  arg1: [foo.bar.arg1]                 <-- unique arg name.
    #  arg2: [foo.bar.arg2, foo.baz.arg2]   <-- conflict resolved by prefixes!
    mapping = defaultdict(list)
    for a in args:
        short_name = a.split('.')[-1]
        mapping[short_name].append(a)

    return mapping
