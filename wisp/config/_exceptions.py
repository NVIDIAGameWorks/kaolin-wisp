# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import tyro
import types
from functools import partial
from typing import Optional
import contextlib
import io


""" A module for handling help & exceptions in wisp's config system.
    Most of the logic here requires overriding the default behavior of libraries such as argparse & tyro.
"""


class InvalidCLISubcommand(Exception):
    """ An custom exception that rises when a CLI arg error have been identified by wisp,
    rather than some config library.
    """
    pass


class AmbiguousArgument(Exception):
    """ An custom exception that rises when a short CLI arg name is given, but wisp identifies it matches
    more than one config class. In this case, users should pass the full arg name including config class prefixes.
    This error is used to alert such ambiguities need to be taken care of.
    """
    pass


def handle_custom_errors(decorated_config_type) -> Optional[str]:
    """Under the hood, tyro uses argparse to validate the CLI arguments.
    Upon an error, the default behavior of argparse is to print a message to stderr and invoke a SystemExit.
    Since wisp makes heavy use of union configs (which amend to argparse subcommands), we wish to override these
    messages with structured ones that avoid the clutter and give proper instructions to users.

    Unfortunately argparse doesn't have a convenient mechanism to override error messages, so in the following,
    we're forced to be creative and instrument custom errors instead.

    Args:
        decorated_config_type (type): The type for the config object, possibly decorated with tyro markers.
            This config should appear exactly as it is fed into tyro.cli

    Returns:
        This function operates in 3 modes:
        1. Custom error messages this function knows how to handle:
            This function will raise custom error and print a message for errors with custom messages. It may also
            exit the program.
        2. Error messages this function recognizes, but cannot fully control:
            This function will return "hint" strings to add further information about upcoming errors tyro may trigger.
        3. Otherwise, for unhandled (non-custom) errors or valid arguments this function will return silently.
    """
    def _catchable_check_value(self, action, value, _check_value):
        """ Overrides argparse's _check_value function with our own custom version which wraps with try-except.
        Specifically, argparse uses this function to prompt an ArgumentError if an invalid choice is given to a
        subcommand. This error eventually propagates to argparse's error() message.
        Here, we replace ArgumentError with wisp's error type and custom message.
        """
        try:
            _check_value(action, value)  # _check_value is a method bound to some instance, no need to pass self.
        except argparse.ArgumentError as e:
            choices = [x.strip('{').strip('}') for x in e.argument_name.split(',')]
            subcommand = choices[0].split(':')[0]
            choices = [x.split(':')[1] for x in choices]
            message = f'Invalid configuration choice for "{subcommand}", expected to be any of the following:\n'
            message += '\n'.join([f'({idx+1}) {choice}' for idx, choice in enumerate(choices)])
            message += f'\n If using a YAML, set a "{subcommand}.constructor" entry to one of the choices above.'
            message += f'\n If using the CLI, add a subcommand "{subcommand}:[CHOICE]" to one of the choices above.'
            raise InvalidCLISubcommand(message) from e

    # The following block crawls through a replica of the main parser that tyro.cli uses under the hood.
    # Each of the subparser instances is instrumented with the custom method defined above, instead of the default one.
    # This makes sure that when parse_args() is called, our function will prompt the desired custom message.
    main_parser = tyro.extras.get_parser(decorated_config_type)
    parsers = [main_parser]
    while parsers:
        parser = parsers.pop(0)
        # This syntax defines binds with types.MethodType a new _check_value method to the parser instance.
        # We also pass the original _check_value function as an arg within the new function, to be able to invoke
        # this logic in a "super()" like syntax (there is no inheritance here, as we don't have access to the
        # ArgumentParser class which tyro initializes under the hood).
        # The usage of partial ensures we can capture the original parser._check_value and still keep the original
        # function signature on the caller side.
        parser._check_value = types.MethodType(partial(_catchable_check_value, _check_value=parser._check_value), parser)
        action_choices = [action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
        for choices in action_choices:
            for subcommand, subcommand_parser in choices.items():
                parsers.append(subcommand_parser)

    # Silence all error messages printed to std.err.
    # argparse's default behavior is to dump messages when an error occurs, but unless we have a custom error message
    # to handle, we prefer to let tyro handle the formatting (it does it better :)  )
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            # Invoke the parser just for the sake of handling custom error messages
            # If all args are valid, this line returns gracefully and we ignore the returned value
            # (we call this again via tyro to obtain the returned config instance)
            _, unknown_args = main_parser.parse_known_args()
            if unknown_args:
                return "\nThis means that the config system cannot handle some args you specified. " \
                       "Common reasons this happens: \n" \
                       "1. Are your arguments properly typed within the code? " \
                       "i.e: `foo(my_arg: int)` works but `foo(my_arg)` should not. \n" \
                       "2. Are your arguments types supported? e.g: int, str, float, lists, tuples, etc. " \
                       "For example, torch.Tensors and objects are not supported. " \
                       "Such args should be not be included in the yaml / cli. \n" \
                       "   Instead they should be passed when constructing objects with " \
                       "`instantiate(cfg, arg=that_unsupported_object).` \n" \
                       "3. Double check your arg names and categories for typos. \n" \
                       "4. Don't forget args are cAsE SeNsItIvE! \n"
        except InvalidCLISubcommand as e:
            # Custom error message raised. Restore std.err and call exit() via argparse to let it execute its default
            # behavior
            with contextlib.redirect_stderr(None):
                main_parser.exit(e)
        except SystemExit:
            # Let tyro handle any other error messages
            pass
    return None

"""
A module for parsing the CLI into concrete config dataclass instances.
tyro is used by wisp to consolidate configuration files and CLI args, and handle multiple config options
due to unions of configs generated by autoconfig().  
"""
class TyroFlatSubcommandHelpFormatter(tyro._argparse_formatter.TyroArgparseHelpFormatter):
    """
    A custom formatter that ensures --help prints arguments for all subcommands
    (tyro's default is to print for current / next subcommand).
    """
    def add_usage(self, usage, actions, groups, prefix=None):
        aggregated_subcommand_group = []
        for action_name, sub_parser in self.collect_subcommands_parsers(actions).items():
            for sub_action_group in sub_parser._action_groups:
                sub_group_actions = sub_action_group._group_actions
                if len(sub_group_actions) > 0:
                    is_subparser_action = lambda x: isinstance(x, argparse._SubParsersAction)
                    is_help_action = lambda x: isinstance(x, argparse._HelpAction)
                    if any([is_subparser_action(a) and not is_help_action(a) for a in sub_group_actions]):
                        aggregated_subcommand_group.append(sub_action_group)

        # Remove duplicate subcommand parsers
        aggregated_subcommand_group = list({a._group_actions[0].metavar: a
                                            for a in aggregated_subcommand_group}.values())
        next_actions = [g._group_actions[0] for g in aggregated_subcommand_group]
        actions.extend(next_actions)
        super().add_usage(usage, actions, groups, prefix)

    def add_arguments(self, action_group):
        if len(action_group) > 0 and action_group[0].container.title == 'subcommands':
            # If a subcommands action group - rename first subcommand (for which this function was invoked)
            choices_header = next(iter(action_group[0].choices))
            choices_title = choices_header.split(':')[0] + ' choices'
            action_group[0].container.title = choices_title
            self._current_section.heading = choices_title  # Formatter have already set a section, override heading

        # Invoke default
        super().add_arguments(action_group)

        aggregated_action_group = []
        aggregated_subcommand_group = []
        for action in action_group:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for action_name, sub_parser in self.collect_subcommands_parsers([action]).items():
                sub_parser.formatter_class = self
                for sub_action_group in sub_parser._action_groups:
                    sub_group_actions = sub_action_group._group_actions
                    if len(sub_group_actions) > 0:
                        is_subparser_action = lambda x: isinstance(x, argparse._SubParsersAction)
                        is_help_action = lambda x: isinstance(x, argparse._HelpAction)
                        if any([not is_subparser_action(a) and not is_help_action(a) for a in sub_group_actions]):
                            for a in sub_group_actions:
                                a.container.title = action_name + ' arguments'
                            aggregated_action_group.append(sub_action_group)
                        elif any([not is_help_action(a) for a in sub_group_actions]):
                            for a in sub_group_actions:
                                choices_header = next(iter(sub_group_actions[0].choices))
                                a.container.title = choices_header.split(':')[0] + ' choices'
                            aggregated_subcommand_group.append(sub_action_group)

        # Remove duplicate subcommand parsers
        aggregated_subcommand_group = list({a._group_actions[0].metavar: a
                                            for a in aggregated_subcommand_group}.values())
        for aggregated_group in (aggregated_subcommand_group, aggregated_action_group):
            for next_action_group in aggregated_group:
                self.end_section()
                self.start_section(next_action_group.title)
                self.add_text(next_action_group.description)
                super().add_arguments(next_action_group._group_actions)

    def collect_subcommands_parsers(self, actions):
        collected_titles = list()
        collected_subparsers = list()
        parsers = list()

        def _handle_actions(_actions):
            action_choices = [action.choices for action in _actions if isinstance(action, argparse._SubParsersAction)]
            for choices in action_choices:
                for subcommand, subcommand_parser in choices.items():
                    collected_titles.append(subcommand)
                    collected_subparsers.append(subcommand_parser)
                    parsers.append(subcommand_parser)

        _handle_actions(actions)
        while parsers:
            parser = parsers.pop(0)
            _handle_actions(parser._actions)

        # Eliminate duplicates and preserve order (dicts are guaranteed to preserve insertion order from python >=3.7)
        return dict(zip(collected_titles, collected_subparsers))


# A monkey patch to apply the custom help formatter on top of tyro's default one.
# Makes sure that all subparsers are printed with --help.
tyro._argparse_formatter.TyroArgparseHelpFormatter = TyroFlatSubcommandHelpFormatter
