# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import sys
import pprint


def default_log_setup(level=logging.INFO):
    """
    Sets up default logging, always logging to stdout.

    :param level: logging level, e.g. logging.INFO
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    # TODO: better to also use loggers per file and add %(name)15s
    logging.basicConfig(level=level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)


def args_to_log_format(args_dict) -> str:
    """Convert args hierarchy to string representation suitable for logging (i.e. with Tensorboard).

    Args:
        args_dict : The parsed arguments, grouped within a dictionary.

    Returns:
        arg_str : The args encoded in a string format.
    """
    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'
    return args_str