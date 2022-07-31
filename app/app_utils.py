# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import sys


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


def add_log_level_flag(parser):
    parser.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
