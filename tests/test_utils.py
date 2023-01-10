# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys
import shlex
import re
import logging
import subprocess
import json
import pytest
from collections import defaultdict


logger = logging.getLogger(__name__)

def run_wisp_script(cmd, cli_args):
    """ Run the wisp script and it's cli arguments as a subprocess and return the output.
        Assert if the return code indicated some error.
    """
    # Get current env python interpreter
    pyinterpreter = sys.executable
    # Get current env variables and override to headless mode
    env_vars = {**os.environ, **dict(PYTHONUNBUFFERED='1', WISP_HEADLESS='1')}
    # Split args str to list of tokens
    args = shlex.split(cli_args)

    logger.info(f'Executing Wisp Script: {cmd} {cli_args}')
    out = []
    with subprocess.Popen([pyinterpreter, cmd, *args],
                          env=env_vars,
                          cwd=os.getcwd(),
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          universal_newlines=True,
                          bufsize=1,
                          preexec_fn=os.setsid) as cp:
        for out_line in cp.stdout:  # Redirect stdout / stderr to stdout of main process, but also aggregate
            print(out_line, end='')
            out.append(out_line)
    logger.info(f'Script execution completed.')
    out = ''.join(out)

    if cp.returncode:
        raise RuntimeError(f'Pytest failed with the error code {cp.returncode}')

    return out


def _get_metric_from_log_line(line_with_metrics, metric_name):
    # Get all lines that contain the metric title
    # Break line to tokens by space and other separators
    tokens = re.split('/| |:|=', line_with_metrics)
    # Remove empty tokens, they're unhelpful
    tokens = list(filter(None, tokens))
    # Convert to lower, less brittle that way
    tokens = [t.lower() for t in tokens]
    try:
        epoch = tokens[tokens.index('epoch') + 1]
    except ValueError:
        epoch = -1
    metric = tokens[tokens.index(metric_name.lower()) + 1]

    return int(epoch), metric


def collect_metrics_from_log(out, metric_names):
    """ out is a string containing stdout,
        metric_names is a list of metric names to look for inside the log.

        This function assumes the metric name and it's value exist within the same line, possibly separated by some
        non-alphanumeric delimeter.
        If the epoch number is listed in the same line, it will be returned along with the metric.
        Otherwise, -1 will be returned (to signify the last occurrence of the metric in the log).
    """
    metrics = defaultdict(lambda: dict())
    for metric_name in metric_names:
        all_lines_with_criteria = list(filter(lambda line: metric_name in line, re.split('\n', out)))
        for line in all_lines_with_criteria:
            epoch, metric = _get_metric_from_log_line(line, metric_name)
            metrics[epoch][metric_name] = metric
    return metrics


def report_metrics(metrics):
    if metrics is not None:
        metrics = {(f'Epoch #{k}' if k != -1 else f'No epoch data'): v for k, v in metrics.items()}
        logger.info(f'Metrics:\n-----------\n{json.dumps(metrics, indent=4)}')


class TestWispApp:

    def setup_method(self, method):
        logger.info(f'Test {method.__name__} started.')

    def teardown_method(self, method):
        logger.info('Test completed.')
