# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import List, Tuple

white = (1.0, 1.0, 1.0)
black = (0.0, 0.0, 0.0)
dark_gray = (0.25, 0.25, 0.25)
light_purple = (0.788, 0.580, 1.0)
lime = (0.746, 1.0, 0.0)
red = (1.0, 0.0, 0.0)
green = (0.0, 1.0, 0.0)
blue = (0.0, 0.0, 1.0)
orange = (1.0, 0.5, 0.0)
light_cyan = (0.796, 1.0, 1.0)
light_pink = (1.0, 0.796, 1.0)
light_yellow = (1.0, 1.0, 0.796)
light_teal = (0.757, 1.0, 0.949)
gray = (0.5, 0.5, 0.5)
soft_blue = (0.721, 0.90, 1.0)
soft_red = (1.0, 0.0, 0.085)
lime_green = (0.519, 0.819, 0.0)
purple = (0.667, 0.0, 0.429)
gold = (1.0, 0.804, 0.0)


def color_wheel():
    """ Returns:
        (list) a list of all colors defined in the color module.
        Each entry is a tuple of 3 floats (RGB values).
    """
    return [
        white, black, dark_gray, light_purple, lime, red, green, blue, orange, light_cyan, light_pink,
        light_yellow, light_teal, gray, soft_blue, soft_red, lime_green, purple, gold
    ]


def colors_generator(skip_colors: List = None) -> Tuple[float, float, float]:
    """ Generates the next color in the color wheel on each invocation.
        This generator repeats the color wheel cyclically when exhausted.

    Args:
        skip_colors
    """
    if skip_colors is None:
        skip_colors = []
    while True:
        for color in color_wheel():
            if color in skip_colors:
                continue
            yield color
