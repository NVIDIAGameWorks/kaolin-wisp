# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Callable, Any
import torch
import torch.nn.functional as F


BlendFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""
Blend functions receive: channel_a, channel_b, alpha_a, alpha_b, and return a blended channel value.
Channel A is assumed to be in front of channel B
"""

NormalizeFunction = Callable[[torch.Tensor, Any, Any], torch.Tensor]
"""
Normalization functions receive a channel, and possible min / max values acceptable for this channel type.
These functions output the normalized values for the channel in the range [0, 1]
"""

###############################################################################################################
#                                           How to use channels:                                              #
###############################################################################################################
#  Blending:                                                                                                  #
#  - Channel dimensions are flexible and assumed to be (..., C), where C is the number of channel dimensions. #
#  - Channel dtypes are assumed to be floating points (of arbitrary precision)                                #
#  - Alpha values are assumed to be in the range of [0, 1].                                                   #
#  - TODO (operel): add support for premultiplied alphas /w Porter Duff operators                             #
#                                                                                                             #
#  Normalization:                                                                                             #
#  - Channels may define a min value, a max value, none of them or both. It is up to the normalization        #
#    function to query for the existence of such boundaries. If any of them is None, a fallback should        #
#    exist.                                                                                                   #
#  - A normalized channel is constrains values to the range [0, 1].                                           #
###############################################################################################################


###############################################################################################################
#                                       Normalization Functions:                                              #
###############################################################################################################

def identity(c: torch.Tensor) -> torch.Tensor:
    """ A naive normalization function which assumes the value is already normalized and returned as is.

    Args:
        c (torch.Tensor): A single channel tensor of an arbitrary shape.

    Returns:
        (torch.Tensor): Input channel c is returned without a change.

    """
    return c


def normalize(c: torch.Tensor, min_val: Any = None, max_val: Any = None) -> torch.Tensor:
    """ A linear normalization function which maps the channel c to the range of [0, 1].
    If the min / max values bounds of the channel are not explicitly specified, they're determined by
    c's values.

    Args:
        c (torch.Tensor): A single channel tensor of an arbitrary shape.
        min_val (Any): Optional, the lower boundary for possible values in channel C
        max_val (Any): Optional, the upper boundary for possible values in channel C

    Returns:
        (torch.Tensor): Input channel c is linearly mapped to the range [0, 1].

    """
    min_val = torch.min(c) if min_val is None else min_val
    max_val = torch.max(c) if max_val is None else max_val
    return (c - min_val) / (max_val - min_val)


def normalize_linear_scale(c: torch.Tensor, min_val: Any = None, max_val: Any = None,
                           linear_scale: float = 1.0) -> torch.Tensor:
    """ A normalization function which linear scales the channel before normalizing it to the range of [0, 1].
    If the min / max values bounds of the channel are not explicitly specified, they're determined by
    c's values. If explicitly specified, the bounds are scaled as well.

    Args:
        c (torch.Tensor): A single channel tensor of an arbitrary shape.
        min_val (Any): Optional, the lower boundary for possible values in channel C
        max_val (Any): Optional, the upper boundary for possible values in channel C
        linear_scale (float): Channel scale value

    Returns:
        (torch.Tensor): Input channel c is linearly mapped to the range [0, 1].

    """
    c *= linear_scale
    min_val = linear_scale * min_val if min_val is not None else min_val
    max_val = linear_scale * max_val if max_val is not None else max_val
    return normalize(c=c, min_val=min_val, max_val=max_val)


def normalize_log_scale(c: torch.Tensor, min_val: Any = None, max_val: Any = None,
                        linear_scale: float = 1.0, log_scale: float = 1.0) -> torch.Tensor:
    """ A normalization function which applies log and linear scales to the channel before normalizing it
    to the range of [0, 1].
    If the min / max values bounds of the channel are not explicitly specified, they're determined by
    c's values. If explicitly specified, the bounds are scaled as well.

    Args:
        c (torch.Tensor): A single channel tensor of an arbitrary shape.
        min_val (Any): Optional, the lower boundary for possible values in channel C
        max_val (Any): Optional, the upper boundary for possible values in channel C
        linear_scale (float): Channel linear scale value
        log_scale (float: Channel log scale value.

    Returns:
        (torch.Tensor): Input channel c is mapped to the range [0, 1] by the formula:
            ``c' = a * log(b * c)`` where a is the linear scale, b is the log scale and c is the channel value.

    """
    c = linear_scale * torch.log(log_scale * c)
    min_val = linear_scale * torch.log(log_scale * min_val) if min_val is not None else min_val
    max_val = linear_scale * torch.log(log_scale * max_val) if max_val is not None else max_val
    return normalize(c=c, min_val=min_val, max_val=max_val)


def normalize_vector(c: torch.Tensor) -> torch.Tensor:
    """ A normalization function which applies a L2 normalization over a channel of vector data.

    Args:
        c (torch.Tensor): A single channel tensor of an arbitrary shape.

    Returns:
        (torch.Tensor): Input channel c is normalized by the L2 norm.
    """
    return F.normalize(c, dim=-1)


###############################################################################################################
#                                           Blending Functions:                                               #
###############################################################################################################


def blend_linear(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ A direct linear interpolation between c1 and c2.
    Useful for blending channels which do not consider the alpha value (i.e. the alpha channel itself).

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused
        alpha2 (torch.Tensor): Unused

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1 + c2 * (1.0 - c1)


def blend_alpha_composite_over(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ An alpha compositing op where a front pixel is alpha blended with the background pixel
    (in a usual painter's algorithm manner).
    Useful for blending channels such as RGB.
    See: https://en.wikipedia.org/wiki/Alpha_compositing

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): alpha channel tensor, corresponding to first channel, in the shape of c1.
        alpha2 (torch.Tensor): alpha channel tensor, corresponding to second channel, in the shape of c1.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    alpha_out = alpha1 + alpha2 * (1.0 - alpha1)
    c_out = torch.where(condition=alpha_out > 0,
                        input=(c1 * alpha1 + c2 * alpha2 * (1.0 - alpha1)) / alpha_out,
                        other=torch.zeros_like(c1))
    return c_out


def blend_alpha_lerp(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ A linear interpolation between c1 and c2, which uses the alpha channel as a weighting factor.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): alpha channel tensor, corresponding to first channel, in the shape of c1.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1 * alpha1 + c2 * (1.0 - alpha1)


def blend_alpha_slerp(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ A spherical linear interpolation, useful for interpolating rotations or blending directional vectors.
    c1 and c2 are normalized and interpolated over the unit hypersphere.
    alpha1 acts as the interpolation weight.
    See: https://en.wikipedia.org/wiki/Slerp

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): alpha channel tensor, corresponding to first channel, in the shape of c1.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    # To unit directions
    t = alpha1  # alpha1 is used as the interpolation weight
    c1 = F.normalize(c1, dim=1)
    c2 = F.normalize(c2, dim=1)
    dot = (c1*c2).sum(1)         # batched dot prod
    omega = torch.acos(dot)      # angle between directions
    sin_omega = torch.sin(omega)    # TODO (operel): Be careful of omega=0.0 case
    c2_weight = (torch.sin((1.0 - t) * omega) / sin_omega).unsqueeze(1)
    c1_weight = (torch.sin(t * omega) / sin_omega).unsqueeze(1)
    res = c2_weight * c2 + c1_weight * c1
    return res


def blend_normal(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ A standard blend mode which uses the front pixel value, without mixing.
        Useful when alpha blending is undesired, or the channel contains categorical info (i.e. semantic class ids).

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): Unused
        alpha1 (torch.Tensor): Unused
        alpha2 (torch.Tensor): Unused

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1


def blend_multiply(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ Commutative blend mode which preserves dark colors.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1 * c2


def blend_screen(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ Commutative blend mode which preserves light colors.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return 1.0 - (1.0 - c1)(1.0 - c2)


def blend_add(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ An additive blend mode, for aggregation of channel information.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1 + c2


def blend_sub(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ An subtractive blend mode, for removing channel information.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return c1 - c2


def blend_logical_and(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ For boolean channels, blends with a logical AND function.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return torch.logical_and(c1, c2).to(c1.dtype)


def blend_logical_or(c1: torch.Tensor, c2: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):
    """ For boolean channels, blends with a logical OR function.

    Args:
        c1 (torch.Tensor): first channel tensor of an arbitrary shape.
        c2 (torch.Tensor): second channel tensor, in the shape of c1.
        alpha1 (torch.Tensor): Unused.
        alpha2 (torch.Tensor): Unused.

    Returns:
        (torch.Tensor): Blended channel in the shape of c1
    """
    return torch.logical_or(c1, c2).to(c1.dtype)
