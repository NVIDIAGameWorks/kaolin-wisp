# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from wisp.core.channel_fn import *
from dataclasses import dataclass
from typing import Any, Optional, Dict
from functools import partial


@dataclass
class Channel:
    """ Defines how a Renderbuffer channel should behave in terms of functionalities like blending, normalization,
        and boundaries.
    """

    blend_fn: BlendFunction = None
    """" How to blend information from this channel between 2 RenderBuffers """

    normalize_fn: NormalizeFunction = normalize
    """ How to normalize the channel information to a scale of [0, 1] """

    min_val: Optional[Any] = None
    """ Minimal valid value supported by this channel type. None indicates the valid values range from -inf. """

    max_val: Optional[Any] = None
    """ Maximal valid value supported by this channel type . None indicates the valid values range to inf. """


def create_default_channel() -> Channel:
    """ A general channel template, to be used if no information about a channel have been recorded """
    return Channel(
        blend_fn=blend_alpha_composite_over,
        normalize_fn=normalize,
        min_val=None,
        max_val=None
    )


def channels_starter_kit() -> Dict[str, Channel]:
    """ Creates a predefined kit of channels commonly useful in the context of Wisp.
        Users may augment or replace this kit with additional custom channels.
    """
    return dict(
        rgb=Channel(
            blend_fn=blend_alpha_composite_over,
            normalize_fn=identity,
            min_val=0.0,
            max_val=1.0
        ),
        alpha=Channel(
            blend_fn=blend_linear,
            normalize_fn=normalize,
            min_val=0.0,
            max_val=1.0
        ),
        depth=Channel(
            blend_fn=blend_normal,
            normalize_fn=partial(normalize_linear_scale, linear_scale=1000.0),
            min_val=0.0
        ),
        normal=Channel(
            blend_fn=blend_alpha_slerp,
            normalize_fn=normalize_vector
        ),
        hit=Channel(
            blend_fn=blend_logical_or,
            normalize_fn=identity
        ),
        err=Channel(
            blend_fn=blend_add,
            normalize_fn=normalize
        ),
        gt=Channel(
            blend_fn=blend_alpha_composite_over,
            normalize_fn=identity,
            min_val=0.0,
            max_val=1.0
        )
    )
