# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

# Closest point function + texture sampling
# https://en.wikipedia.org/wiki/Closest_point_method

import torch
import numpy as np

def closest_point(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor):

    assert False and "This function is not supported in the main branch. Please use ttakikawa/pysdf for this function."


