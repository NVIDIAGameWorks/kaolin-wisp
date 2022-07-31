# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np

class BaseTracer(object):
    """Virtual base class for tracer"""

    def __init__(self,
        step_size     : float = 1.0,
        num_steps     : int   = 64, # samples for raymaching, iterations for sphere trace
        min_dis       : float = 1e-3,
        bg_color      : str   = 'white',
        raymarch_type : str   = 'voxel',
        **kwargs): 

        self.step_size = step_size
        self.num_steps = num_steps
        self.min_dis = min_dis
        self.bg_color = bg_color
        self.raymarch_type = raymarch_type

        self.inv_num_steps = 1.0 / self.num_steps
        self.diagonal = np.sqrt(3) * 2.0
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Base implementation for forward"""
        raise NotImplementedError
