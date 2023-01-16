# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.datasets.batch import MultiviewBatch


class SampleRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    @torch.cuda.nvtx.range("SampleRays")
    def __call__(self, inputs: MultiviewBatch):
        device = inputs['rays'].origins.device
        ray_idx = torch.randint(0, inputs['rays'].shape[0], [self.num_samples], device=device)

        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()

        # Loop over ray values in this batch
        for channel_name, ray_value in inputs.ray_values().items():
            out[channel_name] = ray_value[ray_idx].contiguous()
        return out
