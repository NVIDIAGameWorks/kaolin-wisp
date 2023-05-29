# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch.nn as nn


class RasterizationPipeline(nn.Module):
    """ Wrapper class for implementing neural / non-neural Rasterization pipelines.
    RasterizationPipeline is a thin wrapper around existing rasterizers, which simply hints wisp
    the wrapped object is a rasterizer which relies on camera input, rather than rays.
    """
    
    def __init__(self, rasterizer):
        """Initialize the Pipeline.

        Args:
            rasterizer: A general model of a rasterizer.
                No assumptions are made on the rasterizer object. The only requirement is
                for this object to be callable.
                Rasterizers are encouraged to return a Renderbuffer object, but are not required to do so.
        """
        super().__init__()
        self.rasterizer = rasterizer

    def forward(self, *args, **kwargs):
        """The forward function will invoke the underlying rasterizer (the forward model).
        Rasterizer is any general callable interface.
        """
        return self.rasterizer(*args, **kwargs)
