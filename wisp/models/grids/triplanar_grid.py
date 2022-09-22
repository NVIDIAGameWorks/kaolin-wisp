# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math

from wisp.utils import PsDebugger, PerfTimer

import wisp.ops.spc as wisp_spc_ops

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

from wisp.accelstructs import OctreeAS
import kaolin.ops.spc as spc_ops
import kaolin.render.spc as spc_render

class TriplanarFeatureVolume(nn.Module):
    """Triplanar feature module implemented with a lod of grid_sample swizzling.
    """
    def __init__(self, fdim, fsize, std, bias):
        """Initializes the feature volume.

        Args:
            fdim (int): The feature dimension.
            fsize (int): The height and width of the texture map.
            std (float): The standard deviation for the Gaussian initialization.
            bias (float): The mean for the Gaussian initialization.

        Returns:
            (void): Initializes the feature volume.
        """
        super().__init__()
        self.fsize = fsize
        self.fdim = fdim
        self.fmx = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1) * std + bias)
        self.fmy = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1) * std + bias)
        self.fmz = nn.Parameter(torch.randn(1, fdim, fsize+1, fsize+1) * std + bias)
        self.padding_mode = 'reflection'

    def forward(self, x):
        """Interpolates from the feature volume.

        Args:
            x (torch.FloatTensor): Coordinates of shape [batch, num_samples, 3] or [batch, 3].

        Returns:
            (torch.FloatTensor): Features of shape [batch, num_samples, fdim] or [batch, fdim].
        """
        # TODO(ttakikawa): Maybe cleaner way of writing this?
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3) # [N, 1, 1, 3]    
            samplex = F.grid_sample(self.fmx, sample_coords[...,[1,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            sampley = F.grid_sample(self.fmy, sample_coords[...,[0,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            samplez = F.grid_sample(self.fmz, sample_coords[...,[0,1]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            sample = torch.stack([samplex, sampley, samplez], dim=1).permute(0,3,1,2)
        else:
            sample_coords = x.reshape(1, N, 1, 3) # [N, 1, 1, 3]    
            samplex = F.grid_sample(self.fmx, sample_coords[...,[1,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
            sampley = F.grid_sample(self.fmy, sample_coords[...,[0,2]], 
                                    align_corners=True, padding_modes=self.padding_mode)[0,:,:,0].transpose(0,1)
            samplez = F.grid_sample(self.fmz, sample_coords[...,[0,1]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
            sample = torch.stack([samplex, sampley, samplez], dim=1)
        return sample

class TriplanarGrid(BLASGrid):
    """This is a feature grid where the features are defined on a pyramid of multiresolution triplanar maps.

    Since the triplanar feature grid means the support region is bounded by an AABB, this uses an AABB
    as the BLAS. Hence the class is compatible with the usual packed tracers.
    """

    def __init__(self, 
        feature_dim        : int,
        base_lod           : int   = 0,
        num_lods           : int   = 1, 
        interpolation_type : str   = 'linear',
        multiscale_type    : str   = 'cat',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        **kwargs
    ):
        """Initialize the octree grid class.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid. This is the lowest LOD of the SPC octree
                            for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
        
        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        # The actual feature_dim is multiplied by 3 because of the triplanar maps.
        self.feature_dim = feature_dim * 3
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias

        # TODO(ttakikawa) The Triplanar API should look more like the ImagePyramid class with 
        # similar initialization mechanisms since it's not limited to octrees.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = self.num_lods + self.base_lod - 1

        log.info(f"Active LODs: {self.active_lods}")

        self.blas = OctreeAS()
        self.blas.init_aabb()

        self._init()

    def _init(self):
        """Initializes everything that is not the BLAS.
        """

        self.features = nn.ModuleList([])
        self.num_feat = 0
        for i in self.active_lods:
            self.features.append(
                    TriplanarFeatureVolume(self.feature_dim//3, 2**i, self.feature_std, self.feature_bias))
            self.num_feat += ((2**i + 1)**2) * self.feature_dim * 3

        log.info(f"# Feature Vectors: {self.num_feat}")
        
    def freeze(self):
        """Freezes the feature grid.
        """
        self.features.requires_grad_(False)

    def interpolate(self, coords, lod_idx, pidx=None):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): point_hiearchy indices of shape [batch]. Unused in this function.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        batch, num_samples, _ = coords.shape
        
        feats = []
        for i in range(lod_idx + 1):
            feats.append(self._interpolate(coords, self.features[i], i))
        
        feats = torch.cat(feats, dim=-1)

        if self.multiscale_type == 'sum':
            feats = feats.reshape(batch, num_samples, lod_idx + 1, feats.shape[-1] // (lod_idx + 1)).sum(-2)
        
        return feats

    def _interpolate(self, coords, feats, lod_idx):
        """Interpolates the given feature using the coordinates x. 

        This is a more low level interface for optimization.

        Inputs:
            coords     : float tensor of shape [batch, num_samples, 3]
            feats : float tensor of shape [num_feats, feat_dim]
            pidx  : long tensor of shape [batch]
            lod   : int specifying the lod
        Returns:
            float tensor of shape [batch, num_samples, feat_dim]
        """
        batch, num_samples = coords.shape[:2]

        if self.interpolation_type == 'linear':
            fs = feats(coords).reshape(batch, num_samples, 3 * feats.fdim)
        else:
            raise ValueError(f"Interpolation mode '{self.interpolation_type}' is not supported")
        
        return fs
    
    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raymarch(rays, level=0, num_samples=num_samples,
                                  raymarch_type=raymarch_type)
    
    def raytrace(self, rays, level=None, with_exit=False):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        
        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raytrace(rays, level=0, with_exit=with_exit)
