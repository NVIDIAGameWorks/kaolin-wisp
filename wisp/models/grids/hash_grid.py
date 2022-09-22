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
from wisp.ops.spc import sample_spc

import wisp.ops.spc as wisp_spc_ops
import wisp.ops.grid as grid_ops

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

import kaolin.ops.spc as spc_ops

from wisp.accelstructs import OctreeAS

class HashGrid(BLASGrid):
    """This is a feature grid where the features are defined in a codebook that is hashed.
    """

    def __init__(self, 
        feature_dim        : int,
        interpolation_type : str   = 'linear',
        multiscale_type    : str   = 'cat',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 16,
        blas_level         : int   = 7,
        **kwargs
    ):
        """Initialize the hash grid class.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            interpolation_type (str): The type of interpolation function.
            multiscale_type (str): The type of multiscale aggregation. Usually 'sum' or 'cat'.
                                   Note that 'cat' will change the decoder input dimension.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The mean of the Gaussian distribution.
            codebook_bitwidth (int): The bitwidth of the codebook.
            blas_level (int): The level of the octree to be used as the BLAS.
        
        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type

        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth
        self.blas_level = blas_level

        self.kwargs = kwargs
    
        self.blas = OctreeAS()
        self.blas.init_dense(self.blas_level)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.ones(self.num_cells) * 20.0

    def init_from_octree(self, base_lod, num_lods):
        """Builds the multiscale hash grid with an octree sampling pattern.
        """
        octree_lods = [base_lod + x for x in range(num_lods)]
        resolutions = [2 ** lod for lod in octree_lods]
        self.init_from_resolutions(resolutions)

    def init_from_geometric(self, min_width, max_width, num_lods):
        """Build the multiscale hash grid with a geometric sequence.

        This is an implementation of the geometric multiscale grid from 
        instant-ngp (https://nvlabs.github.io/instant-ngp/).

        See Section 3 Equations 2 and 3 for more details.
        """
        b = np.exp((np.log(max_width) - np.log(min_width)) / num_lods) 
        resolutions = [int(np.floor(min_width*(b**l))) for l in range(num_lods)]
        self.init_from_resolutions(resolutions)
    
    def init_from_resolutions(self, resolutions):
        """Build a multiscale hash grid from a list of resolutions.
        """
        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        log.info(f"Active Resolutions: {self.resolutions}")
        
        self.codebook_size = 2 ** self.codebook_bitwidth

        self.codebook = nn.ParameterList([])
        for res in resolutions:
            num_pts = res ** 3
            fts = torch.zeros(min(self.codebook_size, num_pts), self.feature_dim)
            fts += torch.randn_like(fts) * self.feature_std
            self.codebook.append(nn.Parameter(fts))

    def freeze(self):
        """Freezes the feature grid.
        """
        self.codebook.requires_grad_(False)

    def interpolate(self, coords, lod_idx, pidx=None):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): Primitive indices of shape [batch]. Unused here.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        timer = PerfTimer(activate=False, show_memory=False)

        batch, num_samples, _ = coords.shape
        
        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth,
                                  lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            return feats
        elif self.multiscale_type == 'sum':
            return feats.reshape(batch, num_samples, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, level=self.blas_level, num_samples=num_samples,
                                  raymarch_type=raymarch_type)
