# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import logging as log
import kaolin.ops.spc as spc_ops
import wisp.ops.grid as grid_ops
from wisp.accelstructs import OctreeAS
from wisp.models.grids import BLASGrid


class HashGrid(BLASGrid):
    """A feature grid where hashed feature pointers are stored as multi-LOD grid nodes,
    and actual feature contents are stored in a hash table.
    (see: Muller et al. 2022, Instant-NGP: https://nvlabs.github.io/instant-ngp/)

    The occupancy state (e.g. BLAS, Bottom Level Acceleration Structure) is tracked separately from the feature
    volume, and relies on heuristics such as pruning for keeping it aligned with the feature structure.
    """
    def __init__(self,
        feature_dim        : int,
        base_lod           : int   = None,
        num_lods           : int   = 1,
        multiscale_type    : str   = 'sum',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 8,
        tree_type          : str   = None,
        min_grid_res       : int   = None,
        max_grid_res       : int   = None,
        blas_level: int = 7,
    ):
        """Builds a HashGrid instance, including the feature structure and an underlying BLAS for fast queries.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of for which features are defined. This arg is only used
                            when the HashGrid is initialized with the tree_type="quad" pattern.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            tree_type (str): How to determine the resolution of the grid.
                "geometric" - uses the geometric sequence initialization from InstantNGP,
                "quad" -  uses an octree sampling pattern.
            min_grid_res (int): min resolution of the feature grid. Used only for "geometric" initialization.
            max_grid_res (int): max resolution of the feature grid. Used only for "geometric" initialization.
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
        """
        super().__init__()
        # Feature Structure
        self.feature_dim = feature_dim
        self.num_lods = num_lods
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth

        # Occupancy Structure
        self.blas_level = blas_level
        self.blas = OctreeAS.make_dense(level=self.blas_level)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points, self.blas.pyramid, self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)

        if tree_type == 'quad':
            if base_lod is None:
                raise ValueError("'base_lod' must be specified with tree_type == 'quad'")
            self._init_from_octree(base_lod)
        elif tree_type == 'geometric':
            if max_grid_res is None or min_grid_res is None:
                raise ValueError("'max_grid_res' must be specified with tree_type == 'geometric'")
            self._init_from_geometric(min_grid_res, max_grid_res)
        else:
            raise ValueError(f"tree_type == '{tree_type}' not supported")

    def _init_from_octree(self, base_lod):
        """Builds the multiscale hash grid with an octree sampling pattern.
        """
        octree_lods = [base_lod + x for x in range(self.num_lods)]
        resolutions = [2 ** lod for lod in octree_lods]
        self._init_from_resolutions(resolutions)

    def _init_from_geometric(self, min_width, max_width):
        """Build the multiscale hash grid with a geometric sequence.

        This is an implementation of the geometric multiscale grid from 
        instant-ngp (https://nvlabs.github.io/instant-ngp/).

        See Section 3 Equations 2 and 3 for more details.
        """
        b = np.exp((np.log(max_width) - np.log(min_width)) / (self.num_lods-1))
        resolutions = [int(1 + np.floor(min_width*(b**l))) for l in range(self.num_lods)]
        self._init_from_resolutions(resolutions)
    
    def _init_from_resolutions(self, resolutions):
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

    @classmethod
    def from_octree(cls,
                    feature_dim        : int,
                    base_lod           : int   = 2,
                    num_lods           : int   = 1,
                    multiscale_type    : str   = 'sum',
                    feature_std        : float = 0.0,
                    feature_bias       : float = 0.0,
                    codebook_bitwidth  : int   = 8,
                    blas_level         : int   = 7) -> HashGrid:
        """
        Builds a hash grid using an octree sampling pattern.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
                The HashGrid is backed
        """
        return cls(feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level, tree_type='quad')

    @classmethod
    def from_geometric(cls,
                       feature_dim        : int,
                       num_lods           : int,
                       multiscale_type    : str   = 'sum',
                       feature_std        : float = 0.0,
                       feature_bias       : float = 0.0,
                       codebook_bitwidth  : int   = 8,
                       min_grid_res       : int   = 16,
                       max_grid_res       : int   = None,
                       blas_level: int = 7) -> HashGrid:
        """
        Builds a hash grid using the geometric sequence initialization pattern from Muller et al. 2022 (Instant-NGP).

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            min_grid_res (int): min resolution of the feature grid.
            max_grid_res (int): max resolution of the feature grid.
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
        """
        return cls(feature_dim=feature_dim, num_lods=num_lods, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level, min_grid_res=min_grid_res, max_grid_res=max_grid_res, tree_type='geometric')

    def freeze(self):
        """Freezes the feature grid.
        """
        self.codebook.requires_grad_(False)

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
                For some grid implementations, specifying num_samples may allow for slightly faster trilinear
                interpolation. HashGrid doesn't use this optimization, but allows this input type for compatability.
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of shape
             [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape
        output_shape = coords.shape[:-1]
        if coords.ndim >= 2:
            coords = coords.reshape(-1, coords.shape[-1])

        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            return feats.reshape(*output_shape, feats.shape[-1])
        elif self.multiscale_type == 'sum':
            return feats.reshape(*output_shape, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, raymarch_type, num_samples, level=None):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=self.blas_level)

    def name(self) -> str:
        return "Hash Grid"
