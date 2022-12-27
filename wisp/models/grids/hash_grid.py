# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Dict, Set, Any, Type, List
import torch
import torch.nn as nn
import numpy as np
import kaolin.ops.spc as spc_ops
import wisp.ops.grid as grid_ops
from wisp.accelstructs import OctreeAS, BaseAS, ASRaymarchResults
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
        resolutions        : List[int],
        multiscale_type    : str   = 'sum',
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 8,
        blas_level         : int = 7
    ):
        """Builds a HashGrid instance, including the feature structure and an underlying BLAS for fast queries.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
             resolutions (List[int]): A list of resolutions to be used for each feature grid lod of the hash structure.
                i.e. resolutions=[562, 777, 1483, 2048] means that at LOD0, a grid of 562x562x562 nodes will be used,
                where each node is a hashed pointer to the feature table
                (note that feature table size at level L >= resolution of level L).
            multiscale_type (str): The type of multiscale aggregation.
                'sum' - aggregates features from different LODs with summation.
                'cat' - aggregates features from different LODs with concatenation.
                Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
        """
        # Occupancy Structure
        self.blas_level = blas_level
        blas = OctreeAS.make_dense(level=blas_level)
        super().__init__(blas)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points,
                                                               self.blas.pyramid,
                                                               self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)

        # Feature Structure - dims
        self.feature_dim = feature_dim
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth

        # Feature Structure - setup grid LODs
        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

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
        octree_lods = [base_lod + x for x in range(num_lods)]
        resolutions = [2 ** lod for lod in octree_lods]
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level)

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
        This is an implementation of the geometric multiscale grid from
        instant-ngp (https://nvlabs.github.io/instant-ngp/).
        See Section 3 Equations 2 and 3 for more details.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            num_lods (int): The number of LODs for which features are defined. Starts at lod=0.
                            i.e.  num_lods=16 means features are kept for levels 0, 1, 2, .., 14, 15.
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
        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_lods-1))
        resolutions = [int(1 + np.floor(min_grid_res*(b**l))) for l in range(num_lods)]
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level)

    @classmethod
    def from_resolutions(cls,
                         feature_dim: int,
                         resolutions: List[int],
                         multiscale_type: str = 'sum',
                         feature_std: float = 0.0,
                         feature_bias: float = 0.0,
                         codebook_bitwidth: int = 8,
                         blas_level: int = 7) -> HashGrid:
        """
        Builds a hash grid from a list of resolution sizes (each entry contains a RES for the RES x RES x RES
        lod of nodes pointing at the actual hash table) .

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            resolutions (List[int]): A list of resolutions to be used for each feature grid lod of the hash structure.
                i.e. resolutions=[562, 777, 1483, 2048] means that at LOD0, a grid of 562x562x562 nodes will be used,
                where each node is a hashed pointer to the feature table
                (note that feature table size at level L >= resolution of level L).
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure)
        """
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level)

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
        if coords.ndim == 3:    # flatten num_samples dim with batch for cuda call
            batch, num_samples, coords_dim = coords.shape  # batch x num_samples
            coords = coords.reshape(batch * num_samples, coords_dim)

        feats = grid_ops.hashgrid(coords, self.resolutions, self.codebook_bitwidth, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            return feats.reshape(*output_shape, feats.shape[-1])
        elif self.multiscale_type == 'sum':
            return feats.reshape(*output_shape, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=self.blas_level)

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return {OctreeAS}

    def name(self) -> str:
        return "Hash Grid"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        parent_properties = super().public_properties()
        active_lods = None if self.active_lods is None or len(self.active_lods) == 0 else \
            f'{min(self.active_lods)} - {max(self.active_lods)}'
        properties = {
            "Feature Dims": self.feature_dim,
            "Total LODs": self.max_lod,
            "Active feature LODs": active_lods,
            "Interpolation": 'linear',
            "Multiscale aggregation": self.multiscale_type,
            "HashTable Size": f"2^{self.codebook_bitwidth}"
        }
        return {**parent_properties, **properties}
