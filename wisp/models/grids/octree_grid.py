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

from wisp.models.grids import BLASGrid
from wisp.models.decoders import BasicDecoder

import kaolin.ops.spc as spc_ops

from wisp.accelstructs import OctreeAS

class OctreeGrid(BLASGrid):
    """This is a multiscale feature grid where the features are defined on the BLAS, the octree.
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
        self.feature_dim = feature_dim
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type

        self.feature_std = feature_std
        self.feature_bias = feature_bias

        self.kwargs = kwargs

        # List of octree levels which are optimized.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = self.num_lods + self.base_lod - 1

        log.info(f"Active LODs: {self.active_lods}")
        
        self.blas = OctreeAS()

    def init_from_mesh(self, mesh_path, level=None, sample_tex=False, num_samples=100000000):
        """Builds the grid from a path to the mesh.
        """
        if level is None:
            level = self.max_lod
        self.blas.init_from_mesh(mesh_path, level, sample_tex=sample_tex, num_samples=num_samples)
        self._init()

    def init_from_pointcloud(self, pointcloud, level=None):
        """Builds the grid from a pointcloud.

        Args:
            pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords, 3] in 
                                            normalized space [-1, 1].
            level (int): The depth of the octree. If None, uses the max LOD.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        if level is None:
            level = self.max_lod
        self.blas.init_from_pointcloud(pointcloud, level)
        self._init()

    def init_dense(self, level=None):
        """Builds a dense octree grid.

        Args:
            level (int): The depth of the octree. If None, uses the max LOD.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        if level is None:
            level = self.max_lod
        self.blas.init_dense(level)
        self._init()

    def init(self, octree):
        """Initializes auxillary state from an octree tensor.

        Args:
            octree (torch.ByteTensor): SPC octree tensor.

        Returns:
            (void): Will initialize the OctreeAS object.
        """
        self.blas.init(octree)
        self._init()

    def _init(self):
        """Initializes everything that is not the BLAS.
        """
        if not self.blas_initialized():
            assert False and "Octree BLAS not initialized. _init is private."

        # Build the trinket structure
        if self.interpolation_type in ['linear']:
            self.points_dual, self.pyramid_dual, self.trinkets, self.parents = \
                wisp_spc_ops.make_trilinear_spc(self.blas.points, self.blas.pyramid)
            log.info("Built dual octree and trinkets")
        
        # Build the pyramid of features
        fpyramid = []
        for al in self.active_lods:
            if self.interpolation_type == 'linear':
                fpyramid.append(self.pyramid_dual[0,al]+1)
            elif self.interpolation_type == 'closest':
                fpyramid.append(self.blas.pyramid[0,al]+1)
            else:
                raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        self.num_feat = sum(fpyramid).long()
        log.info(f"# Feature Vectors: {self.num_feat}")

        self.features = nn.ParameterList([])
        for i in range(len(self.active_lods)):
            fts = torch.zeros(fpyramid[i], self.feature_dim) + self.feature_bias
            fts += torch.randn_like(fts) * self.feature_std
            self.features.append(nn.Parameter(fts))
        
        log.info(f"Pyramid:{self.blas.pyramid[0]}")
        log.info(f"Pyramid Dual: {self.pyramid_dual[0]}")

    def freeze(self):
        """Freezes the feature grid.
        """
        for lod_idx in range(self.num_lods):
            self.features[lod_idx].requires_grad_(False)

    def _index_features(self, feats, idx):
        """Internal function. Returns the feats based on indices.

        This function exists to override in case you want to implement a different method of indexing,
        i.e. a differentiable one as in Variable Bitrate Neural Fields (VQAD).

        Args:
            feats (torch.FloatTensor): tensor of feats of shape [num_feats, feat_dim]
            idx (torch.LongTensor): indices of shape [num_indices]

        Returns:
            (torch.FloatTensor): tensor of feats of shape [num_indices, feat_dim]
        """
        return feats[idx.long()]

    def _interpolate(self, coords, feats, pidx, lod_idx):
        """Interpolates the given feature using the coordinates x. 

        This is a more low level interface for optimization.

        Inputs:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            feats (torch.FloatTensor): feats of shape [num_feats, feat_dim]
            pidx (torch.LongTensor) : point_hiearchy indices of shape [batch]
            lod_idx (int) : int specifying the index fo ``active_lods``
        Returns:
            (torch.FloatTensor): acquired features of shape [batch, num_samples, feat_dim]
        """
        batch, num_samples = coords.shape[:2]
        lod = self.active_lods[lod_idx]

        if self.interpolation_type == 'linear':
            fs = spc_ops.unbatched_interpolate_trilinear(
                coords, pidx.int(), self.blas.points, self.trinkets.int(),
                feats.half(), lod).float()
        elif self.interpolation_type == 'closest':
            fs = self._index_features(feats, pidx.long()-self.blas.pyramid[1, lod])[...,None,:]
            fs = fs.expand(batch, num_samples, feats.shape[-1])
       
        # Keep as backup
        elif self.interpolation_type == 'trilinear_old':
            corner_feats = feats[self.trinkets.index_select(0, pidx).long()]
            coeffs = spc_ops.coords_to_trilinear_coeffs(coords, self.points.index_select(0, pidx)[:,None].repeat(1, coords.shape[1], 1), lod)
            fs = (corner_feats[:, None] * coeffs[..., None]).sum(-2)

        else:
            raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        
        return fs

    def interpolate(self, coords, lod_idx, pidx=None):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): point_hiearchy indices of shape [batch]
            features (torch.FloatTensor): features to interpolate. If ``None``, will use `self.features`.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        timer = PerfTimer(activate=False, show_memory=False)

        batch, num_samples, _ = coords.shape

        if lod_idx == 0:
            if pidx is None:
                pidx = self.blas.query(coords[:,0], self.active_lods[lod_idx], with_parents=False)
            feat = self._interpolate(coords, self.features[0], pidx, 0)
            return feat
        else:
            feats = []
            
            # In the multiscale case, the raytrace _currently_  does not return multiscale indices.
            # As such, no matter what the pidx will be recalculated to get the multiscale indices.
            num_feats = lod_idx + 1
            
            # This might look unoptimal since it assumes that samples are _not_ in the same voxel.
            # This is the correct assumption here, because the point samples are from the base_lod,
            # not the highest LOD.
            pidx = self.blas.query(
                coords.reshape(-1, 3), self.active_lods[lod_idx], with_parents=True
            )[...,self.base_lod:]
            pidx = pidx.reshape(-1, coords.shape[1], num_feats)
            pidx = torch.split(pidx, 1, dim=-1)
            
            # list of [batch, num_samples, 1]

            for i in range(num_feats):
                feat = self._interpolate(
                    coords.reshape(-1, 1, 3), self.features[i], pidx[i].reshape(-1), i)[:,0]
                feats.append(feat)
            
            feats = torch.cat(feats, dim=-1)

            if self.multiscale_type == 'sum':
                feats = feats.reshape(*feats.shape[:-1], num_feats, self.feature_dim)
                if self.training:
                    feats = feats.sum(-2)
                else:
                    feats = feats.sum(-2)
            
            timer.check("aggregate")
            
            return feats.reshape(batch, num_samples, self.feature_dim)

    def raymarch(self, rays, level=None, num_samples=64, raymarch_type='voxel'):
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, level=self.base_lod, num_samples=num_samples,
                                  raymarch_type=raymarch_type)

class CodebookOctreeGrid(OctreeGrid):
    """This is a multiresolution feature grid where the octree stores indices into a fixed size codebook.
    """

    def _init(self):
        """Initializes everything that is not the BLAS.
        """
        if not self.blas_initialized():
            assert False and "Octree BLAS not initialized. _init is private."
        if self.interpolation_type == 'linear':
            self.points_dual, self.pyramid_dual, self.trinkets, self.parents = \
                    wisp_spc_ops.make_trilinear_spc(self.blas.points, self.blas.pyramid)
            log.info("Built dual octree and trinkets")
            
        # Create the pyramid of features.
        fpyramid = []
        for al in self.active_lods:
            if self.interpolation_type == 'linear':
                fpyramid.append(self.pyramid_dual[0,al]+1)
            elif self.interpolation_type == 'closest':
                fpyramid.append(self.blas.pyramid[0,al]+1)
            else:
                raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        self.num_feat = sum(fpyramid).long()
        log.info(f"# Feature Vectors: {self.num_feat}")

        self.bitwidth = self.kwargs['codebook_bitwidth']

        self.dictionary_size = 2**self.bitwidth

        self.dictionary = nn.ParameterList([])
        for i in range(len(self.active_lods)):
            fts = torch.zeros(self.dictionary_size, self.feature_dim)
            fts += torch.randn_like(fts) * self.feature_std
            self.dictionary.append(nn.Parameter(fts))

        self.features = nn.ParameterList([]) 
        
        for i in range(len(self.active_lods)):
            fts = torch.zeros(fpyramid[i], self.dictionary_size)
            fts += torch.randn_like(fts) * self.feature_std
            self.features.append(nn.Parameter(fts))

    def bake(self):
        for i, f in enumerate(self.features):
            self.features[i] = nn.Parameter(f.max(dim=-1)[1].float())

    def _index_features(self, feats, idx, lod_idx):
        """Internal function. Returns the feats based on indices.

        Args:
            feats (torch.FloatTensor): tensor of feats of shape [num_feats, feat_dim]
            idx (torch.LongTensor): indices of shape [num_indices]
            lod_idx (int): index to `self.active_lods`.

        Returns:
            (torch.FloatTensor): tensor of feats of shape [num_indices, feat_dim]
        """
        # idx -> [N, 8]
        # [1, 1, 256, 32] * [N, 8, 256, 1] -> [N, 8, 32]
        
        if self.training:
            logits = feats[idx.long()]
            y_soft = F.softmax(logits, dim=-1)
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, index, 1.0)
            keys = y_hard - y_soft.detach() + y_soft
            return (self.dictionary[lod_idx][None, None] * keys[..., None]).sum(-2)
            
            # TODO(ttakikawa): Replace with a cleaner / faster softmax implementation
            #keys = F.softmax(feats[idx.long()], dim=-1)
            #return softmax_dictionary(keys, self.dictionary[lod_idx])
        else:
            # [N, 8, 256] -> [N, 8]
            #keys = feats[idx.long()].long()
            keys = torch.max(feats[idx.long()], dim=-1)[1]
            return self.dictionary[lod_idx][keys]
    
    def _interpolate(self, coords, feats, pidx, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            lod_idx  (int): int specifying the index to ``active_lods`` 
            pidx (torch.LongTensor): point_hiearchy indices of shape [batch]
            features (torch.FloatTensor): features to interpolate. If ``None``, will use `self.features`.

        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        batch, num_samples = coords.shape[:2]

        if self.interpolation_type == 'linear':
            
            fs = torch.zeros(batch, num_samples, self.feature_dim, device=coords.device)
            
            valid_mask = pidx > -1
            valid_pidx = pidx[valid_mask]
            if valid_pidx.shape[0] == 0:
                return fs

            corner_feats = self._index_features(feats, 
                    self.trinkets.index_select(0, valid_pidx).long(), lod_idx)[:, None]
            
            pts = self.blas.points.index_select(0, valid_pidx)[:,None].repeat(1, coords.shape[1], 1)

            coeffs = spc_ops.coords_to_trilinear_coeffs(coords[valid_mask], pts, self.active_lods[lod_idx])[..., None]
            fs[valid_mask] = (corner_feats * coeffs).sum(-2)

        elif self.interpolation_type == 'closest':
            raise NotImplementedError
        else:
            raise Exception(f"Interpolation mode {self.interpolation_type} is not supported.")
        
        return fs

