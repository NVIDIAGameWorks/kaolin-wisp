# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging as log
from typing import Dict, Set, Any, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from wisp.core import WispModule
from wisp.models.grids import BLASGrid
from wisp.accelstructs import AxisAlignedBBoxAS, BaseAS, ASRaytraceResults, ASRaymarchResults


class TriplanarGrid(BLASGrid):
    """A feature grid where the features are stored on a multiresolution pyramid of triplanes.
    Each LOD consists of a triplane, e.g. a triplet of orthogonal planes.

    The shape of the triplanar feature grid means the support region is bounded by an AABB,
    therefore spatial queries / ray tracing ops can use an AABB as an acceleration structure.
    Hence the class is compatible with BaseTracer implementations.
    """

    def __init__(self,
                 feature_dim: int,
                 base_lod: int,
                 num_lods: int = 1,
                 interpolation_type: str = 'linear',
                 multiscale_type: str = 'sum',
                 feature_std: float = 0.0,
                 feature_bias: float = 0.0
                 ):
        """Constructs an instance of a TriplanarGrid.

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
        # The bottom level acceleration structure is an axis aligned bounding box
        super().__init__(blas=AxisAlignedBBoxAS())
        # The actual feature_dim is multiplied by 3 because of the triplanar maps.
        self.feature_dim = feature_dim * 3
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias

        # TODO(ttakikawa) The Triplanar API should look more like the ImagePyramid class with
        #   similar initialization mechanisms since it's not limited to octrees.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = self.num_lods + self.base_lod - 1

        log.info(f"Active LODs: {self.active_lods}")

        self.num_feat = 0
        self.init_feature_structure()

    def init_feature_structure(self):
        """ Initializes everything related to the features stored in the triplanar grid structure. """
        self.features = nn.ModuleList([])
        self.num_feat = 0
        for i in self.active_lods:
            self.features.append(
                TriplanarFeatureVolume(self.feature_dim // 3, 2 ** i, self.feature_std, self.feature_bias))
            self.num_feat += ((2 ** i + 1) ** 2) * self.feature_dim * 3

        log.info(f"# Feature Vectors: {self.num_feat}")

    def freeze(self):
        """Freezes the feature grid.
        """
        self.features.requires_grad_(False)

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of
            shape [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        output_shape = coords.shape[:-1]
        if coords.ndim < 3:
            coords = coords[:, None]  # (batch, 3) -> (batch, num_samples, 3)

        feats = []
        for i in range(lod_idx + 1):
            feats.append(self._interpolate(coords, self.features[i], i))

        feats = torch.cat(feats, dim=-1)

        if self.multiscale_type == 'sum':
            feats = feats.reshape(*output_shape, lod_idx + 1, feats.shape[-1] // (lod_idx + 1)).sum(-2)

        return feats

    def _interpolate(self, coords, feats, lod_idx):
        """Interpolates the given feature using the coordinates x.

        This is a more low level interface for optimization.

        Inputs:
            coords     : float tensor of shape [batch, num_samples, 3]
            feats : float tensor of shape [num_feats, feat_dim]
            lod_idx   : int specifying the lod
        Returns:
            float tensor of shape [batch, num_samples, feat_dim]
        """
        batch, num_samples = coords.shape[:2]

        if self.interpolation_type == 'linear':
            fs = feats(coords).reshape(batch, num_samples, 3 * feats.fdim)
        else:
            raise ValueError(f"Interpolation mode '{self.interpolation_type}' is not supported")

        return fs

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=0)

    def raytrace(self, rays, level=None, with_exit=False) -> ASRaytraceResults:
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.

        Important detail: this is just used as an AABB tracer.
        """
        return self.blas.raytrace(rays, level=0, with_exit=with_exit)

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return {AxisAlignedBBoxAS}

    def name(self) -> str:
        return "Triplanar Grid"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        parent_properties = super().public_properties()
        properties = {
            "Feature Dims": self.feature_dim,
            "Total LODs": self.max_lod,
            "Active feature LODs": [str(x) for x in self.active_lods],
            "Interpolation": self.interpolation_type,
            "Multiscale aggregation": self.multiscale_type,
        }
        for idx, module in enumerate(self.features):
            properties[f"Pyramid Layer #{idx + 1}"] = module

        return {**parent_properties, **properties}


class TriplanarFeatureVolume(WispModule):
    """Triplanar feature volume represents a single triplane, e.g. a single LOD in a TriplanarGrid. """

    def __init__(self, fdim, fsize, std, bias):
        """Initializes the feature triplane.

        Args:
            fdim (int): The feature dimension.
            fsize (int): The height and width of the texture map.
            std (float): The standard deviation for the Gaussian initialization.
            bias (float): The mean for the Gaussian initialization.
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
        N = x.shape[0]
        if len(x.shape) == 3:
            sample_coords = x.reshape(1, N, x.shape[1], 3)  # [N, 1, 1, 3]
            samplex = F.grid_sample(self.fmx, sample_coords[...,[1,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            sampley = F.grid_sample(self.fmy, sample_coords[...,[0,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            samplez = F.grid_sample(self.fmz, sample_coords[...,[0,1]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,:].transpose(0,1)
            sample = torch.stack([samplex, sampley, samplez], dim=1).permute(0,3,1,2)
        else:
            sample_coords = x.reshape(1, N, 1, 3)  # [N, 1, 1, 3]
            samplex = F.grid_sample(self.fmx, sample_coords[...,[1,2]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
            sampley = F.grid_sample(self.fmy, sample_coords[...,[0,2]], 
                                    align_corners=True, padding_modes=self.padding_mode)[0,:,:,0].transpose(0,1)
            samplez = F.grid_sample(self.fmz, sample_coords[...,[0,1]], 
                                    align_corners=True, padding_mode=self.padding_mode)[0,:,:,0].transpose(0,1)
            sample = torch.stack([samplex, sampley, samplez], dim=1)
        return sample

    def name(self) -> str:
        """ A human readable name for the given wisp module. """
        return "TriplanarFeatureVolume"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            'Resolution': f'3x{self.fsize}x{self.fsize}'
        }
