# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import logging as log
from typing import Dict, Set, Any, Type
import torch
import torch.nn as nn
import wisp.ops.spc as wisp_spc_ops
from wisp.models.grids import BLASGrid
import kaolin.ops.spc as spc_ops
from wisp.accelstructs import BaseAS, OctreeAS, ASRaymarchResults


class OctreeGrid(BLASGrid):
    """This is a multiscale feature grid where the features are defined on the BLAS, the octree.
    """

    def __init__(
        self,
        accelstruct: OctreeAS,
        feature_dim         : int,
        base_lod            : int,
        num_lods            : int          = 1,
        interpolation_type  : str          = 'linear',
        multiscale_type     : str          = 'cat',
        feature_std         : float        = 0.0,
        feature_bias        : float        = 0.0
    ):
        """Initialize the octree grid class.

        Args:
            accelstruct: Spatial acceleration structure which tracks the occupancy state of this grid.
                         Used to speed up spatial queries and ray tracing operations.
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
            sample_tex (bool): If True, will also sample textures and store it in the accelstruct.
            num_samples (int): The number of samples to be generated on the mesh surface.
        Returns:
            (void): Initializes the class.
        """
        super().__init__(accelstruct)
        self.feature_dim = feature_dim
        self.base_lod = base_lod
        self.num_lods = num_lods
        self.interpolation_type = interpolation_type
        self.multiscale_type = multiscale_type

        self.feature_std = feature_std
        self.feature_bias = feature_bias

        # List of octree levels which are optimized.
        self.active_lods = [self.base_lod + x for x in range(self.num_lods)]
        self.max_lod = OctreeGrid.max_octree_lod(self.base_lod, self.num_lods)

        log.info(f"Active LODs: {self.active_lods}")    # TODO(operel): move into trainer

        if self.num_lods > 0:
            self.init_feature_structure()

    @staticmethod
    def max_octree_lod(base_lod, num_lods) -> int:
        """
        Returns:
            (int): The highest level-of-detail maintaining features in this Octree grid.
        """
        return base_lod + num_lods - 1

    @classmethod
    def make_dense(cls,
                   feature_dim        : int,
                   base_lod           : int,
                   num_lods           : int          = 1,
                   interpolation_type : str          = 'linear',
                   multiscale_type    : str          = 'cat',
                   feature_std        : float        = 0.0,
                   feature_bias       : float        = 0.0) -> OctreeGrid:
        """Builds a fully occupied OctreeGrid with `base_lod + num_lods - 1` levels.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of the  octree for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            interpolation_type (str): The type of interpolation function used when querying features on the grid.
                                      'linear' - uses trilinear interpolation from nearest 8 nodes.
                                      'closest' - uses feature from nearest grid node.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.

        Returns:
            (OctreeGrid): A new instance of an OctreeGrid at full occupancy.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.make_dense(level=max_lod)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias)
        return grid

    @classmethod
    def from_mesh(cls,
                  mesh_path: str,
                  num_samples_on_mesh: int,
                  feature_dim: int,
                  base_lod: int,
                  num_lods: int = 1,
                  interpolation_type: str = 'linear',
                  multiscale_type: str = 'cat',
                  feature_std: float = 0.0,
                  feature_bias: float = 0.0) -> OctreeGrid:
        """Builds the OctreeGrid from point samples over mesh faces.
        Samples will be quantized to populate the octree cells.
        This method is not guaranteed to reproduce a perfect mesh structure without "holes",
        but achieves a very good approximation with high probability.

        Args:
            mesh_path (str): Path to an OBJ file with a mesh to initialize the octree (only supports OBJ for now).
            num_samples (int): The number of samples to be generated on the mesh surface.
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of the  octree for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            interpolation_type (str): The type of interpolation function used when querying features on the grid.
                                      'linear' - uses trilinear interpolation from nearest 8 nodes.
                                      'closest' - uses feature from nearest grid node.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.

        Returns:
            (OctreeGrid): A new instance of an OctreeGrid with occupancy initialized from samples over the mesh faces.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.from_mesh(mesh_path, level=max_lod, sample_tex=False, num_samples=num_samples_on_mesh)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias)
        return grid

    @classmethod
    def from_pointcloud(cls,
                        pointcloud: torch.FloatTensor,
                        feature_dim: int,
                        base_lod: int,
                        num_lods: int = 1,
                        interpolation_type: str = 'linear',
                        multiscale_type: str = 'cat',
                        feature_std: float = 0.0,
                        feature_bias: float = 0.0) -> OctreeGrid:
        """Builds the OctreeGrid, initializing it with a pointcloud.
        The cells occupancy will be determined by points occupying the octree cells.

        Args:
            pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords, 3] in normalized space [-1, 1].
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of the  octree for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            interpolation_type (str): The type of interpolation function used when querying features on the grid.
                                      'linear' - uses trilinear interpolation from nearest 8 nodes.
                                      'closest' - uses feature from nearest grid node.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.

        Returns:
            (OctreeGrid): A new instance of an OctreeGrid with occupancy initialized from the pointcloud.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.from_pointcloud(pointcloud, level=max_lod)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias)
        return grid

    @classmethod
    def from_spc(cls,
                 spc_octree: torch.FloatTensor,
                 feature_dim: int,
                 base_lod: int,
                 num_lods: int = 1,
                 interpolation_type: str = 'linear',
                 multiscale_type: str = 'cat',
                 feature_std: float = 0.0,
                 feature_bias: float = 0.0) -> OctreeGrid:
        """Builds the OctreeGrid, initializing it's occupancy state with a Structured Point Cloud (SPC).
        The octree cells occupancy will be determined by the SPC cells.
        SPCs are compressed, sparse volumetric data structures used for accelerated point queries and ray tracing ops.

        Args:
            spc_octree (torch.ByteTensor):
                A tensor which holds the topology of a Structured Point Cloud (SPC).
                Each byte represents a single octree cell's occupancy (that is, each bit of that byte represents
                the occupancy status of a child octree cell), yielding 8 bits for 8 cells.
                See also https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of the  octree for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            interpolation_type (str): The type of interpolation function used when querying features on the grid.
                                      'linear' - uses trilinear interpolation from nearest 8 nodes.
                                      'closest' - uses feature from nearest grid node.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.

        Returns:
            (OctreeGrid): A new instance of an OctreeGrid with occupancy initialized from the SPC topology.
        """
        blas = OctreeAS(spc_octree)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias)
        return grid

    def init_feature_structure(self):
        """ Initializes everything related to the features stored in the codebook octree structure. """

        # Assumes the occupancy structure have been initialized (the BLAS: Bottom Level Accelerated Structure).
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

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to ``active_lods``
            features (torch.FloatTensor): features to interpolate. If ``None``, will use `self.features`.

        Returns:
            (torch.FloatTensor): interpolated features of shape
            [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape, and inflate to (batch, num_samples, 3) format
        output_shape = coords.shape[:-1]
        if coords.ndim < 3:
            coords = coords[:, None]    # (batch, 3) -> (batch, num_samples, 3)

        if lod_idx == 0:
            query_results = self.blas.query(coords[:,0], self.active_lods[lod_idx], with_parents=False)
            pidx = query_results.pidx
            feat = self._interpolate(coords, self.features[0], pidx, 0)
            return feat.reshape(*output_shape, feat.shape[-1])
        else:
            feats = []
            
            # In the multiscale case, the raytrace _currently_  does not return multiscale indices.
            # As such, no matter what the pidx will be recalculated to get the multiscale indices.
            num_feats = lod_idx + 1
            
            # This might look unoptimal since it assumes that samples are _not_ in the same voxel.
            # This is the correct assumption here, because the point samples are from the base_lod,
            # not the highest LOD.
            query_results = self.blas.query(coords.reshape(-1, 3), self.active_lods[lod_idx], with_parents=True)
            pidx = query_results.pidx[...,self.base_lod:]
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
            
            return feats.reshape(*output_shape, self.feature_dim)

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=self.base_lod)

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return {OctreeAS}

    def name(self) -> str:
        return "Octree Grid"

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
            "Multiscale aggregation": self.multiscale_type
        }
        return {**parent_properties, **properties}
