# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Dict, Any
import logging as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import kaolin.ops.spc as spc_ops
import wisp.ops.spc as wisp_spc_ops
from wisp.models.grids.octree_grid import OctreeGrid
from wisp.accelstructs import OctreeAS


class CodebookOctreeGrid(OctreeGrid):
    """This is a multiresolution feature grid where the octree stores indices into a fixed size codebook.
    """
    def __init__(
        self,
        accelstruct,
        feature_dim         : int,
        base_lod            : int,
        num_lods            : int          = 1,
        interpolation_type  : str          = 'linear',
        multiscale_type     : str          = 'cat',
        feature_std         : float        = 0.0,
        feature_bias        : float        = 0.0,
        codebook_bitwidth   : int          = 8
    ):
        """
        Args:
            accelstruct: Spatial acceleration structure which tracks the occupancy state of this grid.
                         Used to speed up spatial queries and ray tracing operations.
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
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
        """
        self.bitwidth = codebook_bitwidth
        super().__init__(accelstruct=accelstruct,
                         feature_dim=feature_dim,
                         base_lod=base_lod,
                         num_lods=num_lods,
                         interpolation_type=interpolation_type,
                         multiscale_type=multiscale_type,
                         feature_std=feature_std,
                         feature_bias=feature_bias)

    @classmethod
    def make_dense(cls,
                   feature_dim        : int,
                   base_lod           : int,
                   num_lods           : int          = 1,
                   interpolation_type : str          = 'linear',
                   multiscale_type    : str          = 'cat',
                   feature_std        : float        = 0.0,
                   feature_bias       : float        = 0.0,
                   codebook_bitwidth   : int         = 8) -> CodebookOctreeGrid:
        """Builds a fully occupied CodebookOctreeGrid with `base_lod + num_lods - 1` levels.

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
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth

        Returns:
            (CodebookOctreeGrid): A new instance of an OctreeGrid at full occupancy.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.make_dense(level=max_lod)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth)
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
                  feature_bias: float = 0.0,
                  codebook_bitwidth: int = 8) -> CodebookOctreeGrid:
        """Builds the CodebookOctreeGrid from point samples over mesh faces.
        Samples will be quantized to populate the octree cells.
        This method is not guaranteed to reproduce a perfect mesh structure without "holes",
        but achieves a very good approximation with high probability.

        Args:
            mesh_path (str): Path to an OBJ file with a mesh to initialize the octree (only supports OBJ for now).
            num_samples_on_mesh (int): The number of samples to be generated on the mesh surface.
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
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth

        Returns:
            (CodebookOctreeGrid): A new instance of an CodebookOctreeGrid with occupancy initialized from
            samples over the mesh faces.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.from_mesh(mesh_path, level=max_lod, sample_tex=False, num_samples=num_samples_on_mesh)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth)
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
                        feature_bias: float = 0.0,
                        codebook_bitwidth: int = 8) -> CodebookOctreeGrid:
        """Builds the CodebookOctreeGrid, initializing it with a pointcloud.
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
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth

        Returns:
            (CodebookOctreeGrid): A new instance of an CodebookOctreeGrid with occupancy initialized from the pointcloud.
        """
        max_lod = OctreeGrid.max_octree_lod(base_lod, num_lods)
        blas = OctreeAS.from_pointcloud(pointcloud, level=max_lod)
        grid = cls(accelstruct=blas, feature_dim=feature_dim, base_lod=base_lod, num_lods=num_lods,
                   interpolation_type=interpolation_type, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth)
        return grid

    def init_feature_structure(self):
        """ Initializes everything related to the features stored in the codebook octree structure. """
        # Assumes the occupancy structure have been initialized (the BLAS: Bottom Level Accelerated Structure).
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

        self.dictionary_size = 2 ** self.bitwidth

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

    def name(self) -> str:
        return "Codebook Grid"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        parent_properties = super().public_properties()
        properties = {
            "Bitwidth": self.bitwidth
        }
        return {**parent_properties, **properties}
