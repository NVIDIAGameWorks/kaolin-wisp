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

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import *

class NeuralSDFTex(BaseNeuralField):
    """Model for encoding implicit surfaces (usually SDF) with textures.
    """

    def init_embedder(self):
        """Creates positional embedding functions for the position and view direction.
        """
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(self.pos_multires, 
                                                                       self.embedder_type == "positional")
        log.info(f"Position Embed Dim: {self.pos_embed_dim}")

    def init_decoder(self):
        """Initializes the decoder object.
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim *= self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim
        
        self.input_dim = self.effective_feature_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        self.decoder = BasicDecoder(self.input_dim, 4, get_activation_class(self.activation_type), True,
                                    layer=get_layer_class(self.activation_type), num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, skip=[])

    def init_grid(self):
        """Initialize the grid object.
        """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'sdf'

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgbsdf, ["rgb", "sdf"])

    def rgbsdf(self, coords, pidx=None, lod_idx=None):
        """Computes the RGB + SDF for some samples.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            {"rgb": torch.FloatTensor, "sdf": torch.FloatTensor}:
            - RGB of shape [batch, num_samples, 3]
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        
        if shape[0] == 0:
            return dict(rgb=torch.zeros_like(coords)[...,:3], sdf=torch.zeros_like(coords)[...,0:1])

        if lod_idx is None:
            lod_idx = self.num_lods - 1
        
        if len(shape) == 2:
            coords = coords[:, None]

        # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
        feats = self.grid.interpolate(coords, lod_idx, pidx=pidx)

        if self.position_input:
            feats = torch.cat([self.pos_embedder(coords), feats], dim=-1)

        rgbsdf = self.decoder(feats)

        if len(shape) == 2:
            rgbsdf = rgbsdf[:,0]
            
        return dict(rgb=torch.sigmoid(rgbsdf[...,:3]), sdf=rgbsdf[...,3:4])


class NeuralSDF(BaseNeuralField):
    """Model for encoding implicit surfaces (usually SDF).
    """
    
    def init_embedder(self):
        """Creates positional embedding functions for the position and view direction.
        """
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(self.pos_multires, 
                                                                       self.embedder_type == "positional")
        log.info(f"Position Embed Dim: {self.pos_embed_dim}")

    def init_decoder(self):
        """Initializes the decoder object.
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim *= self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim
        
        self.input_dim = self.effective_feature_dim

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        self.decoder = BasicDecoder(self.input_dim, 1, get_activation_class(self.activation_type), True,
                                    layer=get_layer_class(self.layer_type), num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, skip=[])

    def init_grid(self):
        """Initialize the grid object.
        """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'sdf'

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.sdf, ["sdf"])

    def sdf(self, coords, pidx=None, lod_idx=None):
        """Computes the RGB + SDF for some samples.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            (torch.FloatTensor):
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        
        if shape[0] == 0:
            return dict(sdf=torch.zeros_like(coords)[...,0:1])

        if lod_idx is None:
            lod_idx = self.num_lods - 1
        
        # TODO(ttakikawa): Does the SDF really need num_samples? Note to myself to rethink this through...
        if len(shape) == 2:
            coords = coords[:, None]
        num_samples = coords.shape[1]

        # TODO(ttakikawa): this should return [batch, ns, f] but it returns [batch, f]
        feats = self.grid.interpolate(coords, lod_idx, pidx=pidx)

        if self.position_input:
            feats = torch.cat([self.pos_embedder(coords.view(-1, 3)).view(-1, num_samples, self.pos_embed_dim), 
                               feats], dim=-1)

        sdf = self.decoder(feats)

        if len(shape) == 2:
            sdf = sdf[:,0]
            
        return dict(sdf=sdf)
