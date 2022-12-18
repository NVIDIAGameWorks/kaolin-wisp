# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import logging as log

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import *


class NeuralSDFTex(BaseNeuralField):
    """Model for encoding neural signed distance functions + plenoptic color, e.g., implicit surfaces with albedo.
    """

    def __init__(self,
                 grid: BLASGrid = None,
                 # embedder args
                 embedder_type: str = 'none',
                 pos_multires: int = 10,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1
                 ):
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.embedder_type = embedder_type
        self.pos_multires = pos_multires
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(embedder_type, pos_multires)

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.position_input = embedder_type != 'none'
        self.decoder, self.effective_feature_dim, self.input_dim = \
            self.init_decoder(activation_type, layer_type, num_layers, hidden_dim,
                              self.position_input, self.pos_embed_dim)

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, pos_multires):
        """Creates positional embedding functions for the position and view direction.
        """
        is_active = embedder_type == "positional"
        pos_embedder, pos_embed_dim = get_positional_embedder(frequencies=pos_multires, active=is_active)
        log.info(f"Position Embed Dim: {pos_embed_dim}")
        return pos_embedder, pos_embed_dim

    def init_decoder(self, activation_type, layer_type, num_layers, hidden_dim, position_input, pos_embed_dim):
        """Initializes the decoder object.
        """
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        
        input_dim = effective_feature_dim
        if position_input:
            input_dim += pos_embed_dim

        decoder = BasicDecoder(input_dim=input_dim,
                               output_dim=4,
                               activation=get_activation_class(activation_type),
                               bias=True,
                               layer=get_layer_class(layer_type),
                               num_layers=num_layers,
                               hidden_dim=hidden_dim,
                               skip=[])
        return decoder, effective_feature_dim, input_dim

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgbsdf, ["rgb", "sdf"])

    def rgbsdf(self, coords, lod_idx=None):
        """Computes the RGB + SDF for some samples.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
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

        feats = self.grid.interpolate(coords, lod_idx)

        if self.position_input:
            feats = torch.cat([self.pos_embedder(coords), feats], dim=-1)

        rgbsdf = self.decoder(feats)

        if len(shape) == 2:
            rgbsdf = rgbsdf[:,0]
            
        return dict(rgb=torch.sigmoid(rgbsdf[...,:3]), sdf=rgbsdf[...,3:4])
