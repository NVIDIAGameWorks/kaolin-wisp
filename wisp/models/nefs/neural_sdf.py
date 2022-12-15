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
from wisp.models.grids import BLASGrid


class NeuralSDF(BaseNeuralField):
    """Model for encoding neural signed distance functions (implicit surfaces).
    This field implementation uses feature grids for faster and more efficient queries.
    For example, the usage of Octree follows the idea from Takikawa et al. 2021 (Neural Geometric Level of Detail).
    """
    def __init__(self,
                 grid: BLASGrid = None,
                 # embedder args
                 pos_embedder: str = 'none',
                 pos_multires: int = 10,
                 position_input: bool = True,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1
                 ):
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.pos_multires = pos_multires
        self.position_input = position_input
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires, position_input)

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder = self.init_decoder(activation_type, layer_type, num_layers, hidden_dim)

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, position_input=True):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none':
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity':
            embedder, embed_dim = torch.nn.Identity(), 0
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, position_input=position_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralSDF: {embedder_type}')
        return embedder, embed_dim

    def init_decoder(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder = BasicDecoder(input_dim=self.decoder_input_dim,
                               output_dim=1,
                               activation=get_activation_class(activation_type),
                               bias=True,
                               layer=get_layer_class(layer_type),
                               num_layers=num_layers,
                               hidden_dim=hidden_dim,
                               skip=[])
        return decoder

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.sdf, ["sdf"])

    def sdf(self, coords, lod_idx=None):
        """Computes the Signed Distance Function for input samples.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Outputs:
            (torch.FloatTensor):
            - SDF of shape [batch, num_samples, 1]
        """
        shape = coords.shape
        
        if shape[0] == 0:
            return dict(sdf=torch.zeros_like(coords)[...,0:1])

        if lod_idx is None:
            lod_idx = self.grid.num_lods - 1

        if len(shape) == 2:
            coords = coords[:, None]
        num_samples = coords.shape[1]

        feats = self.grid.interpolate(coords, lod_idx)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            feats = torch.cat([self.pos_embedder(coords.view(-1, 3)).view(-1, num_samples, self.pos_embed_dim), 
                               feats], dim=-1)

        sdf = self.decoder(feats)

        if len(shape) == 2:
            sdf = sdf[:,0]
            
        return dict(sdf=sdf)

    @property
    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    @property
    def decoder_input_dim(self):
        input_dim = self.effective_feature_dim
        if self.position_input:
            input_dim += self.pos_embed_dim
        return input_dim