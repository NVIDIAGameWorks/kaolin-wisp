# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from wisp.models.nefs import BaseNeuralField
from wisp.models.grids import BLASGrid
from wisp.models.embedders import get_positional_embedder


class FunnyNeuralField(BaseNeuralField):
    """Model for encoding radiance fields (density and plenoptic color),
    the decoder is specialized to output sigmoid latents
    """
    def __init__(self, grid: BLASGrid = None):
        super().__init__()
        self.grid = grid
        self.pos_embedder, self.pos_embed_dim = self.init_embedder()
        self.density_decoder, self.rgb_decoder = self.init_decoders()

    def init_embedder(self):
        """Creates positional embedding functions for input coordinates"""
        embedder, embed_dim = get_positional_embedder(frequencies=10, input_dim=3, include_input=True)
        return embedder, embed_dim

    def init_decoders(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders are used to convert features to output values (such as: rgb, density, sdf, etc), for example:
         """
        density_decoder = SigDecoder(input_dim=self.input_dim,
                                     output_dim=1,  # Density
                                     activation=torch.relu,
                                     bias=True,
                                     hidden_dim=128)
        rgb_decoder = SigDecoder(input_dim=self.input_dim,
                                 output_dim=3,   # RGB
                                 activation=torch.relu,
                                 bias=True,
                                 hidden_dim=128)
        return density_decoder, rgb_decoder

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])
        self._register_forward_function(self.color_feature, ["color_feature"])

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'funny_nerf'

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
                - Density tensor of shape [batch, 1]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            positional_embedding = self.pos_embedder(coords.reshape(-1, 3))
            feats = torch.cat([feats, positional_embedding], dim=-1)

        # Decode high-dimensional vectors to RGBA.
        rgb = self.rgb_decoder(feats)
        alpha = self.density_decoder(feats)

        # Colors are values [0, 1] floats
        colors = torch.sigmoid(rgb).reshape(batch, 3)
        # Density is [particles / meter], so need to be multiplied by distance
        density = torch.relu(alpha).reshape(batch, 1)

        return dict(rgb=colors, density=density)

    def color_feature(self, coords, ray_d, lod_idx=None):
        """Compute the color's latent feature [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"color_feature": torch.FloatTensor}:
                - 3 Latent dimensions tensor of shape [batch, 3]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            positional_embedding = self.pos_embedder(coords.reshape(-1, 3))
            feats = torch.cat([feats, positional_embedding], dim=-1)

        # Decode high-dimensional vectors to RGBA.
        color_feature = self.rgb_decoder.forward_feature(feats)
        color_feature = color_feature.reshape(batch, 3)

        return dict(color_feature=color_feature)

    @property
    def effective_feature_dim(self):
        """ Determine: What is the effective feature dimension?
        (are we using concatenation or summation to consolidate features from multiple LODs?)
        """
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        elif self.grid.multiscale_type == 'sum':
            effective_feature_dim = self.grid.feature_dim
        else:
            raise NotImplementedError(f'FunnyNeuralField does not support multiscale type: {self.grid.multiscale_type}')
        return effective_feature_dim

    @property
    def input_dim(self):
        """ Calculates the decoder input dim.
        If using positional embedding, the input to the decoder has additional dimensions
        """
        input_dim = self.effective_feature_dim
        if self.pos_embedder is not None:
            input_dim += self.pos_embedder.out_dim
        return input_dim


class SigDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation, bias):
        """Initialize the SigDecoder.
        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            hidden_dim (int): Hidden dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.hidden_layer = nn.Linear(self.input_dim, hidden_dim, bias=bias)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim, bias=bias)

    def forward_feature(self, x):
        """A specialized forward function for the MLP, to obtain 3 hidden channels, post sigmoid activation.
        after

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., 3]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        return x_h[..., :3]

    def forward(self, x):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]

        Returns:
            (torch.FloatTensor): The output tensor of shape [batch, ..., output_dim]
        """
        x_h = self.hidden_layer(x)
        x_h[..., :3] = torch.sigmoid(x_h[..., :3])
        x_h[..., 3:] = self.activation(x_h[..., 3:])
        out = self.output_layer(x_h)
        return out
