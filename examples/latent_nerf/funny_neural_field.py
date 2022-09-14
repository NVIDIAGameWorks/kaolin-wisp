# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import PositionalEmbedder
from wisp.models.grids import *


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


class FunnyNeuralField(BaseNeuralField):
    """Model for encoding radiance fields (density and plenoptic color),
    the decoder is specialized to output sigmoid latents
    """

    def init_embedder(self):
        """Creates here positional embedding functions for the position and view direction. """
        # Create your positional embedding here, for example:
        self.pos_embedder = PositionalEmbedder(num_freq=10,
                                               max_freq_log2=9,
                                               log_sampling=True,
                                               include_input=True,
                                               input_dim=3)

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field """

        # Create your decoder from features to output values here (such as: rgb, density, sdf, etc), for example:

        # Determine: What is the effective feature dimensions?
        # (are we using concatenation or summation to consolidate features from multiple LODs?)
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        elif self.multiscale_type == 'sum':
            self.effective_feature_dim = self.grid.feature_dim

        # If using positional embedding, the input to the decoder has additional dimensions:
        self.input_dim = self.effective_feature_dim + self.pos_embedder.out_dim

        self.density_decoder = SigDecoder(input_dim=self.input_dim,
                                          output_dim=1,  # Density
                                          activation=torch.relu,
                                          bias=True,
                                          hidden_dim=128)
        self.rgb_decoder = SigDecoder(input_dim=self.input_dim,
                                      output_dim=3,   # RGB
                                      activation=torch.relu,
                                      bias=True,
                                      hidden_dim=128)

    def init_grid(self):
        """ Creates the feature the grid object. """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        else:
            grid_class = HashGrid

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

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

    ## -- Your new functions here --

    def rgba(self, coords, ray_d, pidx=None, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3]
                - Density tensor of shape [batch, num_samples, 1]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_rgba_preprocess")

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("rf_rgba_interpolate")

        # Optionally concat the positions to the embedding, and also concatenate embedded view directions.
        fdir = torch.cat([feats, self.pos_embedder(coords.reshape(-1, 3))], dim=-1)
        timer.check("rf_rgba_embed_cat")

        # Decode high-dimensional vectors to RGBA.
        rgb = self.rgb_decoder(fdir)
        alpha = self.density_decoder(fdir)
        timer.check("rf_rgba_decode")

        # Colors are values [0, 1] floats
        colors = torch.sigmoid(rgb).reshape(batch, num_samples, -1)

        # Density is [particles / meter], so need to be multiplied by distance
        density = torch.relu(alpha).reshape(batch, num_samples, -1)
        timer.check("rf_rgba_activation")

        return dict(rgb=colors, density=density)

    def color_feature(self, coords, ray_d, pidx=None, lod_idx=None):
        """Compute the color's latent feature [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.

        Returns:
            {"color_feature": torch.FloatTensor}:
                - 3 Latent dimensions tensor of shape [batch, num_samples, 3]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_rgba_preprocess")

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("rf_rgba_interpolate")

        # Optionally concat the positions to the embedding, and also concatenate embedded view directions.
        fdir = torch.cat([feats, self.pos_embedder(coords.reshape(-1, 3))], dim=-1)
        timer.check("rf_rgba_embed_cat")

        # Decode high-dimensional vectors to RGBA.
        color_feature = self.rgb_decoder.forward_feature(fdir)
        color_feature = color_feature.reshape(batch, num_samples, -1)

        return dict(color_feature=color_feature)
