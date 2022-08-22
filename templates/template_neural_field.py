# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.models.nefs import BaseNeuralField
from wisp.models.activations import get_activation_class
from wisp.models.embedders import PositionalEmbedder
from wisp.models.grids import *


class TemplateNeuralField(BaseNeuralField):
    """ An exemplary template for quick creation of new user neural fields.
        Clone this file and modify to create your own customized neural field.
    """

    def init_embedder(self):
        """ Create your positional embedding here, for example: """
        self.pos_embedder = PositionalEmbedder(num_freq=10,
                                               max_freq_log2=9,
                                               log_sampling=True,
                                               include_input=True,
                                               input_dim=3)

    def init_decoder(self):
        """Create here any decoder networks to be used by the neural field.
        Decoders should map from features to output values (such as: rgb, density, sdf, etc), for example:
        """
        # Determine: What is the effective feature dimensions?
        # (are we using concatenation or summation to consolidate features from multiple LODs?)
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        elif self.multiscale_type == 'sum':
            self.effective_feature_dim = self.grid.feature_dim
        else:
            raise NotImplementedError('This neural field supports only concatenation or summation '
                                      'of features from multiple LODs')

        # The input to the decoder is the effective feature dimension + positional embedding
        self.input_dim = self.effective_feature_dim + self.pos_embedder.out_dim

        # The actual decoder is created here
        self.decoder = BasicDecoder(input_dim=self.input_dim,
                                    output_dim=4,   # RGBA
                                    activation=get_activation_class(self.activation_type),
                                    bias=True,
                                    layer=nn.Linear,
                                    num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim,
                                    skip=[])

    def init_grid(self):
        """ Creates the feature structure this neural field uses, i.e: Octree, Triplane, Hashed grid and so forth.
        The feature grid is queried with coordinate samples during ray tracing / marching.
        The feature grid may also include an occupancy acceleration structure internally to speed up
        tracers.
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
        """Returns a text keyword describing the neural field type.

        Returns:
            (str): The key type
        """
        return 'template_nerf'

    def register_forward_functions(self):
        """Register the forward functions.
        Forward functions define the named output channels this neural field supports.
        By registering forward functions, a tracer knows which neural field methods to use to obtain channels values.
        """
        # Here the rgba() function handles both the rgb and density channels at the same time
        self._register_forward_function(self.rgba, ["density", "rgb"])

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

        # Concat the positions to the embedding, see also nerf.py for an example of embedded view directions.
        fdir = torch.cat([feats, self.pos_embedder(coords.reshape(-1, 3))], dim=-1)
        timer.check("rf_rgba_embed_cat")
        
        # Decode high-dimensional vectors to RGBA.
        rgba = self.decoder(fdir)
        timer.check("rf_rgba_decode")

        # Colors are values [0, 1] floats
        colors = torch.sigmoid(rgba[...,:3]).reshape(batch, num_samples, -1)

        # Density is [particles / meter], so need to be multiplied by distance
        density = torch.relu(rgba[...,3:4]).reshape(batch, num_samples, -1)
        timer.check("rf_rgba_activation")
        
        return dict(rgb=colors, density=density)
