# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from typing import Dict, Any
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid

class NeuralRadianceField(BaseNeuralField):
    """Model for encoding Neural Radiance Fields (Mildenhall et al. 2020), e.g., density and view dependent color.
    Different to the original NeRF paper, this implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self,
                 grid: BLASGrid = None,
                 # embedder args
                 pos_embedder: str = 'none',
                 view_embedder: str = 'none',
                 pos_multires: int = 10,
                 view_multires: int = 4,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 # pruning args
                 prune_density_decay: float = None,
                 prune_min_density: float = None,
                 ):
        """
        Creates a new NeRF instance, which maps 3D input coordinates + view directions to RGB + density.

        This neural field consists of:
         * A feature grid (backed by an acceleration structure to boost raymarching speed)
         * Color & density decoders
         * Optional: positional embedders for input position coords & view directions, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation
          * Pruning scheme for HashGrids

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, TriplanarGrid, CodebookGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the density decoder.
                 - 'identity': The sample coordinates are fed as is into the density decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the density decoder.
            view_embedder (str): Type of positional embedder to use for view directions.
                Options:
                 - 'none': No positional input is fed into the color decoder.
                 - 'identity': The view directions are fed as is into the color decoder.
                 - 'positional': The view directions are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the color decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            view_multires (int): Number of frequencies used for 'positional' embedding of view_embedder.
                 Used only if view_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of both decoders.
            num_layers (int): Number of hidden layers in both decoders.
            prune_density_decay (float): Decay rate of density per "prune step",
                 using the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
            prune_min_density (float): Minimal density allowed for "cells" before they get pruned during a "prune step".
                 Used within the pruning scheme from Muller et al. 2022. Used only for grids which support pruning.
        """
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires,
                                                                   include_input=position_input)
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires,
                                                                     include_input=True)

        # Init Decoder
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder_density, self.decoder_color = \
            self.init_decoders(activation_type, layer_type, num_layers, hidden_dim)

        self.prune_density_decay = prune_density_decay
        self.prune_min_density = prune_min_density

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none' and not include_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or (embedder_type == 'none' and include_input):
            embedder, embed_dim = torch.nn.Identity(), 3    # Assumes pos / view input is always 3D
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, include_input=include_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder_density = BasicDecoder(input_dim=self.density_net_input_dim(),
                                       output_dim=16,
                                       activation=get_activation_class(activation_type),
                                       bias=True,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])
        decoder_density.lout.bias.data[0] = 1.0

        decoder_color = BasicDecoder(input_dim=self.color_net_input_dim(),
                                     output_dim=3,
                                     activation=get_activation_class(activation_type),
                                     bias=True,
                                     layer=get_layer_class(layer_type),
                                     num_layers=num_layers + 1,
                                     hidden_dim=hidden_dim,
                                     skip=[])
        return decoder_density, decoder_color

    def prune(self):
        """Prunes the blas based on current state.
        """
        if self.grid is not None:
            if isinstance(self.grid, HashGrid):
                density_decay = self.prune_density_decay
                min_density = self.prune_min_density

                self.grid.occupancy = self.grid.occupancy.cuda()
                self.grid.occupancy = self.grid.occupancy * density_decay
                points = self.grid.dense_points.cuda()
                res = 2.0**self.grid.blas_level
                samples = torch.rand(points.shape[0], 3, device=points.device)
                samples = points.float() + samples
                samples = samples / res
                samples = samples * 2.0 - 1.0
                sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                with torch.no_grad():
                    density = self.forward(coords=samples, ray_d=sample_views, channels="density")
                self.grid.occupancy = torch.stack([density[:, 0], self.grid.occupancy], -1).max(dim=-1)[0]

                mask = self.grid.occupancy > min_density

                _points = points[mask]

                if _points.shape[0] == 0:
                    return

                if hasattr(self.grid.blas.__class__, "from_quantized_points"):
                    self.grid.blas = self.grid.blas.__class__.from_quantized_points(_points, self.grid.blas_level)
                else:
                    raise Exception(f"The BLAS {self.grid.blas.__class__.__name__} does not support initialization " 
                                     "from_quantized_points, which is required for pruning.")

            else:
                raise NotImplementedError(f'Pruning not implemented for grid type {self.grid}')

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def rgba(self, coords, ray_d, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 3]
            ray_d (torch.FloatTensor): tensor of shape [batch, 3]
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
        feats = self.grid.interpolate(coords, lod_idx).reshape(batch, self.effective_feature_dim())

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(batch, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # Decode high-dimensional vectors to density features.
        density_feats = self.decoder_density(feats)

        # Concatenate embedded view directions.
        if self.view_embedder is not None:
            embedded_dir = self.view_embedder(-ray_d).view(batch, self.view_embed_dim)
            fdir = torch.cat([density_feats, embedded_dir], dim=-1)
        else:
            fdir = density_feats

        # Colors are values [0, 1] floats
        # colors ~ (batch, 3)
        colors = torch.sigmoid(self.decoder_color(fdir))

        # Density is [particles / meter], so need to be multiplied by distance
        # density ~ (batch, 1)
        density = torch.relu(density_feats[...,0:1])
        return dict(rgb=colors, density=density)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def density_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def color_net_input_dim(self):
        return 16 + self.view_embed_dim

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {
            "Grid": self.grid,
            "Pos. Embedding": self.pos_embedder,
            "View Embedding": self.view_embedder,
            "Decoder (density)": self.decoder_density,
            "Decoder (color)": self.decoder_color
        }
        if self.prune_density_decay is not None:
            properties['Pruning Density Decay'] = self.prune_density_decay
        if self.prune_min_density is not None:
            properties['Pruning Min Density'] = self.prune_min_density
        return properties
