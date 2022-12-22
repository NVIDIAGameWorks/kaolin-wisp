# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid
from wisp.accelstructs import OctreeAS
import kaolin.ops.spc as spc_ops


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
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.position_input = position_input
        if self.position_input:
            self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires)
        else:
            self.pos_embedder, self.pos_embed_dim = None, 0
        self.view_embedder, self.view_embed_dim = self.init_embedder(view_embedder, view_multires)

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

    def init_embedder(self, embedder_type, frequencies=None):
        """Creates positional embedding functions for the position and view direction.
        """
        if embedder_type == 'none':
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity':
            embedder, embed_dim = torch.nn.Identity(), 0
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralRadianceField: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder_density = BasicDecoder(input_dim=self.density_net_input_dim,
                                       output_dim=16,
                                       activation=get_activation_class(activation_type),
                                       bias=True,
                                       layer=get_layer_class(layer_type),
                                       num_layers=num_layers,
                                       hidden_dim=hidden_dim,
                                       skip=[])
        decoder_density.lout.bias.data[0] = 1.0

        decoder_color = BasicDecoder(input_dim=self.color_net_input_dim,
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

                # TODO (operel): This will soon change to support other blas types
                octree = spc_ops.unbatched_points_to_octree(_points, self.grid.blas_level, sorted=True)
                self.grid.blas = OctreeAS(octree)
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
        feats = self.grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(-1, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # Decode high-dimensional vectors to density features.
        density_feats = self.decoder_density(feats)

        # Concatenate embedded view directions.
        if self.view_embedder is not None:
            embedded_dir = self.view_embedder(-ray_d).view(-1, self.view_embed_dim)
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

    @property
    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    @property
    def density_net_input_dim(self):
        return self.effective_feature_dim + self.pos_embed_dim

    @property
    def color_net_input_dim(self):
        return 16 + self.view_embed_dim
