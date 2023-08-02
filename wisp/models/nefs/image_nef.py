# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math
from typing import Optional

from wisp.models.decoders import BasicDecoder
from wisp.models.nefs import BaseNeuralField
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.embedders import get_positional_embedder
from wisp.models.grids import BLASGrid

class ImageNeuralField(BaseNeuralField):
    """Model for encoding images.
    """
    
    def __init__(self, 
                 grid: BLASGrid,
                 activation_type: str = 'relu',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers:int = 1):
        
        super().__init__()
 
        self.grid = grid
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if self.grid.multiscale_type == 'cat':
            self.feature_dim = self.grid.feature_dim * len(self.grid.resolutions)
        else:
            self.feature_dim = self.grid.feature_dim

        self.embedder, self.embed_dim = get_positional_embedder(frequencies=3, include_input=True)
        self.embed_dim = 14
        self.input_dim = self.feature_dim + self.embed_dim

        self.decoder = BasicDecoder(self.input_dim, 3, get_activation_class(self.activation_type), True,
                                    layer=get_layer_class(self.layer_type), num_layers=self.num_layers,
                                    hidden_dim=self.hidden_dim, skip=[])

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgb, ["rgb"])

    def rgb(self, coords, lod=None):
        """Compute color for some locations

        Inputs:
            coords            : packed float tensor of shape [batch, 3]
            lod               : int of lod
        Outputs:
            float tensor of shape [batch, 3]
        """
        if lod is None:
            lod = len(self.grid.resolutions) - 1
        
        batch, _ = coords.shape
        
        feats = self.grid.interpolate(coords, lod).reshape(-1, self.feature_dim)

        embedded_pos = self.embedder(coords).view(batch, self.embed_dim)
        fpos = torch.cat([feats, embedded_pos], dim=-1)
        
        # fpos = feats

        rgb = torch.sigmoid(self.decoder(fpos))

        return rgb
