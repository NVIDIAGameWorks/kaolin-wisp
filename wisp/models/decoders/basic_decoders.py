# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Dict, Any
import torch
import torch.nn as nn
from wisp.core import WispModule

from scipy.stats import ortho_group

class BasicDecoder(WispModule):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation,
        bias,
        layer = nn.Linear,
        num_layers = 1, 
        hidden_dim = 128, 
        skip       = []
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = self.activation(l(h))
                h = torch.cat([x, h], dim=-1)
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        ms = []
        for i, w in enumerate(self.layers):
            m = get_weight(w.weight)
            ms.append(m)
        for i in range(len(self.layers)):
            self.layers[i].weight = nn.Parameter(ms[i])
        m = get_weight(self.lout.weight)
        self.lout.weight = nn.Parameter(m)

    def name(self) -> str:
        """ A human readable name for the given wisp module. """
        return "BasicDecoder"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Input Dim": self.input_dim,
            "Hidden Dim": self.hidden_dim,
            "Outpt Dim": self.output_dim,
            "Num. Layers": self.num_layers,
            "Layer Type": self.layer.__name__,
            "Activation": self.activation.__name__,
            "Bias": self.bias,
            "Skip Connections": self.skip,
        }


def orthonormal(weight):
    """Initialize the layer as a random orthonormal matrix.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N]. Only used for the shape.

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    m = ortho_group.rvs(dim=max(weight.shape))
    #m = np.dot(m.T, m)
    m = m[:weight.shape[0],:weight.shape[1]]
    return torch.from_numpy(m).float()

def svd(weight):
    """Initialize the layer with the U,V of SVD.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    U,S,V = torch.svd(weight)
    return torch.matmul(U, V.T)

def spectral_normalization(weight):
    """Initialize the layer with spectral normalization.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    U,S,V = torch.svd(weight)
    return weight / S.max()

def identity(weight):
    """Initialize the layer with identity matrix.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    return torch.diag(torch.ones(weight.shape[0]))

def average(weight):
    """Initialize the layer by normalizing the weights.

    Args:
        weight (torch.FloatTensor): Matrix of shape [M, N].

    Returns:
        (torch.FloatTensor): Matrix of shape [M, N].
    """
    return weight / weight.sum()

