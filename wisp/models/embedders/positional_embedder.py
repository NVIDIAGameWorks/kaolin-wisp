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


class PositionalEmbedder(WispModule):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers.
    """
    def __init__(self, num_freq, max_freq_log2, log_sampling=True, include_input=True, input_dim=3):
        """Initialize the module.

        Args:
            num_freq (int): The number of frequency bands to sample. 
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = 2.0**torch.linspace(0.0, max_freq_log2, steps=num_freq)
        else:
            self.bands = torch.linspace(1, 2.0**max_freq_log2, steps=num_freq)

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)
    
    def forward(self, coords):
        """Embeds the coordinates.

        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        winded = (coords[:,None] * self.bands[None,:,None]).reshape(
            N, coords.shape[1] * self.num_freq)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """ A human readable name for the given wisp module. """
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input
        }


def get_positional_embedder(frequencies, input_dim=3, include_input=True):
    """Utility function to get a positional encoding embedding.

    Args:
        frequencies (int): The number of frequencies used to define the PE:
            [2^0, 2^1, 2^2, ... 2^(frequencies - 1)].
        input_dim (int): The input coordinate dimension.
        include_input (bool): If true, will concatenate the input coords.

    Returns:
        (nn.Module, int):
        - The embedding module
        - The output dimension of the embedding.
    """
    encoder = PositionalEmbedder(frequencies, frequencies-1, input_dim=input_dim, include_input=include_input)
    return encoder, encoder.out_dim
