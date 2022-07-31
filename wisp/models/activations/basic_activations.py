# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn

class FullSort(nn.Module):
    """The "FullSort" activation function from https://arxiv.org/abs/1811.05381.
    """
    def forward(self, x):
        """Sorts the feature dimension.
        
        Args:
            x (torch.FloatTensor): Some tensor of shape [..., feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [..., feature_size]
        """
        return torch.sort(x, dim=-1)[0]

class MinMax(nn.Module):
    """The "MinMax" activation function from https://arxiv.org/abs/1811.05381.
    """
    def forward(self, x):
        """Partially sorts the feature dimension.
        
        The feature dimension needs to be a multiple of 2.

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, feature_size]
        
        Returns:
            (torch.FloatTensor): Activation of shape [batch, feature_size]
        """
        N, M = x.shape
        x = x.reshape(N, M//2, 2)
        return torch.cat([x.min(-1, keepdim=True)[0], x.max(-1, keepdim=True)[0]], dim=-1).reshape(N, M)

class Identity(nn.Module):
    """Identity function. Occasionally useful.
    """
    def forward(self, x):
        """Returns the input. :)

        Args:
            x (Any): Anything

        Returns:
            (Any): The input!
        """
        return x

def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'none':
        return Identity()
    elif activation_type == 'fullsort':
        return FullSort()
    elif activation_type == 'minmax':
        return MinMax()
    elif activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    else:
        assert False and "activation type does not exist"
