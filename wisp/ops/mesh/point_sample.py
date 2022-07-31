# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from .sample_near_surface import sample_near_surface
from .sample_surface import sample_surface
from .sample_uniform import sample_uniform
from .area_weighted_distribution import area_weighted_distribution

def point_sample(
    V : torch.Tensor, 
    F : torch.Tensor, 
    techniques : list, 
    num_samples : int):
    """Sample points from a mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique
    
    Returns:
        (torch.FloatTensor): Samples of shape [len(techniques)*num_samples, 3]
    """
    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(V, F)

    samples = []
    for technique in techniques:
        if technique =='trace':
            samples.append(sample_surface(V, F, num_samples, distrib=distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(V, F, num_samples, distrib=distrib))
        elif technique == 'rand':
            samples.append(sample_uniform(num_samples).to(V.device))
    samples = torch.cat(samples, dim=0)
    return samples

