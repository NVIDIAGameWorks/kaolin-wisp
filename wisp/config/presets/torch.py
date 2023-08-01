# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.  #
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from typing import Tuple
from wisp.config.utils import configure

# Premade configurations for various torch modules (useful for torch signatures with unreliable typing).
# How to use:
# ```
#     import from wisp.trainers import ConfigAdam
#     from wisp.config import instantiate
#
#     adam_cfg = ConfigAdam(...)                              # Usually this is initialized during parse_config()
#     optimizer = instantiate(adam_cfg, params=model.params)  # Add any missing args during instantiation
# ```

try:
    import apex
    @configure(target=apex.optimizers.FusedAdam)
    class ConfigFusedAdam:
        lr: float = 1e-3
        betas: Tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0.0
except:
    print("apex import failed. apex optimizer will not be available")
    class ConfigFusedAdam:
        def __init__(self):
            raise Exception("FusedAdam not available since apex import failed")


@configure(target=torch.optim.Adam)
class ConfigAdam:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@configure(target=torch.optim.RMSprop)
class ConfigRMSprop:
    lr: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0


@configure
class ConfigDataloader:
    batch_size: int
    """ Batch size for number of samples returned per batch. """
    num_workers: int = 0
    """ Number of cpu workers used to fetch data. """
