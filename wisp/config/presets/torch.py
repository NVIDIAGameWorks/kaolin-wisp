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

class FusedAdam:
    def __init__(self):
        raise Exception("apex optimizer is not be available, "
                        "install from https://github.com/nvidia/apex#from-source")

try:
    import apex
    FusedAdamTarget = apex.optimizers.FusedAdam
    import_err_msg = None
except:
    import_err_msg = "Cannot load FusedAdam optimizer. The apex package is not available, install from https://github.com/nvidia/apex#from-source"
    class FusedAdam:    # Dummy, will raise an exception
        def __init__(self, *args, **kwargs):
            raise Exception(import_err_msg)
    FusedAdamTarget = FusedAdam
@configure(target=FusedAdamTarget, import_error=import_err_msg)
class ConfigFusedAdam:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@configure(target=torch.optim.Adam)
class ConfigAdam:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@configure(target=torch.optim.AdamW)
class ConfigAdamW:
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
