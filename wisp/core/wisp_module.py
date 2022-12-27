# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class WispModule(nn.Module, ABC):
    """ A general interface for all Wisp building blocks, such as neural fields, grids and tracers.
        WispModules should:
        1. Provide their name & dictionary of public properties. That makes them compatible with systems like
        logging & gui.
        2. WispModules extend torch's nn.Module out of convenience.
        Modules are not required however, to implement a forward() function.
    """

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        """
        Returns:
            (str) A WispModule should be given a meaningful, human readable name.
            By default, the class name is used.
        """
        return type(self).__name__

    @abstractmethod
    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        raise NotImplementedError('Wisp modules should implement the `public_properties` method')
