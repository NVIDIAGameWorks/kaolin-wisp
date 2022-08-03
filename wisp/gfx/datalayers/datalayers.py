# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Dict
from wisp.core import PrimitivesPack

class Datalayers(ABC):

    @abstractmethod
    def needs_redraw(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    @abstractmethod
    def regenerate_data_layers(self, *args, **kwargs) -> Dict[str, PrimitivesPack]:
        raise NotImplementedError
