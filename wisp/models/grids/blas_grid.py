# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from abc import ABC, abstractmethod
from typing import Dict, Any, Set, Type
from wisp.core import WispModule
from wisp.accelstructs import BaseAS, ASQueryResults, ASRaytraceResults, ASRaymarchResults


class BLASGrid(WispModule, ABC):
    """
    BLASGrids (commonly referred in documentation as simply "grids"), represent feature grids in Wisp.
    BLAS: "Bottom Level Acceleration Structure", to signify this structure is the backbone that captures
    a neural field's contents, in terms of both features and occupancy for speeding up queries.

    This is an abstract base class that uses some spatial acceleration structure under the hood, to speed up operations
    such as coordinate based queries or ray tracing.
    Classes which inherit the BLASGrid are generally compatible with BaseTracers to support such operations
    (see: raymarch(), raytrace(), query()).

    Grids are usually employed as building blocks within neural fields (see: BaseNeuralField),
    possibly paired with decoders to form a neural field.
    """

    def __init__(self, blas: BaseAS):
        """ BLASGrids are generally assumed to contain bottom level acceleration structures. """
        super().__init__()
        self.blas = blas

    def raymarch(self, *args, **kwargs) -> ASRaymarchResults:
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raymarch(*args, **kwargs)

    def raytrace(self, *args, **kwargs) -> ASRaytraceResults:
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raytrace(*args, **kwargs)

    def query(self, *args, **kwargs) -> ASQueryResults:
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.query(*args, **kwargs)

    @abstractmethod
    def interpolate(self, coords, lod_idx):
        """ Interpolates a feature value for the given coords using the grid support, in the given lod_idx
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
            lod_idx  (int): int specifying the index to the desired level of detail, if supported.
        """
        raise NotImplementedError('A BLASGrid should implement the interpolation functionality according to '
                                  'the grid structure.')

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return set()

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.

        BLASGrids are generally assumed to contain a bottom level acceleration structure.
        """
        return {
            "Acceleration Structure": self.blas
        }
