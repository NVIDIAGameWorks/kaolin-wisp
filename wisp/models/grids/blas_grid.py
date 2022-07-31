# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn

class BLASGrid(nn.Module):
    """This is an abstract base class for a feature grid that uses an accelstruct to speed up operations.
    
    Classes which inherit the BLASGrid are generally compatible with PackedTracers.
    """

    def blas_initialized(self):
        """This function is used to determine if the BLAS needs to be built.

        Will return False if the blas construction is deferred.
        """
        return self.blas.initialized

    def raymarch(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raymarch(*args, **kwargs)

    def raytrace(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.raytrace(*args, **kwargs)

    def query(self, *args, **kwargs):
        """By default, this function will use the equivalent BLAS function unless overridden for custom behaviour.
        """
        return self.blas.query(*args, **kwargs)
