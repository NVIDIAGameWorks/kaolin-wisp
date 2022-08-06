# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch.nn as nn
from abc import abstractmethod, ABC
import inspect

class BaseTracer(nn.Module, ABC):
    """Virtual base class for tracer"""
    output_channels = None
    input_channels = None
    
    def __init__(self):
        """Initializes the class.
        """
        super().__init__()

    def set_defaults(self):
        """Sets the default arguments for trace. 

        This should be overrided and called if you want to pass custom defaults into the renderer.
        If overrided, it should keep the arguments to `self.trace` in `self.` class variables.
        Then, if these variables exist and no function arguments are passed into forward,
        it will override them as the default.

        TODO(ttakikawa): This is hacky.
        """
        pass

    @abstractmethod
    def get_output_channels(self):
        """Returns the output channels that are supported by this class.

        Implement the function to return the supported channels, e.g.       
        return set(["depth", "rgb"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def get_input_channels(self):
        """Returns the input channels that are supported by this class.
        
        Implement the function to return the supported channels, e.g.       
        return set(["rgb", "density"])

        Returns:
            (set): Set of channel strings.
        """
        pass


    @abstractmethod
    def trace(self, nef, channels, *args, **kwargs):
        """Apply the forward map on the nef. 

        This is the function to implement to implement a custom
        This can take any number of arguments, but `nef` always needs to be the first argument and 
        `channels` needs to be the second argument.
        
        Args:
            nef (nn.Module): A neural field that uses a grid class.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        pass

    def forward(self, nef, channels=None, **kwargs):
        """Queries the tracer with channels.

        Args:
            channels (str or list of str or set of str): Requested channels. See return value for details.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        nef_channels = nef.get_supported_channels()
        unsupported_inputs = self.get_input_channels() - nef_channels
        if unsupported_inputs:
            raise Exception(f"The neural field class {type(nef)} does not support the required channels {unsupported_inputs}.")
        
        if channels is None:
            requested_channels = self.get_output_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        unsupported_outputs = requested_channels - self.get_output_channels()
        if unsupported_outputs:
            raise Exception(f"Channels {unsupported_outputs} are not supported in {type(self)}")

        argspec = inspect.getfullargspec(self.trace)

        # Skip first element (self), second element (nef) and third element (channel)
        required_args = argspec.args[:-len(argspec.defaults)][3:] 
        optional_args = argspec.args[-len(argspec.defaults):]
        
        input_args = {}
        for _arg in required_args:
            # TODO(ttakiakwa): This doesn't actually format the string, fix :) 
            if _arg not in kwargs:
                raise Exception(f"Argument {_arg} not found as input to in {type(self)}.trace()")
            input_args[_arg] = kwargs[_arg]
        for _arg in optional_args:
            if _arg in kwargs:
                # By default, the function args will take priority
                input_args[_arg] = kwargs[_arg]
            else:
                # Check if default_args are set, and use them if they are.
                default_arg = getattr(self, _arg, None)
                if default_arg is not None:
                    input_args[_arg] = default_arg

        return self.trace(nef, requested_channels, **input_args)
