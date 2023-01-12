# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from abc import abstractmethod, ABC
from typing import Dict, Any
import inspect
import torch.nn as nn
from wisp.core import Rays
from wisp.core import WispModule


class BaseTracer(WispModule, ABC):
    """Base class for all tracers within Wisp.
    Tracers drive the mapping process which takes an input "Neural Field", and outputs a RenderBuffer of pixels.
    Different tracers may employ different algorithms for querying points, tracing and marching rays through the
    neural field.
    A common paradigm for tracers to employ is as follows:
    1. Take input in the form of rays
    2. Generate samples by tracing / marching rays, or querying coordinates over the neural field.
       Possibly make use of the neural field spatial structure for high performance.
    3. Invoke neural field's methods to decode sample features into actual channel values, such as color, density,
       signed distance, and so forth.
    4. Aggregate the sample values to decide on the final pixel value.
       The exact output may depend on the requested channel type, blending mode or other parameters.
    Wisp tracers are therefore flexible, and designed to be compatible with specific neural fields,
    depending on the forward functions they support and internal grid structures they use.
    Tracers are generally expected to be differentiable (e.g. they're part of the training loop),
    though non-differentiable tracers are also allowed.
    """

    def __init__(self):
        """Initializes the tracer class and sets the default arguments for trace.
        This should be overrided and called if you want to pass custom defaults into the renderer.
        If overridden, it should keep the arguments to `self.trace` in `self.` class variables.
        Then, if these variables exist and no function arguments are passed into forward,
        it will override them as the default.
        """
        super().__init__()

    @abstractmethod
    def get_supported_channels(self):
        """Returns the set of channel names this tracer may output.

        Implement the function to return the supported channels, e.g.       
        return set(["depth", "rgb"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def get_required_nef_channels(self):
        """Returns the channels required by neural fields to be compatible with this tracer.
        
        Implement the function to return the required channels, e.g.
        return set(["rgb", "density"])

        Returns:
            (set): Set of channel strings.
        """
        pass

    @abstractmethod
    def trace(self, nef, rays, channels, extra_channels, *args, **kwargs):
        """Apply the forward map on the nef. 

        Tracers are required to implement this function, which commonly follows these paradigm:
        1. Take input in the form of rays
        2. Generate samples by tracing / marching rays, or querying coordinates over the neural field.
           Possibly make use of the neural field spatial structure for high performance.
        3. Invoke neural field's methods to decode sample features into actual channel values, such as color, density,
           signed distance, and so forth.
        4. Aggregate the sample values to decide on the final pixel value.
           The exact output may depend on the requested channel type, blending mode or other parameters.
        
        Args:
            nef (nn.Module): A neural field that uses a grid class.
            rays (Rays): Pack of rays to trace through the neural field.
            channels (set): The set of requested channels. The trace method can return channels that 
                            were not requested since those channels often had to be computed anyways.
            extra_channels (set): Requested extra channels, which are not first class channels supported by
                the tracer but will still be able to handle with some fallback options.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        pass

    def forward(self, nef, rays: Rays, channels=None, **kwargs):
        """Queries the tracer with channels.

        Args:
            nef (BaseNeuralField): Neural field to be traced. The nef will be queried for decoded sample values.
            rays (Rays): Pack of rays to trace through the neural field.
            channels (str or list of str or set of str): Requested channel names.
            This list should include at least all channels in tracer.get_supported_channels(),
            and may include extra channels in addition.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (wisp.RenderBuffer): A dataclass which holds the output buffers from the tracer.
        """
        nef_channels = nef.get_supported_channels()
        unsupported_inputs = self.get_required_nef_channels() - nef_channels
        if unsupported_inputs:
            raise Exception(f"The neural field class {type(nef)} does not output the required channels {unsupported_inputs}.")

        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)
        extra_channels = requested_channels - self.get_supported_channels()
        unsupported_outputs = extra_channels - nef_channels
        if unsupported_outputs:
            raise Exception(f"Channels {unsupported_outputs} are not supported in the tracer {type(self)} or neural field {type(nef)}.")
    
        if extra_channels is None:
            requested_extra_channels = set()
        elif isinstance(extra_channels, str):
            requested_extra_channels = set([extra_channels])
        else:
            requested_extra_channels = set(extra_channels)

        argspec = inspect.getfullargspec(self.trace)

        # Skip self, nef, rays, channel, extra_channels
        required_args = argspec.args[:-len(argspec.defaults)][5:]   # TODO (operel): this is brittle
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
        return self.trace(nef, rays, requested_channels, requested_extra_channels, **input_args)

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return dict()
