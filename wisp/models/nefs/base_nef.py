# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import inspect
from abc import abstractmethod
from typing import Dict, Any
from wisp.core import WispModule


class BaseNeuralField(WispModule):
    """The base class for all Neural Fields within Wisp.
    Neural Fields are defined as modules which take coordinates as input and output signals of some form.
    The term "Neural" is loosely used here to imply these modules are generally subject for optimization.

    The domain of neural fields in Wisp is flexible, and left up for the user to decide when implementing the subclass.
    Popular neural fields from the literature, such as Neural Radiance Fields (Mildenhall et al. 2020),
    and Neural Signed Distance Functions (SDFs) can be implemented by creating and registering
    the required forward functions (for i.e. rgb, density, sdf values).

    BaseNeuralField subclasses  usually consist of several optional components:
    - A feature grid (BLASGrid), sometimes also known as 'hybrid representations'.
      These are responsible for querying and interpolating features, often in the context of some 3D volume
      (but not limited to).
      Feature grids often employ some acceleration structure (i.e. OctreeAS),
      which can be used to accelerate spatial queries or raytracing ops,
      hence the term "BLAS" (Bottom Level Acceleration Structure).
    - A decoder (i.e. BasicDecoder) which can feeds on features (or coordinates / pos embeddings) and coverts
      them to output signals.
    - Other components such as positional embedders may be employed.

    BaseNeuralFields are generally meant to be compatible with BaseTracers, thus forming a complete pipeline of
    render-able neural primitives.
    """
    def __init__(self):
        super().__init__()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([channel for channels in self._forward_functions.values() for channel in channels])

    @property
    def device(self):
        """ Returns the device used to process inputs in this neural field.
        By default, the device is queried from the first registered torch nn.parameter.
        Override this property to explicitly specify the device.

        Returns:
            (torch.device): The expected device for inputs to this neural field.
        """
        return next(self.parameters()).device

    def _register_forward_function(self, fn, channels):
        """Registers a forward function.

        Args:
            fn (function): Function to register.
            channels (list of str): Channel output names.
        """
        if isinstance(channels, str):
            channels = [channels]
        self._forward_functions[fn] = set(channels)

    @abstractmethod
    def register_forward_functions(self):
        """Register forward functions with the channels that they output.
        
        This function should be overrided and call `self._register_forward_function` to 
        tell the class which functions output what output channels. The function can be called
        multiple times to register multiple functions.

        Example:

        ```
        self._register_forward_function(self.rgba, ["density", "rgb"])
        self._register_forward_function(self.sdf, ["sdf"])
        ```
        """
        pass

    def get_forward_function(self, channel):
        """Will return the function that will return the channel.
        
        Args: 
            channel (str): The name of the channel to return.

        Returns:
            (function): Function that will return the function. Will return None if the channel is not supported.
        """
        if channel not in self.get_supported_channels():
            raise Exception(f"Channel {channel} is not supported in {self.__class__.__name__}")
        for fn in self._forward_functions:
            output_channels = self._forward_functions[fn]
            if channel in output_channels:
                return lambda *args, **kwargs: fn(*args, **kwargs)[channel]

    def get_supported_channels(self):
        """Returns the channels that are supported by this class.

        Returns:
            (set): Set of channel strings.
        """
        return self.supported_channels

    def forward(self, channels=None, **kwargs):
        """Queries the neural field with channels.

        Args:
            channels (str or list of str or set of str): Requested channels. See return value for details.
            kwargs: Any keyword argument passed in will be passed into the respective forward functions.

        Returns:
            (list or dict or torch.Tensor): 
                If channels is a string, will return a tensor of the request channel. 
                If channels is a list, will return a list of channels.
                If channels is a set, will return a dictionary of channels.
                If channels is None, will return a dictionary of all channels.
        """
        if not (isinstance(channels, str) or isinstance(channels, list) or isinstance(channels, set) or channels is None):
            raise Exception(f"Channels type invalid, got {type(channels)}." \
                             "Make sure your arguments for the nef are provided as keyword arguments.")
        if channels is None:
            requested_channels = self.get_supported_channels()
        elif isinstance(channels, str):
            requested_channels = set([channels])
        else:
            requested_channels = set(channels)

        unsupported_channels = requested_channels - self.get_supported_channels()
        if unsupported_channels:
            raise Exception(f"Channels {unsupported_channels} are not supported in {self.__class__.__name__}")
        
        return_dict = {}
        for fn in self._forward_functions:

            output_channels = self._forward_functions[fn]
            # Filter the set of channels supported by the current forward function
            supported_channels = output_channels & requested_channels

            # Check that the function needs to be executed
            if len(supported_channels) != 0:

                # Filter args to the forward function and execute
                argspec = inspect.getfullargspec(fn)
                required_args = argspec.args[:-len(argspec.defaults)][1:] # Skip first element, self
                optional_args = argspec.args[-len(argspec.defaults):]
                
                input_args = {}
                for _arg in required_args:
                    # TODO(ttakiakwa): This doesn't actually format the string, fix :) 
                    if _arg not in kwargs:
                        raise Exception(f"Argument {_arg} not found as input to in {self.__class__.__name__}.{fn.__name__}()")
                    input_args[_arg] = kwargs[_arg]
                for _arg in optional_args:
                    if _arg in kwargs:
                        input_args[_arg] = kwargs[_arg]
                output = fn(**input_args)

                for channel in supported_channels:
                    return_dict[channel] = output[channel]
        
        if isinstance(channels, str):
            if channels in return_dict:
                return return_dict[channels]
            else:
                return None
        elif isinstance(channels, list):
            return [return_dict[channel] for channel in channels]
        else:
            return return_dict

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return dict()
