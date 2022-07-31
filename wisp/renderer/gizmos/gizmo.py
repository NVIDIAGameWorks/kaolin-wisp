# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from abc import ABC, abstractmethod
from kaolin.render.camera import Camera


class Gizmo(ABC):
    """
    A template for representing gizmos the interactive renderer is able to render over the canvas.
    Gizmos are entities rendered by the graphics api (i.e: OpenGL).
    Normally they're used to draw transient markups or tools over the canvas (such as a world grid or axes).
    """

    @abstractmethod
    def render(self, camera: Camera):
        """ Renders the gizmo using the graphics api. """
        raise NotImplementedError("Gizmos must implement the render function")

    @abstractmethod
    def destroy(self):
        """ Release GL resources, must be called from the rendering thread which owns the GL context """
        raise NotImplementedError("Gizmos must implement the destroy function")
