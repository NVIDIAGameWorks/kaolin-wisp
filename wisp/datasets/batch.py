# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from attrdict import AttrDict
from typing import Optional, List, Dict, Any
import torch
from kaolin.render.camera import Camera
from wisp.core import Rays


class Batch(AttrDict):
    """ Represents a single batch of information sampled and collated from a WispDataset.
    Batches in Wisp keep a general structure by subclassing python's dictionaries and using their semantics.
    The exact fields each batch contain depend on the dataset type.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    @property
    def fields(self) -> List[str]:
        """ Returns a list of all field names the batch currently contains.
        Note that the content of some field values may be None in practice.
        """
        return list(self.keys())


class MultiviewBatch(Batch):
    """ Represents a single batch of multiview information sampled and collated from a WispDataset.
    MultiviewBatch objects are always assumed to carry ray information, and may optionally return the camera information
    used to generate them.
    Batches in Wisp keep a general structure by subclassing python's dictionaries and using their semantics,
    so the exact channels of information each ray carries is flexible and are left up to datasets to initialize.
    Callers should be mindful of which fields are available by calling the `fields()` function.
    Out of convenience, the common rgb channels is explicitly listed here (though are left as optional).
    """

    def __init__(self,
                 rays: Rays,
                 cameras: Optional[List[Camera]] = None,
                 rgb: Optional[torch.Tensor] = None,    # RGB is default for convenience, but not mandatory
                 *args, **kwargs):
        """
        Creates a new MultiviewBatch of ray samples and optional supervision signals they carry.

        Args:
            rays (wisp.core.Rays): a pack of ray origins and directions, usually generated from the dataset cameras.
            cameras (kaolin.render.camera.Camera): Optional.
                The list of Camera objects used to generate the rays in this batch.
            rgb (torch.Tensor): Optional.
                A torch.Tensor of rgb color which corresponds the the gt image pixel rgb each ray intersects.

                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the the gt image pixel rgb
                    each ray intersects.
                *args, **kwargs - may specify any additional channels of information a ray or view can carry.
        """
        super().__init__(rays=rays, cameras=cameras, rgb=rgb)

    def ray_values(self) -> Dict[str, Any]:
        """ Specifies a dictionary of the ray specific supervision channels this MultiviewBatch carries. """
        out = dict()
        if self['rgb'] is not None:
            out['rgb'] = self['rgb']
        return out


class SDFBatch(Batch):
    """ Represents a single batch of coordinates + sdf information sampled and collated from a WispDataset.
    SDFBatch objects are always assumed to carry the coordinate information and signed distance from some sampled
    surface, and may optionally return additional channels of information as well such as sampled materials of
    the surface.
    Batches in Wisp keep a general structure by subclassing python's dictionaries and using their semantics,
    so the exact channels of information each coordinate carries is flexible and are left up to datasets to initialize.
    Callers should be mindful of which fields are available by calling the `fields()` function.
    Out of convenience, the common rgb and normals channels are explicitly listed here (though are left as optional).
    """

    def __init__(self,
                 coords: torch.Tensor,
                 sdf: torch.Tensor,
                 rgb: Optional[torch.Tensor] = None,
                 normals: Optional[torch.Tensor] = None,
                 *args, **kwargs):
        """
        Creates a new SDFBatch of coordinate samples and optional supervision signals they carry.

        Args:
            coords (torch.Tensor): a torch Tensor of coordinate samples of shape (B, d) where B is the number of
                samples in this batch and d is their dimensionality (usually d is the number of spatiotemporal dims).
            sdf (torch.Tensor): A torch.Tensor of the signed distance of each sample coordinate from the surface it
                was generated against. The shape is (B, 1).
            rgb (torch.Tensor): Optional.
                A torch.Tensor of rgb color which corresponds the interpolated texture value of the coordinate.
            normals (torch.Tensor): Optional.
                A torch.Tensor of normal value which corresponds the estimated normal value of the coordinate against
                the nearest point on the surface.
                *args, **kwargs - may specify any additional channels of information a coordinate or batch can carry.
        """
        super().__init__(coords=coords, sdf=sdf, rgb=rgb, normals=normals)

    def coord_values(self) -> Dict[str, Any]:
        """ Specifies a dictionary of the coordinate specific supervision channels this SDFBatch carries. """
        out = dict(sdf=self['sdf'])
        if self['rgb'] is not None:
            out['rgb'] = self['rgb']
        if self['normals'] is not None:
            out['normals'] = self['normals']
        return out