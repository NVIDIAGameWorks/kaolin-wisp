# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Union, Tuple, List
from dataclasses import dataclass
import torch


INFINITY = torch.finfo().max
""" Largest representable float in this system """


@dataclass
class Rays:
    """ A pack of rays represented as origin and direction.
    Ray packs are flexible and may use an arbitrary amount of spatial dimensions (e.g. flat, 2D, etc).
    """

    origins: torch.Tensor
    """ Ray's origin """

    dirs: torch.Tensor
    """ Ray's normalized direction """

    dist_min: Union[float, torch.Tensor] = 0.0
    """ Distance in which ray intersection test begins. Can be defined globally or per ray. """

    dist_max: Union[float, torch.Tensor] = INFINITY
    """ Distance in which ray intersection test ends. Can be defined globally or per ray. """

    # TODO (operel): Handle tensor case in functions below

    def __len__(self) -> int:
        """
        Returns:
            (int): Number of rays in the pack
        """
        if self.origins.shape != self.dirs.shape:
            raise Exception(
                f"Rays.origins shape should match Rays.dirs shape, but got {self.origins.shape} and {self.dirs.shape}.")
        return self.origins.shape[0]

    @property
    def shape(self) -> Tuple[...]:
        """
        Returns:
            (int): Shape of ray pack tensors, excluding the last dimension.
        """
        return self.origins.shape[:-1]

    @property
    def ndim(self) -> int:
        """
        Returns:
            (int): Number of spatial dimensions for ray pack
        """
        return self.origins.ndim - 1

    @classmethod
    def cat(cls, rays_list: List[Rays], dim: int = 0) -> Rays:
        """ Concatenates multiple ray packs into a single pack.

        Args:
            ray_list (List[Rays]): List of ray packs to concatenate, expected to have the same spatial dimensions,
                                  except of dimension dim.
            dim (int): Spatial dimension along which the concatenation should take place

        Returns:
            (Rays): A single ray pack with the concatenation of given ray packs
        """
        if dim < 0:
            dim -= 1
        if dim > rays_list[0].ndim - 1 or dim < -rays_list[0].ndim:
            raise IndexError("Dimension out of range (expected to be in range of "
                             f"[{-rays_list[0].ndim}, {rays_list[0].ndim-1}, but got {dim})")

        return Rays(
            origins=torch.cat([rays.origins for rays in rays_list], dim=dim),
            dirs=torch.cat([rays.dirs for rays in rays_list], dim=dim),
            dist_min=min([rays.dist_min for rays in rays_list]),
            dist_max=max([rays.dist_max for rays in rays_list])
        )

    @classmethod 
    def stack(cls, rays_list: List[Rays], dim: int = 0) -> Rays:
        """ Stacks multiple ray packs into a single pack.

        Args:
            ray_list (List[Rays): List of ray packs to stack, expected to have the same spatial dimensions,
                                  except of dimension dim.

        Returns:
            (Rays): A single ray pack with the stacked ray packs
        """
        return Rays(
            origins=torch.stack([rays.origins for rays in rays_list], dim=dim),
            dirs=torch.stack([rays.dirs for rays in rays_list], dim=dim),
            dist_min=min([rays.dist_min for rays in rays_list]),
            dist_max=max([rays.dist_max for rays in rays_list])
        )

    def __getitem__(self, idx) -> Rays:
        """ Returns a sliced view of the rays.

        Args:
            idx: Indices of rays to slice.

        Returns:
            (Rays): A sliced view of the rays struct
        """
        return Rays(
            origins=self.origins[idx],
            dirs=self.dirs[idx],
            dist_min=self.dist_min[idx] if isinstance(self.dist_min, torch.Tensor) else self.dist_min,
            dist_max=self.dist_max[idx] if isinstance(self.dist_max, torch.Tensor) else self.dist_max
        )

    def split(self, split_size) -> List[Rays]:
        """ Splits the rays pack to equally sized batches.
        The last chunk may be smaller if the number of rays is not divisible by batch_size.

        Args:
            batch_size: Size of each batch returned by this split.

        Returns:
            (List[Rays]): A list of smaller Rays batches, split from the current rays pack.
        """
        zipped = zip(torch.split(self.origins, split_size), torch.split(self.dirs, split_size))
        return [Rays(origins=origins, dirs=dirs, dist_min=self.dist_min, dist_max=self.dist_max)
                for origins, dirs in zipped]

    def reshape(self, *dims: Tuple) -> Rays:
        """ Reshapes the dimensions of the rays struct.

        Args:
            *dims: Tuple of new dimensions after reshape.

        Returns:
            (Rays): The reshaped Rays struct
        """
        return Rays(origins=self.origins.reshape(*dims),
                   dirs=self.dirs.reshape(*dims),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def squeeze(self, dim: int) -> Rays:
        """ Squeezes the tensors of the rays struct.

        Args:
            dims: Dimension to squeeze

        Returns:
            (Rays): The squeezed Rays struct
        """
        return Rays(origins=self.origins.squeeze(dim),
                   dirs=self.dirs.squeeze(dim),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def contiguous(self) -> Rays:
        """ Forces the rays tensors to use contiguous memory.
        If they already do, no modification will take place on the tensors.

        Returns:
            (Rays): The contiguous Rays struct
        """
        return Rays(origins=self.origins.contiguous(),
                   dirs=self.dirs.contiguous(),
                   dist_min=self.dist_min,
                   dist_max=self.dist_max)

    def to(self, *args, **kwargs) -> Rays:
        """ Shifts the rays struct to a different device / dtype.
        seealso::func:`torch.Tensor.to` for an elaborate explanation of using this method.

        Returns:
            The rays tensors will be ast to a different device / dtype
        """

        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        if origins is not self.origins or dirs is not self.dirs:
            return Rays(
                origins=origins,
                dirs=dirs,
                dist_min=self.dist_min,
                dist_max=self.dist_max
            )
        else:
            return self
