# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Tuple, Union
import torch


@dataclass
class PrimitivesPack:
    """Packs of geometric primitives.
    Useful for visualizing vectorial data layers with visualizers such as wisp's interactive renderer,
    or external tools like Polyscope..
    """
    _lines_start: List[torch.Tensor] = field(default_factory=list)
    _lines_end: List[torch.Tensor] = field(default_factory=list)
    _lines_color: List[torch.Tensor] = field(default_factory=list)
    _points_pos: List[torch.Tensor] = field(default_factory=list)
    _points_color: List[torch.Tensor] = field(default_factory=list)

    line_width = 1.0
    point_size = 1.0

    def append(self, other: PrimitivesPack):
        """ Appends other to self, changes the dtype and device if needed """
        def _append_field(field_name):
            # Get field attribute by name
            _self_field = getattr(self, field_name)
            _other_field = getattr(other, field_name)
            if len(_self_field) > 0:    # If self's field has any entry, we map other's field to this dtype / device
                _other_field = map(lambda t: t.to(_self_field[0].device, _self_field[0].dtype), _other_field)
            _self_field.extend(_other_field)    # Concat the lists of tensors

        for f in fields(self):
            _append_field(f.name)

    def add_lines(self, start: torch.Tensor, end: torch.Tensor, color: torch.Tensor):
        if start.ndim == 1:
            start = start.unsqueeze(0)
        if end.ndim == 1:
            end = end.unsqueeze(0)
        if color.ndim == 1:
            color = color.unsqueeze(0)
        self._lines_start.append(start)
        self._lines_end.append(end)
        self._lines_color.append(color)

    @property
    def lines(self) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        start = end = color = None
        if len(self._lines_start) > 0:
            start = torch.cat(self._lines_start, dim=0)
        if len(self._lines_end) > 0:
            end = torch.cat(self._lines_end, dim=0)
        if len(self._lines_color) > 0:
            color = torch.cat(self._lines_color, dim=0)
        if start is None or end is None:
            return None
        return start, end, color

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise Exception("PrimitivePacks only support equality checks with other PrimitivePacks")
        bools = []
        for f in fields(self):
            self_attr = getattr(self, f.name)
            other_attr = getattr(other, f.name)
            if isinstance(self_attr, float):
                bools.append(self_attr == other_attr)
            else:
                bools.append(all(torch.equal(s, o) for s, o in zip(self_attr, other_attr)))
        return all(bools)
