# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import unittest

from wisp.datasets import RandomViewDataset
from wisp.models.nefs import NeuralRadianceField
from wisp.tracers import PackedRFTracer
from wisp.models import Pipeline


class Test(unittest.TestCase):
    def test_extra_channels(self):
        device = "cuda:0"
        nef = NeuralRadianceField(grid_type="HashGrid", multiscale_type="cat")
        nef.grid.init_from_geometric(min_width=2, max_width=4, num_lods=1)
        nef.to(device)
        tracer = PackedRFTracer()
        pipeline = Pipeline(nef, tracer)
        dataset = RandomViewDataset(num_rays=128, device=device)
        datum = dataset[0]
        rb = pipeline(rays=datum.rays, channels=["rgb", "density"])
        assert hasattr(rb, "density")
        assert rb.rgb.shape[0] == rb.density.shape[0]
