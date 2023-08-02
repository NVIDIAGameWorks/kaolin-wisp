# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys

import glob

import numpy as np
import logging as log

import torch
from torch.utils.data import Dataset

import wisp.ops.image as img_ops
import wisp.ops.geometric as geo_ops
from PIL import Image

class ImageDataset(Dataset):
    """This is a single image dataset class.

    This class should be used for training tasks where the task is to fit a single
    image using a neural field. 
    """
    def __init__(self, 
        dataset_path       : str,
        num_pixels_per_image : int = 4096,
    ):
        self.root = os.path.abspath(os.path.expanduser(dataset_path))
        self.image = torch.from_numpy(np.array(Image.open(self.root)))
        self.image = self.image / 255.0

        self.num_pixels_per_image = num_pixels_per_image

        if not self.image.shape[-1] == 3:
            raise Exception("Alpha channel detected for image." 
                            "You should create a 3 channel RGB with alpha channels dealt in whatever way makes sense.")

        self.h, self.w = self.image.shape[:2]
        self.coords = geo_ops.normalized_grid(self.h, self.w, use_aspect=False).reshape(-1, 2).cpu()
        self.pixels = self.image.reshape(-1, 3)

    def get_image(self):
        return self.image
    
    def __len__(self):
        return 100

    def __getitem__(self, idx : int):
        rand_idx = torch.randint(0, self.coords.shape[0], (self.num_pixels_per_image,))
        return self.coords[rand_idx], self.pixels[rand_idx]
