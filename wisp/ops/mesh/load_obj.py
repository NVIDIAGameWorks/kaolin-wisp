# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import sys

import numpy as np
import tinyobjloader
import torch

from PIL import Image

import logging as log

Image.MAX_IMAGE_PIXELS = None

# Refer to 
# https://github.com/tinyobjloader/tinyobjloader/blob/master/tiny_obj_loader.h
# for conventions for tinyobjloader data structures.

texopts = [
    'ambient_texname',
    'diffuse_texname',
    'specular_texname',
    'specular_highlight_texname',
    'bump_texname',
    'displacement_texname',
    'alpha_texname',
    'reflection_texname',
    'roughness_texname',
    'metallic_texname',
    'sheen_texname',
    'emissive_texname',
    'normal_texname'
]

def load_mat(fname : str):
    """Loads material.
    """

    img = torch.FloatTensor(np.array(Image.open(fname)))
    img = img / 255.0

    return img


def load_obj(
    fname : str, 
    load_materials : bool = False):
    """Load .obj file using TinyOBJ and extract info.
    This is more robust since it can triangulate polygon meshes 
    with up to 255 sides per face.
    
    Args:
        fname (str): path to Wavefront .obj file
    """

    assert os.path.exists(fname), \
        'Invalid file path and/or format, must be an existing Wavefront .obj'

    reader = tinyobjloader.ObjReader()
    config = tinyobjloader.ObjReaderConfig()
    config.triangulate = True # Ensure we don't have any polygons

    reader.ParseFromFile(fname, config)

    # Get vertices
    attrib = reader.GetAttrib()
    vertices = torch.FloatTensor(attrib.vertices).reshape(-1, 3)

    # Get triangle face indices
    shapes = reader.GetShapes()
    faces = []
    for shape in shapes:
        faces += [idx.vertex_index for idx in shape.mesh.indices]
    faces = torch.LongTensor(faces).reshape(-1, 3)
    
    mats = {}

    if load_materials:
        # Load per-faced texture coordinate indices
        texf = []
        matf = []
        for shape in shapes:
            texf += [idx.texcoord_index for idx in shape.mesh.indices]
            matf.extend(shape.mesh.material_ids)
        # texf stores [tex_idx0, tex_idx1, tex_idx2, mat_idx]
        texf = torch.LongTensor(texf).reshape(-1, 3)
        matf = torch.LongTensor(matf).reshape(-1, 1)
        texf = torch.cat([texf, matf], dim=-1)

        # Load texcoords
        texv = torch.FloatTensor(attrib.texcoords).reshape(-1, 2)
        
        # Load texture maps
        parent_path = os.path.dirname(fname) 
        materials = reader.GetMaterials()
        for i, material in enumerate(materials):
            mats[i] = {}
            diffuse = getattr(material, 'diffuse')
            if diffuse != '':
                mats[i]['diffuse'] = torch.FloatTensor(diffuse)

            for texopt in texopts:
                mat_path = getattr(material, texopt)
                if mat_path != '':
                    img = load_mat(os.path.join(parent_path, mat_path))
                    mats[i][texopt] = img
                    #mats[i][texopt.split('_')[0]] = img
        return vertices, faces, texv, texf, mats

    return vertices, faces

