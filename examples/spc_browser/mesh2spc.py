# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import pathlib
import argparse
from kaolin.io.obj import import_mesh
from kaolin.ops.mesh import sample_points
from kaolin.render.mesh.utils import texture_mapping
from kaolin.ops.conversions.pointcloud import unbatched_pointcloud_to_spc


def convert_texture_to_torch_sample_format(texture, sampled_uvs):
    """ Convert to (1, C, Tex-H, Tex-W) format """
    return texture.unsqueeze(0).type(sampled_uvs.dtype).permute(0, 3, 1, 2)


def convert_mesh_to_spc(mesh_path, level, output_path, num_samples):
    """ Loads obj and converts it to a SPC. Output will reside in output_path."""
    mesh = import_mesh(mesh_path, with_materials=True)

    print(f'Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces and {len(mesh.materials)} materials.')

    # Load the uv coordinates per face-vertex like "features" per face-vertex,
    # which sample_points will interpolate for new sample points.
    # mesh.uvs is a tensor of uv coordinates of shape (#num_uvs, 2), which we consider as "features" here
    # mesh.face_uvs_idx is a tensor of shape (#faces, 3), indexing which feature to use per-face-per-vertex
    # Therefore, face_features will be of shape (#faces, 3, 2)
    face_features = mesh.uvs[mesh.face_uvs_idx]

    # Kaolin assumes an exact batch format, we make sure to convert from:
    # (V, 3) to (1, V, 3)
    # (F, 3, 2) to (1, F, 3, 2)
    # where 1 is the batch size
    batched_vertices = mesh.vertices.unsqueeze(0)
    batched_face_features = face_features.unsqueeze(0)

    # sample_points is faster on cuda device
    batched_vertices = batched_vertices.cuda()
    faces = mesh.faces.cuda()
    batched_face_features = batched_face_features.cuda()

    sampled_verts, _, sampled_uvs = sample_points(batched_vertices, faces,
                                                  num_samples=num_samples, face_features=batched_face_features)

    print(f'Sampled {sampled_verts.shape[1]} points over the mesh surface.')

    # Convert texture to sample-compatible format
    diffuse_color = mesh.materials[0]['map_Kd']    # Assumes a shape with a single material
    texture_maps = convert_texture_to_torch_sample_format(diffuse_color, sampled_uvs)  # (1, C, Th, Tw)
    texture_maps = texture_maps.cuda()

    # Sample colors according to uv-coordinates
    sampled_uvs = texture_mapping(texture_coordinates=sampled_uvs, texture_maps=texture_maps, mode='nearest')
    # Unbatch
    vertices = sampled_verts.squeeze(0)
    vertex_colors = sampled_uvs.squeeze(0)

    # Normalize to [0,1], and convert to RGBA if needed
    vertex_colors /= 255
    if vertex_colors.shape[-1] == 3:
        vertex_colors = torch.cat([vertex_colors, torch.ones_like(vertex_colors[:, :1])], dim=1)

    spc = unbatched_pointcloud_to_spc(vertices, level, features=vertex_colors)
    print(f'SPC generated with {spc.point_hierarchies.shape[0]} cells.')

    octrees_entry = spc.octrees
    colors_entry = (255 * spc.features.reshape(-1))
    npz_record = dict(
        octree=octrees_entry.cpu().numpy().astype(np.uint8),
        colors=colors_entry.cpu().numpy().astype(np.uint8)
    )

    # Default output path: take filename and save in current working directory
    if output_path is None:
        output_path = pathlib.Path(mesh_path).stem
        output_path = f'{output_path}.npz'

    np.savez(output_path, **npz_record)

    return output_path


if __name__ == "__main__":
    assert torch.cuda.is_available(), 'mesh2spc script requires a CUDA device, which is unavailable for pytorch.'

    parser = argparse.ArgumentParser(description='Converts meshes with a single material to SPC objects.'
                                                 'Requires a CUDA device.')
    parser.add_argument('--obj_path', type=str,
                        help='Path of input .obj file to convert to SPC. '
                             'Expected to point at a .mtl file with Kd material')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path + filename of output .npz file which stores the SPC topology and features.'
                             'By default, the obj filename with npz suffix will be used.')
    parser.add_argument('--level', type=int, default=8,
                        help='Number of level to use for the SPC. e.g. the SPC resolution will be 2^level.')
    parser.add_argument('--num_samples', type=int, default=1000000,
                        help='Number of samples to sample over the mesh faces, used to populate the SPC object.')
    args = parser.parse_args()

    # Returns the actual output path where the spc *.npz resides
    print(f'mesh2spc starting conversion of {args.obj_path}')
    output_path = convert_mesh_to_spc(mesh_path=args.obj_path, level=args.level,
                                      output_path=args.output_path, num_samples=args.num_samples)
    print(f'mesh2spc finished successfully, output resides in {output_path}')
