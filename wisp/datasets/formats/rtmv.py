# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import copy
import os
import glob
import time
import futureproof
import json
from tqdm import tqdm
import numpy as np
import torch
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.image import load_exr
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud


""" A module for loading data files in the RTMV format.
    See: http://www.cs.umd.edu/~mmeshry/projects/rtmv/
"""


def rescale_rtmv_intrinsics(camera, target_size, original_width, original_height):
    """ Rescale the intrinsics. """
    assert original_height == original_width, 'Current code assumes a square image.'
    resize_ratio = target_size * 1. / original_height
    camera.x0 *= resize_ratio
    camera.y0 *= resize_ratio
    camera.focal_x *= resize_ratio
    camera.focal_y *= resize_ratio

def load_rtmv_camera(path):
    """Loads a RTMV camera object from json metadata.
    """
    with open(path.replace("exr", "json"), 'r') as f:
        meta = json.load(f)

    # About RTMV conventions:
    # RTMV: meta['camera_data']['cam2world']), row major
    # kaolin cameras: cam2world is camera.inv_view_matrix(), column major
    # Therefore the following are equal up to a 1e-7 difference:
    # camera.inv_view_matrix() == torch.tensor(meta['camera_data']['cam2world']).T
    cam_data = meta['camera_data']
    camera = Camera.from_args(eye=torch.Tensor(cam_data['camera_look_at']['eye']),
                              at=torch.Tensor(cam_data['camera_look_at']['at']),
                              up=torch.Tensor(cam_data['camera_look_at']['up']),
                              width=cam_data['width'],
                              height=cam_data['height'],
                              focal_x=cam_data['intrinsics']['fx'],
                              focal_y=cam_data['intrinsics']['fy'],
                              x0=0.0,
                              y0=0.0,
                              near=0.0,
                              far=6.0, # inheriting default for nerf-synthetic
                              dtype=torch.float64,
                              device='cpu')

    # RTMV cameras use Blender coordinates, which are right handed with Z axis pointing upwards instead of Y.
    camera.change_coordinate_system(blender_coords())

    return camera.cpu()


def transform_rtmv_camera(camera, mip):
    """Transforms the RTMV camera according to the mip.
    """
    original_width, original_height = camera.width, camera.height
    if mip is not None:
        camera.width = camera.width // (2 ** mip)
        camera.height = camera.height // (2 ** mip)

    # assume no resizing
    rescale_rtmv_intrinsics(camera, camera.width, original_width, original_height)
    return camera


def _parallel_load_rtmv_data(task_basename, exr_args, camera_args):
    """ A wrapper function to allow rtmv load faster with multiprocessing.
        All internal logic must therefore occur on the cpu.
    """
    image, alpha, depth = load_exr(**exr_args)
    camera = load_rtmv_camera(camera_args['path'])
    transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=camera_args['mip'])
    return dict(task_basename=task_basename, image=image, alpha=alpha, depth=depth, camera=transformed_camera.cpu())


def load_rtmv_data(root, split, mip=None, normalize=True, return_pointcloud=False, bg_color='white',
                   num_workers=-1):
    """Load the RTMV data and applies dataset specific transforms required for compatibility with the framework.

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split to use from 'train', 'val', 'test'.
        mip (int): If provided, will rescale images by 2**mip.
        normalize (bool): If True, will normalize the ray origins by the point cloud origin and scale.
        return_pointcloud (bool): If True, will also return the pointcloud and the scale and origin.
        bg_color (str): The background color to use for when alpha=0.
        num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.

    Returns:
        (dict of torch.FloatTensors): Different channels of information from RTMV.
    """
    json_files = sorted(glob.glob(os.path.join(root, '*.json')))

    # Hard-coded train-val-test splits for now (TODO(ttakikawa): pass as args?)
    train_split_idx = len(json_files) * 2 // 3
    eval_split_idx = train_split_idx + (len(json_files) * 10 // 300)

    if split == 'train':
        subset_idxs = np.arange(0, train_split_idx)
    elif split in 'val':
        subset_idxs = np.arange(train_split_idx, eval_split_idx)
    elif split == 'test':
        subset_idxs = np.arange(eval_split_idx, len(json_files))
    else:
        assert False and "Unimplemented split, check the split"

    images = []
    alphas = []
    depths = []
    rays = []
    cameras = dict()
    basenames = []

    json_files = [json_files[i] for i in subset_idxs]
    assert (len(json_files) > 0 and "No JSON files found")

    if num_workers > -1:
        # threading loading images

        try:
            executor = futureproof.ProcessPoolExecutor(max_workers=num_workers)

            timeall = time.time()
            with futureproof.TaskManager(executor) as tm:
                for img_index, json_file in enumerate(json_files):
                    basename = os.path.splitext(os.path.basename(json_file))[0]
                    exr_path = os.path.join(root, basename + ".exr")
                    json_path = os.path.join(root, basename + ".json")

                    load_exr_kwargs = dict(path=exr_path, use_depth=True, mip=mip, srgb=True, bg_color=bg_color)
                    load_camera_kwargs = dict(path=json_path, mip=mip)

                    tm.submit(_parallel_load_rtmv_data, task_basename=basename, exr_args=load_exr_kwargs,
                              camera_args=load_camera_kwargs)

                for task in tqdm(tm.as_completed(), desc='loading data'):
                    images.append(task.result["image"])
                    alphas.append(task.result["alpha"])
                    depths.append(task.result["depth"])
                    cameras[task.result['task_basename']] = task.result["camera"]
        finally:
            if executor is not None:
                executor.join()

    else:
        for img_index, json_file in tqdm(enumerate(json_files), desc='loading data'):
            basename = os.path.splitext(os.path.basename(json_file))[0]
            exr_path = os.path.join(root, basename + ".exr")
            image, alpha, depth = load_exr(exr_path, use_depth=True, mip=mip, srgb=True, bg_color=bg_color)
            json_path = os.path.join(root, basename + ".json")
            camera = load_rtmv_camera(path=json_path)
            transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=mip)

            images.append(image)
            alphas.append(alpha)
            depths.append(depth)
            cameras[basename] = transformed_camera

    for idx in cameras:
        camera = cameras[idx]
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, camera.width, camera.height,
                                                  device='cuda')
        _rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width,
                                                                                       3).to('cpu')
        rays.append(_rays.to(dtype=torch.float32))

    # Normalize
    if normalize:

        coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)
        normalized_coords, coords_center, coords_scale = normalize_pointcloud(coords, return_scale=True)

        depths = torch.stack(depths)
        rays = Rays.stack(rays)
        depths = depths * coords_scale
        rays.origins = (rays.origins - coords_center) * coords_scale
        depths = list(depths)
        rays = list(rays)

        for cam_id, cam in cameras.items():
            cam.translate(-coords_center.to(cam.dtype))
            cam.t = cam.t * coords_scale.to(cam.dtype)

        coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)

    for idx in cameras:
        camera = cameras[idx]

    images = torch.stack(images)[..., :3]
    alphas = torch.stack(alphas)
    depths = torch.stack(depths)
    rays = Rays.stack(rays)

    output = {"imgs": images, "masks": alphas, "rays": rays, "depths": depths, "cameras": cameras}
    if return_pointcloud:
        output.update({"coords": coords, "coords_center": coords_center, "coords_scale": coords_scale})

    return output