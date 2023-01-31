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
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.multiprocessing import Pool, cpu_count
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
import wisp.ops.image as img_ops
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud
import cv2

""" A module for loading data files in the RTMV format.
    See: http://www.cs.umd.edu/~mmeshry/projects/rtmv/
"""

def load_rtmv_images(root, basename, use_depth=False, mip=None, srgb=False, bg_color='white'):
    """Loads a set of RTMV images by path and basename.

    Args:
        root (str): Path to the root of the dataset.
        basename (str): Basename of the RTMV image set to load.
        use_depth (bool): if True, loads the depth data
                          by default, this assumes the depth is stored in the "depth" buffer
        mip (int): if not None, then each image will be resized by 2^mip
        srgb (bool): if True, convert to SRGB

    Returns:
        (dictionary of torch.Tensors)

        Keys:
            image : torch.FloatTensor of size [H,W,3]
            alpha : torch.FloatTensor of size [H,W,1]
            depth : torch.FloatTensor of size [H,W,1]
    """
    # TODO(ttakikawa): There is a lot that this function does... break this up
    from wisp.ops.image import resize_mip, linear_to_srgb

    image_exts = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    exr_exts = ['.exr', '.EXR']
    npz_exts = ['.npz', '.NPZ']

    img = None
    depth = None

    # Try to load RGB first
    found_image_path = None
    for ext in image_exts + exr_exts:
        image_path = os.path.join(root, basename + ext)
        if os.path.exists(image_path):
            found_image_path = image_path
            break
    if found_image_path is None:
        raise Exception("No images found! Check if your dataset path contains an actual RTMV dataset.")

    img_ext = os.path.splitext(found_image_path)[1]
    if img_ext in image_exts:
        img = img_ops.load_rgb(found_image_path)
    elif img_ext in exr_exts:
        try:
            import pyexr
        except:
            raise Exception(
                "The RTMV dataset provided uses EXR, but module pyexr is not available. "
                "To install, run `pip install pyexr`. "
                "You will likely also need `libopenexr`, which through apt you can install with "
                "`apt-get install libopenexr-dev` and on Windows you can install with "
                "`pipwin install openexr`.")
        f = pyexr.open(found_image_path)
        img = f.get("default")
    else:
        raise Exception(f"Invalid image extension for the image path {found_image_path}")

    found_depth_path = None
    for ext in [".depth.npz", ".depth.exr"] + exr_exts:
        depth_path = os.path.join(root, basename + ext)
        if os.path.exists(depth_path):
            found_depth_path = depth_path
            break
    if found_depth_path is None:
        raise Exception("No depth found! Check if your dataset path contains an actual RTMV dataset.")

    depth_ext = os.path.splitext(found_depth_path)[1]
    # Load depth
    if depth_ext == ".npz":
        depth = np.load(found_depth_path)['arr_0'][..., 0]
    elif depth_ext == ".exr":
        try:
            import pyexr
        except:
            raise Exception(
                "The RTMV dataset provided uses EXR, but module pyexr is not available. "
                "To install, run `pip install pyexr`. "
                "You will likely also need `libopenexr`, which through apt you can install with "
                "`apt-get install libopenexr-dev` and on Windows you can install with "
                "`pipwin install openexr`.")
        
        f = pyexr.open(found_depth_path)
        
        components = os.path.basename(found_depth_path).split('.')
        if len(components) > 2 and components[-1] == "exr" and components[-2] == "depth":
            depth = f.get('default')[:, :, 0]
        else:
            if len(f.channel_map['depth']) > 0:
                depth = f.get("depth")
            else:
                raise Exception("Depth channel not found in the EXR file provided!")
    else:
        raise Exception(f"Invalid depth extension for the depth path {found_depth_path}")

    alpha = img[..., 3:4]

    if bg_color == 'black':
        img[..., :3] -= (1 - alpha)
        img = np.clip(img, 0.0, 1.0)
    else:
        img[..., :3] *= alpha
        img[..., :3] += (1 - alpha)
        img = np.clip(img, 0.0, 1.0)

    if mip is not None:
        # TODO(ttakikawa): resize_mip causes the mask to be squuezed... why?
        img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        if use_depth:
            depth = resize_mip(depth, mip, interpolation=cv2.INTER_NEAREST)
            # mask_depth = resize_mip(mask_depth[...,None].astype(np.float), mip, interpolation=cv2.INTER_NEAREST)

    img = torch.from_numpy(img)
    if use_depth:
        depth = torch.from_numpy(depth)

    if use_depth:
        mask_depth = torch.logical_and(depth > -1000, depth < 1000)
        depth[~mask_depth] = -1.0
        depth = depth[:, :, np.newaxis]

    if srgb:
        img = linear_to_srgb(img)

    alpha = mask_depth

    return img, alpha, depth



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


def _parallel_load_rtmv_data(args):
    """ A wrapper function to allow rtmv load faster with multiprocessing.
        All internal logic must therefore occur on the cpu.
    """
    torch.set_num_threads(1)
    with torch.no_grad():
        image, alpha, depth = load_rtmv_images(**args['exr_args'])
        camera = load_rtmv_camera(args['camera_args']['path'])
        transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=args['camera_args']['mip'])
        return dict(
            task_basename=args['task_basename'],
            image=image,
            alpha=alpha,
            depth=depth,
            camera=transformed_camera.cpu()
        )


def load_rtmv_data(root, split, mip=None, normalize=True, return_pointcloud=False, bg_color='white',
                   num_workers=0):
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
        raise RuntimeError("Unimplemented split, check the split")

    images = []
    alphas = []
    depths = []
    rays = []
    cameras = dict()
    basenames = []

    json_files = [json_files[i] for i in subset_idxs]
    assert (len(json_files) > 0 and "No JSON files found")
    if num_workers > 0:
        # threading loading images

        p = Pool(num_workers)
        try:
            basenames = (os.path.splitext(os.path.basename(json_file))[0] for json_file in json_files)
            iterator = p.imap(_parallel_load_rtmv_data, [
                dict(
                    task_basename=basename,
                    exr_args=dict(
                        root=root,
                        basename=basename,
                        use_depth=True,
                        mip=mip,
                        srgb=True,
                        bg_color=bg_color),
                    camera_args=dict(
                        path=os.path.join(root, basename + '.json'),
                        mip=mip)) for basename in basenames])
            for _ in tqdm(range(len(json_files)), desc='loading data'):
                result = next(iterator)
                images.append(result["image"])
                alphas.append(result["alpha"])
                depths.append(result["depth"])
                cameras[result['task_basename']] = result["camera"]
        finally:
            p.close()
            p.join()
    else:
        for img_index, json_file in tqdm(enumerate(json_files), desc='loading data'):
            with torch.no_grad():
                basename = os.path.splitext(os.path.basename(json_file))[0]
                image, alpha, depth = load_rtmv_images(root, basename, use_depth=True, mip=mip, srgb=True, bg_color=bg_color)
                json_path = os.path.join(root, basename + ".json")
                camera = load_rtmv_camera(path=json_path)
                transformed_camera = transform_rtmv_camera(copy.deepcopy(camera), mip=mip)

                images.append(image)
                alphas.append(alpha)
                depths.append(depth)
                cameras[basename] = transformed_camera

    for idx in cameras:
        camera = cameras[idx]
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height,
                                                  device='cuda')
        _rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(
            camera.height, camera.width, 3).to('cpu')
        rays.append(_rays.to(dtype=torch.float32))

    # Normalize
    if normalize:

        coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)
        normalized_coords, coords_center, coords_scale = normalize_pointcloud(
            coords, return_scale=True)

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

    output = {
        "imgs": images,
        "masks": alphas,
        "rays": rays,
        "depths": depths,
        "cameras": cameras
    }
    if return_pointcloud:
        output.update({
            "coords": coords,
            "coords_center": coords_center,
            "coords_scale": coords_scale
        })

    return output
