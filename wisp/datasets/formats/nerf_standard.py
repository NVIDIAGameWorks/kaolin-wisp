# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import glob
import time
import cv2
import skimage
import imageio
import json
from tqdm import tqdm
import skimage.metrics
import logging as log
import numpy as np
import torch
from torch.multiprocessing import Pool
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip

""" A module for loading data files in the standard NeRF format, including extensions to the format
    supported by Instant Neural Graphics Primitives.
    See: https://github.com/NVlabs/instant-ngp
"""

# Local function for multiprocess. Just takes a frame from the JSON to load images and poses.
def _load_standard_imgs(frame, root, mip=None):
    """Helper for multiprocessing for the standard dataset. Should not have to be invoked by users.

    Args:
        root: The root of the dataset.
        frame: The frame object from the transform.json.
        mip: If set, rescales the image by 2**mip.

    Returns:
        (dict): Dictionary of the image and pose.
    """
    fpath = os.path.join(root, frame['file_path'].replace("\\", "/"))

    basename = os.path.basename(os.path.splitext(fpath)[0])
    if os.path.splitext(fpath)[1] == "":
        # Assume PNG file if no extension exists... the NeRF synthetic data follows this convention.
        fpath += '.png'

    # For some reason instant-ngp allows missing images that exist in the transform but not in the data.
    # Handle this... also handles the above case well too.
    if os.path.exists(fpath):
        img = imageio.imread(fpath)
        img = skimage.img_as_float32(img)
        if mip is not None:
            img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        return dict(basename=basename,
                    img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
    else:
        # log.info(f"File name {fpath} doesn't exist. Ignoring.")
        return None

def _parallel_load_standard_imgs(args):
    """Internal function for multiprocessing.
    """
    torch.set_num_threads(1)
    result = _load_standard_imgs(args['frame'], args['root'], mip=args['mip'])
    if result is None:
        return dict(basename=None, img=None, pose=None)
    else:
        return dict(basename=result['basename'], img=result['img'], pose=result['pose'])

def load_nerf_standard_data(root, split='train', bg_color='white', num_workers=-1, mip=None):
    """Standard loading function.

    This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

    There are two pairs of standard file structures this follows:

    ```
    /path/to/dataset/transform.json
    /path/to/dataset/images/____.png
    ```

    or

    ```
    /path/to/dataset/transform_{split}.json
    /path/to/dataset/{split}/_____.png
    ```

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split to use from 'train', 'val', 'test'.
        bg_color (str): The background color to use for when alpha=0.
        num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.
        mip: If set, rescales the image by 2**mip.

    Returns:
        (dict of torch.FloatTensors): Different channels of information from NeRF.
    """
    transforms = sorted(glob.glob(os.path.join(root, "*.json")))

    transform_dict = {}

    train_only = False

    if mip is None:
        mip = 0

    if len(transforms) == 1:
        transform_dict['train'] = transforms[0]
        train_only = True
    elif len(transforms) == 3:
        fnames = [os.path.basename(transform) for transform in transforms]

        # Create dictionary of split to file path, probably there is simpler way of doing this
        for _split in ['test', 'train', 'val']:
            for i, fname in enumerate(fnames):
                if _split in fname:
                    transform_dict[_split] = transforms[i]
    else:
        raise RuntimeError("Unsupported number of splits, there should be ['test', 'train', 'val']")

    if split not in transform_dict:
        raise RuntimeError(f"Split type ['{split}'] unsupported in the dataset provided")

    for key in transform_dict:
        with open(transform_dict[key], 'r') as f:
            transform_dict[key] = json.load(f)

    imgs = []
    poses = []
    basenames = []

    if num_workers > 0:
        # threading loading images

        p = Pool(num_workers)
        try:
            iterator = p.imap(_parallel_load_standard_imgs,
                [dict(frame=frame, root=root, mip=mip) for frame in transform_dict[split]['frames']])
            for _ in tqdm(range(len(transform_dict[split]['frames']))):
                result = next(iterator)
                basename = result['basename']
                img = result['img']
                pose = result['pose']
                if basename is not None:
                    basenames.append(basename)
                if img is not None:
                    imgs.append(img)
                if pose is not None:
                    poses.append(pose)
        finally:
            p.close()
            p.join()
    else:
        for frame in tqdm(transform_dict[split]['frames'], desc='loading data'):
            _data = _load_standard_imgs(frame, root, mip=mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])

    imgs = torch.stack(imgs)
    poses = torch.stack(poses)

    # TODO(ttakikawa): Assumes all images are same shape and focal. Maybe breaks in general...
    h, w = imgs[0].shape[:2]

    if 'x_fov' in transform_dict[split]:
        # Degrees
        x_fov = transform_dict[split]['x_fov']
        fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
        if 'y_fov' in transform_dict[split]:
            y_fov = transform_dict[split]['y_fov']
            fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
        else:
            fy = fx
    elif 'fl_x' in transform_dict[split] and False:
        fx = float(transform_dict[split]['fl_x']) / float(2**mip)
        if 'fl_y' in transform_dict[split]:
            fy = float(transform_dict[split]['fl_y']) / float(2**mip)
        else:
            fy = fx
    elif 'camera_angle_x' in transform_dict[split]:
        # Radians
        camera_angle_x = transform_dict[split]['camera_angle_x']
        fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

        if 'camera_angle_y' in transform_dict[split]:
            camera_angle_y = transform_dict[split]['camera_angle_y']
            fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
        else:
            fy = fx

    else:
        fx = 0.0
        fy = 0.0

    if 'fix_premult' in transform_dict[split]:
        log.info("WARNING: The dataset expects premultiplied alpha correction, "
                 "but the current implementation does not handle this.")

    if 'k1' in transform_dict[split]:
        log.info \
            ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

    if 'rolling_shutter' in transform_dict[split]:
        log.info("WARNING: The dataset expects rolling shutter correction,"
                 "but the current implementation does not handle this.")

    # The principal point in wisp are always a displacement in pixels from the center of the image.
    x0 = 0.0
    y0 = 0.0
    # The standard dataset generally stores the absolute location on the image to specify the principal point.
    # Thus, we need to scale and translate them such that they are offsets from the center.
    if 'cx' in transform_dict[split]:
        x0 = (float(transform_dict[split]['cx']) / (2**mip)) - (w//2)
    if 'cy' in transform_dict[split]:
        y0 = (float(transform_dict[split]['cy']) / (2**mip)) - (h//2)

    offset = transform_dict[split]['offset'] if 'offset' in transform_dict[split] else [0 ,0 ,0]
    scale = transform_dict[split]['scale'] if 'scale' in transform_dict[split] else 1.0
    aabb_scale = transform_dict[split]['aabb_scale'] if 'aabb_scale' in transform_dict[split] else 1.25

    # TODO(ttakikawa): Actually scale the AABB instead? Maybe
    poses[..., :3, 3] /= aabb_scale
    poses[..., :3, 3] *= scale
    poses[..., :3, 3] += torch.FloatTensor(offset)

    # nerf-synthetic uses a default far value of 6.0
    default_far = 6.0

    rays = []

    cameras = dict()
    for i in range(imgs.shape[0]):
        view_matrix = torch.zeros_like(poses[i])
        view_matrix[:3, :3] = poses[i][:3, :3].T
        view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
        view_matrix[3, 3] = 1.0
        camera = Camera.from_args(view_matrix=view_matrix,
                                  focal_x=fx,
                                  focal_y=fy,
                                  width=w,
                                  height=h,
                                  far=default_far,
                                  near=0.0,
                                  x0=x0,
                                  y0=y0,
                                  dtype=torch.float64)
        camera.change_coordinate_system(blender_coords())
        cameras[basenames[i]] = camera
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device='cuda')
        rays.append \
            (generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to
                ('cpu'))

    rays = Rays.stack(rays).to(dtype=torch.float)

    rgbs = imgs[... ,:3]
    alpha = imgs[... ,3:4]
    if alpha.numel() == 0:
        masks = torch.ones_like(rgbs[... ,0:1]).bool()
    else:
        masks = (alpha > 0.5).bool()

        if bg_color == 'black':
            rgbs[... ,:3] -= ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
        else:
            rgbs[... ,:3] *= alpha
            rgbs[... ,:3] += ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)

    return {"imgs": rgbs, "masks": masks, "rays": rays, "cameras": cameras}
