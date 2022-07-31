# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy
import os
import sys

import glob
import time
import futureproof

import pyexr
import cv2
import skimage
import imageio
import json
from PIL import Image
from tqdm import tqdm
import skimage.metrics

import logging as log
import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS

from wisp.ops.geometric import look_at, generate_rays_from_tf
from kaolin.render.camera import Camera, blender_coords
from wisp.renderer.core.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.debug import PsDebugger

import wisp.ops.mesh as mesh_ops
from wisp.core import Rays

### File IO Operations

def hwc_to_chw(img):
    """Converts [H,W,C] to [C,H,W] for TensorBoard output.
    
    Args:
        img (np.array): [H,W,C] image.

    Returns:
        (np.array): [C,H,W] image.
    """
    return np.array(img).transpose(2,0,1)

def write_exr(path, data):
    """Writes an EXR image to some path.

    Data is a dict of form { "default" = rgb_array, "depth" = depth_array }
   
    Args:
        path (str): Path to save the EXR
        data (dict): Dictionary of EXR buffers.
    
    Returns:
        (void): Writes to path.
    """
    pyexr.write(path, data,
                channel_names={'normal': ['X','Y','Z'], 
                               'x': ['X','Y','Z'],
                               'view': ['X','Y','Z']},
                precision=pyexr.HALF)

def write_png(path, data):
    """Writes an PNG image to some path.

    Args:
        path (str): Path to save the PNG.
        data (np.array): HWC image.
    
    Returns:
        (void): Writes to path.
    """
    Image.fromarray(data).save(path)

def glob_imgs(path, exts=['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG']):
    """Utility to find images in some path.

    Args:
        path (str): Path to search images in.
        exts (list of str): List of extensions to try.

    Returns:
        (list of str): List of paths that were found.
    """
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(path, ext)))
    return imgs

def load_rgb(path):
    """Loads an image. 

    TODO(ttakikawa): Currently ignores the alpha channel.

    Args:
        path (str): Path to the image.

    Returns:
        (np.array): Image as an array.
    """
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    img = img[:,:,:3]
    return img

def load_mask(path):
    """Loads an alpha mask.

    Args:
        path (str): Path to the image.

    Returns:
        (np.array): Image as an array.
    """
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5
    object_mask = object_mask.transpose(1, 0)

    return object_mask

def srgb_to_linear(img):
    """Converts from SRGB to Linear colorspace.

    Args:
        img (torch.FloatTensor): SRGB image.

    Returns:
        (torch.FloatTensor): Linear image.
    """
    limit = 0.04045
    return torch.where(img > limit, torch.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
    """Converts from Linear to SRGB colorspace.

    Args:
        img (torch.FloatTensor): Linear image.

    Returns:
        (torch.FloatTensor): SRGB image.
    """
    limit = 0.0031308
    img = torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)
    img[img > 1] = 1
    return img

def load_exr(path, use_depth=False, mip=None, srgb=False, bg_color='white', 
            loader_type='pyexr'):
    """Loads a EXR by path.

    Args:
        path (str): path to the .exr file
        use_depth (bool): if True, loads the depth data
                          by default, this assumes the depth is stored in the "depth" buffer
        mip (int): if not None, then each image will be resized by 2^mip
        srgb (bool): if True, convert to SRGB
        loader_type (str): options [cv2, pyexr, imageio]. 
                           TODO(ttakikawa): Not sure quite yet what options should be supported here

    Returns:
        (dictionary of torch.Tensors)

        Keys:
            image : torch.FloatTensor of size [H,W,3]
            alpha : torch.FloatTensor of size [H,W,1]
            depth : torch.FloatTensor of size [H,W,1]
            ray_o : torch.FloatTensor of size [H,W,3]
            ray_d : torch.FloatTensor of size [H,W,3]
    """
    #TODO(ttakikawa): There is a lot that this function does... break this up

    # Load RGB and Depth
    if loader_type == 'cv2':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if use_depth:
            depth = cv2.imread(path.replace(".exr", ".depth.exr"), cv2.IMREAD_UNCHANGED)[:,:,0]

    elif loader_type == 'pyexr':
        f = pyexr.open(path)
        img = f.get("default")

        if use_depth:
            if len(f.channel_map['depth']) > 0:
                depth = f.get("depth")
            else:
                f = pyexr.open(path.replace(".exr", ".depth.exr"))
                depth = f.get('default')[:,:,0]
                
    elif loader_type == 'imageio':
        img = imageio.imread(path)
        if use_depth:
            depth = imageio.imread(path.replace(".exr", ".depth.exr"))[:,:,0]
    else:
        raise ValueError(f'Invalid loader_type: {loader_type}')
    
    alpha = img[...,3:4]
    
    if bg_color == 'black':
        img[...,:3] -= (1-alpha)
        img = np.clip(img, 0.0, 1.0)
    else:
        img[...,:3] *= alpha
        img[...,:3] += (1-alpha)
        img = np.clip(img, 0.0, 1.0)
    
    if mip is not None:
        # TODO(ttakikawa): resize_mip causes the mask to be squuezed... why?
        img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        if use_depth:
            depth = resize_mip(depth, mip, interpolation=cv2.INTER_NEAREST)
            #mask_depth = resize_mip(mask_depth[...,None].astype(np.float), mip, interpolation=cv2.INTER_NEAREST)

    img = torch.from_numpy(img)
    if use_depth:
        depth = torch.from_numpy(depth)

    if use_depth:
        mask_depth = torch.logical_and(depth > -1000, depth < 1000)
        depth[~mask_depth] = -1.0
        depth = depth[:,:,np.newaxis]

    if loader_type == 'cv2' or loader_type == 'imageio':
        # BGR to RGB
        img[...,:3] = img[...,:3][...,::-1]

    if srgb:
        img = linear_to_srgb(img)

    alpha = mask_depth

    return img.cpu(), alpha.cpu(), depth.cpu()

### Loading Camera Parameters

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
        return dict(basename=basename, img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
    else:
        #log.info(f"File name {fpath} doesn't exist. Ignoring.")
        return None

def _parallel_load_standard_imgs(task_basename, args):
    """Internal function for multiprocessing.
    """
    result = _load_standard_imgs(args['frame'], args['root'], mip=args['mip'])
    if result is None:
        return dict(task_basename=task_basename, basename=None, img=None, pose=None)
    else:
        return dict(task_basename=task_basename, basename=result['basename'], img=result['img'], pose=result['pose'])

def load_standard_transforms(root, split='train', bg_color='white', num_workers=-1, mip=None):
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
        assert False and "Unsupported number of splits, there should be ['test', 'train', 'val']"
    
    if split not in transform_dict:
        assert False and f"Split type ['{split}'] unsupported in the dataset provided"

    for key in transform_dict:
        with open(transform_dict[key], 'r') as f:
            transform_dict[key] = json.load(f)

    imgs = []
    poses = []
    basenames = []
    
    if num_workers > -1:
        # threading loading images 

        try:
            executor = futureproof.ProcessPoolExecutor(max_workers=num_workers)

            timeall = time.time()
            with futureproof.TaskManager(executor) as tm:
                for i, frame in enumerate(transform_dict[split]['frames']):
                    _args = dict(frame=frame, root=root, mip=mip)
                    tm.submit(_parallel_load_standard_imgs, task_basename=i, args=_args)
                for task in tqdm(tm.as_completed(), desc='loading data'):
                    basename = task.result['basename']
                    img = task.result['img']
                    pose = task.result['pose']
                    if basename is not None:
                        basenames.append(basename)
                    if img is not None:
                        imgs.append(img)
                    if pose is not None:
                        poses.append(pose)
        finally:
            if executor is not None:
                executor.join()
    
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
        log.info("WARNING: The dataset expects premultiplied alpha correction, but the current implementation does not handle this.")

    if 'k1' in transform_dict[split]:
        log.info("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

    if 'rolling_shutter' in transform_dict[split]:
        log.info("WARNING: The dataset expects rolling shutter correction, but the current implementation does not handle this.")
        
    # The principal point in wisp are always a displacement in pixels from the center of the image.
    x0 = 0.0
    y0 = 0.0
    # The standard dataset generally stores the absolute location on the image to specify the principal point.
    # Thus, we need to scale and translate them such that they are offsets from the center.
    if 'cx' in transform_dict[split]:
        x0 = (float(transform_dict[split]['cx']) / (2**mip)) - (w//2)
    if 'cy' in transform_dict[split]:
        y0 = (float(transform_dict[split]['cy']) / (2**mip)) - (h//2)

    offset = transform_dict[split]['offset'] if 'offset' in transform_dict[split] else [0,0,0]
    scale = transform_dict[split]['scale'] if 'scale' in transform_dict[split] else 1.0
    aabb_scale = transform_dict[split]['aabb_scale'] if 'aabb_scale' in transform_dict[split] else 1.0
    
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
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, camera.width, camera.height, device='cuda')
        rays.append(generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to('cpu'))

    rays = Rays.stack(rays).to(dtype=torch.float)

    rgbs = imgs[...,:3]
    alpha = imgs[...,3:4]
    if alpha.numel() == 0:
        masks = torch.ones_like(rgbs[...,0:1]).bool()
    else:
        masks = (alpha > 0.5).bool()
        
        if bg_color == 'black':
            rgbs[...,:3] -= (1-alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
        else:
            rgbs[...,:3] *= alpha
            rgbs[...,:3] += (1-alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
    
    return {"imgs": rgbs, "masks": masks, "rays": rays, "cameras": cameras}

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


def load_rtmv_transforms(root, split, mip=None, normalize=True, return_pointcloud=False, bg_color='white', num_workers=-1):
    """Load the RTMV transforms.

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
    assert(len(json_files) > 0 and "No JSON files found")

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
        # Leaving this here just in case...
        #ray_o, ray_d = generate_rays_from_tf(camera)
        #ray_o = ray_o.to(torch.float32)
        #ray_d = ray_d.to(torch.float32)
        ray_grid = generate_centered_pixel_coords(camera.width, camera.height, camera.width, camera.height, device='cuda')
        _rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to('cpu')
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

    images = torch.stack(images)[...,:3]
    alphas = torch.stack(alphas)
    depths = torch.stack(depths)
    rays = Rays.stack(rays)

    output = {"imgs": images, "masks": alphas, "rays": rays, "depths": depths, "cameras": cameras}
    if return_pointcloud:
        output.update({"coords": coords, "coords_center": coords_center, "coords_scale": coords_scale})

    return output

def normalize_pointcloud(coords, return_scale=False):
    """Normalizes pointcloud to an AABB within [-1, 1].

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape [N, 3]
        return_scale (bool): If True, will return the center of the cloud and the scaling factor.

    Returns:
        (torch.FloatTensor) or (torch.FloatTensor, torch.FloatTensor, float):
        - Normalized 3D coordinates of shape [N, 3]
        - Center of the cloud of shape [3]
        - Scaling factor (floating point value)
    """
    coords_max, _ = torch.max(coords, dim=0)
    coords_min, _ = torch.min(coords, dim=0)
    coords_center = (coords_max + coords_min) / 2.0

    # AABB normalize
    coords = coords - coords_center
    max_dist = torch.max(coords)
    coords_scale = 1.0 / max_dist
    coords *= coords_scale

    if return_scale:
        return coords, coords_center, coords_scale
    else:
        return coords


def create_pointcloud_from_images(rgbs, masks, rays, depths):
    """Given depth images, will create a RGB pointcloud.

    TODO (ttakikawa): Probably make the input a tensor not a list...

    Args:
        rgbs (list of torch.FloatTensor): List of RGB tensors of shape [H, W, 3].
        masks (list of torch.FloatTensor): List of mask tensors of shape [H, W, 1].
        rays (list of wisp.core.Rays): List of rays.origins and rays.dirs of shape [H, W, 3].
        depths (list of torch.FloatTensor): List of depth tensors of shape [H, W, 1].

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        - 3D coordinates of shape [N*H*W, 3]
        - colors of shape [N*H*W, 3]
    """
    cloud_coords = []
    cloud_colors = []

    for i in range(len(rgbs)):
        mask = masks[i].bool()
        h, w = mask.shape[:2]
        mask = mask.reshape(h, w)
        depth = depths[i].reshape(h, w, 1)
        assert(len(mask.shape) == 2 and "Mask shape is not correct... it should be [H,W], check size here")
        coords = rays[i].origins[mask] + rays[i].dirs[mask] * depth[mask]
        colors = rgbs[i][mask]
        cloud_coords.append(coords.reshape(-1, 3))
        cloud_colors.append(colors[...,:3].reshape(-1, 3))

    return torch.cat(cloud_coords, dim=0), torch.cat(cloud_colors, dim=0)

### Image / Tensor Processing

def resize_mip(img, mip, interpolation=cv2.INTER_LINEAR):
    """Resize image with cv2.

    Args:
        img (torch.FloatTensor): Image of shape [H, W, 3]
        mip (int): Rescaling factor. Will rescale by 2**mip.
        interpolation: Interpolation modes used by `cv2.resize`.

    Returns:
        (torch.FloatTensor): Rescaled image of shape [H/(2**mip), W/(2**mip), 3]
    """
    resize_factor = 2**mip
    # WARNING: cv2 expects (w,h) for the shape. God knows why :)
    shape = (int(img.shape[1] // resize_factor), int(img.shape[0] // resize_factor))
    img = cv2.resize(img, dsize=shape, interpolation=interpolation)
    return img

### Image Metrics

def psnr(rgb, gts):
    """Calculate the PSNR metric.
    
    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The PSNR score
    """
    assert(rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert(gts.max() <= 1.05 and gts.min() >= -0.05)
    assert(rgb.shape[-1] == 3)
    assert(gts.shape[-1] == 3)

    mse = torch.mean((rgb[...,:3]-gts[...,:3])**2).item()
    return 10 * np.log10(1.0/mse)

def lpips(rgb, gts, lpips_model=None):
    """Calculate the LPIPS metric.
    
    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The LPIPS score
    """
    assert(rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert(gts.max() <= 1.05 and gts.min() >= -0.05)
    assert(rgb.shape[-1] == 3)
    assert(gts.shape[-1] == 3)
    

    if lpips_model is None:
        lpips_model = LPIPS(net='vgg').cuda()
    
    return lpips_model(
            (2.0 * rgb[...,:3] - 1.0).permute(2,0,1),
            (2.0 * gts[...,:3] - 1.0).permute(2,0,1)).mean().item()

def ssim(rgb, gts):
    """Calculate the SSIM metric.
    
    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.FloatTensor): Image tensor of shape [H,W,3]
        gts (torch.FloatTensor): Image tensor of shape [H,W,3]

    Returns:
        (float): The SSIM score
    """
    assert(rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert(gts.max() <= 1.05 and gts.min() >= -0.05)
    return skimage.metrics.structural_similarity(
            rgb[...,:3].cpu().numpy(),
            gts[...,:3].cpu().numpy(),
            multichannel=True,
            data_range=1,
            gaussian_weights=True,
            sigma=1.5)

