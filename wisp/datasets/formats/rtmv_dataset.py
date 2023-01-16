# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import copy
import os
import re
import glob
import json
import cv2
from tqdm import tqdm
import math
import numpy as np
import torch
from torch.multiprocessing import Pool
from typing import Callable, List, Dict
from kaolin.render.camera import Camera, blender_coords
from wisp.core import Rays
import wisp.ops.image as img_ops
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud
from wisp.datasets.base_datasets import MultiviewDataset
from wisp.datasets.batch import MultiviewBatch


class RTMVDataset(MultiviewDataset):
    """ A dataset for loading data files in the RTMV format:
        RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis
        See: http://www.cs.umd.edu/~mmeshry/projects/rtmv/
        RTMV scenes include RGB + depth information.
    """

    def __init__(self, dataset_path: str, bg_color: str, mip: int = 0, split: str = 'train',
                 train_ratio: float = 2.0 / 3.0, val_ratio: float = 1.0 / 30.0,
                 coords_center: torch.Tensor = None, coords_scale: torch.Tensor = None,
                 dataset_num_workers: int = -1, transform: Callable = None):
        """Loads the RTMV data and applies dataset specific transforms required for compatibility with the framework.
        The loaded data is cached inside the `data` field.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            bg_color (str): The background color to use for when alpha=0.
                Options: 'black', 'white'.
            mip (int): If provided, will rescale images by 2**mip. Useful when large images are loaded.
            split (str): The dataset split to use. The exact split's content
                are determined by train_ratio, val_ratio.
                Options: 'train', 'val', 'test'.
            train_ratio (float): Ratio of files allocated for the 'train' split.
            val_ratio (float): Ratio of files allocated for the 'validation' split.
                Note: test_ratio = 1 - train_ratio - val_ratio
            coords_center Optional[torch.Tensor]: optional, since rtmv contains depth information, if coords_center is
                specified, a point cloud will be computed from the depth rays, and centered using coords_center,
                and then used to normalize the ray origins.
                If coords center and scale are both not specified, they're calculated from the data.
                Usually this parameter is specified to ensure multiple splits are normalized in the same way
                (i.e. train set computes coords_center, and validation / test set normalize using that value).
            coords_scale Optional[torch.Tensor]: optional, since rtmv contains depth information, if coords_scale is
                specified, a point cloud will be computed from the depth rays, and scaled using coords_scale,
                and then used to normalize the ray origins.
                If coords center and scale are both not specified, they're calculated from the data.
                Usually this parameter is specified to ensure multiple splits are normalized in the same way
                (i.e. train set computes coords_scale, and validation / test set normalize using that value).
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        super().__init__(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                         transform=transform, split=split)
        self.mip = mip
        self.bg_color = bg_color

        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._json_files = self._list_jsons_for_split(dataset_path, split, train_ratio, val_ratio)

        images, alphas, depths, cameras = self.load()
        rays = self._raygen(cameras)
        self.coords = None

        # If coords center and scale were not specified, calculate them from data
        # This logic will also normalize the ray origins and calculate the depth values
        should_normalize = coords_center is None or coords_scale is None
        if should_normalize:
            depths, rays, coords_center, coords_scale = self._normalize(images, alphas, depths, cameras, rays)
            self.coords, _ = create_pointcloud_from_images(images, alphas, rays, depths)  # RGBD -> point cloud
        self.coords_center = coords_center
        self.coords_scale = coords_scale

        self.data = dict(
            rgb=torch.stack(images)[..., :3],
            alpha=torch.stack(alphas),
            rays=Rays.stack(rays),
            depth=torch.stack(depths),
            cameras=cameras
        )

        if not should_normalize:
            self.data["depth"] = self.data["depth"] * self.coords_scale
            self.data["rays"].origins = (self.data["rays"].origins - self.coords_center) * self.coords_scale

        self._img_shape = self.data["rgb"].shape[1:3]
        self.flatten_tensors()

    def create_split(self, split: str, transform: Callable = None) -> RTMVDataset:
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation and test datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.
        RTMV will also make sure that the created split uses the same normalization values of
        coords_center, coords_scale, such that both dataset splits are compatible.

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val', 'test'.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        return RTMVDataset(
            dataset_path=self.dataset_path,
            bg_color=self.bg_color,
            mip=self.mip,
            split=split,
            train_ratio=self._train_ratio,
            val_ratio=self._val_ratio,
            coords_center=self.coords_center,
            coords_scale=self.coords_scale,
            dataset_num_workers=self.dataset_num_workers,
            transform=transform
        )

    def __getitem__(self, idx: int) -> MultiviewBatch:
        """Retrieve a batch of rays and their corresponding values.
        Rays are precomputed from the dataset's cameras, and are cached within the dataset.
        By default, rays are assumed to have corresponding rgb and depth values, sampled from the dataset's images.

        Returns:
            (MultiviewBatch): A batch of rays and their rgbd values. The fields can be accessed as a dictionary:
                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the gt image pixel rgb
                    each ray intersects.
                "depth" - a torch.Tensor of depth which corresponds the gt image pixel depth
                    each ray intersects.
                "alpha" - a torch.Tensor of alpha value which corresponds to the gt image alpha.
        """
        out = MultiviewBatch(
            rays=self.data["rays"][idx],
            rgb=self.data["rgb"][idx],
            depth=self.data["depth"][idx],
            alpha=self.data["alpha"][idx]
        )

        if self.transform is not None:
            out = self.transform(out)

        return out

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.

        Args:
                root (str): A path to the root directory of the dataset.
                files_list (List[str]): List of files within the dataset root, without their prefix path.
        Returns:
                True if the root folder points to content loadable by this dataset.
        """
        # Assume the existance of:
        # 00000.json, 00000.exr, 00001.json, 00001.exr, ...
        json_regex = re.compile(r"\d{5}\.json")
        json_files = list(filter(json_regex.match, files_list))
        exr_regex = re.compile(r"\d{5}\.exr")
        exr_files = list(filter(exr_regex.match, files_list))
        return len(json_files) == len(exr_files) and len(json_files) > 0

    @staticmethod
    def _list_jsons_for_split(dataset_path: str, split: str, train_ratio: float, val_ratio: float) -> List[str]:
        """
        Returns a list of dataset json files to use for the given split.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            split (str): The dataset split to use. The exact split's content are determined by train_ratio, val_ratio.
                Options: 'train', 'val', 'test'.
            train_ratio (float): Ratio of files allocated for the 'train' split.
            val_ratio (float): Ratio of files allocated for the 'validation' split.
                Note: test_ratio = 1 - train_ratio - val_ratio
        """
        json_files = sorted(glob.glob(os.path.join(dataset_path, '*.json')))

        # Hard-coded train-val-test splits for now
        train_split_idx = math.floor(len(json_files) * train_ratio)
        eval_split_idx = train_split_idx + math.floor(len(json_files) * val_ratio)

        if split == 'train':
            subset_idxs = np.arange(0, train_split_idx)
        elif split in 'val':
            subset_idxs = np.arange(train_split_idx, eval_split_idx)
        elif split == 'test':
            subset_idxs = np.arange(eval_split_idx, len(json_files))
        else:
            raise RuntimeError(f"Unknown split type: {split}, split should be any of: ('train', 'val', 'test')")

        json_files = [json_files[i] for i in subset_idxs]
        if len(json_files) == 0:
            raise ValueError(f"RTMVDataset: No JSON files found for split {split} under {dataset_path}")
        return json_files

    def load_singleprocess(self):
        """Standard parsing function for loading rtmv files on the main process.
        This follows the conventions defined in http://www.cs.umd.edu/~mmeshry/projects/rtmv/

        Returns:
            (tuple of torch.FloatTensors): Channels of information from NeRF:
                - images: a list of torch.FloatTensor of size [H,W,3], each entry corresponds to a single rgb image.
                - alphas: a list of torch.FloatTensor of size [H,W,1], each entry corresponds to a single alpha image.
                - depths: a list of torch.FloatTensor of size [H,W,1], each entry corresponds to a single depth image.
                - cameras: a dictionary of filenames (view id) -> kaolin Camera object
        """
        images = []
        alphas = []
        depths = []
        cameras = dict()

        for img_index, json_file in tqdm(enumerate(self._json_files), desc='loading data'):
            with torch.no_grad():
                basename = os.path.splitext(os.path.basename(json_file))[0]
                image, alpha, depth = RTMVDataset._load_rtmv_images(self.dataset_path, basename,
                                                                    use_depth=True, mip=self.mip,
                                                                    srgb=True, bg_color=self.bg_color)
                json_path = os.path.join(self.dataset_path, basename + ".json")
                camera = RTMVDataset._load_rtmv_camera(path=json_path)
                transformed_camera = RTMVDataset._transform_rtmv_camera(copy.deepcopy(camera), mip=self.mip)

                images.append(image)
                alphas.append(alpha)
                depths.append(depth)
                cameras[basename] = transformed_camera
        return images, alphas, depths, cameras

    @staticmethod
    def _load_rtmv_images(root, basename, use_depth=False, mip=None, srgb=False, bg_color='white'):
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

    @staticmethod
    def _parallel_load_rtmv_data(args):
        """ An internal wrapper function to allow rtmv load faster with multiprocessing.
        Represents a task performed by a single worker.
        Note: Internal logic should occur on the cpu, cuda behavior for multiprocessing is unstable.
        """
        torch.set_num_threads(1)
        with torch.no_grad():
            image, alpha, depth = RTMVDataset._load_rtmv_images(**args['exr_args'])
            camera = RTMVDataset._load_rtmv_camera(args['camera_args']['path'])
            transformed_camera = RTMVDataset._transform_rtmv_camera(copy.deepcopy(camera),
                                                                    mip=args['camera_args']['mip'])
            return dict(
                task_basename=args['task_basename'],
                image=image,
                alpha=alpha,
                depth=depth,
                camera=transformed_camera.cpu()
            )

    def load_multiprocess(self):
        """Standard parsing function for loading rtmv files with multiple workers.
        This follows the conventions defined in http://www.cs.umd.edu/~mmeshry/projects/rtmv/

        Returns:
            (tuple of torch.FloatTensors): Channels of information from NeRF:
                - images: a list of torch.FloatTensor of size [H,W,3], each entry corresponds to a single rgb image.
                - alphas: a list of torch.FloatTensor of size [H,W,1], each entry corresponds to a single alpha image.
                - depths: a list of torch.FloatTensor of size [H,W,1], each entry corresponds to a single depth image.
                - cameras: a dictionary of filenames (view id) -> kaolin Camera object
        """
        images = []
        alphas = []
        depths = []
        cameras = dict()

        p = Pool(self.dataset_num_workers)
        try:
            basenames = (os.path.splitext(os.path.basename(json_file))[0] for json_file in self._json_files)
            iterator = p.imap(RTMVDataset._parallel_load_rtmv_data, [
                dict(
                    task_basename=basename,
                    exr_args=dict(
                        root=self.dataset_path,
                        basename=basename,
                        use_depth=True,
                        mip=self.mip,
                        srgb=True,
                        bg_color=self.bg_color),
                    camera_args=dict(
                        path=os.path.join(self.dataset_path, basename + '.json'),
                        mip=self.mip)) for basename in basenames])
            for _ in tqdm(range(len(self._json_files)), desc='loading data'):
                result = next(iterator)
                images.append(result["image"])
                alphas.append(result["alpha"])
                depths.append(result["depth"])
                cameras[result['task_basename']] = result["camera"]
        finally:
            p.close()
            p.join()
        return images, alphas, depths, cameras

    @staticmethod
    def _rescale_rtmv_intrinsics(camera: Camera, target_size, original_width, original_height):
        """ Internal loading function: rescales the camera intrinsics according to the given ratios. """
        assert original_height == original_width, 'Current code assumes a square image.'
        resize_ratio = target_size * 1. / original_height
        camera.x0 *= resize_ratio
        camera.y0 *= resize_ratio
        camera.focal_x *= resize_ratio
        camera.focal_y *= resize_ratio

    @staticmethod
    def _load_rtmv_camera(path: str) -> Camera:
        """ Internal loading function: loads a single RTMV camera object from json metadata.
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
                                  far=6.0,  # inheriting default for nerf-synthetic
                                  dtype=torch.float64,
                                  device='cpu')

        # RTMV cameras use Blender coordinates, which are right handed with Z axis pointing upwards instead of Y.
        camera.change_coordinate_system(blender_coords())

        return camera.cpu()

    @staticmethod
    def _transform_rtmv_camera(camera: Camera, mip: int) -> Camera:
        """ Transforms the RTMV camera's image plane according to the mip (essentially downsampling the image plane
        resolution).
        Returns the transformed camera with updated image plane and intrinsics.
        """
        original_width, original_height = camera.width, camera.height
        if mip is not None:
            camera.width = camera.width // (2 ** mip)
            camera.height = camera.height // (2 ** mip)

        # assume no resizing
        RTMVDataset._rescale_rtmv_intrinsics(camera, camera.width, original_width, original_height)
        return camera

    @staticmethod
    def _raygen(cameras: List[Camera]) -> List[Rays]:
        """ Generates a full ray pack for each camera view. """
        rays = []
        for idx in cameras:
            camera = cameras[idx]
            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                      camera.width, camera.height,
                                                      device='cuda')
            _rays = generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(
                camera.height, camera.width, 3).to('cpu')
            rays.append(_rays.to(dtype=torch.float32))
        return rays

    @staticmethod
    def _normalize(images: torch.Tensor, alphas: torch.Tensor, depths: torch.Tensor,
                   cameras: Dict[str, Camera], rays: List[Rays]):
        """ Normalizes the content of all views to fit within an axis aligned bounding box of [-1, 1]:
        1. RTMV scenes contain depth information, which is used to compute a point cloud from the depth rays.
        2. The pointcloud is normalized within the AABB of [-1, 1].
        3. The depth information, generated rays and cameras are rescaled according to normalization factors:
            coords_center, coords_scale.

        Returns:
            - (torch.Tensor) depths: the rescaled depth values of each ray
            - (wisp.core.Rays) rays: the rescaled rays
            - (torch.Tensor) coords_center: Value used to centeralize the point cloud around 0, 0, 0.
            - (torch.Tensor) coords_scale: Value used to scale the point cloud within [-1, 1].
        """
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

        return depths, rays, coords_center, coords_scale

    def flatten_tensors(self) -> None:
        """ Flattens the cached data tensors to (NUM_VIEWS, NUM_RAYS, *).
        """
        num_imgs = len(self)
        self.data["rays"] = self.data["rays"].reshape(num_imgs, -1, 3)
        self.data["rgb"] = self.data["rgb"].reshape(num_imgs, -1, 3)
        self.data["depth"] = self.data["depth"].reshape(num_imgs, -1, 1)
        self.data["alpha"] = self.data["alpha"].reshape(num_imgs, -1, 1)

    @property
    def img_shape(self) -> torch.Size:
        """ Returns the shape of the rescaled dataset images (cached values are flattened) """
        return self._img_shape

    @property
    def cameras(self) -> List[Camera]:
        """ Returns the list of camera views used to generate rays for this dataset. """
        return self.data["cameras"]

    def as_pointcloud(self) -> torch.Tensor:
        """ If supports_depth=True, this function can be used to query the depth information in
        the form of a pointcloud tensor.
        """
        return self.coords

    def supports_depth(self) -> bool:
        """ Returns if this dataset have loaded depth information. """
        return self.coords is not None

    @property
    def num_images(self) -> int:
        """ Returns the number of views this dataset stores. """
        return self.data["rgb"].shape[0]
