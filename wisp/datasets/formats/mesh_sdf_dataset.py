# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
from typing import Callable, List
import logging as log
import torch
from wisp.datasets.base_datasets import SDFDataset
from wisp.datasets.batch import SDFBatch
import wisp.ops.mesh as mesh_ops


_SUPPORTED_FORMATS = ['obj']
""" Supported mesh formats this dataset can load """


class MeshSampledSDFDataset(SDFDataset):
    """ A dataset class for signed-distance functions sampled over a single mesh shape's faces.
    The dataset supports online resampling during training. However, sampling time can take up to tens of seconds.
    Samples can also optionally include normals and rgb information when the mesh is textured with materials.
    """

    def __init__(self,
                 dataset_path: str,
                 split: str,
                 transform: Callable = None,
                 sample_mode: list = None,
                 num_samples: int = 100000,
                 get_normals: bool = False,
                 sample_tex: bool = False,
                 mode_norm: str = 'sphere'
                 ):
        """Construct by loading the mesh and generating an initial dataset of coords + sdf samples.

        Args:
            dataset_path (str): Path to file with mesh to initialize the dataset.
                Supported formats: .obj only.
            split (str): Any of 'test', 'train', 'val'.
                Currently used for keeping track of the dataset purpose and not used internally.
            transform (Optional[Callable]):
                A transform function applied per batch when data is accessed with __get_item__.
            sample_mode (list of str): List of different sample methods to apply over the mesh.
                Any sequential combo of:
                    'tracer' - samples generated on the mesh surface
                    'near' - samples generated near the mesh surface
                    'rand' - samples generated unifromly in space
                    See `mesh_ops.point_sample` for additional implementation details for the sampler.
                    Default: ['rand', 'rand', 'near', 'near', 'trace']
            num_samples (int): Number of data points to keep in the working set at any given time.
                A new working set can be regenerated with resample().
            get_normals (bool): If True, __get_item__ will also return normals (estimated by the SDF field).
            sample_tex (bool): If True, __get_item__  will also return RGB values from the nearest texture.
            mode_norm (str): The mode at which the mesh will be normalized in [-1, 1] space:
                'sphere' - the mesh will be normalized within the unit sphere.
                'aabb' - the mesh will be normalized within an axis aligned bounding box extending [-1, 1].
                'planar' - the mesh will be normalized using a non-uniform scale applied to the x,z axes separately.
                'none' - no normalization will take place.
        """
        super().__init__(dataset_path=dataset_path, transform=transform, split=split)

        # Sampling args
        self.sample_mode = sample_mode if sample_mode is not None else ['rand', 'rand', 'near', 'near', 'trace']
        self.num_samples = num_samples
        self.get_normals = get_normals
        self.sample_tex = sample_tex
        self.mode_norm = mode_norm

        # Mesh info
        self.verts = self.faces = self.texv = self.texf = self.mats = None

        # Validate & load mesh, then sample from it
        self.validate(dataset_path)
        self.data = None
        self.load()

    def validate(self, dataset_path) -> None:
        """ Validates the dataset path represents a valid mesh.
        If the mesh is valid, this function returns gracefully, otherwise it asserts with an informative error.

        Args:
            dataset_path (str): Path to a mesh file to initialize the dataset.
                Supported formats: .obj only.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"MeshSampledSDFDataset requires a mesh path, "
                                    f"the dataset_path does not exist: {self.dataset_path}")
        if not any([dataset_path.endswith(ext) for ext in _SUPPORTED_FORMATS]):
            raise FileNotFoundError(f"MeshSampledSDFDataset does not support the mesh format of {self.dataset_path}. "
                                    f"Please use any of the supported formats: {_SUPPORTED_FORMATS}")

    def _load_mesh(self, dataset_path, mode_norm='sphere', sample_tex=False):
        """ Loads obj file, optionally with texture & materials information.

        Args:
            dataset_path (str): Path to a mesh file to initialize the dataset.
                Supported formats: .obj only.
            sample_tex (bool): If True, will load texture coordinates and materials.
            mode_norm (str): The mode at which the mesh will be normalized in [-1, 1] space:
                'sphere' - the mesh will be normalized within the unit sphere.
                'aabb' - the mesh will be normalized within an axis aligned bounding box extending [-1, 1].
                'planar' - the mesh will be normalized using a non-uniform scale applied to the x,z axes separately.
                'none' - no normalization will take place.
        Returns:
            (torch.FloatTensor) mesh: vertex coordinates of triangular faces, of shape [F, 3]
            (torch.FloatTensor) vertices: mesh vertices of shape [V, 3]
            (torch.LongTensor) faces: mesh face indices of shape [F, 3]
            (torch.LongTensor or None) texv
            (torch.LongTensor or None) texf
            (torch.FloatTensor or None) materials
        """
        texv = texf = materials = None
        if sample_tex:
            out = mesh_ops.load_obj(dataset_path, load_materials=True)
            verts, faces, texv, texf, materials = out
        else:
            verts, faces, = mesh_ops.load_obj(dataset_path)

        verts, faces, = mesh_ops.normalize(verts, faces, mode_norm)
        mesh = verts[faces]
        return mesh, verts, faces, texv, texf, materials

    def load_singleprocess(self) -> None:
        """Initializes the dataset by loading a mesh and sampling SDFs from it.
        This function uses the main process to load the dataset, without spawning any workers.
        """
        mesh_fields = self._load_mesh(self.dataset_path, self.mode_norm, self.sample_tex)
        _, self.verts, self.faces, self.texv, self.texf, self.mats = mesh_fields
        self.resample()

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
        # Sampled meshes are currently always assumed to be given as mesh files, check for any of the compatible formats
        return any([root.endswith(ext) for ext in _SUPPORTED_FORMATS])

    def __len__(self):
        """Return length of dataset, as number of samples currently resampled."""
        return self.data["coords"].size()[0]

    def __getitem__(self, idx) -> SDFBatch:
        """Retrieve a batch of sample coordinates and their sdf values.
        If the dataset was loaded with rgb and normals information, these channels will be returned as well.

        Returns:
            (SDFBatch): A batch of coordinates and their SDF values. The fields can be accessed as a dictionary:
                "coords" - a torch.Tensor of the 3d coordinates of each sample
                "sdf" - a torch.Tensor of the signed distance of each sample from the mesh surface.
                "rgb" - optional, a torch.Tensor of the "textured color" the sample interpolates from the nearest point
                    on the mesh surface.
                    If the dataset with built with `sample_tex`=False, this entry is None.
                "normals" - optional,  a torch.Tensor of the normal to the mesh surface.
                    If the dataset with built with `get_normals`=False, this entry is None.
        """
        out = SDFBatch(
            coords=self.data["coords"][idx],
            sdf=self.data["sdf"][idx],
            rgb=self.data["rgb"][idx] if "rgb" in self.data else None,
            normals=self.data["normals"][idx] if "normals" in self.data else None
        )

        if self.transform is not None:
            out = self.transform(out)

        return out

    def resample(self) -> None:
        """Resamples a new working set of SDFs, by computing an entirely new set of SDF samples.
        resample() updates the self.data field in place.
        !! Warning: this operation can take up to a minute !!
        """
        log.info(f"Resampling mesh for new sdf samples...")
        self.data = dict()
        rgb = nrm = None
        if self.get_normals:
            pts, nrm = mesh_ops.sample_surface(self.verts, self.faces, self.num_samples * len(self.sample_mode))
        else:
            pts = mesh_ops.point_sample(self.verts, self.faces, self.sample_mode, self.num_samples)

        if self.sample_tex:
            rgb, _, d = mesh_ops.closest_tex(self.verts.cuda(), self.faces.cuda(), self.texv, self.texf, self.mats, pts)
        else:
            d = mesh_ops.compute_sdf(self.verts.cuda(), self.faces.cuda(), pts.cuda())

        self.data['coords'] = pts.cpu()
        self.data['sdf'] = d.cpu()
        if rgb is not None:
            self.data['rgb'] = rgb.cpu()
        if nrm is not None:
            self.data['normals'] = nrm.cpu()

    @property
    def coordinates(self) -> torch.Tensor:
        """ Returns the coordinates of samples stored in this dataset. """
        return self.data["coords"]
