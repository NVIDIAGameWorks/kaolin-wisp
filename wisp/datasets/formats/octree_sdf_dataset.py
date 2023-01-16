# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from typing import Callable, List
import torch
import logging as log
import kaolin.ops.spc as spc_ops
from wisp.datasets.base_datasets import SDFDataset
from wisp.datasets.batch import SDFBatch
from wisp.accelstructs import BaseAS, OctreeAS
import wisp.ops.mesh as mesh_ops
import wisp.ops.spc as wisp_spc_ops


class OctreeSampledSDFDataset(SDFDataset):
    """ A dataset class for signed-distance functions sampled from a sparse octree acceleration structure,
    which was constructed from single mesh shape. Sampling is executed on the mesh faces,
    and is limited to occupied octree cells. This dataset has the benefit of limiting the sampling region to areas
    which are actually occupied by the mesh. It also allows for equal distribution of samples per the octree cells.
    The dataset supports online resampling during training. Resampling is quick as a pool of narrowband values is
    cached for the duration of training, this pool is quickly subsampled each time resample() is invoked.
    Samples can also optionally include rgb information when the mesh is textured with materials.
    """

    def __init__(self,
                 occupancy_struct: OctreeAS,
                 split: str,
                 transform: Callable = None,
                 sample_mode: list = None,
                 num_samples: int = 100000,
                 sample_tex: bool = False,
                 samples_per_voxel: int = 32
                 ):
        """Construct a sample-set coords + sdf samples by sampling the mesh faces and dropping samples
        that fall into empty octree cells.

        Args:
            occupancy_struct (OctreeAS): An octree acceleration structure, initialized from a mesh.
            split (str): Any of 'test', 'train', 'val'.
                Currently used for keeping track of the dataset purpose and not used internally.
            transform (Optional[Callable]):
                A transform function applied per batch when data is accessed with __get_item__.
            sample_mode (list of str): List of different sample methods to apply over the mesh.
                Any sequential combo of:
                    'tracer' - samples generated on the mesh surface
                    'near' - samples generated near the mesh surface
                    'rand' - samples generated uniformly in the space of occupied grid cells
                    See `mesh_ops.point_sample` for additional implementation details for the sampler.
                    Default: ['rand', 'rand', 'near', 'near', 'trace']
            num_samples (int): Number of data points to keep in the working set at any given time.
                A new working set can be regenerated with resample().
            sample_tex (bool): If True, __get_item__  will also return RGB values from the nearest texture.
            samples_per_voxel (int): Number of samples to generate per octree grid cell to create a preliminary
                equally distributed pool of samples.
        """
        super().__init__(transform=transform, split=split)

        self.blas = occupancy_struct
        self.sample_mode = sample_mode if sample_mode is not None else ['rand', 'rand', 'near', 'near', 'trace']
        self.num_samples = num_samples
        self.sample_tex = sample_tex
        self.samples_per_voxel = samples_per_voxel

        self.validate()
        self.data_pool = None
        self.data = None
        self.load()

    @staticmethod
    def supports_blas(blas: BaseAS) -> bool:
        """ This dataset requires an octree to compute the narrowband.
        Given a bottom-level-acceleration-structure, this function checks if it contains the information needed
        to create this dataset from it (i.e. the blas was initialized from a mesh).
        """
        # This dataset requires the bottom level acceleration structure to keep the mesh info in its optional fields
        return isinstance(blas, OctreeAS) and 'vertices' in blas.extent and 'faces' in blas.extent

    def validate(self) -> None:
        """ Ensures the current dataset is valid. If it is valid, this function will terminate gracefully.
        Otherwise an informative error is prompted.
        """
        if not self.supports_blas(self.blas):
            raise RuntimeError("The Octree acceleration structure was not initialized from a mesh. To use "
                               "an OctreeAS with this dataset, make sure to construct it with a mesh.")

    def _sample_from_grid(self, blas: OctreeAS, samples_per_voxel=32):
        """Initializes the dataset by sampling SDFs from an OctreeGrid created from a mesh.

        Args:
            blas (OctreeAS): An OctreeAS class initialized from mesh.
            samples_per_voxel (int): Number of data points to uniformly sample per voxel,
                irrespective of samples generated on / near the mesh surface.
        """
        # TODO (operel): TBD when kaolin adds a mesh class:
        #  grid is only really needed for filtering out points and more efficient 'rand',
        #  better give the mesh as another input and not store the mesh contents in the extent field
        vertices = blas.extent['vertices']
        faces = blas.extent['faces']
        level = blas.max_level

        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = spc_ops.unbatched_get_level_points(blas.points, blas.pyramid, level)

        # Two pass sampling to figure out sample size
        pts = []
        for mode in self.sample_mode:
            if mode == "rand":
                # Sample the points.
                pts.append(wisp_spc_ops.sample_spc(corners, level, samples_per_voxel).cpu())
        for mode in self.sample_mode:
            if mode == "rand":
                pass
            elif mode == "near":
                pts.append(mesh_ops.sample_near_surface(vertices.cuda(), faces.cuda(), pts[0].shape[0],
                                                        variance=1.0 / (2 ** level)).cpu())
            elif mode == "trace":
                pts.append(mesh_ops.sample_surface(vertices.cuda(), faces.cuda(), pts[0].shape[0])[0].cpu())
            else:
                raise Exception(f"Sampling mode {mode} not implemented")

        # Filter out points which do not belong to the narrowband
        pts = torch.cat(pts, dim=0)
        query_results = blas.query(pts.cuda(), 0)
        pts = pts[query_results.pidx > -1]

        # Sample distances and textures.
        rgb = None
        if self.sample_tex:
            texv = blas.extent['texv']
            texf = blas.extent['texf']
            mats = blas.extent['mats']
            rgb, hit_pts, d = mesh_ops.closest_tex(vertices, faces, texv, texf, mats, pts)
        else:
            d = mesh_ops.compute_sdf(vertices, faces, pts)
            assert (d.shape[0] == pts.shape[0]), 'SDF validation logic failed: the number of returned sdf samples' \
                                                 'does not match the number of input coordinates.'

        data = dict(
            coords=pts.cpu(),
            sdf=d.cpu(),
        )
        if rgb is not None:
            data['rgb'] = rgb.cpu()
        return data

    def resample(self) -> None:
        """Resamples a working set of coordaintes + sdf from the existing pool of samples.
        This is essentially a quick subsampling operation which permutes and selects random indices.
        """
        log.info(f"Resampling {self.num_samples} samples..")
        idx = torch.randperm(self.pool_size - 1, device='cuda')
        if self.num_samples is not None and self.num_samples < self.pool_size:
            idx = idx[:self.num_samples]
        self.data = {k: v[idx] for k, v in self.data_pool.items() if v is not None}

    def load_singleprocess(self):
        """Initializes the dataset by sampling SDF values from a grid and a mesh contained within it.
        This function uses the main process to load the dataset, without spawning any workers.
        """
        log.info(f"Computing SDFs for entire samples pool (may take a while)..")
        self.data_pool = self._sample_from_grid(blas=self.blas, samples_per_voxel=self.samples_per_voxel)
        log.info(f"Total Samples in Pool: {self.data_pool['coords'].shape[0]}")
        self.resample()

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.
        This class does not load external files and therefore always returns False (e.g. should be created explicitly).
        """
        # This dataset is created from a preloaded grid and not from a file. Therefore it has to be created explicitly
        return False

    def __len__(self):
        """Return length of dataset, as number of samples currently resampled."""
        return self.data["coords"].size()[0]

    @property
    def pool_size(self) -> int:
        """ Total number of samples available in data pool, prior to subsampling the data from it with resample() """
        return self.data_pool['coords'].shape[0]

    def __getitem__(self, idx: int) -> SDFBatch:
        """Retrieve a batch of sample coordinates and their sdf values.
        If the dataset was loaded with rgb information, this channel will be returned as well.

        Returns:
            (SDFBatch): A batch of coordinates and their SDF values. The fields can be accessed as a dictionary:
                "coords" - a torch.Tensor of the 3d coordinates of each sample
                "sdf" - a torch.Tensor of the signed distance of each sample from the mesh surface.
                "rgb" - optional, a torch.Tensor of the "textured color" the sample interpolates from the nearest point
                    on the mesh surface.
                    If the dataset with built with `sample_tex`=False, this entry is None.
        """
        out = SDFBatch(
            coords=self.data["coords"][idx],
            sdf=self.data["sdf"][idx],
            rgb=self.data["rgb"][idx] if "rgb" in self.data else None,
        )

        if self.transform is not None:
            out = self.transform(out)

        return out

    @property
    def coordinates(self) -> torch.Tensor:
        """ Returns the coordinates of samples stored in this dataset. """
        return self.data["coords"]
