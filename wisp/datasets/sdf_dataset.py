# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from torch.utils.data import Dataset
import logging as log
import kaolin.ops.spc as spc_ops
from wisp.models.grids import OctreeGrid
import wisp.ops.mesh as mesh_ops
import wisp.ops.spc as wisp_spc_ops


class SDFDataset(Dataset):
    """Base class for single mesh datasets with points sampled only at a given octree sampling region.
    """
    def __init__(self, 
        sample_mode       : list       = ['rand', 'rand', 'near', 'near', 'trace'],
        num_samples       : int        = 100000,
        get_normals       : bool       = False,
        sample_tex        : bool       = False,
        dataset_path      : str        = None,
        mode_norm         : str        = 'sphere',
        grid              : OctreeGrid = None,
        samples_per_voxel : int        = 32 
    ):
        """Construct dataset. This dataset also needs to be initialized.

        Args:
            sample_mode (list of str): List of different sample modes. 
                                       See `mesh_ops.point_sample` for more details.
            num_samples (int): Number of data points to keep in the working set.
                               Whatever samples are not in the working set
                               can be sampled with resample.
            get_normals (bool): If True, will also return normals (estimated by the SDF field).
            sample_tex (bool): If True, will also sample RGB from the nearest texture.
            dataset_path (str): If set, Path to OBJ file to initialize the dataset.
            mode_norm (str): Used with dataset_path,
                             the mode at which the mesh will be normalized in [-1, 1] space.
            grid (OctreeGrid): If set, an OctreeGrid class initialized from mesh to initialize the 
        """
        self.sample_mode = sample_mode
        self.num_samples = num_samples
        self.get_normals = get_normals
        self.sample_tex = sample_tex
        if dataset_path is not None:
            if grid is not None:
                raise ValueError("'dataset_path' should not be set with 'grid'")
            self._init_from_mesh(dataset_path, mode_norm)
        elif grid is not None:
            self._init_from_grid(grid, samples_per_voxel)
        else:
            raise ValueError("'dataset_path' or 'grid' must be set")
    
    def _init_from_mesh(self, dataset_path, mode_norm='sphere'):
        """Initializes the dataset by sampling SDFs from a mesh.

        Args:
            dataset_path (str): Path to OBJ file.
            mode_norm (str): The mode at which the mesh will be normalized in [-1, 1] space.
        """
        self.initialization_mode = "mesh"
        
        if self.sample_tex:
            out = mesh_ops.load_obj(dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            self.V, self.F = mesh_ops.load_obj(dataset_path)

        self.V, self.F = mesh_ops.normalize(self.V, self.F, mode_norm)

        self.mesh = self.V[self.F]
        self.resample()

    # TODO (operel): grid is only really needed for filtering out points and more efficient 'rand',
    #  better not store the mesh contents in the extent field
    def _init_from_grid(self, grid, samples_per_voxel=32):
        """Initializes the dataset by sampling SDFs from an OctreeGrid created from a mesh.

        Args:
            grid (wisp.models.grids.OctreeGrid): An OctreeGrid class initialized from mesh.
            samples_per_voxel (int): Number of data points to sample per voxel.
                                     Right now this class will sample upto 3x the points in reality since it will
                                     augment the samples with surface samples. Only used if the SDFs are sampled
                                     from a grid.
        """
        if grid.__class__.__name__ != "OctreeGrid" and "OctreeGrid" not in [pclass.__name__ for pclass in grid.__class__.__bases__]:
            raise Exception("Only the OctreeGrid class or derivatives are supported for this initialization mode")
    
        if not hasattr(grid, 'blas') and hasattr(grid.blas, 'extent') and 'vertices' in grid.blas.extent:
            raise Exception("Only the OctreeGrid class or derivatives initialized from mesh are supported for this initialization mode")

        if self.get_normals:
            raise Exception("Grid initialization does not currently support normals")

        self.initialization_mode = "grid"
        self.samples_per_voxel = samples_per_voxel

        vertices = grid.blas.extent['vertices']
        faces = grid.blas.extent['faces']

        level = grid.active_lods[-1]

        # Here, corners mean "the bottom left corner of the voxel to sample from"
        corners = spc_ops.unbatched_get_level_points(grid.blas.points, grid.blas.pyramid, level)

        # Two pass sampling to figure out sample size
        self.pts_ = []
        for mode in self.sample_mode: 
            if mode == "rand":
                # Sample the points.
                self.pts_.append(wisp_spc_ops.sample_spc(corners, level, self.samples_per_voxel).cpu())
        for mode in self.sample_mode:
            if mode == "rand":
                pass
            elif mode == "near":
                self.pts_.append(mesh_ops.sample_near_surface(vertices.cuda(),
                                                   faces.cuda(),
                                                   self.pts_[0].shape[0], 
                                                   variance=1.0 / (2 ** level)).cpu())
            elif mode == "trace":
                self.pts_.append(mesh_ops.sample_surface(vertices.cuda(),
                                 faces.cuda(),
                                 self.pts_[0].shape[0])[0].cpu())
            else:
                raise Exception(f"Sampling mode {mode} not implemented")

        # Filter out points which do not belong to the narrowband
        self.pts_ = torch.cat(self.pts_, dim=0)
        query_results = grid.query(self.pts_.cuda(), 0)
        self.pidx = query_results.pidx
        self.pts_ = self.pts_[self.pidx>-1]
    
        # Sample distances and textures.
        if self.sample_tex:
            texv = grid.blas.extent['texv']
            texf = grid.blas.extent['texf']
            mats = grid.blas.extent['mats']
            self.rgb_, self.hit_pts_, self.d_ = mesh_ops.closest_tex(vertices, faces, texv, texf, mats, self.pts_)
        else:
            log.info(f"Computing SDFs for {self.pts_.shape[0]} samples (may take a while)..")
            self.d_ = mesh_ops.compute_sdf(vertices, faces, self.pts_)
            assert(self.d_.shape[0] == self.pts_.shape[0])
        
        log.info(f"Total Samples: {self.pts_.shape[0]}")
        
        self.resample()

    @classmethod
    def from_mesh(cls, 
                  sample_mode  : list = ['rand', 'rand', 'near', 'near', 'trace'],
                  num_samples  : int  = 100000,
                  get_normals  : bool = False,
                  sample_tex   : bool = False,
                  dataset_path : str  = None,
                  mode_norm    : str  = 'sphere'):
        return cls(sample_mode, num_samples, get_normals, sample_tex,
                   dataset_path=dataset_path, mode_norm=mode_norm)

    @classmethod
    def from_grid(cls, 
                  sample_mode       : list       = ['rand', 'rand', 'near', 'near', 'trace'],
                  num_samples       : int        = 100000,
                  get_normals       : bool       = False,
                  sample_tex        : bool       = False,
                  grid              : OctreeGrid = None,
                  samples_per_voxel : int        = 32):
        return cls(sample_mode, num_samples, get_normals, sample_tex,
                   grid=grid, samples_per_voxel=samples_per_voxel)

    def resample(self):
        """Resamples a new working set of SDFs.
        """
        
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        elif self.initialization_mode == "mesh":
            # Compute new sets of SDFs entirely

            self.nrm = None
            if self.get_normals:
                self.pts, self.nrm = mesh_ops.sample_surface(self.V, self.F,
                                                             self.num_samples * len(self.sample_mode))
                self.nrm = self.nrm.cpu()
            else:
                self.pts = mesh_ops.point_sample(self.V, self.F, self.sample_mode, self.num_samples)

            if self.sample_tex:
                self.rgb, _, self.d = mesh_ops.closest_tex(self.V.cuda(), self.F.cuda(), 
                                                           self.texv, self.texf, self.mats, self.pts)
                self.rgb = self.rgb.cpu()
            else:
                self.d = mesh_ops.compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   

            self.d = self.d.cpu()
            self.pts = self.pts.cpu()

            log.info(f"Resampling...")
        
        elif self.initialization_mode == "grid":
            # Choose a new working set of SDFs
            self.pts = self.pts_
            self.d = self.d_
            
            if self.sample_tex:
                self.rgb = self.rgb_
                self.hit_pts = self.hit_pts_

            _idx = torch.randperm(self.pts.shape[0] - 1, device='cuda')

            self.pts = self.pts[_idx]
            self.d = self.d[_idx]
            if self.sample_tex:
                self.rgb = self.rgb[_idx]
                self.hit_pts = self.hit_pts[_idx]

            total_samples = self.num_samples
            self.pts = self.pts[:total_samples]
            self.d = self.d[:total_samples]
            if self.sample_tex:
                self.rgb = self.rgb[:total_samples]
                self.rgb = self.rgb.cpu()
                self.hit_pts = self.hit_pts[:total_samples]
                self.hit_pts = self.hit_pts.cpu()

            self.d = self.d.cpu()
            self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        # TODO(ttakikawa): Do this channel-wise instead
        if self.get_normals and self.sample_tex:
            return self.pts[idx], self.d[idx], self.nrm[idx], self.rgb[idx]
        elif self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]

    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        return self.pts.size()[0]
