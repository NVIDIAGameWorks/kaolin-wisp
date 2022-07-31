# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
import cv2
import wisp._C as _C

# This is a collection of utility functions that broadly work with different 3D constructs.

def find_depth_bound(query, nug_depth, info, curr_idxes=None):
    r"""Associate query points to the closest depth bound in-order.
    
    TODO: Document the input.
    """
    if curr_idxes is None:
        curr_idxes = torch.nonzero(info).contiguous()
    return _C.render.find_depth_bound_cuda(query.contiguous(), curr_idxes.contiguous(), nug_depth.contiguous())

def compute_sdf_iou(pred, gts):
    """Compute intersection over union for SDFs.
    
    Args:
        pred (torch.FloatTensor): Predicted signed distances
        gts (torch.FloatTensor): Groundtruth signed distances

    Returns:
        (float): The IOU score between 0 and 100.
    """
    inside_pred = (pred < 0).byte()
    inside_gts = (gts < 0).byte()

    area_union = torch.sum((inside_pred | inside_gts).float()).item()
    area_intersect = torch.sum((inside_pred & inside_gts).float()).item()

    iou = area_intersect / area_union
    return 100.0 * iou

def sample_unif_sphere(n):
    """Sample uniformly random points on a sphere.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """
    u = np.random.rand(2, n)
    z = 1 - 2*u[0,:]
    r = np.sqrt(1. - z * z)
    phi = 2 * np.pi * u[1,:]
    xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).transpose()
    return xyz

def sample_fib_sphere(n):
    """
    Evenly distributed points on sphere using Fibonnaci sequence.
    From <http://extremelearning.com.au/evenly-distributing-points-on-a-sphere>
    WARNING: Order is not randomized.
    
    Args:
        n (int): Number of samples.

    Returns:
        (np.array): Positions of shape [n, 3]
    """

    i = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2*i/n)
    golden_ratio = (1 + 5**0.5)/2
    theta = 2. * np.pi * i / golden_ratio
    xyz = np.array([np.cos(theta) * np.sin(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(phi)]).transpose()
    return xyz

def normalized_grid(height, width, jitter=False, device='cuda', use_aspect=True):
    """Returns grid[x,y] -> coordinates for a normalized window.

    This is generally confusing and terrible, but in the [XYZ] 3D space, the width generally corresponds to 
    the XZ axis and the height corresponds to the Y axis. So in the normalized [XY] space, width also 
    corresponds to the X and the height corresponds to Y.

    However, PyTorch follows HW ordering. So that means, given a [HW] image, the [H+1, W] coordinate will
    see an increase in the Y axis (the 2nd axis) of the actual global coordinates.

    Args:
        height (int): grid height
        width (int): grid width
        jitter (bool): If True, will jitter the coordinates.
        device (str): Device to allocate the grid on.
        use_aspect (bool): If True, will scale the coords by the aspect ratio.

    Returns:
        (torch.FloatTensor): Coords tensor of shape [H, W, 2]
    """
    window_x = torch.linspace(-1, 1, steps=width, device=device)
    window_y = torch.linspace(1, -1, steps=height, device=device)
    
    #window_x = torch.arange(0, width, device=device) + 0.5 
    #window_y = torch.arange(0, height, device=device) + 0.5
    #window_x = window_x * (1.0/width) * 2.0 - 1.0
    #window_y = -1.0 * (window_y * (1.0/width) * 2.0 - 1.0)
    
    if jitter:
        window_x += (2.0 * torch.rand(*window_x.shape, device=device) - 1.0) * (1. / width)
        window_y += (2.0 * torch.rand(*window_y.shape, device=device) - 1.0) * (1. / height)
    
    if use_aspect:
        if width > height:
            window_x = window_x * (width/height)
        elif height > width:
            window_y = window_y * (height/width)

    #coord = torch.stack(torch.meshgrid(window_x, window_y, indexing='ij')).permute(2,1,0)
    coord = torch.stack(torch.meshgrid(window_x, window_y)).permute(2,1,0)   # torch>=1.10: indexing='ij' (default)
    return coord

def generate_rays_from_tf(camera):
    """Returns a set of ray_origins and ray_directions from a [4x4] camera to world transform.

    Currently assumes perspective projection.

    Args:
        camera Camera object

    Returns: 
        (torch.FloatTensor, torch.FloatTensor)
        - [H,W,3] tensor of ray origins
        - [H,W,3] tensor of ray directions
    """
    # TODO (operel): This ray generator doesn't support principal point inputs & focal_y
    width = camera.width
    height = camera.height
    focal_length = camera.focal_x
    camera_to_world_tf = camera.inv_view_matrix()[0]

    window_x = torch.arange(0, width, device=camera_to_world_tf.device) 
    window_y = torch.arange(0, height, device=camera_to_world_tf.device)
    grid = torch.stack(torch.meshgrid(window_x, window_y)).permute(2,1,0)   # torch>=1.10: indexing='ij' (default)
    grid_x = (grid[...,0:1] - width * 0.5 + 0.5) / focal_length
    grid_y = (grid[...,1:2] - height * 0.5 + 0.5) / focal_length
    grid = torch.cat([grid_x, -grid_y, -1.0*torch.ones_like(grid_x)], dim=-1)

    ray_ds = F.normalize(torch.sum(grid[...,None,:] * camera_to_world_tf[:3, :3], dim=-1), dim=-1)
    ray_os = camera_to_world_tf[:3, -1].expand(ray_ds.shape)

    return ray_os, ray_ds

def normalized_slice(height, width, dim=0, depth=0.0, device='cuda'):
    """Returns a set of 3D coordinates for a slicing plane.
    
    Args:
        height (int): Grid height.
        width (int): Grid width.
        dim (int): Dimension to slice along.
        depth (float): The depth (from the 0 on the axis) for which the slicing will happen.
        device (str): Device to allocate the grid on.
    
    Returns:
        (torch.FloatTensor): Coords tensor of shape [height, width, 3].
    """
    window = normalized_grid(height, width, device)
    depth_pts = torch.ones(height, width, 1, device=device) * depth

    if dim==0:
        pts = torch.cat([depth_pts, window[...,0:1], window[...,1:2]], dim=-1)
    elif dim==1:
        pts = torch.cat([window[...,0:1], depth_pts, window[...,1:2]], dim=-1)
    elif dim==2:
        pts = torch.cat([window[...,0:1], window[...,1:2], depth_pts], dim=-1)
    else:
        assert(False, "dim is invalid!")
    pts[...,1] *= -1
    return pts


def look_at(f, t, height, width, mode='persp', fov=90.0, device='cuda'):
    """Vectorized look-at function, returns an array of ray origins and directions
    
    This function is mostly just a wrapper on top of generate_rays, but will calculate for you
    the view, right, and up vectors based on the from and to.
    
    Args:
        f (list of floats): [3] size list or tensor specifying the camera origin
        t (list of floats): [3] size list or tensor specifying the camera look at point
        height (int): height of the image
        width (int): width of the image
        mode (str): string that specifies the camera mode.
        fov (float): field of view of the camera.
        device (str): device the tensor will be allocated on.

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        - [height, width, 3] tensor of ray origins
        - [height, width, 3] tensor of ray directions
    """

    camera_origin = torch.FloatTensor(f).to(device)
    camera_view = F.normalize(torch.FloatTensor(t).to(device) - camera_origin, dim=0)
    camera_right = F.normalize(torch.cross(camera_view, torch.FloatTensor([0,1,0]).to(device)), dim=0)
    camera_up = F.normalize(torch.cross(camera_right, camera_view), dim=0)

    return generate_rays(camera_origin, camera_view, camera_right, camera_up,
            height, width, mode=mode, fov=fov, device=device)

def generate_rays(camera_origin, camera_view, camera_right, camera_up, height, width, 
        mode='persp', fov=90.0, device='cuda'):
    """Generates rays from camera parameters.

    Args:
        camera_origin (torch.FloatTensor): [3] size tensor specifying the camera origin
        camera_view (torch.FloatTensor): [3] size tensor specifying the camera view direction
        camera_right (torch.FloatTensor): [3] size tensor specifying the camera right direction
        camera_up (torch.FloatTensor): [3] size tensor specifying the camera up direction
        height (int): height of the image
        width (int): width of the image
        mode (str): string that specifies the camera mode.
        fov (float): field of view of the camera.
        device (str): device the tensor will be allocated on.

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        - [height, width, 3] tensor of ray origins
        - [height, width, 3] tensor of ray directions
    """
    coord = normalized_grid(height, width, device=device)

    ray_origin = camera_right * coord[...,0,np.newaxis] * np.tan(np.radians(fov/2)) + \
                 camera_up * coord[...,1,np.newaxis] * np.tan(np.radians(fov/2)) + \
                 camera_origin + camera_view
    ray_origin = ray_origin.reshape(-1, 3)
    ray_offset = camera_view.unsqueeze(0).repeat(ray_origin.shape[0], 1)
    
    if mode == 'ortho': # Orthographic camera
        ray_dir = F.normalize(ray_offset, dim=-1)
    elif mode == 'persp': # Perspective camera
        ray_dir = F.normalize(ray_origin - camera_origin, dim=-1)
        ray_origin = camera_origin.repeat(ray_dir.shape[0], 1)
    else:
        raise ValueError('Invalid camera mode!')

    return ray_origin, ray_dir

class FirstPersonCameraController():
    """Super simple first person camera controller to be used in the renderer.
    """
    def __init__(self, origin=[5.0,2.1,4.0], fov=30):
        """Initialize controller variables.

        Args:
            origin (list of floats): Specifies the camera origin
            fov (float): Specifies the field of view of the camera.
        """
        
        self.eye = torch.tensor(origin, dtype=torch.float32)
        self.up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        self.view = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self.right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        self.fov = fov

        # vert
        self.pitch = -0.35
        # horz
        self.yaw = 2.25

        self.camera_up = self.up.clone()
        self.camera_right = self.right.clone()
        self.camera_view = self.view.clone()
        
        self.origin = torch.tensor(origin, dtype=torch.float32)

    def get_view_matrix(self):
        """Returns the 4x4 view matrix for use in the rasterizer.

        Returns:
            (torch.FloatTensor): 4x4 view matrix.
        """
        view_matrix = torch.zeros(4,4)
        view_matrix[:3, 0] = self.camera_right
        view_matrix[:3, 1] = self.camera_up
        view_matrix[:3, 2] = -self.camera_view
        view_matrix[3, 0] = -(self.eye * self.camera_right).sum()
        view_matrix[3, 1] = -(self.eye * self.camera_up).sum()
        view_matrix[3, 2] = -(self.eye * -self.camera_view).sum()
        view_matrix[3,3] = 1
        return view_matrix

    def generate_rays(self, h, w):
        """Generates the ray of the current controller state.

        Args:
            h (int): image height.
            w (int): image width.

        Returns:
            (torch.FloatTensor, torch.FloatTensor):
            - [height, width, 3] tensor of ray origins
            - [height, width, 3] tensor of ray directions
        """
        rot_y = torch.tensor(
            [[np.cos(self.yaw), 0, np.sin(self.yaw)],
             [0, 1, 0],
             [-np.sin(self.yaw), 0, np.cos(self.yaw)]],
            dtype=torch.float32)
        
        rot_x = torch.tensor(
            [[1, 0, 0],
             [0, np.cos(self.pitch), np.sin(self.pitch)],
             [0, np.sin(self.pitch), np.cos(self.pitch)]],
            dtype=torch.float32)

        self.camera_view = torch.matmul(self.view, rot_x)
        self.camera_view = torch.matmul(self.camera_view, rot_y)
        
        self.camera_right = F.normalize(torch.cross(self.camera_view, self.up), dim=0)
        self.camera_up = F.normalize(torch.cross(self.camera_right, self.camera_view), dim=0)

        ray_o, ray_d = generate_rays(
                self.eye.cuda(), 
                self.camera_view.cuda(),
                self.camera_right.cuda(),
                self.camera_up.cuda(),
                h, w, mode='persp', fov=self.fov)

        return ray_o, ray_d

    def move_right(self, speed):
        """Move camera to the right.
        """
        self.eye += self.camera_right * speed

    def move_left(self, speed):
        """Move camera to the left.
        """
        self.eye -= self.camera_right * speed
    
    def move_forward(self, speed):
        """Move camera forwards.
        """
        self.eye += self.camera_view * speed
    
    def move_backward(self, speed):
        """Move camera backwards.
        """
        self.eye -= self.camera_view * speed

    def rotate_horizontal(self, speed):
        """Adjust yaw.
        """
        self.yaw += speed

    def rotate_vertical(self, speed):
        """Adjust pitch.
        """
        self.pitch += speed
        self.pitch = np.clip(self.pitch, -np.pi/4, np.pi/4)

def matcap_sampler(path, interpolate=True):
    """Fetches MatCap texture & converts to a interpolation function (if needed).
    
    TODO(ttakikawa): Replace this with something GPU compatible.

    Args:
        path (str): path to MatCap texture
        interpolate (bool): perform interpolation (default: True)
    
    Returns:
        (np.array) or (scipy.interpolate.Interpolator)
        - The matcap texture
        - A SciPy interpolator function to be used for CPU texture fetch.
    """

    matcap = np.array(Image.open(path)).transpose(1,0,2)
    if interpolate:
        return RegularGridInterpolator((np.linspace(0, 1, matcap.shape[0]),
                                        np.linspace(0, 1, matcap.shape[1])), matcap)
    else:
        return matcap

def spherical_envmap(ray_dir, normal):
    """Computes matcap UV-coordinates from the ray direction and normal.
    
    Args:
        ray_dir (torch.Tensor): incoming ray direction of shape [...., 3]
        normal (torch.Tensor): surface normal of shape [..., 3]

    Returns:
        (torch.FloatTensor): UV coordinates of shape [..., 2]
    """
    # Input should be size [...,3]
    # Returns [N,2] # Might want to make this [...,2]
    
    # TODO(ttakikawa): Probably should implement all this on GPU
    ray_dir_screen = ray_dir.clone()
    ray_dir_screen[...,2] *= -1
    ray_dir_normal_dot = torch.sum(normal * ray_dir_screen, dim=-1, keepdim=True)
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    r[...,2] -= 1.0
    m = 2.0 * torch.sqrt(torch.sum(r**2, dim=-1, keepdim=True))
    vN = (r[...,:2] / m) + 0.5
    vN = 1.0 - vN
    vN = vN[...,:2].reshape(-1, 2)
    vN = torch.clip(vN, 0.0, 1.0)
    vN[torch.isnan(vN)] = 0
    return vN

def spherical_envmap_numpy(ray_dir, normal):
    """Computes matcap UV-coordinates from the ray direction and normal.
    
    Args:
        ray_dir (torch.Tensor): incoming ray direction of shape [...., 3]
        normal (torch.Tensor): surface normal of shape [..., 3]

    Returns:
        (torch.FloatTensor): UV coordinates of shape [..., 2]
    """
    ray_dir_screen = ray_dir * np.array([1,1,-1])
    # Calculate reflection
    ray_dir_normal_dot = np.sum(normal * ray_dir_screen, axis=-1)[...,np.newaxis]
    r = ray_dir_screen - 2.0 * ray_dir_normal_dot * normal
    m = 2.0 * np.sqrt(r[...,0]**2 + r[...,1] **2 + (r[...,2]-1)**2)
    vN = (r[...,:2] / m[...,np.newaxis]) + 0.5
    vN = 1.0 - vN
    vN = vN[...,:2].reshape(-1, 2)
    vN = np.clip(vN, 0, 1)
    vN[np.isnan(vN)] = 0
    return vN
