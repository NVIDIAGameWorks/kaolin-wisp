# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import time
import numpy as np
import torch
import torch.nn.functional as F
from wisp.core import RenderBuffer, Rays
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.shaders import matcap_shader, pointlight_shadow_shader
from wisp.ops.differential import finitediff_gradient
from wisp.ops.geometric import normalized_grid, normalized_slice
from wisp.tracers import *


# --  This module will be deprecated, public usage of this functionality is discouraged -- """


def _look_at(f, t, height, width, mode='persp', fov=90.0, device='cuda'):
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
    camera_right = F.normalize(torch.cross(camera_view, torch.FloatTensor([0, 1, 0]).to(device)), dim=0)
    camera_up = F.normalize(torch.cross(camera_right, camera_view), dim=0)

    return _generate_rays(camera_origin, camera_view, camera_right, camera_up,
                          height, width, mode=mode, fov=fov, device=device)


def _generate_rays(camera_origin, camera_view, camera_right, camera_up, height, width,
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

    ray_origin = camera_right * coord[..., 0, np.newaxis] * np.tan(np.radians(fov / 2)) + \
                 camera_up * coord[..., 1, np.newaxis] * np.tan(np.radians(fov / 2)) + \
                 camera_origin + camera_view
    ray_origin = ray_origin.reshape(-1, 3)
    ray_offset = camera_view.unsqueeze(0).repeat(ray_origin.shape[0], 1)

    if mode == 'ortho':  # Orthographic camera
        ray_dir = F.normalize(ray_offset, dim=-1)
    elif mode == 'persp':  # Perspective camera
        ray_dir = F.normalize(ray_origin - camera_origin, dim=-1)
        ray_origin = camera_origin.repeat(ray_dir.shape[0], 1)
    else:
        raise ValueError('Invalid camera mode!')

    return ray_origin, ray_dir


class OfflineRenderer():
    """Renderer class to do simple offline, non interactive rendering.

    TODO(ttakikawa): This class will be deprecated soon when the main renderer class supports offline renders.
    """
    def __init__(self, 
        render_res   : list = [1024, 720], # [w, h]
        camera_proj  : str  = 'persp', # one of ['persp', 'ortho']
        render_batch : int  = -1, # -1 for no batching
        shading_mode : str  = 'rb', # one of ['matcap', 'rb']
        matcap_path  : str  = './data/matcap/Pearl.png', # set if shadming mode = matcap
        shadow       : bool = False, 
        ao           : bool = False, 
        perf         : bool = False,
        device              = 'cuda',
        **kwargs
    ):
        self.render_res = render_res
        self.render_batch = render_batch
        self.shading_mode = shading_mode
        self.matcap_path = matcap_path
        self.shadow = shadow
        self.ao = ao
        self.perf = perf
        self.device = device
        
        self.width, self.height = self.render_res

        self.kwargs = kwargs

    def render_lookat(self, 
            pipeline, 
            f            = [0,0,1], 
            t            = [0,0,0], 
            fov          = 30.0, 
            camera_proj  = 'persp',  
            device       = 'cuda', 
            mm           = None,
            lod_idx      = None,
            camera_clamp = [0, 5]
        ):
        """Render images from a lookat.

        Args:
            pipeline (wisp.core.Pipeline): the pipeline to render
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fov (float): field of view
            aa (int): Number of samples per pixel to render
            mm (torch.FloatTensor): model transformation matrix
            device: Device to initialize rays on.
            lod_idx (int): LOD to renderer
            camera_clamp (tuple of int): The near and far clipping planes.

        Returns: 
            (wisp.core.RenderBuffer): The rendered image buffers.
        """
        # Generate the ray origins and directions, from camera parameters
        ray_o, ray_d = _look_at(f, t, self.height, self.width, fov=fov, mode=camera_proj, device=device)
        # Rotate the camera into model space
        if mm is not None:
            mm = mm.to('cuda')
            ray_o = torch.mm(ray_o, mm)
            ray_d = torch.mm(ray_d, mm)
        
        rays = Rays(origins=ray_o, dirs=ray_d, dist_min=camera_clamp[0], dist_max=camera_clamp[1])

        rb = self.render(pipeline, rays, lod_idx=lod_idx)
        rb = rb.reshape(self.height, self.width, -1) 
        return rb


    def render(self, pipeline, rays, lod_idx=None):
        """Render images from a lookat.

        Args:
            pipeline (wisp.core.Pipeline): the pipeline to render
            rays (wisp.core.Rays): the rays to render
            lod_idx (int): LOD to renderer

        Returns: 
            (wisp.core.RenderBuffer): The renderer image.
        """
        # Differentiable Renderer
        timer = PerfTimer(activate=self.perf)
        if self.perf:
            _time = time.time()

        with torch.no_grad():
            if self.render_batch > 0:
                rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, ao=None, dirs=None)
                for ray_pack in rays.split(self.render_batch):
                    rb  += pipeline.tracer(pipeline.nef, rays=ray_pack, lod_idx=lod_idx, **self.kwargs)
            else:
                rb = pipeline.tracer(pipeline.nef, rays=rays, lod_idx=lod_idx, **self.kwargs)

        ######################
        # Shading Rendering
        ######################
        
        # This is executed if the tracer does not handle RGB
        # TODO(ttakikawa): Brittle
        if self.shading_mode == 'rb' and rb.rgb is None: 
            rb.rgb = pipeline.nef.rgb(rb.xyz, rays.dirs)

        if self.perf:
            print("Time Elapsed:{:.4f}".format(time.time() - _time))
        
        # Shade the image
        if self.shading_mode == 'matcap':
            # TODO(ttakikawa): Should use the mm
            rb = matcap_shader(rb, rays, self.matcap_path, mm=None)
        elif self.shading_mode == 'rb':
            assert rb.rgb is not None and "No rgb in buffer; change shading-mode"
            pass
        elif self.shading_mode == 'normal':    
            rb.rgb = (rb.normal + 1.0) / 2.0
        else:
            raise NotImplementedError
        
        # Use segmentation
        if rb.normal is not None:
            rb.normal[~rb.hit] = 1.0
        if rb.rgb is not None:
            rb.rgb[~rb.hit] = 1.0

        # Add secondary effects
        if self.shadow:
            rb = pointlight_shadow_shader(rb, rays, pipeline)
        
        if self.ao:
            # TODO(ttakikawa): Mystery function... how did I write this?
            acc = torch.zeros_like(rb.depth).to(rays.origins.device)
            r = torch.zeros_like(rb.depth).to(rays.origins.device)
            with torch.no_grad():
                weight = 3.5
                for i in range(40):
                    
                    # Visual constants
                    ao_width = 0.1
                    _d = ao_width * 0.25 * (float(i+1) / float(40+1)) ** 1.6
                    q = rb.xyz + rb.normal * _d

                    # AO for surface
                    with torch.no_grad():
                        r[rb.hit] = pipeline.nef(q[rb.hit])

                    if self.shadow:
                        net_d = pipeline.nef(q[plane_hit])
                        plane_d = torch.zeros_like(net_d) + _d
                        r[plane_hit] = torch.min(torch.cat([net_d, plane_d], dim=-1), dim=-1, keepdim=True)[0]
                        acc[plane_hit] += 3.5 * F.relu(_d - r[plane_hit] - 0.0015)
                    acc[rb.hit] += 3.5 * F.relu(_d - r[rb.hit] - 0.0015)
                    weight *= 0.84
        
            rb.ao = torch.clamp(1.0 - acc,  0.1, 1.0)
            rb.ao = rb.ao * rb.ao

        if self.ao:
            rb.rgb[...,:3] *= rb.ao        
        
        return rb
    
    # TODO(ttakikawa): These are useful functions but probably does not need to live in the renderer. Migrate.
    def normal_slice(self, fn, dim=0, depth=0.0):
        """Get a slice of the normals.
        """
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device).reshape(-1,3)
        normal = (F.normalize(finitediff_gradient(pts, fn).detach()) + 1.0) / 2.0
        normal = normal.reshape(self.width, self.height, 3).cpu().numpy()
        return normal

    def sdf_slice(self, fn, dim=0, depth=0):
        """Get a slice of SDFs.
        """
        pts = normalized_slice(self.width, self.height, dim=dim, depth=depth, device=self.device)

        d = torch.zeros(self.width * self.height, 1, device=pts.device)
        with torch.no_grad():
            d = fn(pts.reshape(-1,3))

        d = d.reshape(self.width, self.height, 1)

        d = d.squeeze().cpu().numpy()
        dpred = d
        d = np.clip((d + 1.0) / 2.0, 0.0, 1.0)
        blue = np.clip((d - 0.5)*2.0, 0.0, 1.0)
        yellow = 1.0 - blue
        vis = np.zeros([*d.shape, 3])
        vis[...,2] = blue
        vis += yellow[...,np.newaxis] * np.array([0.4, 0.3, 0.0])
        vis += 0.2
        vis[d - 0.5 < 0] = np.array([1.0, 0.38, 0.0])
        for i in range(50):
            vis[np.abs(d - 0.02*i) < 0.0015] = 0.8
        vis[np.abs(d - 0.5) < 0.004] = 0.0
        return vis
    
    def shade_images(self, pipeline, f=[0,0,1], t=[0,0,0], fov=30.0, aa=1, mm=None, 
                     lod_idx=None, camera_clamp=[0,10]):
        """
        Invokes the renderer and outputs images.

        Args:
            pipeline (wisp.core.Pipeline): the pipeline to render
            f (list[f,f,f]): camera from
            t (list[f,f,f]): camera to
            fov (float): field of view
            aa (int): Number of samples per pixel to render
            mm (torch.FloatTensor): model transformation matrix
            lod_idx (int): LOD to renderer
            camera_clamp (tuple of int): The near and far clipping planes.
        
        Returns: 
            (wisp.core.RenderBuffer): The rendered image buffers.
        """
        if mm is None:
            mm = torch.eye(3)
        
        if aa > 1:
            rblst = [] 
            for _ in range(aa):
                rblst.append(self.render_lookat(pipeline, f=f, t=t, fov=fov, mm=mm, lod_idx=lod_idx,
                    camera_clamp=camera_clamp))
            rb = RenderBuffer.mean(*rblst)
        else:
            rb = self.render_lookat(pipeline, f=f, t=t, fov=fov, mm=mm, lod_idx=lod_idx, 
                    camera_clamp=camera_clamp)
        rb = rb.cpu().transpose()
        return rb

