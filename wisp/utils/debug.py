# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import pdb
import polyscope as ps
import wisp.ops.mesh as mesh_ops

""" This module contains utility function and classes for debugging purposes. """


class PsDebugger:
    """This is a small convenience class to use Polyscope.
    
    You can basically inject Polyscope visualization for any function using something like:

    psd = PsDebugger()
    psd.register_point_cloud("test", point_tensor)
    psd.add_vector_quantity("test", "test_vectors", vector_tensor)
    psd.show()

    These functions mostly just follow polyscope, so for more details check out: https://polyscope.run
    """
    def __init__(self):
        ps.init()
        self.pcls = {}

    def register_curve_network(self, name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[name] = ps.register_curve_network(name, tensor.reshape(-1, 3).numpy(), 'line', **kwargs)

    def register_point_cloud(self, name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[name] = ps.register_point_cloud(name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_vector_quantity(self, pcl_name, vec_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_vector_quantity(vec_name, tensor.reshape(-1, 3).numpy(), **kwargs)
    
    def add_scalar_quantity(self, pcl_name, s_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_scalar_quantity(s_name, tensor.reshape(-1).numpy(), **kwargs)
    
    def add_color_quantity(self, pcl_name, c_name, tensor, **kwargs):
        if 'cpu' not in str(tensor.device):
            tensor = tensor.cpu().detach()
        self.pcls[pcl_name].add_color_quantity(c_name, tensor.reshape(-1, 3).numpy(), **kwargs)

    def add_surface_mesh(self, name, obj_path, **kwargs):
        verts, faces = mesh_ops.load_obj(obj_path)
        ps.register_surface_mesh(name, verts.numpy(), faces.numpy(), **kwargs)
    
    def show(self):
        ps.show()
        pdb.set_trace()

