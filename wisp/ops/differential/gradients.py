# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


# Differential operators

def autodiff_gradient(x, f):
    """Compute gradient using the PyTorch autodiff.

    Args:
        x (torch.FloatTensor): Coordinate tensor
        f (nn.Module): The function to perform autodiff on.
    """
    with torch.enable_grad():
        x = x.requires_grad_(True)
        y = f(x)
        grad = torch.autograd.grad(y, x,
                                   grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return grad


def finitediff_gradient(x, f, eps=0.005):
    """Compute 3D gradient using finite difference.

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    eps_x = torch.tensor([eps, 0.0, 0.0], device=x.device)
    eps_y = torch.tensor([0.0, eps, 0.0], device=x.device)
    eps_z = torch.tensor([0.0, 0.0, eps], device=x.device)

    grad = torch.cat([f(x + eps_x) - f(x - eps_x),
                      f(x + eps_y) - f(x - eps_y),
                      f(x + eps_z) - f(x - eps_z)], dim=-1)
    grad = grad / (eps * 2.0)

    return grad


def tetrahedron_gradient(x, f, eps=0.005):
    """Compute 3D gradient using finite difference (using tetrahedron method).

    Args:
        x (torch.FloatTensor): Coordinate tensor of shape [..., 3]
        f (nn.Module): The function to perform autodiff on.
    """
    h = eps
    k0 = torch.tensor([1.0, -1.0, -1.0], device=x.device, requires_grad=False)
    k1 = torch.tensor([-1.0, -1.0, 1.0], device=x.device, requires_grad=False)
    k2 = torch.tensor([-1.0, 1.0, -1.0], device=x.device, requires_grad=False)
    k3 = torch.tensor([1.0, 1.0, 1.0], device=x.device, requires_grad=False)
    h0 = torch.tensor([h, -h, -h], device=x.device, requires_grad=False)
    h1 = torch.tensor([-h, -h, h], device=x.device, requires_grad=False)
    h2 = torch.tensor([-h, h, -h], device=x.device, requires_grad=False)
    h3 = torch.tensor([h, h, h], device=x.device, requires_grad=False)
    h0 = x + h0
    h1 = x + h1
    h2 = x + h2
    h3 = x + h3
    h0 = h0.detach()
    h1 = h1.detach()
    h2 = h2.detach()
    h3 = h3.detach()
    h0 = k0 * f(h0)
    h1 = k1 * f(h1)
    h2 = k2 * f(h2)
    h3 = k3 * f(h3)
    grad = (h0 + h1 + h2 + h3) / (h * 4.0)
    return grad
