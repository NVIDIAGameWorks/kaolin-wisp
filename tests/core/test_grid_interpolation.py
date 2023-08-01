# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import unittest
import pytest
import torch
from wisp.ops.grid import grid_interpolate

params = [(torch.float), (torch.half)]

@pytest.mark.parametrize("dtype", params)
def test_forward_backward(dtype):
    device = "cuda:0"
    N = 100000
    
    x_ = torch.rand([N, 3], device=device, dtype=torch.float)
    _x = 1.0 - x_

    fs = 10 * torch.rand([N, 8, 2], device=device, dtype=dtype)
    fs = fs.requires_grad_(True)

    coeffs_analytic = torch.cat([
        _x[...,0:1] * _x[...,1:2] * _x[...,2:3],
        _x[...,0:1] * _x[...,1:2] * x_[...,2:3],
        _x[...,0:1] * x_[...,1:2] * _x[...,2:3],
        _x[...,0:1] * x_[...,1:2] * x_[...,2:3],
        x_[...,0:1] * _x[...,1:2] * _x[...,2:3],
        x_[...,0:1] * _x[...,1:2] * x_[...,2:3],
        x_[...,0:1] * x_[...,1:2] * _x[...,2:3],
        x_[...,0:1] * x_[...,1:2] * x_[...,2:3]], dim=-1)[..., None].detach()
    
    interpolated_feat0 = (coeffs_analytic * fs.float()).sum(-2).to(dtype)
    loss0 = interpolated_feat0.sum()
    loss0.backward()
    grad0 = fs.grad.clone()

    if fs.grad is not None:
        fs.grad.detach_()
        fs.grad.zero_()

    interpolated_feat1 = grid_interpolate(x_, fs)
    loss1 = interpolated_feat1.sum()
    loss1.backward()
    grad1 = fs.grad.clone()
    
    if dtype == torch.half:
        atol = 1e-2
        rtol = 1e-2
    else:
        atol = 1e-6
        rtol = 1e-4

    assert torch.allclose(loss0, loss1, atol=atol, rtol=rtol)
    assert torch.allclose(interpolated_feat0, interpolated_feat1, atol=atol, rtol=rtol)
    assert torch.allclose(grad0, grad1, atol=atol, rtol=rtol)

