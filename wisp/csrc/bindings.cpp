/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <torch/extension.h>
#include "./render/find_depth_bound.h"
#include "./external/mesh_to_sdf.h"
#include "./ops/hashgrid_interpolate.h"

namespace wisp {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module render = m.def_submodule("render");
    render.def("find_depth_bound_cuda", &find_depth_bound_cuda);
    py::module external = m.def_submodule("external");
    external.def("mesh_to_sdf_cuda", &mesh_to_sdf_cuda);
    py::module ops = m.def_submodule("ops");
    ops.def("hashgrid_interpolate_cuda", &hashgrid_interpolate_cuda);
    ops.def("hashgrid_interpolate_backward_cuda", &hashgrid_interpolate_backward_cuda);
}

}

