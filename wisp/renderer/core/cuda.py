# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from contextlib import contextmanager


if not os.environ.get('ENABLE_PYCUDA') == '1':
    from cuda import cuda
    import torch

    @contextmanager
    def cuda_map_resource(img):
        """Context manager simplifying use of cuda.cuGraphicsMapResources / cuGraphicsSubResourceGetMappedArray.
        Boilerplate code based in part on pytorch-glumpy.
        """
        # args: (count, resource, stream)
        mapping_result = cuda.cuGraphicsMapResources(1, img, torch.cuda.default_stream().cuda_stream)
        if mapping_result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to map GL graphics resource to be accessed by CUDA.")

        # args (resource, arrayIndex, mipLevel)
        mapping_array = cuda.cuGraphicsSubResourceGetMappedArray(img, 0, 0)
        if mapping_array[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to get mapped array from GL resource for CUDA copy.")
        yield mapping_array[1]

        unmapping_result = cuda.cuGraphicsUnmapResources(1, img, torch.cuda.default_stream().cuda_stream)
        if unmapping_result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to unmap GL graphics resource from being accessed by CUDA.")


    def cuda_2d_memcpy(resource_handle, shared_tex, img, height):
        cpy = cuda.CUDA_MEMCPY2D()
        with cuda_map_resource(resource_handle) as ary:
            cpy.srcDevice = img.data_ptr()
            cpy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
            cpy.dstArray = ary
            cpy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_ARRAY
            cpy.WidthInBytes = cpy.srcPitch = cpy.dstPitch = shared_tex.nbytes // height
            cpy.Height = height
            cpy_result = cuda.cuMemcpy2DUnaligned(cpy)
            if cpy_result[0] != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Failed to memcopy cuda buffer to GL.")


    def cuda_register_gl_image(image, target):
        # Create shared GL / CUDA resource
        map_flags = cuda.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD
        register_result = cuda.cuGraphicsGLRegisterImage(image=image, target=target, Flags=map_flags)
        if register_result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Failed to register GL texture as a CUDA shared resource.')
        resource_handle = register_result[1]
        return resource_handle

    def cuda_unregister_resource(handle):
        unregister_result = cuda.cuGraphicsUnregisterResource(handle)
        if unregister_result[0] != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Failed to unregister CUDA resource.')

else:
    import pycuda
    import pycuda.gl as pycuda_gl

    window = None
    try:
        # !!! Should be called when interactive wisp loads, before any torch ops take place !!!
        # The following is a hacky workaround due to a cublas error on interfering streams:
        # pycuda.gl fails to initialize after torch performs batched matrix multiplication
        # (the bugs causes any following torch.dot invocations to fail)..
        # The solution is to initialize pycuda.gl early when wisp loads.
        # To load pycuda.gl, we require a GL context of some window,
        # so here we force glumpy-glfw to create an invisible window, which generates an opengl context.
        # Then immediately import pycuda.gl.autoint to let it initialize properly
        from glumpy import app

        # Tell glumpy to use glfw backend
        app.use("glfw_imgui")
        # Let glumpy use glfw to create an invisible window
        window = app.Window(width=10, height=10, title='dummy', visible=False)

        # pycuda initializes the default context with "cuGLCtxCreate", but this call will fail if a GL context
        # is not currently set. Therefore import is invoked only after glfw obtains a GL context.
        # See: https://documen.tician.de/pycuda/gl.html#module-pycuda.gl.autoinit
        import pycuda.gl.autoinit
        # Next tell torch to initialize the primary cuda context
        import torch
        torch.cuda.init()
        # pycuda should not create a new context, but retain the torch one
        import pycuda.driver as cuda
        pycuda_context = cuda.Device(0).retain_primary_context()

    except (ModuleNotFoundError, ImportError):
        pass  # Don't fail if interactive mode is disabled (e.g: glumpy or pycuda are unavailable)
    finally:
        if window is not None:
            window.close()

    @contextmanager
    def cuda_map_resource(img):
        """Context manager simplifying use of pycuda_gl.RegisteredImage.
        Boilerplate code based in part on pytorch-glumpy.
        """
        mapping = img.map()
        yield mapping.array(0, 0)
        mapping.unmap()

    def cuda_2d_memcpy(resource_handle, shared_tex, img, height):
        cpy = pycuda.driver.Memcpy2D()
        with cuda_map_resource(resource_handle) as ary:
            cpy.set_src_device(img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = shared_tex.nbytes // height
            cpy.height = height
            cpy(aligned=False)

    def cuda_register_gl_image(image, target):
        # Create shared GL / CUDA resource
        map_flags = pycuda_gl.graphics_map_flags.WRITE_DISCARD
        resource_handle = pycuda_gl.RegisteredImage(image, target, map_flags)
        return resource_handle

    def cuda_unregister_resource(handle):
        # Nothing to be done - when ref count reaches zero on proxy object in python, unregister should
        # be called automatically
        pass
