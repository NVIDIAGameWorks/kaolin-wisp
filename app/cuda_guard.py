# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def setup_cuda_context():
    """Carefully load CUDA based frameworks to avoid interference.
       Interactive apps should invoke this function as early as possible.
    """
    import os
    import sys
    if not os.environ.get('WISP_HEADLESS') == '1':
        # glump.app.Window is using argparse
        # We remove sys.argv temporarily to avoid conflict with Wisp's argparse
        argv = sys.argv
        sys.argv = [argv[0]]
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
            # restore sys.argv for Wisp argparse
            sys.argv = argv
