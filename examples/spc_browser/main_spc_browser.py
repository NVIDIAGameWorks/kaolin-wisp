# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

# Set logger display format
import logging as log
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s',
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)


if __name__ == "__main__":
    """
    An example application for browsing and visualizing Structured Point Cloud (SPC) models from kaolin.
    Assumes SPCs are stored as *.npz files, with an "octree" entry and some feature ("color" or "normal").
    
    Usage: python examples/spc_browser/main_spc_browser.py --dataset-dir <PATH_TO_SPC_FOLDER>
    """

    from app.cuda_guard import setup_cuda_context
    setup_cuda_context()  # Must be called before any torch operations take place

    import torch
    from wisp.config_parser import parse_options, argparse_to_str
    from wisp.framework.state import WispState
    from browse_spc_app import BrowseSPCApp

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')

    # Add custom args if needed for app
    app_group.add_argument('--dataset-dir', type=str, help='Folder with SPC files for visualizing.')
    args, args_str = argparse_to_str(parser)

    # Save dir path in extent field, to be queried by the app
    scene_state = WispState()
    scene_state.extent['dataset_path'] = args.dataset_dir

    scene_state.renderer.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # In this example, the SPC Pipeline objects are created within the renderer's widget paint() logic.
    # See widget_spc_selector.py
    renderer = BrowseSPCApp(wisp_state=scene_state, window_name="SPC Browser")
    renderer.run()
