# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
from wisp.renderer.app.interactive_renderer import InteractiveRenderer

if __name__ == "__main__":
    scene_config = {
        'scene': {
            "title": "kaolin-wisp interactive renderer",
            "channel_depth": 4,
            "target_fps": 30,
            "device": 'cuda'
        },
        'cameras': [
            {
                'id': 0,
                'extrinsics': {
                    'cam_at': np.array([-0.05198334725487073, -0.016510273653732227, 0.0716196402346452]),
                    'cam_eye': np.array([-5.5917379969124275, -5.5100606335385356, -5.17243748276105064]),
                    'cam_up': np.array([0.0, 0.0, 1.0]),
                },
                'intrinsics': {
                    'type': "Pinhole",
                    'width': 1600, # 1600,
                    'height': 1600, # 1600,
                    'x0': 0.0,
                    'y0': 0.0,
                    'focal_x': 1931.37133, # 482.8428, # 1931.37133,
                    'focal_y': 1931.37133, # 482.8428, # 1931.37133,
                }
            }
        ],
        'renderers': [
            {
                'id': "LegoV8",
                'type': "PackedRFRenderer",
                'args': {
                    'batch_size': 2 ** 15,
                    'model_path': "<your_model_path>/model.pth",
                }
            }
        ]
    }

    renderer = InteractiveRenderer(width=scene_config['cameras'][0]['intrinsics']['width'],
                                   height=scene_config['cameras'][0]['intrinsics']['height'],
                                   channel_depth=scene_config['scene']['channel_depth'],
                                   window_name=scene_config['scene']['title'],
                                   app_config=scene_config)
    renderer.run()
