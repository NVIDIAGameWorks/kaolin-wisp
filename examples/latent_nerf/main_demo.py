# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


def parse_args():
    from wisp.config_parser import parse_options, argparse_to_str
    parser = parse_options(return_parser=True)
    app_utils.add_log_level_flag(parser)
    args, args_str = argparse_to_str(parser)
    return args, args_str


def create_trainer(args, scene_state):
    """ Create the trainer according to config args """
    from wisp.config_parser import get_modules_from_config, get_optimizer_from_config
    pipeline, train_dataset, device = get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                           optim_cls, args.lr, args.weight_decay,
                                           args.grid_lr_weight, optim_params, args.log_dir, device,
                                           exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                           render_every=args.render_every, save_every=args.save_every,
                                           scene_state=scene_state)
    return trainer


def create_app(scene_state, trainer):
    """ Create the interactive app running the renderer & trainer """
    from demo_app import DemoApp
    scene_state.renderer.device = trainer.device  # Use same device for trainer and renderer
    interactive_app = DemoApp(wisp_state=scene_state, background_task=trainer.iterate, window_name="SIGGRAPH 2022 Demo")
    return interactive_app


if __name__ == "__main__":
    # Must be called before any torch operations take place
    from app.cuda_guard import setup_cuda_context

    setup_cuda_context()

    import os
    import app.app_utils as app_utils
    import logging as log
    from wisp.framework import WispState
    from wisp.trainers import *

    # Register any newly added user classes before running the config parser
    # Registration ensures the config parser knows about these classes and is able to dynamically create them.
    from wisp.config_parser import register_class
    from funny_neural_field import FunnyNeuralField

    register_class(FunnyNeuralField, 'FunnyNeuralField')

    # Parse config yaml and cli args
    args, args_str = parse_args()
    app_utils.default_log_setup(args.log_level)

    # Create the state object, shared by all wisp components
    scene_state = WispState()

    # Create the trainer
    trainer = create_trainer(args, scene_state)

    if not os.environ.get('WISP_HEADLESS') == '1':
        interactive_app = create_app(scene_state, trainer)
        interactive_app.run()
    else:
        log.info("Running headless. For the app, set WISP_HEADLESS=0")
        if args.valid_only:
            trainer.validate()
        else:
            trainer.train()
