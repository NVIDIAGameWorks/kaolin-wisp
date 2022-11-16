# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


if __name__ == "__main__":
    import app_utils

    from wisp.trainers import *
    from wisp.config_parser import parse_options, argparse_to_str, get_modules_from_config, \
        get_optimizer_from_config
    from wisp.framework import WispState
    
    import wandb

    # Usual boilerplate
    parser = parse_options(return_parser=True)
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases Project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases Entity")
    app_utils.add_log_level_flag(parser)
    app_group = parser.add_argument_group('app')
    # Add custom args if needed for app
    args, args_str = argparse_to_str(parser)
    using_wandb = args.wandb_project is not None and args.wandb_entity is not None
    if using_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            entity=args.wandb_entity,
            job_type="validate" if args.valid_only else "train",
            config=vars(args),
            sync_tensorboard=True
        )
    app_utils.default_log_setup(args.log_level)
    pipeline, train_dataset, device = get_modules_from_config(args)
    optim_cls, optim_params = get_optimizer_from_config(args)
    trainer = globals()[args.trainer_type](pipeline, train_dataset, args.epochs, args.batch_size,
                                      optim_cls, args.lr, args.weight_decay,
                                      args.grid_lr_weight, optim_params, args.log_dir, device,
                                      exp_name=args.exp_name, info=args_str, extra_args=vars(args),
                                      render_every=args.render_every, save_every=args.save_every, using_wandb=using_wandb)
    if args.valid_only:
        trainer.validate()
    else:
        trainer.train()
        trainer.validate()
    
    if using_wandb:
        wandb.finish()
