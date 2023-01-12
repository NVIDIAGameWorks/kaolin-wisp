# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.  #
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import time
import logging as log
from datetime import datetime
from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from wisp.offline_renderer import OfflineRenderer
from wisp.framework import WispState, BottomLevelRendererState
from wisp.datasets import default_collate
from wisp.renderer.core.api import add_to_scene_graph

import wandb
import numpy as np


def log_metric_to_wandb(key, _object, step):
    wandb.log({key: _object}, step=step, commit=False)


def log_images_to_wandb(key, image, step):
    wandb.log({key: wandb.Image(np.moveaxis(image, 0, -1))}, step=step, commit=False)


class BaseTrainer(ABC):
    """
    Base class for the trainer.

    The default overall flow of things:

    init()
    |- set_renderer()
    |- set_logger()

    train():
        pre_training()
        (i) for every epoch:
            |- pre_epoch()

            (ii) for every iteration:
                |- pre_step()
                |- step()
                |- post_step()

            post_epoch()
            |- log_tb()
            |- save_model()
            |- render_tb()
            |- resample_dataset()

            |- validate()
        post_training()

    iterate() runs a single iteration step of train() through all internal lifecycle methods,
    meaning a single run over loop (ii), and loop (i) if loop (ii) is finished.
    This is useful for cases like gui apps which run without a training loop.

    Each of these events can be overridden, or extended with super().

    """

    #######################
    # Initialization
    #######################

    def __init__(self, pipeline, dataset, num_epochs, batch_size,
                 optim_cls, lr, weight_decay, grid_lr_weight, optim_params, log_dir, device,
                 exp_name=None, info=None, scene_state=None, extra_args=None,
                 render_tb_every=-1, save_every=-1, trainer_mode='validate', using_wandb=False):
        """Constructor.
        
        Args:
            pipeline (wisp.core.Pipeline): The pipeline with tracer and neural field to train.
            dataset (torch.Dataset): The dataset to use for training.
            num_epochs (int): The number of epochs to run the training for.
            batch_size (int): The batch size used in training.
            optim_cls (torch.optim): The Optimizer object to use
            lr (float): The learning rate to use
            weight_decay (float): The weight decay to use
            optim_params (dict): Optional params for the optimizer.
            device (device): The device to run the training on. 
            log_dir (str): The directory to save the training logs in.
            exp_name (str): The experiment name to use for logging purposes.
            info (str): The args to save to the logger.
            scene_state (wisp.core.State): Use this to inject a scene state from the outside to be synced
                                           elsewhere.
            extra_args (dict): Optional dict of extra_args for easy prototyping.
            render_tb_every (int): The number of epochs between renders for tensorboard logging. -1 = no rendering.
            save_every (int): The number of epochs between model saves. -1 = no saving.
            trainer_mode (str): 'train' or 'validate' for choosing running training or validation only modes.
                Currently used only for titles within logs.
            using_wandb (bool): When True, weights & biases will be used for logging.
        """
        log.info(f'Info: \n{info}')
        log.info(f'Training on {extra_args["dataset_path"]}')

        # initialize scene_state
        if scene_state is None:
            scene_state = WispState()
        self.scene_state = scene_state

        self.extra_args = extra_args
        self.info = info
        self.trainer_mode = trainer_mode

        self.pipeline = pipeline
        log.info("Total number of parameters: {}".format(
            sum(p.numel() for p in self.pipeline.nef.parameters()))
        )
        # Set device to use
        self.device = device
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.init_renderer()

        self.dataset = dataset

        # Optimizer params
        self.optim_cls = optim_cls
        self.lr = lr
        self.weight_decay = weight_decay
        self.grid_lr_weight = grid_lr_weight
        self.optim_params = optim_params
        self.init_optimizer()

        # Training params
        self.epoch = 1
        self.iteration = 1
        self.max_epochs = num_epochs
        self.batch_size = batch_size
        self.exp_name = exp_name if exp_name else "unnamed_experiment"

        # Add object to scene graph: if interactive mode is on, this will make sure the visualizer can display it.
        # batch_size is an optional setup arg here which hints the visualizer how many rays can be processed at once
        # (e.g. this is the pipeline's batch_size used for inference time)
        add_to_scene_graph(state=self.scene_state, name=self.exp_name, obj=self.pipeline, batch_size=2**14)
        # Update optimization state about the current train set used
        self.scene_state.optimization.train_data.append(dataset)

        if hasattr(self.dataset, "data"):
            self.scene_state.graph.cameras = self.dataset.data.get("cameras", dict())

        self.scaler = torch.cuda.amp.GradScaler()

        # In-training variables
        self.train_data_loader_iter = None
        self.val_data_loader = None
        self.dataset_size = None
        self.log_dict = {}
        self.init_dataloader()

        self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(
            log_dir,
            self.exp_name,
            self.log_fname
        )

        self.render_tb_every = render_tb_every
        self.save_every = save_every
        self.using_wandb = using_wandb

    def init_dataloader(self):
        self.train_data_loader = DataLoader(self.dataset,
                                            batch_size=self.batch_size,
                                            collate_fn=default_collate,
                                            shuffle=True, pin_memory=True,
                                            num_workers=self.extra_args['dataloader_num_workers'])
        self.iterations_per_epoch = len(self.train_data_loader)

    def init_optimizer(self):
        """Default initialization for the optimizer.
        """

        params_dict = { name : param for name, param in self.pipeline.nef.named_parameters()}
        
        params = []
        decoder_params = []
        grid_params = []
        rest_params = []

        for name in params_dict:
            
            if 'decoder' in name:
                # If "decoder" is in the name, there's a good chance it is in fact a decoder,
                # so use weight_decay
                decoder_params.append(params_dict[name])

            elif 'grid' in name:
                # If "grid" is in the name, there's a good chance it is in fact a grid,
                # so use grid_lr_weight
                grid_params.append(params_dict[name])

            else:
                rest_params.append(params_dict[name])

        params.append({"params" : decoder_params,
                       "lr": self.lr, 
                       "weight_decay": self.weight_decay})

        params.append({"params" : grid_params,
                       "lr": self.lr * self.grid_lr_weight})
        
        params.append({"params" : rest_params,
                       "lr": self.lr})

        self.optimizer = self.optim_cls(params, **self.optim_params)

    def init_renderer(self):
        """Default initalization for the renderer.
        """
        self.renderer = OfflineRenderer(**self.extra_args)

    #######################
    # Data load
    #######################

    def reset_data_iterator(self):
        """Rewind the iterator for the new epoch.
        """
        self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
        self.train_data_loader_iter = iter(self.train_data_loader)

    def next_batch(self):
        """Actually iterate the data loader.
        """
        return next(self.train_data_loader_iter)

    def resample_dataset(self):
        """
        Override this function if some custom logic is needed.

        Args:
            (torch.utils.data.Dataset): Training dataset.
        """
        if hasattr(self.dataset, 'resample'):
            log.info("Reset DataLoader")
            self.dataset.resample()
            self.init_dataloader()
        else:
            raise ValueError("resample=True but the dataset doesn't have a resample method")

    #######################
    # Training Life-cycle
    #######################

    def is_first_iteration(self):
        return self.total_iterations == 1

    def is_any_iterations_remaining(self):
        return self.total_iterations < self.max_iterations

    def begin_epoch(self):
        """Begin epoch.
        """
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """End epoch.
        """
        current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        self.epoch_start_time = current_time
        # TODO(ttakikawa): Don't always write to TB
        self.writer.add_scalar(f'time/elapsed_ms_per_epoch', elapsed_time * 1000, self.epoch)
        if self.using_wandb:
            log_metric_to_wandb(f'time/elapsed_ms_per_epoch', elapsed_time * 1000, self.epoch)

        self.post_epoch()

        if self.extra_args["valid_every"] > -1 and \
                self.epoch % self.extra_args["valid_every"] == 0 and \
                self.epoch != 0:
            self.validate()

        if self.epoch < self.max_epochs:
            self.iteration = 1
            self.epoch += 1
        else:
            self.is_optimization_running = False

    def grow(self):
        stage = min(self.extra_args["num_lods"],
                    (self.epoch // self.extra_args["grow_every"]) + 1)  # 1 indexed
        if self.extra_args["growth_strategy"] == 'onebyone':
            self.loss_lods = [stage - 1]
        elif self.extra_args["growth_strategy"] == 'increase':
            self.loss_lods = list(range(0, stage))
        elif self.extra_args["growth_strategy"] == 'shrink':
            self.loss_lods = list(range(0, self.extra_args["num_lods"]))[stage - 1:]
        elif self.extra_args["growth_strategy"] == 'finetocoarse':
            self.loss_lods = list(range(
                0, self.extra_args["num_lods"]
            ))[self.extra_args["num_lods"] - stage:]
        elif self.extra_args["growth_strategy"] == 'onlylast':
            self.loss_lods = list(range(0, self.extra_args["num_lods"]))[-1:]
        else:
            raise NotImplementedError

    def iterate(self):
        """Advances the training by one training step (batch).
        """
        if self.is_optimization_running:
            if self.is_first_iteration():
                self.pre_training()
            iter_start_time = time.time()
            try:
                if self.train_data_loader_iter is None:
                    self.begin_epoch()
                self.iteration += 1
                data = self.next_batch()
            except StopIteration:
                self.end_epoch()
                if self.is_any_iterations_remaining():
                    self.begin_epoch()
                    data = self.next_batch()
            if self.is_any_iterations_remaining():
                self.pre_step()
                self.step(data)
                self.post_step()
                iter_end_time = time.time()
            else:
                iter_end_time = time.time()
                self.post_training()
            self.scene_state.optimization.elapsed_time += iter_end_time - iter_start_time

    def save_model(self):
        """
        Override this function to change model saving.
        """

        if self.extra_args["save_as_new"]:
            model_fname = os.path.join(self.log_dir, f'model-ep{self.epoch}-it{self.iteration}.pth')
        else:
            model_fname = os.path.join(self.log_dir, f'model.pth')

        log.info(f'Saving model checkpoint to: {model_fname}')
        if self.extra_args["model_format"] == "full":
            torch.save(self.pipeline, model_fname)
        else:
            torch.save(self.pipeline.state_dict(), model_fname)

        if self.using_wandb:
            name = wandb.util.make_artifact_name_safe(f"{wandb.run.name}-model")
            model_artifact = wandb.Artifact(name, type="model")
            model_artifact.add_file(model_fname)
            wandb.run.log_artifact(model_artifact, aliases=["latest", f"ep{self.epoch}_it{self.iteration}"])

    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """
        self.is_optimization_running = True
        while self.is_optimization_running:
            self.iterate()

    #######################
    # Training Events
    #######################

    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        # Default TensorBoard Logging
        self.writer = SummaryWriter(self.log_dir, purge_step=0)
        self.writer.add_text('Info', self.info)
        
        if self.using_wandb:
            wandb_project = self.extra_args["wandb_project"]
            wandb_run_name = self.extra_args.get("wandb_run_name")
            wandb_entity = self.extra_args.get("wandb_entity")
            wandb.init(
                project=wandb_project,
                name=self.exp_name if wandb_run_name is None else wandb_run_name,
                entity=wandb_entity,
                job_type=self.trainer_mode,
                config=self.extra_args,
                sync_tensorboard=True
            )

    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """
        self.writer.close()
        if self.using_wandb:
            wandb.finish()

    def pre_epoch(self):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        # The DataLoader is refreshed before every epoch, because by default, the dataset refreshes
        # (resamples) after every epoch.

        self.loss_lods = list(range(0, self.extra_args["num_lods"]))
        if self.extra_args["grow_every"] > 0:
            self.grow()

        if self.extra_args["only_last"]:
            self.loss_lods = self.loss_lods[-1:]

        if self.extra_args["resample"] and self.epoch % self.extra_args["resample_every"] == 0:
            self.resample_dataset()

        self.pipeline.train()

    def post_epoch(self):
        """
        Override this function to change the post-epoch post processing.

        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the dataset.

        To keep default behaviour but also augment with other features, do

          super().post_epoch()

        in the derived method.
        """
        self.pipeline.eval()

        total_loss = self.log_dict['total_loss'] / len(self.train_data_loader)
        self.scene_state.optimization.losses['total_loss'].append(total_loss)

        self.log_cli()
        self.log_tb()

        # Render visualizations to tensorboard
        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0:
            self.render_tb()

       # Save model
        if self.save_every > -1 and self.epoch % self.save_every == 0 and self.epoch != 0:
            self.save_model()

    def pre_step(self):
        """
        Override this function to change the pre-step preprocessing (runs per iteration).
        """
        pass

    def post_step(self):
        """
        Override this function to change the pre-step preprocessing (runs per iteration).
        """
        pass

    @abstractmethod
    def step(self, data):
        """Advance the training by one step using the batched data supplied.

        data (dict): Dictionary of the input batch from the DataLoader.
        """
        pass

    @abstractmethod
    def validate(self):
        pass

    #######################
    # Logging
    #######################

    def init_log_dict(self):
        """
        Override this function to use custom logs.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0

    def log_model_details(self):
        # TODO (operel): Brittle
        log.info(f"Position Embed Dim: {self.pipeline.nef.pos_embed_dim}")
        log.info(f"View Embed Dim: {self.pipeline.nef.view_embed_dim}")

    def log_cli(self):
        """
        Override this function to change CLI logging.

        By default, this function only runs every epoch.
        """
        # Average over iterations
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))

    def log_tb(self):
        """
        Override this function to change loss / other numeric logging to TensorBoard / Wandb.
        """
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f'loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)

    def render_tb(self):
        """
        Override this function to change render logging to TensorBoard / Wandb.
        """
        self.pipeline.eval()
        for d in [self.extra_args["num_lods"] - 1]:
            out = self.renderer.shade_images(self.pipeline,
                                             f=self.extra_args["camera_origin"],
                                             t=self.extra_args["camera_lookat"],
                                             fov=self.extra_args["camera_fov"],
                                             lod_idx=d,
                                             camera_clamp=self.extra_args["camera_clamp"])

            # Premultiply the alphas since we're writing to PNG (technically they're already premultiplied)
            if self.extra_args["bg_color"] == 'black' and out.rgb.shape[-1] > 3:
                bg = torch.ones_like(out.rgb[..., :3])
                out.rgb[..., :3] += bg * (1.0 - out.rgb[..., 3:4])

            out = out.image().byte().numpy_dict()

            log_buffers = ['depth', 'hit', 'normal', 'rgb', 'alpha']

            for key in log_buffers:
                if out.get(key) is not None:
                    self.writer.add_image(f'{key}/{d}', out[key].T, self.epoch)
                    if self.using_wandb:
                        log_images_to_wandb(f'{key}/{d}', out[key].T, self.epoch)

    #######################
    # Properties
    #######################

    @property
    def is_optimization_running(self) -> bool:
        return self.scene_state.optimization.running

    @is_optimization_running.setter
    def is_optimization_running(self, is_running: bool):
        self.scene_state.optimization.running = is_running

    @property
    def epoch(self) -> int:
        """ Epoch counter, starts at 1 and ends at max epochs"""
        return self.scene_state.optimization.epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self.scene_state.optimization.epoch = epoch

    @property
    def iteration(self) -> int:
        """ Iteration counter, for current epoch. Starts at 1 and ends at iterations_per_epoch """
        return self.scene_state.optimization.iteration

    @iteration.setter
    def iteration(self, iteration: int):
        """ Iteration counter, for current epoch """
        self.scene_state.optimization.iteration = iteration

    @property
    def iterations_per_epoch(self) -> int:
        """ How many iterations should run per epoch """
        return self.scene_state.optimization.iterations_per_epoch

    @iterations_per_epoch.setter
    def iterations_per_epoch(self, iterations: int):
        """ How many iterations should run per epoch """
        self.scene_state.optimization.iterations_per_epoch = iterations

    @property
    def total_iterations(self) -> int:
        """ Total iteration steps the trainer took so far, for all epochs.
            Starts at 1 and ends at max_iterations
        """
        return (self.epoch - 1) * self.iterations_per_epoch + self.iteration - 1

    @property
    def max_epochs(self) -> int:
        """ Total number of epochs set for this optimization task.
        The first epoch starts at 1 and the last epoch ends at the returned `max_epochs` value.
        """
        return self.scene_state.optimization.max_epochs

    @max_epochs.setter
    def max_epochs(self, num_epochs):
        """ Total number of epochs set for this optimization task.
        The first epoch starts at 1 and the last epoch ends at `num_epochs`.
        """
        self.scene_state.optimization.max_epochs = num_epochs

    @property
    def max_iterations(self) -> int:
        """ Total number of iterations set for this optimization task. """
        return self.max_epochs * self.iterations_per_epoch
