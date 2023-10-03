# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.  #
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import time
import logging as log
import torch
import wisp
from typing import Optional, Union, Tuple
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from wisp.trainers.tracker import Tracker
from wisp.framework import WispState
from wisp.datasets import WispDataset, default_collate
from wisp.config import configure, autoconfig, instantiate, write_config_to_yaml
from wisp.config.presets import ConfigAdam, ConfigRMSprop, ConfigFusedAdam, ConfigAdamW, ConfigDataloader
from wisp.renderer.core.api import add_to_scene_graph


@configure
class ConfigBaseTrainer:
    """ Configuration common to base trainer and its derivatives """

    optimizer: Union[ConfigAdam, ConfigRMSprop, ConfigFusedAdam, ConfigAdamW]
    """ Optimizer to be used, includes optimizer modules available within `torch.optim` 
        and fused optimizers from `apex`.
    """

    dataloader: ConfigDataloader
    """ Dataloader configuration, used during optimization. """

    exp_name: str
    """ Name of the experiment: a unique id to use for logging, model names, etc. """

    mode: str = 'train'  # options: 'train', 'validate'
    """ Trainer mode: 
    'train': Trainer will optimize the given pipeline. 
    'validate': Only run validation, expects to run a pretrained model.
    """ 

    max_epochs: int = 250
    """ Number of epochs to run the training. """

    save_every: int = -1
    """ Saves the optimized model every N epochs """

    save_as_new: bool = False
    """ If True, will save the model as a new file every time the model is saved  """

    model_format: str = 'full'  # options: 'full', 'state_dict'
    """ Format to save the model: 'full' (weights+model) or 'state_dict' """

    render_every : int = 100
    """ Renders an image of the neural field every N epochs """

    valid_every: int = -1
    """ Runs validation every N epochs """
    
    valid_split : str = 'test'
    """ Split to use for validation """

    enable_amp: bool = True
    """ If enabled, the step() training function will use mixed precision. """

    profile_nvtx: bool = True
    """ If enabled, nvtx markers will be emitted by torch for profiling. See: torch.autograd.profiler.emit_nvtx"""

    grid_lr_weight : float = 1.0
    """ Learning rate weighting applied only for the grid parameters
        (e.g. parameters which contain "grid" in their name)
    """
    
    scheduler : bool = False
    """ If enabled, will use learning rate scheduling. """

    scheduler_milestones : Tuple[float, ...] = (0.5, 0.75, 0.9)
    """ The milestones during training (as a ratio of total iterations) to adjust learning rate. """

    scheduler_gamma : float = 0.333 
    """ The amount to adjust learning rate at the milestones. """
    
    valid_metrics : Tuple[str, ...] = ('psnr', ) # lpips, ssim are also supported
    """ The validation metrics to use. Will tend to vary based on application. """

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
            |- log_console()
            |- log_tracker()
            |- render_snapshot()
            |- save_model()
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
    def __init__(self,
        cfg: ConfigBaseTrainer,
        pipeline: wisp.models.Pipeline,
        train_dataset: wisp.datasets.WispDataset,
        tracker: Tracker,
        device: torch.device = 'cuda',
        scene_state: Optional[wisp.framework.WispState] = None):
        """Constructor.

        Args:
            cfg (ConfigBaseTrainer): Trainer base configuration, includes optimization, logging and other definitions.
            pipeline (wisp.core.Pipeline): Target neural field to optimize.
                The pipeline includes both the neural field to train, and a differentiable tracer to render it.
            train_dataset (wisp.datasets.WispDataset): Dataset used for generating training batches.
            tracker (wisp.trainers.tracker.Tracker): An experiments tracker object, used for logging the optimization progress.
            device (torch.device): Device used to run the optimization.
            scene_state (wisp.core.State): Global information and definitions shared between the trainer and external components (i.e. interactive visualizer).
        """
        self.device = device
        self.cfg = cfg
        self.pipeline = pipeline.to(device)
        self.train_dataset = train_dataset
        self.tracker = tracker

        # Initialize global scene_state, if not set already
        if scene_state is None:
            scene_state = WispState()
        self.scene_state = scene_state

        self.scaler = None
        self.train_data_loader_iter = None
        self.val_data_loader = None
        self.train_dataset_size = None

        # Training params
        self.enable_amp = cfg.enable_amp
        self.max_epochs = cfg.max_epochs
        self.epoch = 1
        self.iteration = 0

        # Dictionary for any return values at the end of training
        # (useful for when integrating with external hyperparameter optimizers, etc)
        self.return_dict = {}

        # Add object to scene graph: if interactive mode is on, this will make sure the visualizer can display it.
        self.populate_scenegraph()

        # Update optimization state about the current train set used
        self.scene_state.optimization.train_data.append(train_dataset)

        self.init_optimizer()
        self.init_dataloader()

        device_name = torch.cuda.get_device_name(device=self.device)
        num_parameters = sum(p.numel() for p in self.pipeline.nef.parameters())
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')
        log.info(f"Total number of parameters: {num_parameters}")

    def populate_scenegraph(self):
        """ Updates the scenegraph with information about available objects.
        Doing so exposes these objects to other components, like visualizers and loggers.
        """
        # Add object to scene graph: if interactive mode is on, this will make sure the visualizer can display it.
        # batch_size is an optional setup arg here which hints the visualizer how many rays can be processed at once
        # (e.g. this is the pipeline's batch_size used for inference time)
        add_to_scene_graph(state=self.scene_state, name=self.cfg.exp_name, obj=self.pipeline, batch_size=2**14)

    def init_dataloader(self):
        self.train_data_loader = DataLoader(self.train_dataset,
                                            batch_size=self.cfg.dataloader.batch_size,
                                            collate_fn=default_collate,
                                            shuffle=True, pin_memory=True,
                                            num_workers=self.cfg.dataloader.num_workers)
        self.iterations_per_epoch = len(self.train_data_loader)

    def init_optimizer(self):
        """Default initialization for the optimizer.
        """
        params_dict = { name : param for name, param in self.pipeline.nef.named_parameters()}

        params = []
        decoder_params = []
        grid_params = []
        rest_params = []

        # TODO (operel): Better to use wisp interfaces here, as names may be brittle
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

        lr = self.cfg.optimizer.lr
        eps = self.cfg.optimizer.eps
        weight_decay = self.cfg.optimizer.weight_decay
        
        params.append({"params": decoder_params, "lr": lr, "eps": eps, "weight_decay": weight_decay})
        params.append({"params": grid_params, "eps": eps, "lr": lr * self.cfg.grid_lr_weight})
        params.append({"params": rest_params, "eps": eps, "lr": lr})

        max_steps = len(self.train_dataset) * self.cfg.max_epochs
        milestone_iters = [max_steps * x for x in self.cfg.scheduler_milestones]
        self.optimizer = instantiate(self.cfg.optimizer, params=params)
        self.scaler = torch.cuda.amp.GradScaler()
        if self.cfg.scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestone_iters,
                gamma=self.cfg.scheduler_gamma,
            )

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
        if hasattr(self.train_dataset, 'resample'):
            log.info("Reset DataLoader")
            self.train_dataset.resample()
            self.init_dataloader()
        else:
            raise ValueError("resample=True but the training dataset doesn't have a resample method")

    #######################
    # Training Life-cycle
    #######################

    def is_first_iteration(self):
        return self.total_iterations == 0

    def is_any_iterations_remaining(self):
        return self.total_iterations < self.max_iterations

    def begin_epoch(self):
        """Begin epoch.
        """
        self.reset_data_iterator()
        self.pre_epoch()
        self.epoch_start_time = time.time()

    def end_epoch(self):
        """End epoch.
        """
        current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        self.epoch_start_time = current_time

        self.tracker.log_metric(f'time/elapsed_ms_per_epoch', elapsed_time * 1000, self.epoch)

        self.post_epoch()

        if self.cfg.valid_every > -1 and \
                self.epoch % self.cfg.valid_every == 0 and \
                self.epoch != 0:
            self.validate()

        if self.epoch < self.max_epochs:
            self.iteration = 0
            self.epoch += 1
        else:
            self.is_optimization_running = False

    def iterate(self):
        """Advances the training by one training step (batch).
        """
        if self.is_optimization_running:
            # import pdb; pdb.set_trace()
            if self.is_first_iteration():
                self.pre_training()
            iter_start_time = time.time()
            try:
                if self.train_data_loader_iter is None:
                    self.begin_epoch()
                self.iteration += 1
                data = self.next_batch()
            except StopIteration:
                self.end_epoch()    # determines if optimization keeps running
                if self.is_any_iterations_remaining():
                    self.begin_epoch()
                    data = self.next_batch()
                else:
                    self.post_training()
            if self.is_any_iterations_remaining():
                self.pre_step()
                with torch.cuda.amp.autocast(self.enable_amp):
                    self.step(data)
                self.post_step()
            iter_end_time = time.time()
            self.scene_state.optimization.elapsed_time += iter_end_time - iter_start_time

    def save_model(self):
        """
        Override this function to change model saving.
        """
        if self.cfg.save_as_new:
            model_fname = os.path.join(self.tracker.log_dir, f'model-ep{self.epoch}-it{self.iteration}.pth')
        else:
            model_fname = os.path.join(self.tracker.log_dir, f'model.pth')

        log.info(f'Saving model checkpoint to: {model_fname}')
        if self.cfg.model_format == "full":
            torch.save(self.pipeline, model_fname)
        else:
            torch.save(self.pipeline.state_dict(), model_fname)

        self.tracker.log_artifact(model_fname=model_fname, names=["latest", f"ep{self.epoch}_it{self.iteration}"])

    def train(self):
        """
        Override this if some very specific training procedure is needed.
        
        Returns:
            (dict): The dictionary with validation metrics and other information as needed.
        """
        with torch.autograd.profiler.emit_nvtx(enabled=self.cfg.profile_nvtx):
            self.is_optimization_running = True
            while self.is_optimization_running:
                self.iterate()
            log.info('Training completed.')

        return self.return_dict

    #######################
    # Training Events
    #######################

    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        app_config = self.tracker.get_app_config()
        if app_config is not None:
            write_config_to_yaml(app_config, os.path.join(self.tracker.log_dir, "config.yaml"))

            app_config = self.tracker.get_app_config(as_dict=True)
            self.tracker.log_config(app_config)

        self.tracker.metrics.define_metric('total_loss', aggregation_type=float)
        # self.log_model_details()

    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """
        self.tracker.teardown()

    def pre_epoch(self):
        """
        Override this function to change the pre-epoch preprocessing.
        This function runs once before the epoch.
        """
        self.pipeline.train()
        self.tracker.metrics.clear()

    def post_epoch(self):
        """
        Override this function to change the post-epoch post processing.

        By default, this function logs to Tensorboard, renders images to Tensorboard, saves the model,
        and resamples the training dataset.

        To keep default behaviour but also augment with other features, do

          super().post_epoch()

        in the derived method.
        """
        self.pipeline.eval()

        # Report average loss for epoch to console
        self.log_console()

        # Update dashboards and external components (i.e. interactive visualizer) about epoch results
        self.log_tracker()
        self.tracker.metrics.finalize_epoch(self.scene_state)

        if self.is_time_to_render():
            self.render_snapshot()  # Render visualizations to disk, possibly log to experiment dashboards

        if self.is_time_to_save():
            self.save_model()       # Save model to disk, possibly log artifact

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

    def is_time_to_render(self):
        return self.cfg.render_every > -1 and self.epoch % self.cfg.render_every == 0

    def is_time_to_save(self):
        return self.cfg.save_every > -1 and self.epoch % self.cfg.save_every == 0 and self.epoch != 0

    def log_model_details(self):
        log.info(f"-- Model Details --")
        if self.pipeline.nef is not None:
            for key, value in self.pipeline.nef.public_properties().items():
                log.info(f"{key}: {value}")

    def log_console(self):
        """
        Override this function to change CLI logging.

        By default, this function only runs every epoch.
        """
        # Average over iterations
        total_loss = self.tracker.metrics.average_metric('total_loss')
        log_text = 'EPOCH {}/{}'.format(self.epoch, self.max_epochs)
        log_text += ' | total loss: {:>.3E}'.format(total_loss)
        log.info(log_text)

    def log_tracker(self):
        for key in self.tracker.metrics.active_metrics:
            if 'loss' in key:
                self.tracker.log_metric(metric=f'loss/{key}', value=self.tracker.metrics[key], step=self.epoch)

    def render_snapshot(self):
        """Renders a snapshot of the neural field
        Override this function to change render logging to TensorBoard / Wandb.
        """
        self.pipeline.eval()
        lod_idx = self.pipeline.nef.grid.num_lods - 1
        out = self.tracker.visualizer.render_snapshot(self.pipeline,
                                                      f=self.tracker.cfg.vis_camera.camera_origin,
                                                      t=self.tracker.cfg.vis_camera.camera_lookat,
                                                      fov=self.tracker.cfg.vis_camera.camera_fov,
                                                      lod_idx=lod_idx,
                                                      camera_clamp=self.tracker.cfg.vis_camera.camera_clamp)

        # Premultiply the alphas since we're writing to PNG (technically they're already premultiplied)
        if self.pipeline.tracer.bg_color == 'black' and out.rgb.shape[-1] > 3:
            bg = torch.ones_like(out.rgb[..., :3])
            out.rgb[..., :3] += bg * (1.0 - out.rgb[..., 3:4])

        out = out.image().byte().numpy_dict()

        log_buffers = ['depth', 'hit', 'normal', 'rgb', 'alpha']

        for key in log_buffers:
            if out.get(key) is not None:
                self.tracker.log_image(f'{key}/{lod_idx}', out[key].T, self.epoch)

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
        return (self.epoch - 1) * self.iterations_per_epoch + self.iteration

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
