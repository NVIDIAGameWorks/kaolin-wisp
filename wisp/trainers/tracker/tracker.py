# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import torch
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, Tuple
import os
from datetime import datetime
import numpy as np
import logging as log
from tqdm import tqdm
from PIL import Image
import dataclasses
import pandas as pd
from wisp.config import configure, autoconfig, instantiate
from wisp.trainers.tracker.offline_renderer import OfflineRenderer
from wisp.trainers.tracker.metrics import MetricsBoard

# Check for availability of tensorboard and wandb
_TENSORBOARD_AVAILABLE, _WANDB_AVAILABLE = True, True
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    _TENSORBOARD_AVAILABLE = False
try:
    import wandb
except:
    _WANDB_AVAILABLE = False


@configure
class ConfigTracker:
    """ Configuration related to logging and offline rendering """

    tensorboard: autoconfig(_Tensorboard)
    """ Tensorboard configuration. Used when enable_tensorboard=True."""

    wandb: autoconfig(_WandB)
    """ Weights & Biases configuration. Used when enable_wandb=True."""

    visualizer: autoconfig(OfflineRenderer)
    """ Configuration for the visualizer module, which saves rendered neural field snapshots """

    vis_camera: ConfigVisCameras
    """ Configuration parameters controlling the camera during visualization of images and animations """

    enable_tensorboard: bool = True
    """ Use Tensorboard for experiment tracking. """

    enable_wandb: bool = False
    """ Use Weights & Biases for experiment tracking. """

    log_dir : str = '_results/logs'
    """ Output folder to save checkpoints and results. """


@configure
class ConfigVisCameras:
    """ Configuration related to visualization properties"""

    camera_origin : Tuple[float, float, float] = (-2.8, 2.8, -2.8)
    """ For rendered snapshots: origin of the camera. """

    camera_lookat : Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """ For rendered snapshots: Lookat point of the camera. """

    camera_fov : float = 30.0
    """ For rendered snapshots: FOV of the camera. """

    camera_clamp : Tuple[float, float] = (0.0, 10.0)
    """ For rendered snapshots: near / fear planes which clamp the view frustum of the camera. """

    viz360_num_angles: int = 20
    """ WandB 360 animation: Number of angles to use in 360 animation. Set to 0 to disable the 360 animation."""

    viz360_radius: float = 3.0
    """ WandB 360 animation: Distance of camera orbit from the object the camera is circling. """

    viz360_render_all_lods: bool = False
    """ WandB 360 animation: for neural fields with grids that store latents on multiple levels. 
        If False, will render the neural field with the highest LOD.
        If True, will render multiple animations of the neural field per LOD (takes longer to generate).
    """

class Tracker:
    """ The Tracker is a module for running various useful validation operations:
    1. self.dashboards   ||  Experiment dashboards (uses Tensorboard and Wandb under the hood, configurable)
    2. self.visualizer ||  Creating snapshot visualizations and animations (uses wisp's OfflineRenderer)
    3. self.metrics    ||  Aggregating metrics (uses an internal MetricsBoard class)
    Under the hood, the Tracker keeps state related to all of the above, which allows thinner trainers and better
    reuse of validation logic.
    Since this module is complex, each of sub-modules uses a different config class.
    """

    def __init__(self, cfg: ConfigTracker, exp_name: str, log_fname: Optional[str] = None):
        """
        Args:
            cfg (ConfigTracker): Configuration object, with default definitions for visualization and
                experiments tracking.
            exp_name (str): Name for identifying sets of experiments.
            log_fname (Optional[str]): Unique ID for experiments. 
                                       Defaults to some generation according to datetime.
        """
        self.cfg = cfg
        self.exp_name = exp_name
        # Unique ID for this experiment (default)
        self.log_fname = log_fname
        if self.log_fname is None:
            self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # dashboards includes tensorboard, wandb or both or neither.
        self.dashboards: Dict[str, _BaseDashboard] = self._setup_dashboards(cfg, self.exp_name, self.log_fname)
        # visualizer is the OfflineRenderer
        self.visualizer = instantiate(self.cfg.visualizer)
        self.metrics = MetricsBoard()

        # Log files will be saved to the following path
        self.log_dir = os.path.join(self.cfg.log_dir, self.exp_name, self.log_fname)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Optional: To be initialized with save_app_config() for apps that log the app configuration
        self.app_config = None

    def save_app_config(self, app_config):
        """ Saves a reference to the app config, for future logging.
        This avoids the need of passing the app_config around simply for logging it.
        """
        self.app_config = app_config

    def get_app_config(self, as_dict: bool = False):
        """ Returns the app logic cached in this Tracker.

        Args:
            as_dict (bool): If True, the returned value is converted to a dict 
                            (e.g. a config dataclass will be converted to a dict).
        """
        if self.app_config is None:
            return None
        if as_dict and self.app_config is not None and dataclasses.is_dataclass(self.app_config):
            return dataclasses.asdict(self.app_config)
        else:
            return self.app_config

    def get_record_dict(self):
        """ Returns the app logic cached in this Tracker for purposes of logging.

        This is close to get_app_config, but will also flatten the dictionary with hierarchical keys,
        and also will filter out any Tensor fields. This is convenient for purposes of logging config
        arguments as a Pandas dataframe or equivalent.
        """
        app_config = self.get_app_config(as_dict=True)
        if app_config is None:
            return None
        else:
            flattened_dict = pd.json_normalize(app_config, sep='.').to_dict(orient='records')[0]
            
            # record_dict contains config args, but omits torch.Tensor fields which were not explicitly converted to
            # numpy or some other format. This is required as parquet doesn't support torch.Tensors
            # (and also for output size considerations)
            def record_dict_filter(k, v):
                is_not_tensor = not isinstance(v, torch.Tensor)
                is_not_underscore = all([not _k.startswith("_") for _k in k.split(".")])
                return is_not_tensor and is_not_underscore
    
            record_dict = {k: v for k, v in flattened_dict.items() if record_dict_filter(k, v)}
            return record_dict


    @staticmethod
    def _setup_dashboards(cfg: ConfigTracker, exp_name: str, log_fname: str):
        """ Set up tensorboard and wand if they're enabled by the config, and their packages are installed.
        This function is invoked from the Tracker constructor, and must run before performing any experiment logging.
        """
        dashboards: Dict[str, _BaseDashboard] = dict()
        if cfg.enable_tensorboard:
            if cfg.tensorboard.exp_name is None:
                cfg.tensorboard.exp_name = exp_name
            if cfg.tensorboard.log_fname is None:
                cfg.tensorboard.log_fname = log_fname
            if _TENSORBOARD_AVAILABLE:
                dashboards['tensorboard'] = instantiate(cfg.tensorboard)
            else:
                log.warning("Tensorboard experiment tracking enabled, "
                            "but couldn't import torch.utils.tensorboard.SummaryBoard")
        if cfg.enable_wandb:
            if cfg.wandb.entity is None:
                raise Exception("You must set your username as the entity to use Wandb")
            if cfg.wandb.project is None:
                cfg.wandb.project = exp_name
            if cfg.wandb.group is None:
                cfg.wandb.group = exp_name
            if cfg.wandb.run_name is None:
                cfg.wandb.run_name = log_fname
            if _WANDB_AVAILABLE:
                dashboards['wandb'] = instantiate(cfg.wandb)
            else:
                log.warning("wandb experiment tracking enabled, but couldn't import wandb")
        return dashboards

    def teardown(self):
        """ Closes the experiment dashboards currently set up. This is used to finalize tensorboard and wandb,
        committing any pending reports, etc.
        """
        for dashboard in self.dashboards.values():
            dashboard.teardown()

    def log_metric(self, metric, value, step=None):
        """ Logs a metric:value to the enabled experiment dashboards (tensorboard, wandb).
        step is an optional step (iteration / epoch) identifier for the current logged metric.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        for dashboard in self.dashboards.values():
            dashboard.log_metric(metric, value, step)

    def log_table(self, caption: str, data: Dict, step=None):
        """ Logs the data as a table to the enabled experiment dashboards (tensorboard, wandb).
        step is an optional step (iteration / epoch) identifier for the current logged table.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        for dashboard in self.dashboards.values():
            dashboard.log_table(caption, data, step)

    def log_config(self, config: Dict[str, Any]):
        """ Logs a config dict to the enabled experiment dashboards (tensorboard, wandb).
        The exact format metrics appear in depends on the experiment dashboard.
        """
        for dashboard in self.dashboards.values():
            dashboard.log_config(config)

    def log_image(self, key, image, step):
        """ Logs an image the enabled experiment dashboards (tensorboard, wandb).
        step (iteration / epoch) is an identifier for the current logged metric.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        for dashboard in self.dashboards.values():
            dashboard.log_image(key, image, step)

    def log_artifact(self, model_fname: str, names):
        """ Logs an artifact (e.g. saved model) to the enabled experiment dashboards (tensorboard, wandb).
        Args:
            model_fname (str): Path where the model is currently saved to the disk.
            names: Various aliases describing this artifact (i.e. 'epoch_100', 'latest', 'best').
        The exact format artifacts appear in depends on the experiment dashboard.
        Note that some experiment dashboards may not support this op, in which case they will silently return if enabled.
        """
        for dashboard in self.dashboards.values():
            dashboard.log_artifact(model_fname, names)

    def log_360_orbit(self, pipeline):
        """ Log a 360 orbit animation of the rendererd pipeline, using the camera configured in self.cfg.vis_camera.
        -- Requires wandb to be enabled --
        """
        # TODO (operel): @cfujitsang's module for camera movement should generalize this
        vis_cfg = self.cfg.vis_camera
        is_render_all_lods = self.cfg.vis_camera.viz360_render_all_lods
        num_angles = self.cfg.vis_camera.viz360_num_angles
        camera_distance = self.cfg.vis_camera.viz360_radius

        wandb_dashboard = self.dashboards.get('wandb')
        if wandb_dashboard is not None and num_angles != 0:
            angles = np.pi * 0.1 * np.array(list(range(num_angles + 1)))
            x = -camera_distance * np.sin(angles)
            y = vis_cfg.camera_origin[1]
            z = -camera_distance * np.cos(angles)

            lods = list(range(pipeline.nef.grid.num_lods)) if is_render_all_lods else [None]
            for d in lods:
                if d is not None:   # Render all lods
                    desc = f"Generating 360 Degree of View for LOD {d}"
                    metric_prefix = f"LOD-{d}-"
                    log_target = f"LOD-{d}"
                    wandb.define_metric(f"LOD-{d}-360-Degree-Scene")
                    wandb.define_metric(f"LOD-{d}-360-Degree-Scene", step_metric=f"LOD-{d}-360-Degree-Scene/step")
                else:               # Render only max lod
                    desc = f"Generating 360 Degree of View"
                    metric_prefix = f""
                    log_target = f"MaxLOD"
                    wandb.define_metric(f"360-Degree-Scene")
                    wandb.define_metric(f"360-Degree-Scene", step_metric=f"360-Degree-Scene/step")

                out_rgb = []
                for idx in tqdm(range(num_angles + 1), desc=desc):
                    wandb_dashboard.log_metric(f"{metric_prefix}360-Degree-Scene/step", idx, step=idx)
                    out = self.visualizer.render_snapshot(
                        pipeline,
                        f=[x[idx], y, z[idx]],
                        t=vis_cfg.camera_lookat,
                        fov=vis_cfg.camera_fov,
                        lod_idx=d,
                        camera_clamp=vis_cfg.camera_clamp
                    )
                    out = out.image().byte().numpy_dict()
                    if out.get('rgb') is not None:
                        wandb_dashboard.log_image(f"{metric_prefix}360-Degree-Scene/RGB", out['rgb'].T, idx)
                        out_rgb.append(Image.fromarray(np.moveaxis(out['rgb'].T, 0, -1)))
                    if out.get('rgba') is not None:
                        wandb_dashboard.log_image(f"{metric_prefix}360-Degree-Scene/RGBA", out['rgba'].T, idx)
                    if out.get('depth') is not None:
                        wandb_dashboard.log_image(f"{metric_prefix}360-Degree-Scene/Depth", out['depth'].T, idx)
                    if out.get('normal') is not None:
                        wandb_dashboard.log_image(f"{metric_prefix}360-Degree-Scene/Normal", out['normal'].T, idx)
                    if out.get('alpha') is not None:
                        wandb_dashboard.log_image(f"{metric_prefix}360-Degree-Scene/Alpha", out['alpha'].T, idx)
                    wandb.log({})

                rgb_gif = out_rgb[0]
                gif_path = os.path.join(self.log_dir, "rgb.gif")
                rgb_gif.save(gif_path, save_all=True, append_images=out_rgb[1:], optimize=False, loop=0)
                wandb.log({f"360-Degree-Scene/RGB-Rendering/{log_target}": wandb.Video(gif_path)})


class _BaseDashboard(ABC):
    """ General interface for experiment dashboards """

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        """ Setup logic for the experiment dashboards.
        """
        raise NotImplementedError()

    @abstractmethod
    def teardown(self):
        """ Closes the experiment dashboards currently set up. This is used to finalize,
        committing any pending reports, etc.
        """
        raise NotImplementedError()

    @abstractmethod
    def log_metric(self, metric, value, step=None):
        """ Logs a metric:value to the enabled experiment dashboards.
        step is an optional step (iteration / epoch) identifier for the current logged metric.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        raise NotImplementedError()

    def log_table(self, caption: str, data: Dict, step=None):
        """ Logs the data as a table to the enabled experiment dashboards.
        step is an optional step (iteration / epoch) identifier for the current logged table.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        pass    # unused

    def log_config(self, config: Dict[str, Any]):
        """ Logs a config dict to the enabled experiment dashboards.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        pass    # unused

    def log_image(self, key, image, step):
        """ Logs an image the enabled experiment dashboards.
        step (iteration / epoch) is an identifier for the current logged metric.
        The exact format metrics appear in depends on the experiment dashboard.
        """
        pass    # unused

    def log_artifact(self, model_fname, names):
        """ Logs an artifact (e.g. saved model) to the enabled experiment dashboards.
        Args:
            model_fname (str): Path where the model is currently saved to the disk.
            names: Various aliases describing this artifact (i.e. 'epoch_100', 'latest', 'best').
        The exact format artifacts appear in depends on the experiment dashboard.
        Note that some experiment dashboards may not support this op, in which case they will silently return.
        """
        pass    # unused


class _Tensorboard(_BaseDashboard):
    """ Wraps around tensorboard functionality """

    def __init__(self, log_dir: str, exp_name: Optional[str], log_fname: Optional[str]):
        """ Tensorboard experiments dashboard.
        Args:
            log_dir (str): Path where the tensorboard runs are saved.
            exp_name (Optional[str]): Path where sets of experiments are saved (equivalent to project in wandb)
            log_fname (Optional[str]): Path where specific experiments are saved (equivalent to run name in wandb)
        """
        super().__init__()
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.log_fname = log_fname
        self.setup()

    def setup(self):
        self.writer = SummaryWriter(os.path.join(self.log_dir, self.exp_name, self.log_fname), purge_step=0)

    def teardown(self):
        self.writer.close()

    def log_table(self, caption: str, data: Dict[str, Any], step=None):
        data_str = str(dict(sorted(data.items())))
        self.writer.add_text(caption, data_str, step)

    def log_metric(self, metric, value, step=None):
        self.writer.add_scalar(metric, value, step)

    def log_image(self, key, image, step):
        self.writer.add_image(key, image, step)

    def log_config(self, config: Dict[str, Any]):
        self.log_table(caption='config', data=config)


class _WandB(_BaseDashboard):
    """ Wraps around wandb functionality """

    def __init__(self, entity: Optional[str] = None, project: Optional[str] = None,
                 group: Optional[str] = None, run_name: Optional[str] = None,
                 job_type: Optional[str] = None, sync_tensorboard: Optional[bool] = True):
        """Weights & Biases experiment dashboard.
        See: https://docs.wandb.ai/ref/python/init

        Args:
            entity (Optional[str]): Weights & Biases entity name (i.e. username)
            project (Optional[str]): Weights & Biases project name. Defaults to the experiment name.
            group (Optional[str]): Weights & Biases group name. Default to the experiment name.
            run_name (Optional[str]): Weights & Biases Run name. Defaults to the run ID.
            job_type (Optional[str]): Weights & Biases Job type, i.e: trainer mode of 'train', 'val', etc.
            sync_tensorboard (Optional[bool]): Sync wandb logs from tensorboard and save the relevant events file.
        """
        super().__init__()
        self.entity = entity
        self.project = project
        self.group = group
        self.run_name = run_name
        self.job_type = job_type
        self.sync_tensorboard = sync_tensorboard
        self.setup()

    def setup(self):
        wandb.init(
            project=self.project,
            group=self.group,
            name=self.run_name,
            entity=self.entity,
            job_type=self.job_type,
            config=dict(),
            sync_tensorboard=self.sync_tensorboard
        )

    def teardown(self):
        wandb.finish()

    def log_table(self, caption: str, data: Dict[str, Any], step=None):
        table = wandb.Table(columns=data.keys(), data=data.values())
        wandb.log({caption: table}, step=step, commit=False)

    def log_metric(self, metric, value, step=None):
        wandb.log({metric: value}, step=step, commit=False)

    def log_image(self, key, image, step):
        wandb.log({key: wandb.Image(np.moveaxis(image, 0, -1))}, step=step, commit=False)

    def log_artifact(self, model_fname, names):
        name = wandb.util.make_artifact_name_safe(f"{wandb.run.name}-model")
        model_artifact = wandb.Artifact(name, type="model")
        model_artifact.add_file(model_fname)
        wandb.run.log_artifact(model_artifact, aliases=names)

    def log_config(self, config: Dict[str, Any]):
        wandb.config.update(config)
