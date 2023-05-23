# Config System

The wisp configuration system aims for a flexible, minimalist api through building on tyro and hydra-zen.

The following example illustrates the usage:

```python
from wisp.config import configure, autoconfig, parse_config, instantiate
from wisp.models.nefs import NeuralRadianceField
from wisp.models.grids import HashGrid, TriplanarGrid
import torch
from typing import Union, Tuple

# For classes and functions which don't use python typing annotations,
# the best practice is to create a config class, which specifies which args are configurable
# and what their types are:
@configure(target=torch.optim.Adam)     # This config can build torch.optim.Adam
class ConfigAdam:
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

@configure(target=torch.optim.RMSprop)  # This config can build torch.optim.RMSprop
class ConfigRMSprop:
    lr: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0

# @configure without a target is equivalent to a @dataclass.
@configure
class AppConfig:
    """  AppConfig is a dataclass holding all configuration fields. """
    # autoconfig() scans the classes for all available constructors, and generate a config dataclass for each.
    # This is useful for classes like grids, which have many types and constructors.
    # The resulting configs are similar to ConfigAdam and ConfigRMSprop above.
    # autoconfig() requires functions to annotate args with typings.
    grid: autoconfig(TriplanarGrid, HashGrid)    # type: Union[ConfigTriplanarGrid, ConfigHashGrid, ConfigHashGridFromGeometric, ...]
    
    # You can also use autoconfig() with a single class or function
    nerf: autoconfig(NeuralRadianceField)        # type: ConfigNeuralRadianceField
    
    # Explicit config set. This is useful, i.e., because the type annotations of torch are incomplete.
    # We specify we want to use the config classes we manually defined above with @configure(target=...)
    optimizer: Union[ConfigAdam, ConfigRMSprop] 
    
# parse_config will read arg values from the CLI to create an AppConfig instance.
# If `--config <path>.yaml` is given, it will also read values from it.
cfg = parse_config(AppConfig, yaml_arg='--config')

grid = instantiate(cfg.grid)                # Build grid instance directly from config
nerf = instantiate(cfg.nerf, grid=grid)     # Build nerf instance from config and grid
optimizer = instantiate(cfg.optimizer, params=nerf.decoder.parameters())      # Build optimizer from config
```

Running the program above and specifying args from CLI:
```shell
python main.py grid:HashGrid.from-geometric optimizer:adam --lr 0.001 --optimizer.eps 0.000001
```

or with a config file:
```yaml
grid:
   constructor: 'HashGrid.from_geometric'
   ...
nerf:
   ...
optimizer:
   constructor: 'Adam'
   lr: 1e-3
   eps: 1e-8
```

# Config Presets
Premade configs for common modules, such as torch optimizers, are included in `wisp.config.presets`.
For example:

```python
from wisp.config.presets import ConfigAdam, ConfigRMSprop, ConfigDataloader
```

# Converting older configs (up to wisp v1.0.2)

The following is a summary of the previous argparse definitions, mapped to the new config system.

## Common args
```
--epochs -> --max-epochs
--render-tb-every -> --render-every
--optimizer-type adam -> trainer.optimizer:Adam
--optimizer-type rmsprop -> trainer.optimizer:RMSProp
--multiview-dataset-format standard -> dataset:NeRFSyntheticDataset
--multiview-dataset-format rtmv -> dataset:RTMVDataset
```

## NeRF App

### Arg groups:
Previously, these are were used to single out multiple choices. 
These args are now replaced with Unions of configs,
which appear in the CLI as subcommands and in the yaml as `constructor` field.
```
--multiview-dataset-format -> dataset:NeRFSyntheticDataset or dataset:RTMVDataset
--grid-type -> grid:OctreeGrid, grid:TriplanarGrid, grid:CodebookOctreeGrid, grid:HashGrid.from-geometric
--blas-type -> blas:OctreeAS.make-dense or blas:AxisAlignedBBoxAS
--optimizer-type -> trainer.optimizer:Adam or trainer.optimizer:RMSprop
```

### Args renamed:
The following args have been renamed to better match the new config hierarchy.

```
logging:
--profile -> --trainer.profile_nvtx

dataset:
--dataloader-num-workers -> --trainer.dataloader.num_workers
--bg-color -> --dataset.bg-color, --tracer.bg-color (two configs use this arg, with an identical name)
--num-rays-sampled-per-img -> dataset-transform.num_samples

grid:
--blas-levels -> --blas.level

trainer:
--epochs -> --trainer.max_epochs
--render-tb-every -> --trainer.render-every
--valid-only -> --trainer.mode (arg is now a string literal, not a bool, with values 'train', 'validate')
--rgb-loss --> --trainer.rgb-lambda

--wandb-project -> --tracker.wandb.project
--wandb-run-name -> --tracker.wandb.run-name
--wandb-entity -> --tracker.wandb.entity
--wandb-viz-nerf-angles -> --tracker.vis_camera.viz360-num-angles
--wandb-viz-nerf-distance -> --tracker.vis_camera.viz360-radius
See also new arg: --tracker.vis_camera.viz360-render-all-lods
```

### Args removed:
The following arguments have been deprecated and removed.

```
--perf (PerfTimer exists but is unused)

--tree-type 
Before: For HashGrids only: how the resolution of the grid is determined.
        "geometric" uses the geometric sequence initialization from InstantNGP,
        "quad" uses an octree sampling pattern.
Now   : Use autoconfig(HashGrid.from_geometric, HashGrid.from_octree) to choose between those two hashgrid constructors.
        For brevity, HashGrid.from_octree was dropped from the default config. 

--blas-level
Before: For HashGrids only: Determines the number of levels in the acceleration structure
        used to track the occupancy status (bottom level acceleration structure).
Now   : This arg was redundant, and is now supported through the new blas arg, see: blas.level

--resample, --resample-every (Resampling datasets is supported, but removed from this script as it's unused by default)
--only-last (unused, removed)
--log-tb-every (unused, removed)
--grow-every (unused, removed)
--growth-strategy (unused, removed)
--camera-proj (unused, removed)
```

### Args unchanged (by old category):
The following arguments are unchanged. Here we specify their complete prefix, e.g. which config
they belong in.

Note: when specifying those args via CLI, if an arg name is unique, specifying
the arg name without the full prefix is enough (i.e. `--exp-name` rather than `--trainer.exp-name`).

```
logging:
--exp-name -> --trainer.exp-name
--log-level -> --log-level

dataset:
--dataset-path -> --dataset.dataset-path
--dataset_num_workers -> --dataset.dataset-num-workers 
--mip -> --dataset.mip

grid:
--interpolation-type -> --grid.interpolation-type
--multiscale-type -> --grid.multiscale-type
--feature-dim -> --grid.feature-dim
--feature-std -> --grid.feature-std
--feature-bias -> --grid.feature-bias 
--log-base-resolution -> --grid.log-base-resolution (only for grid:TriplanarGrid)
--num-lods -> --grid.num-lods (only for grid:OctreeGrid, grid:CodebookGrid)
--codebook-bitwidth -> --grid.codebook-bitwidth (only for grid:CodebookOctreeGrid, grid:HashGrid)
--min-grid-res -> --grid.min_grid_res (only for grid:HashGrid.from-geometric)
--max-grid-res -> --grid.max_grid_res (only for grid:HashGrid.from-geometric)
--prune-min-density -> --nef.prune-min-density      (used with grids which support pruning)
--prune-density-decay -> --nef.prune_density_decay  (used with grids which support pruning)

nef:
--pos-embedder -> --nef.pos-embedder
--view-embedder -> --nef.view-embedder
--position-input -> --nef.position-input
--pos-multires -> --nef.pos-multires
--view-multires -> --nef.view-multires
--layer-type -> --nef.layer-type
--activation-type -> --nef.activation-type
--hidden-dim -> --nef.hidden-dim
--num-layers -> --nef.num-layers

tracer:
--raymarch-type -> --tracer.raymarch-type    (default changed from 'voxel' to 'ray')
--num-steps -> --tracer.num-steps            (default changed from 128 to 1024)

trainer:
# General
--pretrained -> --pretrained
--batch-size -> --trainer.dataloader.batch-size
--model-format -> --trainer.model-format 
--save-as-new -> --trainer.save-as-new
--save_every -> --trainer.save-every
--log-dir -> --tracker.log-dir               (default changed from '_results/logs/runs/' to '_results/logs/'
--prune-every -> --trainer.prune-every       (default changed from -1 to every 100 iterations)
--valid-every -> --trainer.valid-every
--random-lod -> --trainer.random-lod
# Visualizations
--render-res -> --tracker.visualizer.render-res (default set to [512, 512])
--render-batch -> --tracker.visualizer.render-batch (default set to -1)
--camera-origin -> --tracker.vis_camera.camera-origin
--camera-lookat -> --tracker.vis_camera.camera-lookat
--camera-fov -> --tracker.vis_camera.camera-fov
--camera-clamp -> --tracker.vis_camera.camera-clamp
# Optimization
--grid-lr-weight -> --trainer.grid-lr-weight   (see also new arg: feat_lr_weight)
--lr -> --trainer.optimizer.lr        (all optimizer args are now exposed via wisp.config.presets.torch)
--eps -> --trainer.optimizer.eps
--weight-decay -> --trainer.optimizer.weight-decay
```


## NGLOD / SDF App

In addition to the args listed under the NeRF app, the NGLOD (SDF) app changes other specialized args. 

### Arg groups:
Previously, these are were used to single out multiple choices. 
These args are now replaced with Unions of configs,
which appear in the CLI as subcommands and in the yaml as `constructor` field.
```
dataset:MeshSampledSDFDataset or dataset:OctreeSampledSDFDataset
Before: the dataset was selected automatically based on OctreeSampledSDFDataset.supports_blas
(if blas is an OctreeAS initialized from a mesh, OctreeSampledSDFDataset was used).
Now: The dataset used is configurable. OctreeSampledSDFDataset.supports_blas is used for validation only.
 
--grid-type -> grid:OctreeGrid, grid:HashGrid.from-geometric, HashGrid.from-geometric, grid:TriplanarGrid 
--blas-type -> blas:OctreeAS.from-mesh or blas:AxisAlignedBBoxAS
--optimizer-type -> trainer.optimizer:Adam or trainer.optimizer:RMSprop
```

### Args renamed:
The following args have been renamed to better match the new config hierarchy.

```
dataset:
--dataset-path -> --blas.mesh-path (OctreeSampledSDFDataset only)
--dataset-path -> --dataset.mesh-path (MeshSampledSDFDataset only)
--num_samples_on_mesh -> --blas.num_samples_on_mesh (OctreeAS.from_mesh only)
--num-samples -> --dataset.num-samples
--mode-mesh-norm -> --dataset.mode_norm (MeshSampledSDFDataset only)
```

### Arg defaults:
The following default args were updated in the core library, to match the previous argparse defaults:

```
nef:
pos_embedder = 'positional' # previously: 'none'
pos_multires = 4            # previously: 10

tracer:
num_steps = 1024            # previously: 128
step_size = 0.8             # previously: 1.0
```

### Args unchanged (by old category):
The following arguments are unchanged. Here we specify their complete prefix, e.g. which config
they belong in.

Note: when specifying those args via CLI, if an arg name is unique, specifying
the arg name without the full prefix is enough (i.e. `--exp-name` rather than `--trainer.exp-name`).

```
dataset:
--sample-mode -> --dataset.sample-mode
--sample-tex -> --dataset.sample-tex
--get_normals -> --dataset.get_normals (MeshSampledSDFDataset only)
--samples-per-voxel --> --dataset.samples-per-voxel (OctreeSampledSDFDataset only)

trainer:
--log_2d -> --trainer.log_2d
--only_last -> --trainer.only-last
--resample -> --trainer.resample
--matcap-path -> --tracker.visualizer.matcap-path
--ao -> --tracker.visualizer.ao
--shadow -> --tracker.visualizer.shadow
--shading-mode -> --tracker.visualizer.shading-mode
```