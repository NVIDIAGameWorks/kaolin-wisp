# Kaolin Wisp: A PyTorch Library and Engine for Neural Fields Research

<img src="../_static/media/demo.jpg" alt="drawing" width="1000"/>


NVIDIA Kaolin Wisp is a PyTorch library powered by [NVIDIA Kaolin Core](https://github.com/NVIDIAGameWorks/kaolin) to work with
neural fields (including NeRFs, [NGLOD](https://nv-tlabs.github.io/nglod), [instant-ngp](https://nvlabs.github.io/instant-ngp/) and [VQAD](https://nv-tlabs.github.io/vqad)).

NVIDIA Kaolin Wisp aims to provide a set of common utility functions for performing research on neural fields. 
This includes datasets, image I/O, mesh processing, and ray utility functions. 
Wisp also comes with building blocks like differentiable renderers and differentiable data structures 
(like octrees, hash grids, triplanar features) which are useful to build complex neural fields. 
It also includes debugging visualization tools, interactive rendering and training, logging, and trainer classes.

For an overview on neural fields, we recommend you check out the EG STAR report: 
[Neural Fields for Visual Computing and Beyond](https://arxiv.org/abs/2111.11426).

## Installation & Code

See {doc}`Installation Page <install>` and kaolin-wisp on [Github](https://github.com/NVIDIAGameWorks/kaolin-wisp).

## Common Features

<img src="../_static/media/blocks_1_0.jpg" alt="drawing" width="1000"/>


* Differentiable feature grids
    * Octree grids (from NGLOD)
    * Hash grids (from Instant-NGP)
    * Triplanar texture grids (from ConvOccNet, EG3D)
    * Codebook grids (from VQAD)
* Acceleration structures for fast raytracing
    * Octree acceleration structures based on Kaolin Core SPC
* Tracers to trace rays against neural fields
    * PackedSDFTracer for SDFs
    * PackedRFTracer for radiance fields (NeRFs)
* Various datasets for common neural fields
    * Standard Instant-NGP compatible datasets
    * RTMV dataset
    * SDF sampled from meshes
* An interactive renderer where you can train and visualize neural fields
* A set of core framework features (`wisp.core`) for convenience
* A set of utility functions (`wisp.ops`)

Have a feature request? [Leave a GitHub issue](https://github.com/NVIDIAGameWorks/kaolin-wisp/issues/new?title=Issue%20on%20page%20%2Fpages/main.html&body=Your%20issue%20content%20here.)!

## Wisp Structure

### Core Library
The core library resides under [`wisp/`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp), and contains various building blocks useful for optimizing & building neural field based pipelines.

Users are encouraged to browse the following links, and review various aspects of the library:
* [`wisp/model`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp/models) is a subpackage containing modules to construct neural fields.
* [`wisp/trainers`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp/trainerss) is a subpackage containing default trainers which are useful to extend.
* [`wisp/renderer`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp/renderer) is a subpackage containing logic related to the interactive renderer.
* [`wisp/ops`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp/ops) is a subpackage containing useful operators.

## Applications: Training & Rendering with Wisp
The Wisp repository also includes some sample apps aand examples:

* [`app`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/app) - Complete implementations of papers & projects using Wisp components.
  * [`app/nerf`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/app/nerf) is the Wisp's upgraded NeRF app, with support for various grids types.
  * [`app/nglod`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/app/nglod) is an implementation of the paper _Neural Geometric Level of Detail ([Takikawa et al. 2021](https://nv-tlabs.github.io/nglod/)).
* [`examples`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/examples) - Smaller demos demonstrating specific features & use cases of Wisp.
  * [`examples/latent_nerf`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/examples/latent_nerf) - demonstrates how to add a new NeuralField module, which exposess the latent dimensions as output & visualizes it.
  * [`examples/spc_browser`](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/examples/spc_browser) - an app for converting meshes to [Structured Point Clouds](https://kaolin.readthedocs.io/en/latest/notes/spc_summary.html), and visualizing them.


### Configurations

Want to run the Wisp apps with different options? Our configuration system makes this very easy.

Wisp apps use a mixture of config files and CLI arguments, which take higher precedence.
For example, if you want to run NeRF with a different number of hidden layer neurons:
```
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_hash.yaml --dataset-path /path/to/lego --hidden_dim 128
```

Arg values not specified through yaml or CLI will resort to the default value the main script specifies, when available.

Take a look at the respective `main_X.py` script of each app for the list of different options you can pass in, and `configs/Y.yaml` 
for some complete configurations.

Wisp also supports hierarchical configs, by using the `parent` argument in the config to set a parent 
config file path in relative path from the config location or with an absolute path. Note however that 
only a single level of hierarchy is allowed to keep the indirection manageable.

If you get any errors from loading in config files, you likely made a typo in your field names. Check
against the app main as your source of truth. (Or pass in `-h` for help).


### Viewing Results

Most apps, i.e. `main_nerf.py`, will generate logs inside `_results/logs/runs/test-X` in which you can find outputs like the trained 
checkpoint, and `EXR` images of validation outputs. We highly recommend that you install 
[tev](https://github.com/Tom94/tev) as the default application to open EXRs.

To view the logs with TensorBoard:
```
tensorboard --logdir _results/logs/runs
```

### Interactive training

The interactive renderer is intended to be used locally by machines equipped with a GPU (minimum of 10GB memory recommended).

To run the apps interactively using the renderer engine, run:
```
python3 app/main_nerf.py --interactive=True --config app/nerf/configs/nerf_hash.yaml --dataset-path /path/to/lego --dataset-num-workers 4
```

To disable interactive mode, and run wisp _without_ loading the graphics API, use:
```
--interactive=False
```

wisp also supports globally turning off interactive mode by setting the env variable:
```
WISP_HEADLESS=1
```

Toggling this flag is useful for debugging on machines without a display. 
This is also needed if you opt to avoid installing the interactive renderer requirements.

### Jupyter Notebook Mode _(April 2023 Update)_

Machines without a display can still use a simplified version of the interactive renderer.

See: [Jupyter Notebook Example](https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/examples/notebook/view_pretrained.ipynb).

### Experiment Tracking

Wisp's trainers use the [Tracker](https://github.com/NVIDIAGameWorks/kaolin-wisp/tree/main/wisp/trainers/tracker/tracker.py) module to track training and validation metrics, render 3D interactive plots, and reproduce your configurations and results.

The tracker is a one-stop access for various modules, such as tensorboard, wandb, offline visualizations and metrics aggregation.

#### Using [Weights & Biases](https://wandb.ai/site)

To enable wandb tracking in your Weights & Biases workspace just add the additional flag:
`--tracker.enable_wandb=True` when initializing the training script.

Complete list of features supported by Weights & Biases:

- Log training and validation metrics in real time.
- Log system metrics in real time.
- Log RGB, RGBA, Depth renderings etc. during training.
- Log interactive 360 degree renderings post training in all levels of detail.
- Log model checkpoints as [Weights & Biases artifacts](https://wandb.ai/site/artifacts).
- Sync experiment configs for reproducibility.
- Host Tensorboard instance inside Weights & Biases run.

The list of optional arguments related to logging on Weights & Biases include:

- `--tracker.wandb.project`: Name of Weights & Biases project
- `--tracker.wandb.run_name`: Name of Weights & Biases run \[Optional\]
- `--tracker.wandb.entity`: Name of Weights & Biases entity under which your project resides \[Optional\]
- `--tracker.wandb.job_type`: Name of Weights & Biases job type \[Optional\]
- `--tracker.vis_camera.viz360_num_angles`: Number of angles in the 360 degree renderings \[Optional, default set to 20\]
- `--tracker.vis_camera.viz360_radius`: Camera distance to visualize Scene from for 360 degree renderings on Weights & Biases \[Optional, default set to 3\]
- `--tracker.vis_camera.viz360_render_all_lods`: Set to True to render the neural field in all available levels of detail.
