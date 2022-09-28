# Kaolin Wisp: A PyTorch Library and Engine for Neural Fields Research

<img src="media/demo.jpg" alt="drawing" width="1000"/>

NVIDIA Kaolin Wisp is a PyTorch library powered by [NVIDIA Kaolin Core](https://github.com/NVIDIAGameWorks/kaolin) to work with
neural fields (including NeRFs, [NGLOD](https://nv-tlabs.github.io/nglod), [instant-ngp](https://nvlabs.github.io/instant-ngp/) and [VQAD](https://nv-tlabs.github.io/vqad)).

NVIDIA Kaolin Wisp aims to provide a set of common utility functions for performing research on neural fields. 
This includes datasets, image I/O, mesh processing, and ray utility functions. 
Wisp also comes with building blocks like differentiable renderers and differentiable data structures 
(like octrees, hash grids, triplanar features) which are useful to build complex neural fields. 
It also includes debugging visualization tools, interactive rendering and training, logging, and trainer classes.

For an overview on neural fields, we recommend you checkout the EG STAR report: 
[Neural Fields for Visual Computing and Beyond](https://arxiv.org/abs/2111.11426).

## License and Citation

This codebase is licensed under the NVIDIA Source Code License. 
Commercial licenses are also available, free of charge. 
Please apply using this link (use "Other" and specify Kaolin Wisp): https://www.nvidia.com/en-us/research/inquiries/

If you find the NVIDIA Kaolin Wisp library useful for your research, please cite:
```
@misc{KaolinWispLibrary,
      author = {Towaki Takikawa and Or Perel and Clement Fuji Tsang and Charles Loop and Joey Litalien and Jonathan Tremblay and Sanja Fidler and Maria Shugrina},
      title = {Kaolin Wisp: A PyTorch Library and Engine for Neural Fields Research},
      year = {2022},
      howpublished={\url{https://github.com/NVIDIAGameWorks/kaolin-wisp}}
}
```

## Key Features

<img src="media/blocks.jpg" alt="drawing" width="750"/>

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

Have a feature request? Leave a GitHub issue!

## Installation

See installation instructions [here](INSTALL.md).

## Quick Links
The following links contain additional information about various aspects of the library.

* [wisp](wisp) is the main library package.
  * [wisp/model](wisp/models) is a subpackage containing modules to construct neural fields.
  * [wiso/renderer](wisp/renderer) is a subpackage containing logic related to the interactive renderer.
* [templates](templates) is a folder of templates for quick start of new projects.
* [examples](examples) is a folder of small exemplary projects demonstrating some use cases of wisp.

## Training & Rendering with Wisp

### Training NGLOD-NeRF from multiview RGB-D data

You will first need to download some sample data to run NGLOD-NeRF. 
Go to this [Google Drive link](https://drive.google.com/file/d/18hY0DpX2bK-q9iY_cog5Q0ZI7YEjephE/view?usp=sharing)
to download a cool Lego V8 engine from the [RTMV dataset](http://www.cs.umd.edu/~mmeshry/projects/rtmv/).

Once you have downloaded and extracted the data somewhere, you can train a NeRF using [NGLOD](https://nv-tlabs.github.io/nglod/) with:
```
python3 app/main.py --config configs/nglod_nerf.yaml --dataset-path /path/to/V8 --dataset-num-workers 4
```
This will generate logs inside `_results/logs/runs/test-nglod-nerf` in which you can find the trained 
checkpoint, and `EXR` images of validation outputs. We highly recommend that you install 
[tev](https://github.com/Tom94/tev) as the default application to open EXRs.
Note that the `--dataset-num-workers` argument is used here to control the multiprocessing used to load
ground truth images. To disable the multiprocessing, you can pass in `--dataset-num-workers -1`.

To view the logs with TensorBoard:
```
tensorboard --logdir _results/logs/runs
```

Want to run the code with different options? Our configuration system makes this very easy.
If you want to run with a different number of levels of details:
```
python3 app/main.py --config configs/nglod_nerf.yaml --dataset-path /path/to/V8 --num-lods 1
```
Take a look at `wisp/config_parser.py` for the list of different options you can pass in, and `configs/nglod_nerf.yaml` 
for the options that are already passed in.

### Interactive training

To run the training task interactively using the renderer engine, run:
```
WISP_HEADLESS=0 python3 app/main_interactive.py --config configs/nglod_nerf_interactive.yaml --dataset-path /path/to/V8 --dataset-num-workers 4
```

Every config file that we ship has a `*_interactive.yaml` counterpart that can be used for better settings
(in terms of user experience)
for the interactive training app. The later examples we show can all be run interactively with
`WISP_HEADLESS=1 python3 app/main_interactive.py` and the corresponding configs.

### Using `wisp` in headless mode

To disable interactive mode, and run wisp _without_ loading the graphics API, set the env variable:
```
WISP_HEADLESS=1
```
Toggling this flag is useful for debugging on machines without a display. 
This is also needed if you opt to avoid installing the interactive renderer requirements.

### Training NGLOD-SDF from meshes

We also support training neural SDFs from meshes. 
You will first need to download a mesh. 
Go to this [link](https://github.com/alecjacobson/common-3d-test-models/blob/master/data/spot.obj) to download 
an OBJ file of the Spot cow. 

Then, run the SDF training with:
```
python3 app/main.py --config configs/nglod_sdf.yaml --dataset-path /path/to/spot.obj
```

Currently the SDF sampler we have shipped with our code can be quite slow for larger meshes. We plan to
release a more optimized version of the SDF sampler soon.

### Training NGP for forward facing scenes

Lastly, we also show an example of training a forward-facing scene: the `fox` scene from `instant-ngp`.
To train a version of the [Instant-NGP](https://nvlabs.github.io/instant-ngp/), first download the `fox` 
dataset from the `instant-ngp` repository somewhere. Then, run the training with:
```
python3 app/main.py --config configs/ngp_nerf.yaml --multiview-dataset-format standard --mip 0 --dataset-path /path/to/fox
```

Our code supports any "standard" NGP-format datasets that has been converted with the scripts from the 
`instant-ngp` library. We pass in the `--multiview-dataset-format` argument to specify the dataset type, which
in this case is different from the RTMV dataset type used for the other examples. 

The `--mip` argument controls the amount of downscaling that happens on the images when they get loaded. This is useful
for datasets with very high resolution images to prevent overload on system memory, but is usually not necessary for the
fox dataset.

Note that our architecture, training, and implementation details still have slight differences from the 
published Instant-NGP. 

### Configuration System

Wisp accepts configuration from both the command line interface (CLI) and a `yaml` config file 
(examples in `configs`). Whatever config file you pass in through the `--config` option will be checked
against the options in `wisp/options.py` and serve as the *default arguments*. This means any CLI argument
you additionally pass in will overwrite the options you pass in through the `--config`. 
The order of arguments does not matter.

Wisp also supports hierarchical configs, by using the `parent` argument in the config to set a parent 
config file path in relative path from the config location or with an absolute path. Note however that 
only a single level of hierarchy is allowed to keep the indirection manageable.

If you get any errors from loading in config files, you likely made a typo in your field names. Check
against `wisp/options.py` as your source of truth. (Or pass in `-h` for help).

## What is "wisp"?

<img src="media/wisp.jpg" alt="drawing" height="300"/>

Our library is named after the atmospheric ghost light, will-o'-the-wisp, 
which are volumetric ghosts that are harder to model with common standard 
geometry representations like meshes. We provide a [multiview dataset](https://drive.google.com/file/d/1jKIkqm4XhdeEQwXTqbKlZw-9dO7kJfsZ/view) of the 
wisp as a reference dataset for a volumetric object. 
We also provide the [blender file and rendering scripts](https://drive.google.com/drive/folders/1Via1TOsnG-3mUkkGteEoRJdEYJEx3wgf?usp=sharing) if you want to generate specific data with this scene, please refer to the [readme.md](https://drive.google.com/file/d/1IrWKjxxrJOlD3C5lDYvejaSXiPtm_XI_/view?usp=sharing) for greater details on how to generate the data. 

## Thanks

We thank James Lucas, Jonathan Tremblay, Valts Blukis, Anita Hu, and Nishkrit Desai for giving us early feedback
and testing out the code at various stages throughout development. 
We thank Rogelio Olguin and Jonathan Tremblay for the Wisp reference data. 
