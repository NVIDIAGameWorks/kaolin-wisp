# NeRF with Traced Latents (SIGGRAPH 2022 Demo)

This example shows how to add a new type of **Neural Field** to wisp.
Specifically, the newly added Neural Field exposes some of its latent channels to the tracer.
To conclude the demo, an interactive demo app which renders the latent color channels is created.

A full video walkthrough is [available here](https://www.nvidia.com/en-us/on-demand/session/siggraph2022-sigg22-s-14/?playlistId=playList-92d9241d-6d4c-4fc7-88f6-eb8484008787).


## Overview

<img src="../../media/example_latentnerf.jpg" alt="Latent NeRF" width="750"/>

## Running the Demo

The simplest configuration uses an Octree grid: 
```
cd /examples/latent_nerf
python3 main_demo.py --config funny_nerf_octree.yaml --dataset-path /path/to/nerf_lego --multiview-dataset-format standard
```

You can also experiment with a faster, alternative configuration which uses the Hashtable:
```
cd /examples/latent_nerf
python3 main_demo.py --config funny_nerf_hash.yaml --dataset-path /path/to/nerf_lego --multiview-dataset-format standard
```

Make sure to use the appropriate `multiview-dataset-format` according to your data type.

## Files

`funny_neural_field.py` contains:
* The implementation of the custom `SigDecoder`, with a specialized forward function that exposes 3 latent channels.
* The initialization of the feature grid, positional encoding, and decoders.
* 2 forward functions for the tracer to invoke: `rgba()` and `color_feature()`. The example shows how to register these.

`demo_app.py` is the interactive app which visualizes the optimization process.
It includes additional logic for defining the new `color_feature` channel.

`funny_nerf_octree.yaml` and `funny_nerf_hash.yaml` are the configuration files used to run this example.

`main_demo,py` is the main script for registering the new neural field class and running the interactive demo.
