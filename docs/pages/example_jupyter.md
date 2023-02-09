# Jupyter Notebook

Starting with wisp 0.1.2, interactive rendering is supported via Jupyter Notebook.

Jupyter supports remote rendering, e.g. does not require a display.


## Prerequisites

Jupyter notebook and ipycanvas are required:

```bash
pip install notebook
pip install ipycanvas
```

## Running the Demo

The demo allows to interactively visualize a pretrained pipeline.

For example, to obtain a trained NeRF, run:
```bash
cd kaolin-wisp
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_hash.yaml --dataset-path /path/to/lego --max_epochs 100 --save-every 100
```

See {doc}`NeRF App <app_nerf>` page for further details about training such pipeline.

Once the trained pipeline is saved, note the saved path and follow the instructions within the notebook.


## Comparison to desktop interactive visualizer

Jupyter-notebook supports interactive rendering features which don't require OpenGL.
This allows remote rendering at the expanse of a reduced feature set.

The desktop interactive renderer requires OpenGL support and a display. 
It packs additional features such as: a gui to review the neural field properties and optimization progress,
an interactive scene graph, and vector data layers to display debug information.