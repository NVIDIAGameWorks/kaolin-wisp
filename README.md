# Kaolin Wisp: A PyTorch Library and Engine for Neural Fields Research

<img src="docs/_static/media/demo.jpg" alt="drawing" width="1000"/>

NVIDIA Kaolin Wisp is a PyTorch library powered by [NVIDIA Kaolin Core](https://github.com/NVIDIAGameWorks/kaolin) to work with
neural fields (including NeRFs, [NGLOD](https://nv-tlabs.github.io/nglod), [instant-ngp](https://nvlabs.github.io/instant-ngp/) and [VQAD](https://nv-tlabs.github.io/vqad)).

NVIDIA Kaolin Wisp aims to provide a set of common utility functions for performing research on neural fields. 
This includes datasets, image I/O, mesh processing, and ray utility functions. 
Wisp also comes with building blocks like differentiable renderers and differentiable data structures 
(like octrees, hash grids, triplanar features) which are useful to build complex neural fields. 
It also includes debugging visualization tools, interactive rendering and training, logging, and trainer classes.

**[Check our docsite for additional information!](https://kaolin-wisp.readthedocs.io/en/latest/pages/main.html)**

For an overview on neural fields, we recommend you check out the EG STAR report: 
[Neural Fields for Visual Computing and Beyond](https://arxiv.org/abs/2111.11426).

## Latest Updates

**wisp 1.0.3** <-- `main`
* _17/04/23_ The configuration system have been replaced! Check out [the config page](https://kaolin-wisp.readthedocs.io/en/latest/pages/config_system.html) for usage instructions and backwards compatability (**breaking change**). Note that the wisp core library remains compatible, mains and trainers should be updated.

**wisp 1.0.2** <-- `stable`
* _15/04/23_ Jupyter notebook support have been added - useful for machines without a display.
* _01/02/23_ `attrdict` dependency added as part of the new datasets framework. If you pull latest, make sure to `pip install attrdict`.
* _17/01/23_ `pycuda` replaced with `cuda-python`. Wisp can be installed from pip now  (If you pull, run **pip install -r requirements_app.txt**)
* _05/01/23_ Mains are now introduced as standalone apps, for easier support of new pipelines (**breaking change**)  

## Installation

See installation instructions [here](https://kaolin-wisp.readthedocs.io/en/latest/pages/install.html).


## External Contributions

We welcome & encourage external contributions to the codebase!
For further details, read the [FAQ](https://kaolin-wisp.readthedocs.io/en/latest/pages/faq.html) and [license page](https://kaolin-wisp.readthedocs.io/en/latest/pages/license.html).

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

## Thanks

We thank James Lucas, Jonathan Tremblay, Valts Blukis, Anita Hu, and Nishkrit Desai for giving us early feedback
and testing out the code at various stages throughout development. 
We thank Rogelio Olguin and Jonathan Tremblay for the Wisp reference data. 

Special thanks for community members:
* [lightfield botanist](https://github.com/3a1b2c3)
* [Soumik Rakshit](https://github.com/soumik12345)


## What is "wisp"?

<img src="docs/_static/media/wisp.jpg" alt="drawing" height="300"/>

Our library is named after the atmospheric ghost light, will-o'-the-wisp, 
which are volumetric ghosts that are harder to model with common standard 
geometry representations like meshes. We provide a [multiview dataset](https://drive.google.com/file/d/1jKIkqm4XhdeEQwXTqbKlZw-9dO7kJfsZ/view) of the 
wisp as a reference dataset for a volumetric object. 
We also provide the [blender file and rendering scripts](https://drive.google.com/drive/folders/1Via1TOsnG-3mUkkGteEoRJdEYJEx3wgf?usp=sharing) if you want to generate specific data with this scene, please refer to the [readme.md](https://drive.google.com/file/d/1IrWKjxxrJOlD3C5lDYvejaSXiPtm_XI_/view?usp=sharing) for greater details on how to generate the data. 

