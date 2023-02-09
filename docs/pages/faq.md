# Frequently Asked Questions

## General

**Q: What is the difference between kaolin and kaolin-wisp?**

A: 
[kaolin](https://github.com/NVIDIAGameWorks/kaolin) is a 3d deep learning library in the broader sense.
It contains many useful operations for handling differentiable 3D graphics, like a mesh toolkit,
a cameras module, structured pointcloud (e.g. octrees), differentiable renderers, and more.
It builds on, and extends PyTorch with useful operations for handling general 3D structures.

kaolin-wisp (abbreviated as just "wisp"), is dedicated for neural fields research.
It uses [kaolin's](https://github.com/NVIDIAGameWorks/kaolin) components to build full pipelines


**Q: Why is the library called "wisp"?**

A: Our library is named after the atmospheric ghost light, will-o'-the-wisp, 
which are volumetric ghosts that are harder to model with common standard 
geometry representations like meshes. We provide a [multiview dataset](https://drive.google.com/file/d/1jKIkqm4XhdeEQwXTqbKlZw-9dO7kJfsZ/view) of the 
wisp as a reference dataset for a volumetric object. 
We also provide the [blender file and rendering scripts](https://drive.google.com/drive/folders/1Via1TOsnG-3mUkkGteEoRJdEYJEx3wgf?usp=sharing) if you want to generate specific data with this scene, please refer to the [readme.md](https://drive.google.com/file/d/1IrWKjxxrJOlD3C5lDYvejaSXiPtm_XI_/view?usp=sharing) for greater details on how to generate the data. 

<img src="../_static/media/wisp.jpg" alt="drawing" height="300"/>



**Q: Is kaolin-wisp affiliated with Instant-NGP?**

A: Both projects come from NVIDIA research labs, but are purposed differently.
[Instant-NGP](https://github.com/NVlabs/instant-ngp) is a well-optimized codebase, which relies on fused CUDA kernels for training high quality Neural Radiance Fields.
Wisp on the other hand takes a flex approach and puts emphasis on modularity and the ability to customize code.
As a design choice, Wisp relies on PyTorch, and where appropriate, CUDA kernels are used to optimize critical sections of the code.

If you're using NeRF as a black-box, [Instant-NGP](https://github.com/NVlabs/instant-ngp) is a great choice.

If you require customized code, e.g. "hacking" into the neural field or training structure, Wisp is the way to go.


**Q: About project's roadmap?**

A: Short-mid term Milestones are [logged on github](https://github.com/NVIDIAGameWorks/kaolin-wisp/milestones).


## Licensing & Contributions

**Q: Is kaolin-wisp an open source?**

A: Wisp is licensed under the NVIDIA Source Code License.
That means you are welcome, and encouraged to use it for research!
The exact terms can be reviewed on the {doc}`License <license>` page.

**Q: Are there any limitations to hosting my pipelines on kaolin-wisp's github?**

A: Not at all! We welcome new pipeline contributions, whether as apps or examples.
Credit will be given due on the README page included with the app.
Please include a short sample script which allows us to test your pipeline with our CI.

**Q: My project forks kaolin-wisp, can I still link it back?**

A: Sure, feel free to contact us on Github and we will link it back.

**Q: How do I contribute to the codebase?**

A: Contributions are expected in the form of [pull requests](https://github.com/NVIDIAGameWorks/kaolin-wisp/pulls).


## Common Issues

**Q: Does wisp support headless rendering on remote machines?**

A: If your machine is not equipped with a GPU or display, you may still run a simplified renderer via Jupyter Notebook.

[See example here](https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/examples/notebook/view_pretrained.ipynb).

The full interactive visualizer is limited to machines equipped with an NVIDIA GPU and a display.
The minimum requirements come from usage of OpenGL.

**Q: Can I run wisp on Windows?**

A: Windows support is provided by external contributions from the community. 
Note that our CI does not track Windows status.

**Q: Do I require a GPU to run wisp?**

A: Both kaolin and kaolin-wisp rely on CUDA kernels, which require an NVIDIA GPU.

**Q: I pulled the latest on `main` branch and now can't run the NeRF app training code.**

A: Since NeRF is moving at a very fast pace, our changes to the `main` branch may break backwards compatability from time to time.
When that happens, we always leave a note within the [main page](https://github.com/NVIDIAGameWorks/kaolin-wisp) (see: "Latest Updates").

In addition, also keep the `stable` branch up to date with the latest tagged version, to ensure users have a point of reference.


## Reproducibility

**Q: I'm unable to reproduce the results of paper X.**

A: The metrics obtained from wisp apps are provided in their respective README pages (see {doc}`NeRF <app_nerf>` for example).
Note that wisp is meant to be used as a library of components: we do not guarantee an identical reimplementation of papers,
The metrics obtained by our reference implementations should have comparable, but not identical metrics.


**Q: Are models trained with wisp compatible between versions?**

A: We track certain configurations and metrics with our CI, to ensure they don't degrade between versions.
If you spot something unusual, please [notify us on Github](https://github.com/NVIDIAGameWorks/kaolin-wisp/issues/new?title=Issue%20on%20page%20%2Fpages/main.html&body=Your%20issue%20content%20here.).