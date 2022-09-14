# Wisp Library

This is the main folder for the wisp library. The folders can be divided up into **framework** (core components and functions to make everything work) and **building blocks**.

## Framework

`core` contains core primitives and structs that are used everywhere throughout the code. The `Rays` class and the `RenderBuffer` are particularly important classes to be aware of. 

`csrc` contains C++ and CUDA kernels which are used in the code.

`framework` contains state objects and event handlers to interface the core library with apps (like the renderer).

`ops` contains useful helper functions.

`renderer` contains everything that is used to build the interactive visualizer.

`config_parser.py` contains the parser for the configs and the command-line interface (CLI) arguments.

## Building Blocks

`accelstructs` contains classes that can be used as spatial acceleration structures for fast raytracing and query.

`datasets` contains classes for PyTorch datasets.

`models` contains various objects like MLPs and grids which have parameters that can be optimized. 

`tracers` contains classes that can be used as **forward maps**, which are functions which map neural fields onto another domain. Currently this mostly contains mechanisms of tracing rays against neural fields.

`trainers` contains classes that are designed per tasks to actually train, log, save different workloads.

## The Full Monty

The architecture figure describes how the major Wisp components relate to each other:

<img src="../media/wisp_architecture.jpg" alt="Wisp Architecture" width="1000"/>

See also further descriptions accompanying the [Templates](../templates/) folder.
