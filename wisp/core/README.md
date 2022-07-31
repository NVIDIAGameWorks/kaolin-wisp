# Wisp Core

This module contains some important key constructs of wisp that is used throughout the code.

## Rays 

The `Rays` is a dataclass (a fancy Python struct) which holds ray origins, directions, and near / far planes. This struct exists because keeping track of all of these objects can quickly become unwieldly.

To create `Rays`:
```python
import torch
from wisp.core import Rays
rays = Rays(origins=torch.zeros(100, 3), dirs=torch.zeros(100, 3), dist_min=0.0, dist_max=10.0)
```

## RenderBuffer

The `RenderBuffer` is a dataclass (a fancy Python struct) which holds a collection of heterogeneous image buffer objects which can represent a useful information like RGB color, depth, alpha, or whatever else you want. This is used as a convenient transaction format around the `tracers` and `renderers` and follow the `channel` model in the codebase naturally. 

This dataclass comes with many convenience functions which will allow you to manipulate multiple buffers with minimal code. You can even save `RenderBuffer` objects as EXR files for analysis.

To create a `RenderBuffer`:
```python
import torch
from wisp.core import RenderBuffer
h, w = (1080, 1920)
rb = RenderBuffer(rgb=torch.zeros(h,w,3), depth=torch.zeors(h,w,3)) 
```
The default channels of a render buffer are `rgb`, `depth`, and `alpha`. You can access them like:
```python
function_that_uses_depth(rb.depth)
rb.rgb = function_that_outputs_rgb()
```
You can also easily add _your own_ channels:
```python
rb = RenderBuffer(rgb=torch.zeros(h,w,3), depth=torch.zeors(h,w,3), my_channel=torch.zeros(h,w,16)) 
```
These custom channels can be accessed in the same ways:
```python
function_that_uses_my_custom_channel(rb.my_channel)
```
and you can save them quickly as EXR files:
```python
import wisp.ops.image as img_ops
img_ops.write_exr("path/to/out.exr", rb.exr_dict())
```
The `RenderBuffer` object acts like a Python list. So you can use addition to concatenate together multiple `RenderBuffer`s:
```python
rb0 = RenderBuffer(rgb=torch.zeros(16,16,3))
rb1 = RenderBuffer(rgb=torch.zeros(16,16,3))
rb0 + rb1 # will have shape (32,16,3)
```
