# Wisp Models

This folder mostly contains building blocks for neural fields, which usually have trainable parameters.

## What is a Neural Field?

In wisp, a **neural field** (**nef**) is commonly referred to as a combination of a feature grid structure, a decoder, and an optional input embedding.
Which channels are supported is decided by implementing and registering forward functions.

Note that Wisp's definition of _neural fields_ is somewhat non-restrictive: in practice nefs can be any differentiable and parameteric function that takes in coordinates as input and outputs some channel. Hence, you can use just a `grid` or just a `decoder` as long as you implement the channel functions which is explained in the next section.

## Building your own Neural Field 

<img src="../../media/nef.jpg" alt="Wisp's Neural Field" width="750"/>

The intent of the building blocks is to make it easier for users to create their own neural fields models. Users are free to pick and choose whatever modules they find useful from these modules, especially if the intent is to integrate these building blocks in to your own existing pipelines.

If you wish to interface with the rest of the `wisp` framework, however, you will need to create your own `NeuralField` class which inherits from `BaseNeuralField`. Compatibility between various different kinds of `NeuralField` models and the rest of the pipeline like the `tracer`, `renderer`, `trainer` is maintained by registering forward functions.

Your own neural field class might look like this:
```python
from wisp.models.nefs import BaseNeuralField

class MyNeuralField(BaseNeuralField):
    
    def __init__(self, grid, rgb_decoder_params, semantic_decoder_params):
        """ Pass whatever components your neural field needs here """
        # Grid is a BLASGrid instance, like: OctreeGrid, HashGrid, and so forth.
        # Create it outside of the neural field class and pass the instance in as a parameter.
        self.grid = grid
        
        # Your decoders can be initialized as you see fit.
        self.rgb_decoder = self.init_decoder(rgb_decoder_params)
        self.semantics_decoder = self.init_decoder(semantic_decoder_params)
    
    def init_decoder(self, decoder_params):
        # MyDecoder is your decoder class.
        # Wisp includes some options already, and you're also free to use your own custom decoders here.
        # decoder_params is any meaningful argument needed to initialize your decoder.
        return MyDecoder(**decoder_params)

    def register_forward_functions(self):     
        # This function tells the BaseNeuralField class what channels exist for this Neural Field
        # By registering forward funcs, the tracer can connect with the neural field to collect values for samples.
        self._register_forward_function(self.semantics, ["semantics"])
        self._register_forward_function(self.rgb, ["rgb"])
    
    def semantics(self, coords, ray_d=None, lod_idx=None):
        # Forward function for semantics.
        features = self.grid(coords).interpolate(coords)
        return self.semantic_decoder(features)
    
    def rgb(self, coords, ray_d, lod_idx=None):
        # Forward function for rgb..
        features = self.grid(coords).interpolate(coords)
        decocder_input = torch.cat([features, ray_d], -1)
        return self.rgb_decoder(decocder_input)
```

Then, you can simply run these forward functions by using the `forward()` interface from the base class:
```python
nef = MyNeuralField()
semantics = nef(coords=coords, channels="semantics")
rgb, semantics = nef(coords=coords, ray_d=ray_d, channels=["rgb", "semantics"])
channel_dict = nef(coords=coords, ray_d=ray_d, channels=set("rgb"))
rgb = channel_dict["rgb"]
```

## Building Full Pipelines
The common use case for Neural Fields is to combine them with a `BaseTracer` subclass in a single `Pipeline` object.
In this case, the tracer object is responsible for marching / tracing rays, or quering samples around the scene.
The samples are then passed to the neural field class.

The exact input to forward functions is provided by `BaseTracer` implementation.
For example, a `PackedRFTracer`, which traces radiance fields, passes the input coordinates and ray direction.
Consequentially, Neural field implementations should be mindful of which tracers they're compatible with.