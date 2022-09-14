# Wisp Models

This folder mostly contains building blocks for neural fields, which usually have trainable parameters.

## What is a Neural Field?

We call these _neural fields_ but really they can be any differentiable and parameteric function that takes in coordinates as input and outputs some channel. Hence, you can use just a `grid` or just a `decoder` as long as you implement the channel functions which is explained in the next section.

## Building your own Neural Field 

<img src="../../media/nef.jpg" alt="Wisp's Neural Field" width="750"/>

The intent of the building blocks is to make it easier for users to create their own neural fields models. Users are free to pick and choose whatever modules they find useful from these modules, especially if the intent is to integrate these building blocks in to your own existing pipelines.

If you wish to interface with the rest of the `wisp` framework, however, you will need to create your own `NeuralField` class which inherits from `BaseNeuralField`. Compatibility between various different kinds of `NeuralField` models and the rest of the pipeline like the `tracer`, `renderer`, `trainer` is maintained by registering forward functions.

Your own neural field class might look like this:
```python
from wisp.models.nefs import BaseNeuralField

class MyNeuralField(BaseNeuralField):
    def init_decoder(self):
        # The trainer finds decoder parameters by finding "decoder" in the parameter name.
        # So make sure you put "decoder" somewhere if you want the trainer to use the decoder specific options.
        self.semantics_decoder = MyDecoder()
        self.rgb_decoder = MyDecoder()

    def init_grid(self):
        # The trainer finds decoder parameters by finding "grid" in the parameter name.
        # So make sure you put "grid" somewhere if you want the trainer to use the grid specific options.
        self.shared_grid = MyGrid()

    def register_forward_functions(self):     
        # This function tells the BaseNeuralField class what channels exist for this Neural Field   
        self._register_forward_function(self.semantics, ["semantics"])
        self._register_forward_function(self.rgb, ["rgb"])
    
    def semantics(self, coords):
        return self.semantic_decoder(self.shared_grid(coords))
    
    def rgb(self, coords, ray_d):
        return self.rgb_decoder(torch.cat([self.shared_grid(coords), ray_d], -1))
```

Then, you can simply run these forward functions by using the `forward()` interface from the base class:
```python
nef = MyNeuralField()
semantics = nef(coords=coords, channels="semantics")
rgb, semantics = nef(coords=coords, ray_d=ray_d, channels=["rgb", "semantics"])
channel_dict = nef(coords=coords, ray_d=ray_d, channels=set("rgb"))
rgb = channel_dict["rgb"]
```

