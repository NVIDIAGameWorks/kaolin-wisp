# Config System

The wisp configuration system aims for a flexible, minimalist api through building on tyro and hydra-zen.

The following example illustrates the usage:

```python
from wisp.config import configure, autoconfig, parse_config, instantiate
from wisp.models.nefs import NeuralRadianceField
from wisp.models.grids import HashGrid, TriplanarGrid
import torch
from typing import Union, Tuple

# For classes and functions which don't use python typing annotations,
# the best practice is to create a config class, which specifies which args are configurable
# and what their types are:
@configure(target=torch.optim.Adam)     # This config can build torch.optim.Adam
class ConfigAdam:
    lr: float
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

@configure(target=torch.optim.RMSprop)  # This config can build torch.optim.RMSprop
class ConfigRMSprop:
    lr: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0

# @configure without a target is equivalent to a @dataclass.
@configure
class AppConfig:
    """  AppConfig is a dataclass holding all configuration fields. """
    # autoconfig() scans the classes for all available constructors, and generate a config dataclass for each.
    # This is useful for classes like grids, which have many types and constructors.
    # The resulting configs are similar to ConfigAdam and ConfigRMSprop above.
    # autoconfig() requires functions to annotate args with typings.
    grid: autoconfig(TriplanarGrid, HashGrid)    # type: Union[ConfigTriplanarGrid, ConfigHashGrid, ConfigHashGridFromGeometric, ...]
    
    # You can also use autoconfig() with a single class or function
    nerf: autoconfig(NeuralRadianceField)        # type: ConfigNeuralRadianceField
    
    # Explicit config set. This is useful, i.e., because the type annotations of torch are incomplete.
    # We specify we want to use the config classes we manually defined above with @configure(target=...)
    optimizer: Union[ConfigAdam, ConfigRMSprop] 
    
# parse_config will read arg values from the CLI to create an AppConfig instance.
# If `--config <path>.yaml` is given, it will also read values from it.
cfg = parse_config(AppConfig, yaml_arg='--config')

grid = instantiate(cfg.grid)                # Build grid instance directly from config
nerf = instantiate(cfg.nerf, grid=grid)     # Build nerf instance from config and grid
optimizer = instantiate(cfg.optimizer, params=nerf.decoder.parameters())      # Build optimizer from config
```

Running the program above and specifying args from CLI:
```shell
python main.py grid:HashGrid.from-geometric optimizer:adam --lr 0.001 --optimizer.eps 0.000001
```

or with a config file:
```yaml
grid:
   constructor: 'HashGrid.from_geometric'
   ...
nerf:
   ...
optimizer:
   constructor: 'Adam'
   lr: 1e-3
   eps: 1e-8
```
