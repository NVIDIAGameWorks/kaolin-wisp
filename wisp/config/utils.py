# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Union, Type, TYPE_CHECKING, List, Callable, Any, Optional
import dataclasses


""" A module for the public api of wisp's config system. A common use of the various methods is as follows:
    
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
    
    @configure  # @configure without a target is equivalent to a @dataclass.
    class AppConfig:
        # AppConfig is a dataclass holding all configuration fields. 
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
"""


def autoconfig(*classes_and_callables: Type, exclude: List[Callable] = None) -> Any:
    """ Generates a list of Config dataclasses for each of the classes or functions (i.e. specific constructors).
    The class constructors / callables must be type annotated for this function to succeed.
    Otherwise, see configure().

    Specifically, this function will:
        1. Inspect the given classes in classes_and_callables and extract all constructors found.
        2. Add specific functions from classes_and_callables to the list of configs to generate.
        3. Build a config dataclass for each constructor variant, using hydra_zen.
           The dataclass fields should mirror the constructor's arguments.
           Types not supported by the config are recorded an internal field and removed.
        4. Annotate each config with a CLI subcommand. Later, these subcommands are used by tyro to choose which
           config class to load, among this generated set.
        5. If more than one config class is generated, this function returns a Union of configs.
           The configuration options are determined by the contents of the union.

    A constructor in wisp is assumed to be simplistic, and defined as either:
    (1) __init__ explicitly defined within the class body.
    (2) A @classmethod that is annotated as returning the same class type, see example below.
    Other cases, i.e. __new__ keyword, are not explicitly supported.


    Role of this function in config lifecycle:
        - Instances of these dataclasses can be populated by calling parse_config(),
        the CLI / reading config yaml contents.
        - Config dataclasses can be directly converted to actual instances with instantiate().

    Usage example:
        ```
            class Foo:
                def __init__(): ...
                @classmethod def from_X() -> Foo
            class Bar:
                def __init__(): ...
                @classmethod def from_X() -> Bar
                @classmethod def from_Y() -> Bar


            FooBarConfig = autoconfig(Foo, Bar.from_Y)      # Typedef: Union[ConfigFoo, ConfigFooFromX, ConfigBarFromY].
            cfg = parse_config(FooBarConfig)                # Parse args from the CLI and yaml,
                                                            # e.g. tyro.cli is used to read from CLI one of these types
            FooOrBarInstance = instantiate(cfg)             # Creates Foo or Bar instance.
                                                            # e.g. hydra_zen.instantiate is used to build the object
                                                            # from the config
        ```

        A convenient pattern is to use autoconfig within a larger dataclass config.
        The actual config class used can be chosen by the CLI, or config yaml, using abbreviated subcommand names,
        for example:

        ```
            @dataclass
            class AppConfig:
                foobar: autoconfig(Foo, Bar.from_Y)         # type: Union[ConfigFoo, ConfigFooFromX, ConfigBarFromY]

            cfg = parse_config(AppConfig)                   # This will call tyro.cli(AppConfig)
        ```
        cfg can be any of the config types: ConfigFoo, ConfigFooFromX, ConfigBarFromY.
        To choose them via CLI, one can specify: foobar:Foo, foobar:Foo.from-x, foobar:Bar.from-y
        Here underscores in constructor names are replaced with dashes.
        Specific constructor arguments can be passed as --arg1 --arg2.

        To choose them via a config yaml, one can specify:
        ```
        foobar:
            constructor: 'Foo.from_x'
            ...
            args1: value    # args that Foo.from_x takes
            arg2: value
            ...
        ```

    Tip:
        Running the script with --help will trigger tyro to print the available configs.

    Args:
        *classes_and_callables (Type): Variable list of types for configuration. Supports the following:
            - classes: Wisp will scan the class for constructions methods, and generate a config dataclass for each.
            - constructors / functions: Specific construction methods can also be stated.
        exclude (List[Callable]): If specified, particular classes or constructors will be omitted from
            the returned configs. exclude can be used to filter specific constructors when generating configs for entire
            class types.

    Returns:
        - (Type) If classes_and_callables yield a single target constructor,
          returns a single dynamic dataclass config type for this target constructor.
        - (typing.Union[Type, *] If classes_and_callables yield multiple target constructors,
          returns a union of dynamic dataclass config types for each target constructor.
    """
    import wisp.config._hydrazen as hydrazen_parser
    import wisp.config._tyro as tyro_parser

    if TYPE_CHECKING:
        from hydra_zen.typing import Builds
        # Enable static code analysis
        # Note: Callables are not handled here. This line assumes classes_and_callables contains
        # only classes. To support callables, we need to retrieve the classes callables belong to.
        return Builds[Type[Union.__getitem__(tuple(classes_and_callables))]]
    else:
        # Runtime: generate actual dynamic dataclasses for each class constructor and callable here
        config_dataclasses = hydrazen_parser.configs_for(*classes_and_callables, exclude=exclude)

        # Before returning the generated configs, we annotate them into a convenient subcommand format, i.e:
        # config class: `ConfigClassNameConstructorName` --> subcommand: `ClassName.constructor-name`
        # This allows CLI to specify the config choice via subcommands in our specified format
        if len(config_dataclasses) == 1:  # For a single callable, return the config class type.
            return tyro_parser.annotate_subcommand(config_dataclasses[0])
        else:  # If a class(es) of multiple callables, or multiple callables were specified, return the union of configs
            annotated_dataclasses = []
            for config_type in config_dataclasses:
                annotated_dataclasses.append(tyro_parser.annotate_subcommand(config_type))
            return Union.__getitem__(tuple(annotated_dataclasses))


def configure(cls=None, /, *, target: Callable[..., Any] = None, import_error: str = None):
    """ @configure decorates a given dataclass type, cls, as a configuration class that instantiates the target type.
        Use this function when configuring non-typed constructors, for example:

        ```
        @configure(target=torch.optim.Adam)  # This config can build torch.optim.Adam
        class ConfigAdam:
            lr: float
            betas: Tuple[float, float] = (0.9, 0.999)
            eps: float = 1e-8
        ```

        torch.optim.Adam.__init__ doesn't define type annotations for the args, but we can still support building
        this object by explicitly stating the types in this config.

        Essentially, this config says: these are the args I want to configure via CLI / yaml.
        By decorating with @configure, we can later call:
        optimizer = instantiate(ConfigAdam, params=params) to obtain an instance of torch.optim.Adam on the parameters
        of some trainable model.

        If target is None, @configure builds nothing, and simply acts as an alias to @dataclass.

        Role of this function in config lifecycle:
            - Instances of these dataclasses can be populated by calling parse_config(),
            the CLI / reading config yaml contents.
            - Config dataclasses can be directly converted to actual instances with instantiate().

        Usage example:
            ```
            @configure(target=torch.optim.Adam)                 # This config can build torch.optim.Adam
            class ConfigAdam:
                lr: float
                betas: Tuple[float, float] = (0.9, 0.999)
                ...

            @configure(target=torch.optim.RMSprop)              # This config can build torch.optim.RMSprop
            class ConfigRMSprop:
                lr: float = 1e-2
                alpha: float = 0.99
                ...

            @dataclass
            class AppConfig:
                optimizer: Union[ConfigAdam, ConfigRMSprop]       # Explicit config set

            cfg = parse_config(AppConfig)                # Parse args from the CLI and yaml to populate cfg
            optimizer = instantiate(cfg.optimizer)             # Creates Adam or RMSprop instance
            ```

            To choose an optimizer via CLI, one can specify: optimizer:adam, optimizer:rmsprop
            Specific constructor arguments can be passed as --lr --optimizer.alpha.

            To choose an optimizer via a config yaml, one can specify:
            ```
            optimizer:
                constructor: 'Adam'
                ...
                lr: 1e-2
                eps: 1e-8
                ...
            ```

        Tip:
            Running the script with --help will trigger tyro to print the available configs.

        Args:
            cls (Type): A simple class, whose fields correspond in name to the configurable args in the target
                (default constructor if target is a class). Users should specify the typing of args to allow
                the configuration system to initialize this object. Default values are not mandatory, and are
                only used when a value is not stated through the CLI / config yaml.
                If cls is missing args that exist in the target's function, default values will be used from the
                target function. Otherwise, instantiating the object will fail due to missing required args.
            target (Callable[..., Any]):
                A target class or constructor to build with the decorated config class.
                Constructors in wisp are assumed to simplistic, and defined as either:
                (1) __init__ explicitly defined within the class body. By default, this is used for target classes.
                (2) A @classmethod that is annotated as returning the same class type.
            import_error (str):
                An optional import error to show in case the target class is missing.
        """
    def _process_class(cls):
        """ Execute the wrapper logic: replace this class with a proper config class which is also compatible with
        instantiate() and parse_config().
        """
        import docstring_parser
        import wisp.config._hydrazen as hydrazen_parser
        import wisp.config._tyro as tyro_parser
        if target is not None:
            config_fields = dict()
            if cls is not None:
                config_fields.update(cls.__annotations__)   # Tell hydra-zen what typed args cls has
            wrapper = hydrazen_parser.build_config_for_target(target=target, config_fields=config_fields,
                                                              import_error=import_error)
            is_annotate_subcommand = True
        else:
            # No target, so defer to act as a @dataclass wrapper
            wrapper = dataclasses.dataclass
            is_annotate_subcommand = False

        config_class = wrapper(cls)

        # Create docstring (or shortened version of it) for help
        if target is not None:
            doc = None
            if target.__doc__ is not None:
                doc = docstring_parser.parse(target.__doc__).short_description
            if not doc:
                doc = f'Builds {target.__name__}.'
            config_class.__doc__ = doc

        if is_annotate_subcommand:  # Make it CLI / yaml compatible
            config_class = tyro_parser.annotate_subcommand(config_class)

        return config_class

    if cls is None:
        return _process_class          # Support @configure(), @configure(target=...) with parenthesis.
    else:
        return _process_class(cls)     # Support @configure without parenthesis.


def instantiate(config, **kwargs):
    """
    Builds an object from a config dataclass.
    Given a config dataclass defined with @configure or autoconfig, and populated with values from CLI / yaml
    with parse_config, instantiate will invoke the constructor of the target and pass the arg values kept in the config.

    A common pattern is to instantiate a hierarchy of objects.
    In this case, instantiate can be used to build each level of this hierarchy, where the inner object is passed
    as a kwarg, for example:
    ```
    # NeuralRadianceField contains a grid, and grid contains a blas
    blas = instantiate(cfg.blas)
    grid = instantiate(cfg.grid, blas=blas)     # If grid doesn't need blas, it will be ignored
    nef = instantiate(cfg.nef, grid=grid)       # If nef doesn't need grid, it will be ignored
    ```
    Note that instantiate will not fail for grid variants that don't use the blas arg: it will be silently ignored,
    as if the target constructor signature had a **kwargs absorbing excessive args.
    This makes the main scripts shorter and flexible, at the cost of strictly matching the instantiated
    object target signature.

    Args:
        config: An instance of a config class.
            config dataclasses are defined with @configure or autoconfig, and populated with values from CLI / yaml
            with parse_config
        **kwargs: If specified, may contain:
            1. Additional args to pass to the invoked target, not contained in the config class.
            This is useful, i.e., for building a hierarchy of composed objects, see example above.
            2. Override values for args defined in the config class.

    Returns:
        - (object) A new instance of the target buildable by the config class.
    """
    from hydra_zen import instantiate, is_partial_builds
    import wisp.config._hydrazen as hydrazen_parser

    try:
        hydrazen_parser.get_target(config)
    except TypeError as e:
        raise TypeError(f'config dataclass cannot be instantiated: {config}. Make sure that:\n'
                        f'1. This config class was defined with @configure or autoconfig'
                        f'2. parse_config() was used to build this config object.') from e

    # Partial build means the config doesn't contain all the args required to build the object.
    is_partial_build = is_partial_builds(config)
    available_args = hydrazen_parser.get_supported_args(config)
    remaining_args = hydrazen_parser.get_missing_args(config)

    overriden_args = {k: v for k, v in kwargs.items() if k in available_args}
    instance = instantiate(config, **overriden_args)

    # In this case, hydra zen instantiates a functools.partial and we have to invoke
    # the returned partial callable again with the remaining args.
    if is_partial_build:
        args_for_partial = {k: v for k, v in kwargs.items() if k in remaining_args}
        instance = instance(**args_for_partial)

    return instance


def parse_config(config_type, yaml_arg: Optional[str] = '--config'):
    """ This function will:
    1. Parse args from the CLI and optional config yaml path.
    2. Create and populate an instance of the config dataclass type.

    Usage example:
        ```
        @dataclass
        class AppConfig:
            grid: autoconfig(TriplanarGrid, HashGrid)         # type: Union[ConfigTriplanarGrid, ConfigHashGrid, ConfigHashGridFromGeometric, ...]
            nerf: autoconfig(NeuralRadianceField)             # type: ConfigNeuralRadianceField
            optimizer: Union[ConfigAdam, ConfigRMSprop]       # Explicit config set. This is useful, i.e., because the
                                                              # type annotations of torch are not 100% reliable
        cfg = parse_config(AppConfig, yaml_arg='--config')
        # python main.py --config my_config.yaml --arg1
        # cfg.grid, cfg.nerf, cfg.optimizer are now filled with values populated from my_config.yaml and arg1
        ```

        The CLI can with this function as follows:
        ```
        Use just the config:
        > python main.py --config my_config.yaml

        Print help:
        > python main.py --help

        Use the config and override dataset path:
        > python main.py --config my_config.yaml --dataset-path data/lego/

        Use the config with an arg that is used by more than one field (i.e. disambiguate from tracer.bg-color):
        > python main.py --config my_config.yaml --dataset.dataset-path data/lego/

        Select a different optimizer variant:
        > python main.py --config app/nerf/configs/nerf_hash.yaml optimizer:adam

        Select a different optimizer and grid variants:
        > python main.py --config app/nerf/configs/nerf_hash.yaml optimizer:adam grid.TriplanarGrid:
        ```

    The priority of args is determined by:
    1. Args specified through CLI.
    2. Args specified through config yaml.
    3. Defaults defined in config dataclass.

    Args:
        config_type (type): The type for the config object.
        yaml_arg (str): Name of config path arg. By default this is --config. Expected to start with --.

    Returns:
        (dataclass): Config dataclass instance, initialized with parsed args.
    """
    from ._tyro import parse_args_tyro
    return parse_args_tyro(config_type, yaml_arg)


def print_config(config, prefix=""):
    """Prettyprint the config dataclass object.

    Args:
        config (dataclass): Dataclass config object.
        prefix (Optional[str]): If a base level indentation is desired, you can pass in a string.
    """
    if hasattr(config, '__ctor_name__'):
        field_name = 'constructor'
        ctor = getattr(config, '__ctor_name__')
        print(f"{prefix}{field_name}: {ctor}")
    for field in dataclasses.fields(config):
        field_obj = getattr(config, field.name)
        if dataclasses.is_dataclass(field_obj):
            print(f"{prefix}{field.name}")
            print_config(field_obj, prefix=prefix + "  ")
        else:
            if field.name.startswith('_') or field.name.endswith('_'):
                continue
            print(f"{prefix}{field.name}: {field_obj}")


def write_config_to_yaml(config, path):
    """Write config to path as a yaml.

    write_config_to_path(config_object, "config.yaml")

    Args:
        config (dataclass): Dataclass config.
        path (str): Path to write the config file to.
    """
    from ._tyro import write_config_to_yaml
    write_config_to_yaml(config=config, path=path)


def get_config_target(config):
    """For config dataclasses generated with autoconfig() or @configure (or hydra-zen in general),
    this function will return the target type this config constructs when calling instantiate().

    If config is not a dataclass generated with autoconfig(), @configure or hydra-zen, a TypeError
    is raised.

    Args:
        config (dataclass): Dataclass config.
    """
    from hydra_zen import get_target
    try:
        return get_target(config)
    except TypeError as e:
        raise TypeError(f'get_config_target recieved a configuration class not generated with autoconfig(), @configure,'
                        f' or hydra-zen in general: {config}. This means this config class does not instantiate'
                        f' any target object.') from e
