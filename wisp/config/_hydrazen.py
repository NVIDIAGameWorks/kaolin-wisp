# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from __future__ import annotations

import inspect
import enum
import copy
import typing
from typing import get_type_hints, Type, Callable, List, Optional, Any
from functools import lru_cache
import docstring_parser
from dataclasses import field
import hydra_zen
from hydra_zen import instantiate, builds, make_config, hydrated_dataclass
from hydra_zen.typing import Builds


"""
A module for generating dynamic config dataclasses from the application classes and functions.
hydra_zen is used by wisp to implement autoconfig(), possibly as a union of possible configuration dataclasses.  
"""

def generate_implicit_field_types(func):
    func_args = typing.get_type_hints(func)
    func_parameters = inspect.signature(func).parameters

    # If arg doesn't have explicit typing but is using a default value, use the default value to infer
    # an implicit arg type
    implicit_arg_types = dict()
    for arg_name, param in func_parameters.items():
        if arg_name not in func_args and param.default is not inspect._empty:
            implicit_arg_types[arg_name] = type(param.default)
    return implicit_arg_types


def generate_implicit_types_from_defaults(func):

    def _recursive_type_infer(_value):
        if _value == 0 and type(_value) is int:
            # values of 0 and 0.0 are always assumed to be floats
            return float
        elif type(_value) is tuple:
            arg_types = [_recursive_type_infer(subvalue) for subvalue in _value]
            return typing.Tuple.__getitem__(tuple(arg_types))
        else:
            return type(_value)

    func_args = typing.get_type_hints(func)
    func_parameters = inspect.signature(func).parameters
    implicit_types = dict()

    # If arg doesn't have explicit typing but is using a default value, use the default value to infer
    # an implicit arg type
    for arg_name, param in func_parameters.items():
        if arg_name not in func_args and param.default is not inspect._empty:
            implicit_types[arg_name] = _recursive_type_infer(param.default)

    return implicit_types


def generate_custom_dataclass_args(func):
    """Obtains a dataclass config of the function signature.
    This class is an alternative logic to hydra-zen builds(populate_with_signature=True, ..).
    The differences are:
    1) Removal of unsupported types like torch.Tensor is using our own filter
    2) The docstring of each parameter is passed to the dataclass field, which enables tyro to show it with --help

    Args:
        func (fn): Any function.

    Returns:
        (dict): A dictionary of field-names -> dataclass.Field, with typing and doc metadata.
        (list): Arguments with unsupported types, not included in the above mapping.
    """
    func_parameters = inspect.signature(func).parameters
    func_args = {k: v.annotation for k, v in func_parameters.items()}
    func_args.update(typing.get_type_hints(func))

    # We don't want to expose _everything_, just things that seem easily configurable.
    # We manually filter for such things.
    func_args.pop("self", None)
    func_args.pop("cls", None)
    func_args.pop("args", None)
    func_args.pop("kwargs", None)
    func_args.pop("return", None)
    keys_to_remove = []
    for key in func_args:
        if not check_valid_type(func_args[key]):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        func_args.pop(key, None)

    # Get the docstring to add to the CLI
    parsed_docs = {}
    parsed = docstring_parser.parse(func.__doc__)
    for param in parsed.params:
        parsed_docs[param.arg_name] = param.description.replace("\n", " ")

    dataclass_args = dict()
    for key in func_args:
        metadata = None
        docs = parsed_docs.get(key, None)
        if docs is not None:
            metadata = {"help": docs}

        field_dict = {'metadata': metadata}
        default = func_parameters[key].default
        if not (default is inspect._empty):
            field_dict['default'] = default
        dataclass_args[key] = field(**field_dict)

    # At this point, keys_to_remove contains all args with unsupported typings.
    # Fields with default values cannot be initialized via CLI / yaml, but their default values can still be used.
    # Fields without defaults are designated as excluded, users must pass those to instantiate the config.
    # excluded_fields = [k for k in keys_to_remove if func_parameters[k].default == inspect._empty]
    excluded_fields = keys_to_remove
    return dataclass_args, excluded_fields


def gather_constructors(cls):
    """ Returns all @classmethod constructors that a class may have.
    A constructor is defined as a @classmethod annotated as returning the class type.
    """
    classmethods = inspect.getmembers(cls, predicate=inspect.ismethod)
    initializers = [m[1] for m in classmethods if get_type_hints(m[1]).get('return') == cls]
    return initializers


@lru_cache(maxsize=None, typed=True)  # Make sure the same config is returned for this callable in the future
def _build_config_for_callable(cls: Type, func: Callable, use_manual_fields: bool) -> Builds[Type]:
    """
    This function takes a class + one of its designated constructor functions, and generates a dynamic config dataclass
    from it. The returned config class is wrapped with hydra-zen's `builds()` function, which allows to later
    instantiate objects directly from this config by callsing hydra-zen's `instantiate()`.

    This function also takes care of copying the docstring of the original func onto the dynamic config dataclass.

    Args:
        cls (Type): A class to build a single config dataclass for. Config will be created for its "func" function.
        func (Callable): A constructor method, either of the following:
            1. def __init__()
            2. @classmethod
               def _(cls, ) -> cls
            func is generally assumed to be an "unbound" method, to utilize the @lru_cache
            (@lru_cache cannot properly cache bound methods as their id may change between invocations).
            Passing a bound func method is supported, but the result may not be cached.
        use_manual_fields (bool):
            If True, config classes will be created from fields specifically filtered by wisp's
            config parser.
            This means wisp will take care of filtering unsupported annotation types (like torch.Tensor),
            and take care of copying over docstring of args to the config class.
            e.g. builds(populate_full_signature=False, ..., **wisp_dataclass_fields)
            If False, full control over the dataclass creation will be given to hydra-zen,
            e.g. see builds(populate_full_signature=True, ...)
            hydra-zen will alert with errors in case of unsupported annotation types, and will not copy docstrings
            by default.
    Returns:
        A custom dataclass wrapped with hydra-zen's build().
        This designates the returned dataclass as a config class which supports hydra-zen's instantiate().
        Calling instantiate on this return value invokes cls.func() with the dataclass fields as args,
        to return an initialized instance of cls.
    """

    def _set_docstring(_cfg, _cls, _func):
        """ Extract docs from original functions to paste over dynamic configs. This makes --help useful.
        This function operates only on the main dataclass docstring, the args are taken care of separately.
        """
        doc = None
        if _func.__doc__ is not None:
            doc = docstring_parser.parse(_func.__doc__).short_description
        if not doc:
            doc = f'Builds {_cls.__name__} with {_func.__name__} constructor.'
        _cfg.__doc__ = doc

    # Dynamic dataclass name is always Config<classname> or Config<ctor>
    if func == cls.__init__:
        # This config will have a build target of an __init__ function
        target = cls
        dynamic_config_class_name = f"Config{cls.__name__}"
        cmd_name = cls.__name__
    else:
        # This config will have a build target of a @classmethod func: cls.func.
        # The target in this case is expected to be a bound method to cls.
        # If func is a function, we re-bind it to cls to get a bound method.
        # If hasattr(func, '__func__') is false, func is already a bound method.
        target = func.__get__(cls) if not hasattr(func, '__func__') else func
        dynamic_config_class_name = ''.join(['Config' + cls.__name__] +
                                            [s.capitalize() for s in func.__name__.split('_')])
        cmd_name = f'{cls.__name__}.{func.__name__}'
    # If true, wisp will generate the dataclass args by itself, taking care of:
    # 1) Removal of unsupported types like torch.Tensor is using our own filter
    # 2) The docstring of each parameter is passed to the dataclass field, which enables tyro to show it with --help
    if use_manual_fields:

        try:
            implicit_types = generate_implicit_types_from_defaults(func)
            original_annotations = None
            if len(implicit_types) > 0:
                original_annotations = copy.deepcopy(func.__annotations__)
                func.__annotations__.update(implicit_types)

            configurable_fields, unsupported_args = generate_custom_dataclass_args(func)
            cfg = builds(target,
                         populate_full_signature=False,
                         zen_dataclass={'cls_name': dynamic_config_class_name, 'frozen': False},
                         zen_meta={'__ctor_name__': cmd_name,
                                   '__supported_args__': '$'.join(configurable_fields.keys()),
                                   '__missing_args__': '$'.join(unsupported_args)},  # Ignored by hydra-zen, used to identify this func name
                         zen_partial=len(unsupported_args) > 0,
                         hydra_convert="all",
                         **configurable_fields)
        finally:
            if original_annotations is not None:
                func.__annotations__ = original_annotations
    else:  # If False, let hydra zen take over completely and generate the dataclass from callable
        cfg = builds(target,
                     populate_full_signature=True,
                     zen_dataclass={'cls_name': dynamic_config_class_name, 'frozen': False},
                     zen_meta={'__ctor_name__': cmd_name},)  # Ignored by hydra-zen, used to identify this func name
    _set_docstring(cfg, cls, func)
    return cfg


def build_config_for_target(target: Callable[..., typing.Any], config_fields: typing.Dict[str, Type] = None,
                            import_error: str = None):

    if config_fields is None:
        config_fields = dict()

    if inspect.isclass(target):  # target is a class
        cls = target
        cmd_name = cls.__name__
    else:                        # target is a callable
        cls = inspect.getmodule(target).__dict__[target.__qualname__.split('.')[0]]  # Get the class
        cmd_name = f'{cls.__name__}.{target.__name__}'

    supported_args = config_fields.keys()
    unsupported_args = [arg for arg in inspect.signature(target).parameters.keys() if arg not in config_fields]

    config_class = hydrated_dataclass(
        target,
        populate_full_signature=False,
        frozen=False,
        zen_meta={'__ctor_name__': cmd_name,
                  '__supported_args__': '$'.join(supported_args),
                  '__missing_args__': '$'.join(unsupported_args),  # Ignored by hydra-zen, used to identify this func name
                  '__import_error__': import_error,
        },
        zen_partial=len(unsupported_args) > 0,
        hydra_convert="all"
    )
    return config_class


def build_config_for_callable(cls: Type, func: Callable, use_manual_fields: bool = True) -> Builds[Type]:
    if hasattr(func, '__func__'):
        # Make sure that callables are persistent between further calls of this methods with the same func.
        # _build_config_for_callable uses a @lru_cache decorator.
        # If func is a bound method, python may assign a different id each time the func reference is generated,
        # which results in a cache miss. Passing the unbound `function` type fixes this problem.
        func = func.__func__
    return _build_config_for_callable(cls, func, use_manual_fields)


def build_configs(cls: Type, exclude: List = None, use_manual_fields=True) -> List[Builds[Type]]:
    """ Given a class, gather all available constructors and build a hydra-zen config for each of them.
    Args:
            cls (Type): A class to build config dataclasses for. Configs will be created for all methods designated as
                initialization methods:
                1. def __init__()
                2. @classmethod
                   def _(cls, ) -> cls
            exclude (List[function, method]): List of methods or functions to omit from the configs.
            use_manual_fields:
                If True, config classes will be created from fields specifically filtered by wisp's
                config parser.
                This means wisp will take care of filtering unsupported annotation types (like torch.Tensor),
                and take care of copying over docstring of args to the config class.
                e.g. builds(populate_full_signature=False, ..., **wisp_dataclass_fields)
                If False, full control over the dataclass creation will be given to hydra-zen,
                e.g. see builds(populate_full_signature=True, ...)
                hydra-zen will alert with errors in case of unsupported annotation types, and will not copy docstrings
                by default.
    """
    configs = []
    exclude = [] if exclude is None else exclude
    if cls in exclude:  # Entire class is filtered
        return []
    for ctor in filter(lambda x: x not in exclude, gather_constructors(cls)):
        cfg = build_config_for_callable(cls=cls, func=ctor, use_manual_fields=use_manual_fields)
        configs.append(cfg)
    # TODO (operel): Classes that don't define an __init__ explicitly will not be gathered here
    if cls.__init__ not in exclude or len(configs) == 0:  # If no configs were found, include at least the default init
        cfg = build_config_for_callable(cls=cls, func=cls.__init__, use_manual_fields=use_manual_fields)
        configs.insert(0, cfg)
    return configs


def configs_for(*classes_and_callables: Type, exclude: List[Callable] = None) -> List[Builds[Type]]:
    """ Returns a list of config dataclasses for each of the classes or their constructors.
    Config dataclasses can be directly converted to actual instances with hydra-zen's instantiate().

    A constructor is either:
        (1) __init__ explicitly defined within the class body.
        (2) A @classmethod that is annotated as returning the same class type, see example below.

    Example:
            ```
            class Foo:
                def __init__(): ...
                @classmethod def from_X() -> Foo
            class Bar:
                def __init__(): ...
                @classmethod def from_X() -> Bar
                @classmethod def from_Y() -> Bar


            FooBarConfig = auto_config(Foo, Bar.from_Y)     # Typedef: Union[ConfigFoo, ConfigFooFromX, ConfigBarFromY].
            args = tyro.cli(FooBarConfig)                   # Use tyro to read from CLI one of these types
            instance = hydra_zen.instantiate(args)          # Returns Foo or Bar
        ```

    Args:
        *classes_and_callables (Type): Var list of types for configuration. Supports of the following:
            - classes: Wisp will scan the class for constructions methods, and generate a config dataclass for each.
            - constructors / functions: Specific construction methods can also be stated.
        exclude (List[Callable]): If specified, particular classes or constructors will be omitted from
            the returned configs. exclude can be used to filter specific constructors when generating configs for entire
            class types.
    """
    configs = []
    exclude = [] if exclude is None else exclude
    for target in classes_and_callables:
        if inspect.isclass(target):  # target is a class
            cls = target
            configs.extend(build_configs(cls, exclude=exclude))
        elif target not in exclude:  # target is a callable
            cls = inspect.getmodule(target).__dict__[target.__qualname__.split('.')[0]]  # Get the class
            configs.append(build_config_for_callable(cls, target))
    return configs


def check_valid_type(_type):
    """Helper function to check if the provided type is supported by our auto config generation.

    This is needed to filter out arguments from APIs that take as input things that cannot be
    easily parsed from the CLI, like general objects.

    Args:
        _type (type): The type to check.

    Returns:
        (bool): True if the type is supported.
    """
    origin = typing.get_origin(_type)
    if origin is None:
        if inspect.isclass(_type) and issubclass(_type, enum.Enum):
            return True
        primitives = (int, float, bool, str, type(None))
        return _type in primitives
    elif origin is typing.Union:
        args = typing.get_args(_type)
        if type(None) in args and len(args) == 2:   # Optional
            return True
        for arg in args:
            if not check_valid_type(arg):
                return False
        return True
    elif origin is typing.Literal:
        args = typing.get_args(_type)
        for arg in args:
            if not check_valid_type(type(arg)):
                return False
        return True
    elif origin is type(()):
        args = typing.get_args(_type)
        for arg in args:
            if not (arg is Ellipsis) and not check_valid_type(arg):
                return False
        return True
    else:
        return False


def get_supported_args(config) -> Optional[List[str]]:
    """ If this config builds a target, this function returns the target function args supported by this config
    (essentially these args are config fields).
    """
    if not hasattr(config, '__supported_args__'):
        return None
    return config.__supported_args__.split('$')


def get_missing_args(config) -> Optional[List[str]]:
    """ If this config builds a target, this function returns the target function args not supported by this config
    (essentially these args are not config fields, and should be passed explicitly when building the target).
    """
    if not hasattr(config, '__missing_args__'):
        return None
    return config.__missing_args__.split('$')


def get_target(config) -> Any:
    """ Returns the target buildable from the given config dataclass. """
    try:
        # Try to import the target. If it's unavailable, and the target is attached to an
        # optional import error, display that. Otherwise, let hydrazen prompt the error message.
        # We use the hydra implementation for dynamic import here, which is on par with hydrazen.
        from hydra._internal.utils import _locate
        _locate(config._zen_target)
    except (ImportError, ModuleNotFoundError) as e:
        if hasattr(config, '__import_error__'):
            raise type(e)(config.__import_error__) from None
    except:
        # In case importing sparked some dubious exception, fail silently and let hydrazen handle it
        pass
    return hydra_zen.get_target(config)