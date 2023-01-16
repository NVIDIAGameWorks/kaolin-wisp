# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
import os
from typing import Callable, Optional, Type
import collections
import inspect
import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import default_convert, default_collate_err_msg_format
from wisp.core import Rays
from wisp.datasets.base_datasets import WispDataset, MultiviewDataset, SDFDataset
from wisp.datasets.batch import Batch


def load_multiview_dataset(dataset_path: str, dataset_num_workers: int = -1, transform: Callable = None,
                           split: str = None, **kwargs) -> MultiviewDataset:
    """
        A convenience method which loads the MultiviewDataset class which best matches the files under dataset_path.
        The implementation relies on the `WispDataset.is_root_of_dataset()` function being implemented by
        WispDataset implementations.
        Dataset classes are allowed to specify unique terms which set them apart from other datasets, i.e.:
        a dataset path with the following contents can be assumed to be attributed to the nerf-synethetic dataset:
            /path/to/dataset/transform.json
            /path/to/dataset/images/____.png

        Caveats:
        1. Where more than one dataset class matches the contents of dataset_path,
        this function will trigger a RuntimeError due to a non-resolved ambiguity.
        In such cases, it is recommended to construct the datasets explicitly rather than using this function.

        2. This function can only search for MultiviewDataset subclasses which were already loaded by python.
        When specifying custom MultiviewDataset classes which do not exist under this folder,
        make sure python imported them before invoking this function.

        Args:
            dataset_path (str): The root directory of the dataset, where dataset files should reside.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Callable): Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
            split (str): The dataset split to use, may correspond to which files to load, or how to partion the data.
                Options: 'train', 'val', 'test'.
            **kwargs: Any other data specific arguments will be passed to the dataset class once it is matched.
                Excessive kwargs the matched dataset class does not use will be safely disregarded.
        """
    return _load_dataset(MultiviewDataset, dataset_path, dataset_num_workers, transform, split, **kwargs)


def _load_dataset(dataset_baseclass: Type[WispDataset],
                  dataset_path: str, dataset_num_workers: int = -1, transform: Callable = None,
                  split: str = None, **kwargs) -> WispDataset:
    """
    A convenience method which loads the dataset class which best matches the files under dataset_path.
    The implementation relies on the `WispDataset.is_root_of_dataset()` function being implemented by
    WispDataset implementations.
    Dataset classes are allowed to specify unique terms which set them apart from other datasets, i.e.:
    a dataset path with the following contents can be assumed to be attributed to the nerf-synethetic dataset:
        /path/to/dataset/transform.json
        /path/to/dataset/images/____.png

    Caveats:
    1. Where more than one dataset class matches the contents of dataset_path, this function will trigger a RuntimeError
    due to a non-resolved ambiguity. In such cases, it is recommended to construct the datasets explicitly rather
    than using this function.

    2. This function can only search for `dataset_baseclass` subclasses which were already loaded by python.
    When specifying custom WispDataset classes which do not exist under this folder,
    make sure python imported them before invoking this function.

    Args:
        dataset_baseclass (Type[WispDataset]): WispDataset or one of it's variants, to further scope which parts
            of the WispDataset class hierarchy are matched against the root directory.
        dataset_path (str): The root directory of the dataset, where dataset files should reside.
        dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
            If dataset_num_workers < 1, processing will take place on the main process.
        transform (Callable): Transform function applied per batch when data is accessed with __get_item__.
            For example: ray sampling, to filter the amount of rays returned per batch.
            When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        split (str): The dataset split to use, may correspond to which files to load, or how to partion the data.
            Options: 'train', 'val', 'test'.
        **kwargs: Any other data specific arguments will be passed to the dataset class once it is matched.
            Excessive kwargs the matched dataset class does not use will be safely disregarded.
    """
    from wisp.config_parser import get_args_for_function
    def __subclasses_hierarchy(cls): # Get hierarchy of subclasses of cls
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in __subclasses_hierarchy(c)])
    files_list = os.listdir(dataset_path)
    matching_dataset: Optional[Type[WispDataset]] = None
    for dataset in __subclasses_hierarchy(dataset_baseclass):
        if inspect.isabstract(dataset):
            continue
        is_match = dataset.is_root_of_dataset(root=dataset_path, files_list=files_list)
        if is_match:
            if matching_dataset is None:
                matching_dataset = dataset
            else:
                raise RuntimeError(f"load_dataset was given an ambiguous path which matches more than one dataset"
                                   f" class: {dataset} and {matching_dataset}. This is a result of "
                                   f"'is_root_of_dataset()' implementations which do not distinguish well enough"
                                   f"between the two dataset classes. A quick workaround is to load the desired "
                                   f"dataset class explicitly. Otherwise, update the dataset "
                                   f"'is_root_of_dataset()' logics. ")
    if matching_dataset is None:
        raise RuntimeError(f"load_dataset did match any dataset class which fits the contents of {dataset_path}. "
                           f"Is the dataset-path valid?")

    # Out of all given kwargs, select those which the dataset __init__ function uses
    ds_args = get_args_for_function(args=kwargs, func=matching_dataset.__init__)
    return matching_dataset(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                            transform=transform, split=split, **ds_args)


def default_collate(batch):
    r"""
    Function that extends torch.utils.data._utils.collate.default_collate
    to support custom wisp structures such as Rays and Batches.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, Rays):
        return Rays.cat(batch)
    elif isinstance(elem, Batch):
        # Assumes that if first element in batch has a certain None field, this field is None for all other entries too
        return elem_type(**{key: default_collate([d[key] for d in batch])
                            for key in elem if elem.get(key) is not None})
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [default_collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([default_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
