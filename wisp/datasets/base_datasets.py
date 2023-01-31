# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union, Tuple
import torch
from torch.utils.data import Dataset
from kaolin.render.camera import Camera
from wisp.datasets.batch import Batch, MultiviewBatch, SDFBatch


class WispDataset(Dataset, ABC):
    """ A general abstract class all wisp dataset classes should implement to integrate with the datasets framework.
    WispDatasets subclass directly from torch.utils.data.Dataset, and make little assumptions on the dataset structure.

    At the bare minimum, load_singleprocess(), __getitem__ and __len__ should be implemented.
    Additional methods may i.e. extend the dataset to allow it to loader faster,
    load more conveniently, or support better management of dataset splits.

    The following assumptions hold:
        - Datasets can be loaded from some path, but this isn't mandatory. In the case of in memory datasets,
        dataset_path may default to None.
        - Datasets should always support loading from the main process. In addition, datasets may support faster loading
         using multiple workers. When `load()` is invoked, `dataset_num_workers` will be queried to decide
         if `load_singleprocess()` or `load_multiprocess()` should be called.
         By default, if `load_multiprocess` is left unimplemented, `load_singleprocess` will be invoked irrespective
         of the number of workers.
        - Datasets should support transformation functors applied before __getitem__() calls are returned.
        These transformations may modify the content of the returned batch.
        - Where applicable, datasets should support loading specific splits of train, validation and test.
        The exact partition is left up to the dataset logic (i.e. by external files, by some specified ratio, etc).
        - Datasets which rely on external files should designate a set of rules to specify if the contents of some
        directory is loadable by the class.

    Pertaining to the last point, where no ambiguity hold over the content of some input dataset files,
    Wisp subclasses can be created with the convenience functions of wisp.utils.load_X_dataset()
    """

    def __init__(self, dataset_path: str = None, dataset_num_workers: int = -1,
                 transform: Callable = None, split: str = None):
        """
        Args:
            dataset_path (str): The root directory of the dataset, where the dataset's external assets should reside.
                Default: None (in memory dataset).
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
                Default: -1 (dataset always loads on main process, without using multiprocessing).
            transform (Callable): Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
                Default: None (no transformation is applied on batches).
            split (str): The dataset split to use. Options: 'train', 'val', 'test'.
                Default: None (the dataset does not distinguish between different splits).
        """
        self.dataset_path = dataset_path
        self.dataset_num_workers = dataset_num_workers
        self.transform = transform
        self.split = split

    def create_split(self, split: str, transform: Callable = None, *args, **kwargs) -> WispDataset:
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation and test datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.

        Example:
            ```
            train_dataset = Dataset(...)
            validation_dataset = train_dataset.create_split(split="val")
            test_dataset = train_dataset.create_split(split="test")
            ```

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val', 'test'.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        raise NotImplementedError("This dataset does not support generation of new splits from a given dataset.")

    def name(self) -> str:
        """ Returns a human readable name of this dataset class. By default, the class name is returned. """
        return type(self).__name__

    def load(self):
        """ Loads the contents of a dataset object.
        This operation should usually take place once when the dataset is first created.
        This function may return some data values to be cached, or update the dataset fields in place.

        load() is usually intended to be invoked by the subclass __init__.
        load() will route between `load_multiprocess()` and `load_singleprocess()`, depending on the number of workers.
        """
        if self.dataset_num_workers > 0:
            return self.load_multiprocess()
        else:
            return self.load_singleprocess()

    @abstractmethod
    def load_singleprocess(self):
        """Loads the contents of a dataset object on the main process.
        This operation is invoked by load() when the number of workers is 0 or negative.
        This function may return some data values to be cached, or update the dataset fields in place.
        """
        raise NotImplementedError('Wisp Datasets should override load_singleprocess()')

    def load_multiprocess(self):
        """Loads the contents of a dataset object using multiple workers.
        This operation is invoked by load() when the number of workers is 1 or higher.
        This function may return some data values to be cached, or update the dataset fields in place.
        """
        return self.load_singleprocess()  # By default, datasets are not obliged to support multiprocessed loading

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.

        Args:
                root (str): A path to the root directory of the dataset.
                files_list (List[str]): List of files within the dataset root, without their prefix path.
        Returns:
                True if the root folder points to content loadable by this dataset.
        """
        return False

    def __len__(self):
        """ Returns the number of entries in the dataset. """
        raise NotImplementedError('Wisp Datasets should override __len__')

    def __getitem__(self, idx: int) -> Batch:
        """ Samples the dataset for a batch of information. The exact batch returned depends on idx. """
        raise NotImplementedError('Wisp Datasets should override __getitem__')


class MultiviewDataset(WispDataset):
    """ Extends the WispDataset with dataset behavior common to all multiview datasets. """

    @property
    @abstractmethod
    def img_shape(self) -> Union[torch.Size, Tuple[torch.Size]]:
        """ Returns the shape of ground truth images of each view used by this dataset.
        If all images have a common size, the return value is torch.Size.
        If the gt images have different sizes, a Tuple of sizes is returned per view.
        """
        pass

    @property
    @abstractmethod
    def num_images(self) -> int:
        """ Returns the number of images / views this dataset contain. """
        pass

    @property
    def cameras(self) -> Dict[str, Camera]:
        """ Returns the set of cameras this dataset uses to generate rays.
        Cameras are identifiable by unique ids / names
        """
        return dict()

    def as_pointcloud(self) -> torch.FloatTensor:
        """ If `supports_depth()` is True, the current dataset contains depth information.
        This function can be used to query the depth information in the form of a pointcloud tensor.
        """
        raise NotImplementedError('MultiviewDatasets that support depth information should '
                                  'return their data as a pointcloud. Otherwise, set supports_depth to return False.')

    def supports_depth(self) -> bool:
        """ Returns if this dataset have loaded depth information. """
        return False

    def __getitem__(self, idx: int) -> MultiviewBatch:
        """ Samples the dataset for a multiview batch of information.
        The exact ray channels the batch contains are up to the dataset implementation to determine.
        Callers may treat the batch as a dictionary, or query MultiviewBatch.ray_values() to view which
        supervision channels are available.
        """
        raise NotImplementedError('MultiviewDatasets should override __getitem__')

    def __len__(self):
        """Length of the dataset as number of views. """
        return self.num_images


class SDFDataset(WispDataset):
    """ Extends the WispDataset with dataset behavior common to all signed distance function based datasets. """

    @property
    @abstractmethod
    def coordinates(self) -> torch.Tensor:
        """ Returns a tensor of sample coordinates this dataset currently holds. """
        raise NotImplementedError('SDFDatasets should return a (N, d) tensor of sample coordinates.')

    def resample(self) -> None:
        """ Resamples to generate a new working set of coordinates values and their supervision values.
        When implemented, this function allows the dataset to refresh the available samples with a new set.
        Resampling is expected to be implemented as an in-place operation.
        """
        pass

    def __getitem__(self, idx: int) -> SDFBatch:
        """ Samples the dataset for a coordinate+sdf batch of information.
        The exact coordinate supervision channels the batch contains are up to the dataset implementation to determine.
        Callers may treat the batch as a dictionary, or query SDFBatch.coord_values() to view which
        supervision channels are available.
        """
        raise NotImplementedError('SDFDataset should override __getitem__')

    def __len__(self):
        """Length of the dataset as number of coordinate samples. """
        raise NotImplementedError('SDFDataset should override __len__')