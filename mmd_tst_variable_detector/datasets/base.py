import abc
import dataclasses
import typing
import typing as ty
import numpy as np
import random

import torch
import torch.utils.data


TensorPair = ty.Tuple[torch.Tensor, torch.Tensor]


class Return_split_train_and_test(ty.NamedTuple):
    train_dataset: "BaseDataset"
    test_dataset: "BaseDataset"
    train_test_sample_ids: ty.Tuple[ty.List[int], ty.List[int]]



class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    # --------------------------------------
    # base attributes
    _x: torch.Tensor
    _y: torch.Tensor
    _dimension_x: ty.Tuple[int,...]
    _dimension_y: ty.Tuple[int,...]

    # --------------------------------------    
    
    @abc.abstractmethod
    def __len__():
        raise NotImplementedError()

    @abc.abstractmethod
    def copy_dataset(self) -> "BaseDataset":
        """copy itself"""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def close(self):
        # delete non-pickle objects.
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def flatten_matrix_to_vector(data: torch.Tensor) -> torch.Tensor:
        """Flattening a matrix into vector if the incoming data is a matrix form."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_all_samples(self) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_samples_at_dimension_d(self, index_d: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """Return the samples at dimension d."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dimension_flattened(self) -> int:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_dimension_data_space(self) -> ty.Tuple[int, ...]:
        raise NotImplementedError()

    @abc.abstractmethod
    def merge_new_dataset(self, dataset_obj: "BaseDataset") -> "BaseDataset":
        raise NotImplementedError()

    @abc.abstractmethod
    def get_random_data_id(self, n_samples: int, random_seed: ty.Optional[int] = None) -> typing.List[int]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_subsample_dataset(self, n_samples: ty.Optional[int] = None, sample_ids: ty.Optional[ty.List[int]] = None, random_seed: ty.Optional[int] = None) -> typing.Tuple[typing.List[int], "BaseDataset"]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_bootstrap_dataset(self, seed: ty.Optional[int] = None) -> ty.Tuple[ty.Tuple[ty.List[int], ty.List[int]], "BaseDataset"]:
        """
        Return
        ----------
            ((sample-id-x, sample-id-y), DatasetObject)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_selected_variables_dataset(self, selected_variables: ty.Tuple[int,...]) -> "BaseDataset":
        raise NotImplementedError()

    @abc.abstractmethod
    def split_train_and_test(self, 
                             train_ratio: float = 0.8,
                             random_seed: int = 42) -> Return_split_train_and_test:
        """Public API. 
        
        Return
        ----------
            Return_split_train_and_test
        """
        raise NotImplementedError()
    
    def is_dataset_on_ram(self) -> bool:
        return False
    
    def generate_dataset_on_ram(self):
        raise NotImplementedError()
    
    def get_torch_random_generator(self, seed: ty.Optional[int] = None) -> torch.Generator:
        if seed is None:
            seed = random.randint(0, 100000)
        return torch.Generator().manual_seed(seed)


# --------------------------------------
# specifying the data source type



class BaseRamBackendDataset(BaseDataset):
    pass


class BaseFileBackendDataset(BaseDataset):
    @abc.abstractmethod
    def _get_list_pairs(self):
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _get_sample_size(self):
        raise NotImplementedError()
        
    @abc.abstractmethod
    def run_files_validation(self):
        """A method to check if input list of files are valid or not.
        """
        raise NotImplementedError()


# --------------------------------------
# specifying the nature of data


class BaseStaticDataset(BaseDataset):
    # when your data is such as image, which does not time-series dependencies.
    pass


class BaseSpatioTemporalDataset(BaseDataset):
    pass


class BaseSensorSTDataset(BaseSpatioTemporalDataset):
    # when your data is in the form of (|S|, |T|)
    pass


class BaseTrajectorySTDataset(BaseSpatioTemporalDataset):
    # when your data is in the form of (|A|, |T|)
    pass




# --------------------------------------
# specifying the algorithm type

class BaseSampleBasedDataset(BaseDataset):
    pass


class BaseTimeSlicingDataset(BaseDataset):
    pass

