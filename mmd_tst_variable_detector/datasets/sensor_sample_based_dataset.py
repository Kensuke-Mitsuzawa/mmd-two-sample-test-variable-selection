from pathlib import Path
import typing as ty
import logging

import torch

from .base import (
    BaseFileBackendDataset,
    BaseSampleBasedDataset,
    BaseSensorSTDataset,
    TensorPair,
    Return_split_train_and_test
)
from ..logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class FileBackendSensorSampleBasedDataset(BaseFileBackendDataset, BaseSensorSTDataset, BaseSampleBasedDataset):
    def __init__(self,
                 path_pair_parent_dir: ty.List[Path],
                 file_name_x: str = 'x.pt',
                 file_name_y: str = 'y.pt',
                 key_name_array: str = 'array',
                 selected_variables: ty.Optional[ty.Tuple[int,...]] = None) -> None:
        """A dataset class for sensor data, sample-based algorithm.
        
        Under `path_pair_parent_dir` parent directory, there must be a set of sub-directories.
        In each sub-directory, there must be `file_name_x` and `file_name_y`.
        These `file_name_x` and `file_name_y` must be `.pt` extension.
        The `.pt` file must contain a dictionary with `key_name_array` key, the value is a tensor.
        The tensor shape is $\mathbb{R}^{|S| \times |T|}$.
        
        Parameters
        ----------
        path_pair_parent_dir: ty.List[Path]
            A list of parent directories.
        file_name_x: str
            A file name of x.
        file_name_y: str
            A file name of y.
        key_name_array: str
            A key name of the dictionary.
        selected_variables: ty.Optional[ty.List[int]]
            A list of selected variables. If None, all variables are selected.
        """
        super(FileBackendSensorSampleBasedDataset).__init__()
                
        self.path_pair_parent_dir = path_pair_parent_dir
        self.file_name_x = file_name_x
        self.file_name_y = file_name_y
        
        assert len(self.path_pair_parent_dir) > 0, "len(self.path_pair_parent_dir) == 0."        
        
        self.key_name_array = key_name_array

        self.__seq_xy_pairs = self._get_list_pairs()
    
        self.selected_variables = selected_variables

    # --------------------------------------
    # Private APIs.    

    def __get_array_shape(self) -> str:
        """Private API.
        
        Checking only the first element of the file list.
        """
        raise NotImplementedError('This API is not available.')
    
        
    def _get_list_pairs(self) -> ty.List[ty.Tuple[Path, Path]]:
        """getting a list of (x, y) pairs.
        """
        seq_pair_path = []
        for __path_parent in self.path_pair_parent_dir:
            __t_pair = (__path_parent / self.file_name_x, __path_parent / self.file_name_y)
            seq_pair_path.append(__t_pair)
        # end for
        return seq_pair_path
        
    def _get_sample_size(self) -> int:
        """getting the sample size.
        """
        return len(self._get_list_pairs())


    # --------------------------------------
    # public API not used.    
    
    def copy_dataset(self) -> "FileBackendSensorSampleBasedDataset":
        """copy itself"""
        raise NotImplementedError()
    
    def close(self):
        # delete non-pickle objects.
        pass

    def get_samples_at_dimension_d(self, index_d: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """Return the samples at dimension d."""
        raise NotImplementedError()

    def get_subsample_dataset(self, n_samples: ty.Optional[int] = None, sample_ids: ty.Optional[ty.List[int]] = None) -> ty.Tuple[ty.List[int], "FileBackendSensorSampleBasedDataset"]:
        raise NotImplementedError()

    def get_bootstrap_dataset(self, seed: ty.Optional[int] = None) -> ty.Tuple[ty.Tuple[ty.List[int], ty.List[int]], "FileBackendSensorSampleBasedDataset"]:
        """
        Return
        ----------
            ((sample-id-x, sample-id-y), DatasetObject)
        """
        raise NotImplementedError()


    # --------------------------------------
    # public API. Pytorch standard
    
    
    def __len__(self) -> int:
        return self._get_sample_size()
    
    def __getitem__(self, index: int) -> TensorPair:
        """getting a pair of tensors."""
        pair_xy_file = self.__seq_xy_pairs[index]
        
        d_x = torch.load(pair_xy_file[0].open('rb'))
        d_y = torch.load(pair_xy_file[1].open('rb'))

        tensor_x = d_x[self.key_name_array]
        tensor_y = d_y[self.key_name_array]

        if self.selected_variables is not None:
            tensor_x = tensor_x[self.selected_variables]
            tensor_y = tensor_y[self.selected_variables]
        # end if

        return tensor_x, tensor_y
    
    # --------------------------------------
    # public API

    @staticmethod
    def flatten_matrix_to_vector(data: torch.Tensor, tensor_shape: str) -> torch.Tensor:
        """Flattening a matrix into vector if the incoming data is a matrix form."""
        
        if tensor_shape == 'vector':
            return data
        elif tensor_shape == 'matrix':
            return data.flatten()
        else:
            raise ValueError(f"Invalid shape: {tensor_shape}")

    
    def get_selected_variables_dataset(self, selected_variables: ty.List[int]) -> "FileBackendSensorSampleBasedDataset":
        """Public API. Do nothing.
        """
        raise NotImplementedError('Currently, this API is not available. \
            I have to consider how to pass selected_variables to the dataset. \
                In current codebase, all vector must be 1D array. But, now, I wanna consider 2D array shape.')
        return FileBackendSensorSampleBasedDataset(
            path_pair_parent_dir=self.path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            selected_variables=selected_variables
        )
    
    
    def get_random_data_id(self, n_samples: int) -> ty.List[int]:
        raise NotImplementedError()
    
    
    def merge_new_dataset(self, dataset_obj: "FileBackendSensorSampleBasedDataset") -> "FileBackendSensorSampleBasedDataset":
        """Public API.
        
        Concatenating the dataset. Simply, concatenating the list of path_pair_parent_dir.
        Caution that `file_name_x` and `file_name_y` should be the same.
        
        Return
        ----------
            FileBackendSensorSampleBasedDataset
        """
        assert self.file_name_x == dataset_obj.file_name_x, f"self.file_name_x != dataset_obj.file_name_x: {self.file_name_x} != {dataset_obj.file_name_x}"
        assert self.key_name_array == dataset_obj.key_name_array, f"self.key_name_array != dataset_obj.key_name_array: {self.key_name_array} != {dataset_obj.key_name_array}"
        
        new_path_pair_parent_dir = self.path_pair_parent_dir + dataset_obj.path_pair_parent_dir
        return FileBackendSensorSampleBasedDataset(
            path_pair_parent_dir=new_path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array
        )
    
    def get_dimension_data_space(self) -> ty.Tuple[int, ...]:
        if self.selected_variables is not None:
            return self.selected_variables
        # end if
        
        first_pair = self.path_pair_parent_dir[0]
        tensor_x = torch.load(first_pair / self.file_name_x)[self.key_name_array]
        
        return tensor_x.shape

    def get_dimension_flattened(self) -> int:
        """Get the dimension of the flattened data.
        """
        path_parent = self.__seq_xy_pairs[0][0]
        tensor_x = torch.load(path_parent / self.file_name_x)[self.key_name_array]
        tensor_y = torch.load(path_parent / self.file_name_y)[self.key_name_array]
        
        assert tensor_x.shape == tensor_y.shape, f"tensor_x.shape != tensor_y.shape: {tensor_x.shape} != {tensor_y.shape}"

        return tensor_x.shape
    
    def get_all_samples(self) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('This API is not available.')
    
    def split_train_and_test(self, 
                             train_ratio: float = 0.8,
                             random_seed: int = 42) -> Return_split_train_and_test:
        """Public API. 
        
        Splitting the dataset into train and test.
        
        Return
        ----------
            Return_split_train_and_test
        """
        n_train = int(self.__len__() * train_ratio)
        sample_ids_all = list(range(self.__len__()))
        
        __files_train = self.__seq_xy_pairs[:n_train]
        files_train = [t_pair[0].parent for t_pair in __files_train]
        sample_ids_train = sample_ids_all[:n_train]
        
        __files_test = self.__seq_xy_pairs[n_train:]
        files_test = [t_pair[0].parent for t_pair in __files_test]
        sample_ids_test = sample_ids_all[n_train:]
        
        dataset_train = FileBackendSensorSampleBasedDataset(
            path_pair_parent_dir=files_train,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array)
        dataset_test = FileBackendSensorSampleBasedDataset(
            path_pair_parent_dir=files_test,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array            
        )
        
        return Return_split_train_and_test(
            train_dataset=dataset_train,
            test_dataset=dataset_test,
            train_test_sample_ids=(sample_ids_train, sample_ids_test)
        )
        
    
    def run_files_validation(self):
        """Checking if all files are valid or not. 
        You have to call this method manually and verify file existence.
        
        Returns
        ----------
        None
        
        Notes
        ----------
        `__init__` does not call this function. That causes slow-down of Dataset class initialization.
        """
        assert self.path_pair_parent_dir is not None, "self.path_pair_parent_dir is None."
        assert len(self.path_pair_parent_dir) > 0, "len(self.path_pair_parent_dir) == 0."
    
        for path_x, path_y in self._get_list_pairs():
            assert path_x.exists(), f"path_x does not exist: {path_x}"
            assert path_y.exists(), f"path_y does not exist: {path_y}"

            __d_x = torch.load(path_x)
            __d_y = torch.load(path_y)
    
            assert self.key_name_array in __d_x, f"path_x is not a tensor: {path_x}"
            assert self.key_name_array in __d_y, f"path_y is not a tensor: {path_y}"
    
            assert isinstance(__d_x[self.key_name_array], torch.Tensor), f"path_x is not a tensor: {path_x}"
            assert isinstance(__d_y[self.key_name_array], torch.Tensor), f"path_y is not a tensor: {path_y}"

            assert __d_x[self.key_name_array].shape == __d_y[self.key_name_array].shape, f"__d_x[self.key_name_array].shape != __d_y[self.key_name_array].shape: {__d_x[self.key_name_array].shape} != {__d_y[self.key_name_array].shape}"
            
            assert len(__d_x[self.key_name_array].shape) == 2, f"len(__d_x[self.key_name_array].shape) != 2: {len(__d_x[self.key_name_array].shape)} != 2"
            assert len(__d_y[self.key_name_array].shape) == 2, f"len(__d_y[self.key_name_array].shape) != 2: {len(__d_y[self.key_name_array].shape)} != 2"
    