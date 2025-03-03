import typing as ty
from pathlib import Path 

import torch

from .base import (
    BaseFileBackendDataset, 
    BaseStaticDataset, 
    BaseSampleBasedDataset,
    TensorPair, 
    Return_split_train_and_test
)


class FileBackendOneTimeLoadStaticDataset(BaseFileBackendDataset, BaseStaticDataset, BaseSampleBasedDataset):
    def __init__(self,
                 path_pair_parent_dir: ty.List[Path],
                 file_name_x: str = 'x.pt',
                 file_name_y: str = 'y.pt',
                 key_name_array: str = 'array',
                 x: ty.Optional[torch.Tensor] = None,
                 y: ty.Optional[torch.Tensor] = None,
                 selected_variables: ty.Optional[ty.Tuple[int,...]] = None) -> None:
        """A dataset class for static data. When a input data is 2D-array, This dataset class returns a flatten 1D-array.
        This dataset holds just a file path until you call `load_file()` method. 
        Please call `load_file()` method before you use this dataset, ex. in Dask functions.
        In fact, I design this dataset class especially for Dask architecture.
        I do not wanna transfer data array via protbuf that Dask uses.
        Thus, I load data in each worker node.
        
        Parameters
        ----------
            path_pair_parent_dir: ty.List[Path]
                A list of path to the parent directory of (x, y) pairs.
                `path_pair_parent_dir` requires a list of parent directories.
                This class does not run `rglob` or `glob`, which makes slow-down of Dataset class initialization.
            file_name_x: str
                A name of the file for x.
            file_name_y: str
                A name of the file for y.
            key_name_array: str
                A key name of the dict object that saves the tensor.
            selected_variables: list
                a list of indices of variables to be selected. Given only when you want to select.
            x: torch.Tensor
                a dummy attribute. Not used when you call `__init__`.
            y: torch.Tensor
                a dummy attribute. Not used when you call `__init__`.
        """
        super(FileBackendOneTimeLoadStaticDataset).__init__()
        
        # ------------------------------
        # declare attributes.
        self._data_tensor_x: ty.Optional[torch.Tensor] = None
        self._data_tensor_y: ty.Optional[torch.Tensor] = None
        
        self.selected_variables: ty.Optional[ty.Tuple[int,...]] = selected_variables
        # ------------------------------
        assert self._data_tensor_y is None, "self._data_tensor_y must be None."
    
        self.path_pair_parent_dir = path_pair_parent_dir
        self.file_name_x = file_name_x
        self.file_name_y = file_name_y
        
        assert len(self.path_pair_parent_dir) > 0, "len(self.path_pair_parent_dir) == 0."        
        self.key_name_array = key_name_array

        self.__seq_xy_pairs = self._get_list_pairs()
        # tensor shape
        self.tensor_shape = self.__get_array_shape()

        if x is None:
            # comment: this if block is when you initialize the dataset object.
            pass
        else:            
            # comment: this if block is when Pytorch dataloader reads the dataset object.
            # Note: Pytorch dataloader calls __init__ method at every epoch.
            assert x is not None, "x is None."
            assert y is not None, "y is None."
            self._data_tensor_x = x
            self._data_tensor_y = y
        
    # --------------------------------------
    # Private APIs.    

    def __get_array_shape(self) -> str:
        """Private API.
        
        Checking only the first element of the file list.
        
        Return
        ----------
            str
                A shape of the array. 'vector', 'matrix'
        """
        path_dir_parent = self.path_pair_parent_dir[0]
        d_x = torch.load(path_dir_parent / self.file_name_x)
        
        if len(d_x[self.key_name_array].shape) == 1:
            return 'vector'
        elif len(d_x[self.key_name_array].shape) == 2:
            return 'matrix'
        else:
            raise ValueError(f"Invalid shape: {d_x[self.key_name_array].shape}")
        
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
    
    def is_dataset_on_ram(self) -> bool:
        return True
    
    def copy_dataset(self) -> "FileBackendOneTimeLoadStaticDataset":
        """copy itself"""
        return FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=self.path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            selected_variables=self.selected_variables)
    
    def close(self):
        # delete non-pickle objects.
        pass

    def get_samples_at_dimension_d(self, index_d: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        """Return the samples at dimension d."""
        x_at_d = torch.zeros((len(self), ))
        y_at_d = torch.zeros((len(self), ))
        for idx in range(0, len(self)):
            __x, __y = self.__getitem__(idx)
            x_at_d[idx] = __x[index_d]
            y_at_d[idx] = __y[index_d]
        # end for
        return x_at_d, y_at_d
            

    def get_random_data_id(self, n_samples: int, random_seed: ty.Optional[int] = None) -> ty.List[int]:
        generator = self.get_torch_random_generator(random_seed)        
        perm = torch.randperm(len(self._x), generator=generator)
        idx = perm[:n_samples]
        return idx.tolist()

    def get_subsample_dataset(self, 
                              n_samples: ty.Optional[int] = None, 
                              sample_ids: ty.Optional[ty.List[int]] = None,
                              random_seed: ty.Optional[int] = None
                              ) -> ty.Tuple[ty.List[int], "FileBackendOneTimeLoadStaticDataset"]:
        assert n_samples is not None or sample_ids is not None, 'Either of n_samples or sample_ids should be specified.'
        if sample_ids is None:
            assert n_samples is not None, 'n_samples should be specified.'
            sub_data_index = self.get_random_data_id(n_samples, random_seed)
        else:
            sub_data_index = sample_ids
        # end if
        
        seq_extracted = [path_pair for __i_file, path_pair in enumerate(self.path_pair_parent_dir) if __i_file in sub_data_index]
            
        sub_datset = FileBackendOneTimeLoadStaticDataset(
            seq_extracted,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            selected_variables=self.selected_variables)
        return sub_data_index, sub_datset

    def get_bootstrap_dataset(self, 
                              seed: ty.Optional[int] = None
                              ) -> ty.Tuple[ty.Tuple[ty.List[int], ty.List[int]], "FileBackendOneTimeLoadStaticDataset"]:
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
        assert self._data_tensor_x is not None, "self.x is None. You forgot to call `generate_dataset_on_ram()` method."
        assert self._data_tensor_y is not None, "self.y is None. You forgot to call `generate_dataset_on_ram()` method."
        
        tensor_x = self._data_tensor_x[index]
        tensor_y = self._data_tensor_y[index]

        if self.selected_variables is not None:
            tensor_x = tensor_x[list(self.selected_variables)]
            tensor_y = tensor_y[list(self.selected_variables)]
        # end if

        return tensor_x, tensor_y
    
    # --------------------------------------
    # public API

    def generate_dataset_on_ram(self) -> "FileBackendOneTimeLoadStaticDataset":
        """Loading data from files to RAM.
        You have to execute this method before starting training loop.
        
        This method loads data and returns a new dataset object.
        The new dataset object has `x` and `y` arguments of `__init__`. 
        These x, y arguments exist in RAM.
        
        Return
        ----------
            FileBackendOneTimeLoadStaticDataset
        """
        _d_size_x = self.get_dimension_flattened()
        _data_tensor_x = torch.zeros((len(self), _d_size_x))
        _data_tensor_y = torch.zeros((len(self), _d_size_x))
        
        for __i, _t_dir_file_path in enumerate(self.__seq_xy_pairs):
            assert _t_dir_file_path[0].exists(), f"File does not exist: {_t_dir_file_path[0]}"
            assert _t_dir_file_path[1].exists(), f"File does not exist: {_t_dir_file_path[1]}"
                    
            d_x = torch.load(_t_dir_file_path[0].open('rb'))
            d_y = torch.load(_t_dir_file_path[1].open('rb'))

            tensor_x = self.flatten_matrix_to_vector(d_x[self.key_name_array], self.tensor_shape)
            tensor_y = self.flatten_matrix_to_vector(d_y[self.key_name_array], self.tensor_shape)
            
            _data_tensor_x[__i] = tensor_x
            _data_tensor_y[__i] = tensor_y
        # end for
        
        return FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=self.path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            x=_data_tensor_x,
            y=_data_tensor_y,
            selected_variables=self.selected_variables)

    @staticmethod
    def flatten_matrix_to_vector(data: torch.Tensor, tensor_shape: str) -> torch.Tensor:
        """Flattening a matrix into vector if the incoming data is a matrix form."""
        
        if tensor_shape == 'vector':
            return data
        elif tensor_shape == 'matrix':
            return data.flatten()
        else:
            raise ValueError(f"Invalid shape: {tensor_shape}")

    
    def get_selected_variables_dataset(self, selected_variables: ty.Tuple[int,...]) -> "FileBackendOneTimeLoadStaticDataset":
        """Public API. Do nothing.
        """
        return FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=self.path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            selected_variables=selected_variables,
            x=self._data_tensor_x,
            y=self._data_tensor_y
        )
    
    
    def merge_new_dataset(self, dataset_obj: "FileBackendOneTimeLoadStaticDataset") -> "FileBackendOneTimeLoadStaticDataset":
        """Public API.
        
        Concatenating the dataset. Simply, concatenating the list of path_pair_parent_dir.
        Caution that `file_name_x` and `file_name_y` should be the same.
        
        Return
        ----------
            FileBackendStaticDataset
        """
        assert self.file_name_x == dataset_obj.file_name_x, f"self.file_name_x != dataset_obj.file_name_x: {self.file_name_x} != {dataset_obj.file_name_x}"
        assert self.key_name_array == dataset_obj.key_name_array, f"self.key_name_array != dataset_obj.key_name_array: {self.key_name_array} != {dataset_obj.key_name_array}"
        
        new_path_pair_parent_dir = self.path_pair_parent_dir + dataset_obj.path_pair_parent_dir
        
        # comment: not need to load data in RAM. You have to call `generate_dataset_on_ram()` method later.
        return FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=new_path_pair_parent_dir,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            selected_variables=self.selected_variables,
            x=self._data_tensor_x,
            y=self._data_tensor_y
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
        path_parent = self.__seq_xy_pairs[0]
        tensor_x = torch.load(path_parent[0])[self.key_name_array]
        tensor_y = torch.load(path_parent[1])[self.key_name_array]

        if self.tensor_shape == 'vector':
            return tensor_x.shape[0]
        elif self.tensor_shape == 'matrix':
            return tensor_x.shape[0] * tensor_x.shape[1]
        else:
            raise ValueError(f"Invalid shape: {self.tensor_shape}")
    
    def get_all_samples(self) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('This API is not available.')
    
    def split_train_and_test(self, 
                             train_ratio: float = 0.8,
                             random_seed: int = 42,
                             ) -> Return_split_train_and_test:
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
        
        dataset_train = FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=files_train,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            x=self._data_tensor_x,
            y=self._data_tensor_y,
            selected_variables=self.selected_variables)
        dataset_test = FileBackendOneTimeLoadStaticDataset(
            path_pair_parent_dir=files_test,
            file_name_x=self.file_name_x,
            file_name_y=self.file_name_y,
            key_name_array=self.key_name_array,
            x=self._data_tensor_x,
            y=self._data_tensor_y,
            selected_variables=self.selected_variables)
        
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
    
            assert self.key_name_array in torch.load(path_x), f"path_x is not a tensor: {path_x}"
            assert self.key_name_array in torch.load(path_y), f"path_y is not a tensor: {path_y}"
    
            assert isinstance(torch.load(path_x)[self.key_name_array], torch.Tensor), f"path_x is not a tensor: {path_x}"
            assert isinstance(torch.load(path_y)[self.key_name_array], torch.Tensor), f"path_y is not a tensor: {path_y}"
