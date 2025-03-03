from pathlib import Path
import typing as ty
import math
import logging
import more_itertools
from tqdm import tqdm


from tempfile import mkdtemp
import dask
import dask.array
from distributed import Client

import numpy as np
import torch

from ..logger_unit import handler

from ..datasets.base import BaseDataset
from ..datasets.file_backend_static_dataset import FileBackendStaticDataset
from ..datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ..datasets.sensor_sample_based_dataset import FileBackendSensorSampleBasedDataset
from ..datasets.ram_backend_static_dataset import SimpleDataset


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)

PossibleDatasetType = ty.Union[BaseDataset, FileBackendStaticDataset]

"""
A high level API of dataset generation/validation.
Converting your own dataset to the format of this project.


There are several cases of input datasets.

Data characteristic
--------------------

There are roughly two types of data characteristics; static and Spatio-Temporal (ST).

The ST type has two sub-types; sensor and trajectory.

Dara Source type
--------------------

RAM or File.


Expected Input Form
--------------------


When `source=ram` and `characteristic=static`, `numpy.ndarray` or `torch.Tensor` is expected.
A sample is either 1D or 2D array. The 2D array is when the sample is image.


When `source=file` and `characteristic=static`, the inputs are data source files.
The possible extention format is torch pt.
A file has to contain one sample.
Directory structure is as follows.

```
+ root-directory
|--sample-pair-1
|  |--sample-x.pt
|  |--sample-y.pt
|--sample-pair-2
...
```



When `source=file`, `characteristic=sensor_st` and `algorithm=time_slicing`, the input X, Y pair is just one pair.
X, Y are both supposed to be saved in a file pt.
The data shape must be $\mathbbb{R}^{|S| \times |T|}$.

When `source=file`, `characteristic=trajectory_st` and `algorithm=time_slicing`, the input X, Y pair is just one pair.
X, Y are both supposed to be saved in a file pt.
The data shape must be $\mathbbb{R}^{|A| \times |T| \times C}$.


"""

PossibleInputType = ty.Union[str, np.ndarray, torch.Tensor, Path, ty.List[Path]]


class _ArgFuncDistributed(ty.NamedTuple):
    timestamp: int
    torch_x: ty.Union[Path, torch.Tensor]
    torch_y: ty.Union[Path, torch.Tensor]
    path_word_dir: Path
    dataset_type_charactersitic: str
    is_value_between_timestamp: bool    



class DatasetGenerater(object):
    def __init__(self,
                 data_x: PossibleInputType,
                 data_y: PossibleInputType,
                 dataset_type_backend: str,
                 dataset_type_charactersitic: str,
                 dataset_type_algorithm: str = 'dataset_type_algorithm',
                 time_aggregation_per: ty.Optional[int] = 100,
                 key_name_array: ty.Optional[str] = 'array',
                 path_work_dir: ty.Optional[Path] = Path('/tmp/dataset_generater')
                 ):
        """
        Parameters
        ----------
        data_x : PossibleInputType
            Input data of x. If dataset_backend is 'ram', data_x must be np.ndarray or torch.Tensor.
            If dataset_backend is 'file', data_x must be Path or List[Path].
        data_y : PossibleInputType
            Same as data_x.
        dataset_type_backend : str
            'ram' or 'file' or 'flexible-file'.
        dataset_type_charactersitic : str
            'static', 'sensor_st'
        dataset_type_algorithm: str
            'sample_based',
        time_aggregation_per: int
            This is only used when dataset_type_algorithm is 'sample_based'.
            This is the number of timestamp to be aggregated.            
        key_name_array: str
            When the data-source is from file, each torch `.pt` file must contain a dictionary.
            The `key_name_array` specifies the key name of the array.
        path_work_dir: Path
            Be informed that your disk has enough space to save the dataset.
        """
        # --------------------------------------------------------------------
        # definitions
        self.array_x: ty.Optional[ty.Union[np.ndarray, torch.Tensor]] = None
        self.array_y: ty.Optional[ty.Union[np.ndarray, torch.Tensor]] = None
        
        self.path_x: ty.Optional[Path] = None
        self.path_y: ty.Optional[Path] = None
        
        self.seq_path_x: ty.Optional[ty.List[Path]] = None
        self.seq_path_y: ty.Optional[ty.List[Path]] = None
        # --------------------------------------------------------------------
        
        assert dataset_type_backend in ('ram', 'file', 'flexible-file')
        assert dataset_type_charactersitic in ('static', 'sensor_st',)
        assert dataset_type_algorithm in ('sample_based',)
        
        self.dataset_type_backend = dataset_type_backend
        self.dataset_type_charactersitic = dataset_type_charactersitic
        self.dataset_type_algorithm = dataset_type_algorithm
        
        self.time_aggregation_per = time_aggregation_per
        self.key_name_array = key_name_array

        if path_work_dir is None:
            self.path_work_dir = Path('/tmp/dataset_generater')
        else:
            self.path_work_dir = path_work_dir
        # end if
        if isinstance(data_x, (np.ndarray, torch.Tensor)):
            # case of ram
            self.__validation_case_ram(data_x, data_y)
        elif isinstance(data_x, Path):
            # case. Currently, a single file x, y is only the case of time-slicing algorithm
            assert isinstance(data_y, Path), 'data_x and data_y must be same type.'
            self.__validate_case_time_slicing(data_x, data_y)
        elif isinstance(data_x, list):
            # case of file backend, can 
            assert all([isinstance(__p, Path) for __p in data_x]), 'data_x and data_y must be same type.'
            assert isinstance(data_y, list), 'data_x and data_y must be same type.'
            assert all([isinstance(__p, Path) for __p in data_y]), 'data_x and data_y must be same type.'
            self.__validate_case_file_backend(data_x, data_y)
            
    def __validate_case_file_backend(self, seq_path_x: ty.List[Path], seq_path_y: ty.List[Path]):
        """
        """
        assert self.dataset_type_backend in ('file', 'flexible-file'), 'dataset_type_backend must be file.'
        assert self.dataset_type_charactersitic in ('static', 'sensor_st'), 'dataset_type_charactersitic must be static, sensor_st.'
        assert self.dataset_type_algorithm in ('sample_based',), 'dataset_type_algorithm must be sample_based.'
        
        # check all file exists.
        assert all([__p.exists() for __p in seq_path_x]), 'All files must exist.'
        assert all([__p.exists() for __p in seq_path_y]), 'All files must exist.'
        
        self.seq_path_x = seq_path_x
        self.seq_path_y = seq_path_y
                        
    def __validation_case_ram(self, array_x, array_y):
        """Currently, thie case is for "ram", "static", "sample-based".
        """
        assert isinstance(array_x, (np.ndarray, torch.Tensor)), 'data_x and data_y must be same type.'
        self.array_x = array_x
        self.array_y = array_y
        
        assert self.dataset_type_backend == 'ram', 'dataset_type_backend must be ram.'
        assert self.dataset_type_charactersitic in ('static', 'sensor_st',), 'dataset_type_charactersitic must be static or sensor_st.'
        assert self.dataset_type_algorithm in ('sample_based',), 'dataset_type_algorithm must be sample_based.'
    
    def __func_key_sort_parent_dir(self, file_path: Path):
        return file_path.name
    
    def __generate_file_backend_static_dataset(self
                                               ) -> ty.Union[FileBackendOneTimeLoadStaticDataset, FileBackendStaticDataset]:
        assert self.seq_path_x is not None, 'self.seq_path_x must be set.'
        assert self.seq_path_y is not None, 'self.seq_path_y must be set.'
        
        # getting parent directory level.
        seq_parent_dir_x = [__p.parent for __p in self.seq_path_x]
        seq_parent_dir_y = [__p.parent for __p in self.seq_path_y]
        
        __x = sorted(seq_parent_dir_x, key=self.__func_key_sort_parent_dir)
        __y = sorted(seq_parent_dir_y, key=self.__func_key_sort_parent_dir)
        
        for __x_dir, __y_dir in zip(__x, __y):
            assert __x_dir == __y_dir, 'X and Y must be in the same directory.'
        # end for
        
        # TODO dataset arguments
        if self.dataset_type_backend == 'file':
            dataset = FileBackendStaticDataset(__x)
        elif self.dataset_type_backend == 'flexible-file':
            dataset = FileBackendOneTimeLoadStaticDataset(__x)
        else:
            raise ValueError('Undefiend case. Aborting.')
        # end if
        
        dataset.run_files_validation()
        
        return dataset
    
    # --------------------------------------------------------------------
    # Private API for manipulating files, for sample-based algorithm.
    
    @staticmethod
    def aggregation_matrix(torch_tensor: torch.Tensor, aggregation_by: int) -> torch.Tensor:
        n_columns = torch_tensor.shape[-1]
        result_tensor = []
        current_agg_point = aggregation_by
        # agg. between [0: current_agg_point]
        sub_tensor = torch_tensor[:, 0:aggregation_by]
        mean_vector = torch.mean(sub_tensor, dim=1)
        _ = mean_vector[:, None]
        result_tensor.append(_)
        # agg. from [current_agg_point:]
        while (current_agg_point + aggregation_by) < n_columns:
            sub_tensor = torch_tensor[:, current_agg_point: (aggregation_by + current_agg_point)]
            mean_vector = torch.mean(sub_tensor, dim=1)
            _ = mean_vector[:, None]  # 2d tensor into 3d tensor
            assert len(_.shape) == 2
            result_tensor.append(_)
            current_agg_point += aggregation_by
        # end while
        sub_tensor = torch_tensor[:, current_agg_point: (aggregation_by + current_agg_point)]
        mean_vector = torch.mean(sub_tensor, dim=1)
        _ = mean_vector[:, None]  # 2d tensor into 3d tensor
        assert len(_.shape) == 2
        result_tensor.append(_)

        __ = torch.cat(result_tensor, dim=1)
        if __.shape[-1] > 2:
            # comment: __.shape[-1] haappens often in DEBUG mode.
            assert __.shape[-1] == math.ceil(n_columns / aggregation_by)

        return __
        
    def __aggregated_spatio_temporal_tensor_by_timestamp(self, path_target_dir: Path) -> torch.Tensor:
        """Private API. Called by `__generate_file_backend_sensor_dataset`."""
        d_tensor = torch.load(path_target_dir)
        target_tensor = d_tensor[self.key_name_array]
        
        assert self.time_aggregation_per is not None, 'self.time_aggregation_per must be set.'
        x_aggregated = self.aggregation_matrix(target_tensor, self.time_aggregation_per)
        
        return x_aggregated
    
    def __generate_file_backend_sensor_dataset(self) -> FileBackendSensorSampleBasedDataset:
        """For sample-based algorithm. The data-characteristic is sensor.
        I strongly recommend to do aggregation by timestamps.
        Too many time-stamps makes a file-size big.
        Pytorch is not good at opening a big file. That causes heavy slow-donw of MMD-Opt.
        """
        assert self.seq_path_x is not None, 'self.seq_path_x must be set.'
        assert self.seq_path_y is not None, 'self.seq_path_y must be set.'
        
        if self.time_aggregation_per is None:
            logger.warning('time_aggregation_per is None. This may cause a heavy slow-down of MMD-Opt.')
        # end if
        
        # getting parent directory level.
        seq_parent_dir_x = [__p.parent for __p in self.seq_path_x]
        seq_parent_dir_y = [__p.parent for __p in self.seq_path_y]
        
        __x = sorted(seq_parent_dir_x, key=self.__func_key_sort_parent_dir)
        __y = sorted(seq_parent_dir_y, key=self.__func_key_sort_parent_dir)
        
        for __x_dir, __y_dir in zip(__x, __y):
            assert __x_dir == __y_dir, 'X and Y must be in the same directory.'
        # end for
        
        if self.time_aggregation_per is not None:
            stack_file_agg = []
            logger.info(f'Aggregating by timestamps. time_aggregation_per={self.time_aggregation_per}.')
            assert self.path_work_dir is not None, 'self.path_work_dir must be set.'
            
            path_new_root_dir = self.path_work_dir / 'aggregated'
            for __x_dir, __y_dir in zip(__x, __y):
                __path_new_data_parent_dir = path_new_root_dir / __x_dir.name
                __path_new_data_parent_dir.mkdir(parents=True, exist_ok=True)
                stack_file_agg.append(__path_new_data_parent_dir)

                tensor_agg_x = self.__aggregated_spatio_temporal_tensor_by_timestamp(__x_dir / 'x.pt')
                tensor_agg_y = self.__aggregated_spatio_temporal_tensor_by_timestamp(__y_dir / 'y.pt')
                
                torch.save({'array': tensor_agg_x}, __path_new_data_parent_dir / 'x.pt')
                torch.save({'array': tensor_agg_y}, __path_new_data_parent_dir / 'y.pt')                
            # end for
                    
            # TODO dataset arguments
            dataset = FileBackendSensorSampleBasedDataset(stack_file_agg)
        else:                    
            # TODO dataset arguments
            dataset = FileBackendSensorSampleBasedDataset(__x)
        # end if
        dataset.run_files_validation()
        
        return dataset
      
    # --------------------------------------------------------------------
    # Public API
    
    
    def get_dataset(self) -> ty.List[PossibleDatasetType]:
        """Public API.
        """	
        generated_datasets = []
        
        if self.dataset_type_backend == 'ram':
            if self.dataset_type_charactersitic == 'static':
                # TODO check data, np.ndarray or torch.Tensor
                assert self.array_x is not None, 'self.array_x must be set.'
                assert self.array_y is not None, 'self.array_y must be set.'
                if isinstance(self.array_x, np.ndarray):
                    assert isinstance(self.array_y, np.ndarray), 'self.array_x and self.array_y must be same type.'
                    logger.info('Case: `SimpleDataset` dataset class.')
                    dataset = SimpleDataset(
                        torch.from_numpy(self.array_x), 
                        torch.from_numpy(self.array_y)
                    )
                    generated_datasets.append(dataset)
                elif isinstance(self.array_x, torch.Tensor):
                    assert isinstance(self.array_y, torch.Tensor), 'self.array_x and self.array_y must be same type.'
                    dataset = SimpleDataset(self.array_x, self.array_y)
                    generated_datasets.append(dataset)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        elif self.dataset_type_backend in ('file', 'flexible-file'):
            if self.dataset_type_charactersitic == 'static':
                logger.info('Case: `FileBackendStaticDataset` dataset class.')
                dataset = self.__generate_file_backend_static_dataset()
                generated_datasets.append(dataset)
            elif self.dataset_type_charactersitic in ('sensor_st',):
                if self.dataset_type_algorithm == 'sample_based':
                    logger.info(f'Case: FileBackendSensorSampleBasedDataset.')
                    dataset = self.__generate_file_backend_sensor_dataset()
                    generated_datasets.append(dataset)
                else:
                    raise NotImplementedError('No cased matched. Aborting.')
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        
            
            
        return generated_datasets  # type: ignore