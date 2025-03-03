from pathlib import Path
import typing as ty
import math
from tqdm import tqdm
import logging

from tempfile import mkdtemp
import dask
import dask.array
from distributed import Client

import numpy as np
import torch

from ..logger_unit import handler

from .base import BaseDataset
from .file_backend_static_dataset import FileBackendStaticDataset
from .file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from .sensor_sample_based_dataset import FileBackendSensorSampleBasedDataset
from .ram_backend_static_dataset import SimpleDataset

from ..utils.get_time_slicing_points import adf_test_time_slicing



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


Algorithm type
--------------------

There are roughly two types of algorithms; sample-based and time-slicing.

The time-slicing dataset takes just a one pair of X, Y.

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
                 dataset_type_algorithm: str,
                 time_slicing_per: ty.Optional[ty.Union[int, str, ty.List[int]]] = 'auto',
                 time_aggregation_per: ty.Optional[int] = 100,
                 key_name_array: ty.Optional[str] = 'array',
                 path_work_dir: ty.Optional[Path] = Path('/tmp/dataset_generater'),
                 dataset_client: ty.Optional[Client] = None,):
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
            'static', 'sensor_st', 'trajectory_st'
        dataset_type_algorithm: str
            'sample_based', 'time_slicing'
        time_slicing_per: int
            This is only used when dataset_type_algorithm is 'time_slicing'.
            This is the number of timestamps per batch.
        time_aggregation_per: int
            This is only used when dataset_type_algorithm is 'sample_based'.
            This is the number of timestamp to be aggregated.
        key_name_array: str
            When the data-source is from file, each torch `.pt` file must contain a dictionary.
            The `key_name_array` specifies the key name of the array.
        path_work_dir: Path
            This is only used when dataset_type_algorithm is 'time_slicing'.
            This is the path to save the time-slicing dataset.
            Be informed that your disk has enough space to save the dataset.
        dataset_client: Client
            This is only used when dataset_type_algorithm is 'time_slicing'.
            This is the dask client to be used for parallel computing.
        """
        # --------------------------------------------------------------------
        # definitions
        self.array_x: ty.Optional[ty.Union[np.ndarray, torch.Tensor]] = None
        self.array_y: ty.Optional[ty.Union[np.ndarray, torch.Tensor]] = None
        
        self.path_x: ty.Optional[Path] = None
        self.path_y: ty.Optional[Path] = None
        
        self.seq_path_x: ty.Optional[ty.List[Path]] = None
        self.seq_path_y: ty.Optional[ty.List[Path]] = None
        
        self.dataset_client = dataset_client
        # --------------------------------------------------------------------
        
        assert dataset_type_backend in ('ram', 'file', 'flexible-file')
        assert dataset_type_charactersitic in ('static', 'sensor_st', 'trajectory_st')
        assert dataset_type_algorithm in ('sample_based', 'time_slicing')
        
        self.dataset_type_backend = dataset_type_backend
        self.dataset_type_charactersitic = dataset_type_charactersitic
        self.dataset_type_algorithm = dataset_type_algorithm
        
        self.time_slicing_per = time_slicing_per
        self.time_aggregation_per = time_aggregation_per
        self.key_name_array = key_name_array
        
        # used only when dataset_type_algorithm=time_slicing.
        if dataset_type_charactersitic == 'sensor_st':
            self.is_value_between_timestamp = False
        else:
            self.is_value_between_timestamp = True
        # end if
        
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
                
    def __validate_case_time_slicing(self, path_x: Path, path_y: Path):
        assert self.dataset_type_backend == 'file', 'dataset_type_backend must be file.'
        assert self.dataset_type_charactersitic in ('sensor_st', 'trajectory_st'), 'dataset_type_charactersitic must be sensor_st or trajectory_st.'
        assert self.dataset_type_algorithm in ('time_slicing',), 'dataset_type_algorithm must be time_slicing.'
        
        self.path_x = path_x
        self.path_y = path_y
        
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
    # Private API for manipulating files, for time-slicing algorithm.
    
    
    @staticmethod
    def __func_sub_distributed__split_into_timestamps(args: _ArgFuncDistributed) -> ty.Tuple[int, Path]:
        """Only for time-slicing algorithm.
        An actual function of splitting a single (|S|, |T|) file into a set of (|S|) files.
        
        Returns
        -------
        ty.Tuple[int, Path]
            timestamp and the path to the directory, a parent directory of x.pt and y.pt.
        """
        timestamp: int = args.timestamp
        torch_x: ty.Union[Path, torch.Tensor] = args.torch_x
        torch_y: ty.Union[Path, torch.Tensor] = args.torch_y
        path_word_dir: Path = args.path_word_dir
        dataset_type_charactersitic: str = args.dataset_type_charactersitic
        is_value_between_timestamp: bool = args.is_value_between_timestamp
        
        if isinstance(torch_x, Path):
            torch_x = torch.load(torch_x)
        # end if            
        if isinstance(torch_y, Path):
            torch_y = torch.load(torch_y)
        # end if
        
        assert isinstance(torch_x, torch.Tensor), 'torch_x must be torch.Tensor.'
        assert isinstance(torch_y, torch.Tensor), 'torch_y must be torch.Tensor.'
        
        __path_data_dir = path_word_dir / f'{timestamp}'
        __path_data_dir.mkdir(parents=True, exist_ok=True)

        if dataset_type_charactersitic == 'sensor_st':
            __one_sample_x = torch_x[:, timestamp]
            __one_sample_y = torch_y[:, timestamp]
        elif dataset_type_charactersitic == 'trajectory_st':
            if is_value_between_timestamp:
                __one_sample_x = torch_x[:, (timestamp + 1), :] - torch_x[:, timestamp, :]
                __one_sample_y = torch_y[:, (timestamp + 1), :] - torch_y[:, timestamp, :]
            else:
                __one_sample_x = torch_x[:, timestamp, :]
                __one_sample_y = torch_y[:, timestamp, :]
        else:
            raise NotImplementedError('Undefiend case. Aborting.')
        # end if
        
        __one_sample_x_clone = __one_sample_x.clone()
        __one_sample_y_clone = __one_sample_y.clone()
        
        torch.save({'array': __one_sample_x_clone}, __path_data_dir / 'x.pt')
        torch.save({'array' :__one_sample_y_clone}, __path_data_dir / 'y.pt')

        return timestamp, __path_data_dir
    
    def __split_into_timestamps(self, 
                                path_data_x: Path, 
                                path_data_y: Path, 
                                path_word_dir: Path,
                                is_value_between_timestamp: bool) -> ty.List[Path]:
        """Private API. Called by `__generate_file_backend_any_time_slicing_dataset`.
        Splitting an array of Spatio-Temporal data into a set of arrays by timestamps.
        
        Parameters
        ----------
        is_value_between_timestamp : bool
            If True, the array element is a value of (t+1) - t, which is a time difference.
            If False, the array element is a value of t.
            When True, the sample size is |T|-1.
            
        Returns
        -------
        a list of parent directory paths. A parent directory is a parent of x.pt and y.pt.
        Ex. the parent directory is named timestamp number, such as `0`, `1`, `2`, ...
        """
        # warning: when is_value_between_timestamp=False and dataset_type_charactersitic=trajectory_st
        if is_value_between_timestamp is False and self.dataset_type_charactersitic == 'trajectory_st':
            logger.warning('ARE YOU SURE THAT is_value_between_timestamp=False and dataset_type_charactersitic=trajectory_st? Just for Confirmation.')
        # end if
        
        d_x = torch.load(path_data_x)    
        torch_x: torch.Tensor = d_x[self.key_name_array]

        d_y = torch.load(path_data_y)    
        torch_y: torch.Tensor = d_y[self.key_name_array]
        
        assert torch_x.shape == torch_y.shape, 'X and Y must be same shape.'
        
        if self.dataset_type_charactersitic == 'sensor_st':
            assert len(torch_x.shape) == 2, 'The data shape must be 2D.'
        elif self.dataset_type_charactersitic == 'trajectory_st':
            assert len(torch_x.shape) == 3, 'The data shape must be 3D.'
        else:
            raise NotImplementedError('Undefiend case. Aborting.')
        # end if
        
        path_word_dir.mkdir(parents=True, exist_ok=True)
        
        if is_value_between_timestamp:
            n_timestamps = torch_x.shape[1] - 1
        else:
            n_timestamps = torch_x.shape[1]
        # end if
        
        seq_parent_dir: ty.List[Path] = []
    
        logger.info(f'n_timestamps={n_timestamps}. Start splitting...')
    
        if self.dataset_client is None:
            seq_args = []
            for __timestamp in range(0, n_timestamps):
                __args = _ArgFuncDistributed(
                    timestamp=__timestamp,
                    torch_x=torch_x,
                    torch_y=torch_y,
                    path_word_dir=path_word_dir,
                    dataset_type_charactersitic=self.dataset_type_charactersitic,
                    is_value_between_timestamp=is_value_between_timestamp)
                seq_args.append(__args)
            # end for            
            t_process_out = [self.__func_sub_distributed__split_into_timestamps(__args) for __args in tqdm(seq_args)]
        else:
            logger.debug(f'Generating timeslicing datasets by dask.')
            seq_args = []

            __path_tmp = Path(mkdtemp())
            __path_tmp.mkdir(parents=True, exist_ok=True)
            torch.save(torch_x, __path_tmp / 'x.pt')
            torch.save(torch_y, __path_tmp / 'y.pt')
            
            for __timestamp in range(0, n_timestamps):
                __args = _ArgFuncDistributed(
                    timestamp=__timestamp,
                    torch_x=__path_tmp / 'x.pt',
                    torch_y=__path_tmp / 'y.pt',
                    path_word_dir=path_word_dir,
                    dataset_type_charactersitic=self.dataset_type_charactersitic,
                    is_value_between_timestamp=is_value_between_timestamp)
                seq_args.append(__args)
            # end for            
            
            __dask_queue = self.dataset_client.map(self.__func_sub_distributed__split_into_timestamps, seq_args)
            t_process_out = self.dataset_client.gather(__dask_queue)
        # end if
        
        sorted_t_process_out = sorted(t_process_out, key=lambda x: x[0])  # sort by timestamp  # type: ignore
        seq_parent_dir = [__t[1] for __t in sorted_t_process_out]
        logger.info(f'Timeslicing datasets are ready to use.')
        
        return seq_parent_dir
  
    # def __generate_file_backend_any_time_slicing_dataset(self) -> ty.List[ty.Union[FileBackendSensorTimeSlicingDataset, FileBackendTrajectoryTimeSlicingDataset]]:
    #     """Generating a set of dataset from time-slicing algorithm.
    #     Since time-slicing algorithm splits a set of timestamps $T$ into batches, this methods returns a list of datasets.
        
    #     The expected input file is a pair of `Path`, `tuple(path_x, path_y)`.
    #     When the `sensor_st` case, the data shape is $\mathbbb{R}^{|S| \times |T|}$.
    #     When the `trajectory_st` case, the data shape is $\mathbbb{R}^{|A| \times |T| \times C}$.
    #     The simulation output is supposed to be only one pair.
        
    #     This method splits the one simulation output array into a set of arrays by the number of $|T|$.
        
    #     Returns
    #     -------
    #     ty.List[ty.Union[FileBackendSensorTimeSlicingDataset, FileBackendSensorTimeSlicingDataset]]
    #         A list of dataset. The list length is the number of $|T|$.
    #     """
    #     assert self.dataset_type_backend == 'file', 'dataset_type_backend must be file.'
    #     assert self.dataset_type_charactersitic in ('sensor_st', 'trajectory_st'), 'dataset_type_charactersitic must be sensor_st or trajectory_st.'
    #     assert self.dataset_type_algorithm in ('time_slicing',), 'dataset_type_algorithm must be time_slicing.'
                
    #     assert self.time_slicing_per is not None, 'self.time_slicing_per must be set.'
    #     assert self.path_x is not None, 'self.path_x must be set.'
    #     assert self.path_y is not None, 'self.path_y must be set.'
        
    #     assert self.path_x.exists(), 'self.path_x must exist.'
    #     assert self.path_y.exists(), 'self.path_y must exist.'
        
    #     if self.path_x.is_file():
    #         # comment: when input x,y is a single file, time-slicing input data files are not ready yet.
    #         assert self.path_y.is_file(), f'path_y must a single input file. is-file -> {self.path_y.is_file()}'
    #         # `seq_parent_dir` is a list of parent directories, which is a parent of x.pt and y.pt. 
    #         seq_parent_dir = self.__split_into_timestamps(path_data_x=self.path_x, 
    #                                             path_data_y=self.path_y, 
    #                                             path_word_dir=self.path_work_dir,
    #                                             is_value_between_timestamp=self.is_value_between_timestamp)
    #     else:
    #         # comment: when input x,y are lists of files, time-slicing input files are ready.
    #         # skip splitting into timestamps. 
    #         __seq_x = self.path_x.glob('**/x.pt')
    #         __seq_y = self.path_x.glob('**/y.pt')
    #         seq_parent_dir = []
    #         for __path_x_file in __seq_x:
    #             __path_parent_dir = __path_x_file.parent  # `__path_parent_dir` is a parent directory of x.pt
    #             assert (__path_parent_dir / 'y.pt').exists(), f'Not found -> {(__path_parent_dir / "y.pt")}'
    #             seq_parent_dir.append(__path_parent_dir)
    #         # end for
    #     # end if
        
        
    #     # auto time-slicing per
    #     seq_t_start_end_bucket: ty.List[ty.Tuple[int, int]] = []
    #     if isinstance(self.time_slicing_per, str):
    #         assert self.time_slicing_per == 'auto', 'self.time_slicing_per must be auto.'
    #         logger.debug(f'Generating time-slicing dataset by auto mode.')
    #         # TODO I must replace here as trend-filtering based.
    #         __seq_end_buckets = adf_test_time_slicing(
    #             array_x=self.path_x,
    #             array_y=self.path_y,
    #             n_span_time_slicing=100,
    #             ratio_sampling_sensor=0.8,
    #             threshold_ratio_sensors=0.8)
    #         logger.debug('Generated time-slicing points.')
    #         for __i, __step_end_bucket in enumerate(__seq_end_buckets):
    #             if __i == 0:
    #                 __time_from = 0
    #             else:
    #                 __time_from = __seq_end_buckets[__i - 1]
    #             # end if
                
    #             if __i == len(__seq_end_buckets) - 1:
    #                 __time_to = len(seq_parent_dir)
    #             else:
    #                 __time_to = __step_end_bucket
    #             # end if
                    
    #             seq_t_start_end_bucket.append((__time_from, __time_to))
    #         # end for
    #     elif isinstance(self.time_slicing_per, int):
    #         assert self.time_slicing_per > 0, 'self.time_slicing_per must be positive.'
    #         _size_one_time_bucket = math.ceil(len(seq_parent_dir) / self.time_slicing_per)
            
    #         for __index_time_bucket in range(0, _size_one_time_bucket):
    #             __time_from: int = __index_time_bucket * self.time_slicing_per
    #             __time_to: int = (__index_time_bucket + 1) * self.time_slicing_per
                
    #             if len(seq_parent_dir) < __time_to:
    #                 __time_to = len(seq_parent_dir)
    #             # end if
    #             seq_t_start_end_bucket.append((__time_from, __time_to))
    #         # end for
            
    #     elif isinstance(self.time_slicing_per, list):
    #         assert len(self.time_slicing_per) > 0, 'self.time_slicing_per must be positive.'
    #         assert all([__n > 0 for __n in self.time_slicing_per]), 'self.time_slicing_per must be positive.'
    #         for __i, __step_end_bucket in enumerate(self.time_slicing_per):
    #             if __i == 0:
    #                 __time_from = 0
    #             else:
    #                 __time_from = self.time_slicing_per[__i - 1]
    #             # end if
                
    #             if __i == len(self.time_slicing_per) - 1:
    #                 __time_to = len(seq_parent_dir)
    #             else:
    #                 __time_to = __step_end_bucket
    #             # end if
                    
    #             seq_t_start_end_bucket.append((__time_from, __time_to))
    #         # end for
            
    #     else:
    #         raise TypeError('self.time_slicing_per must be int, str or ty.List[int].')
    #     # end if
                    
    #     seq_dataset = []
        
    #     # self.time_slicing_per -> 
    #     for __t_time_start_end in seq_t_start_end_bucket:
    #         __time_from = __t_time_start_end[0]
    #         __time_to = __t_time_start_end[1]
            
    #         if len(seq_parent_dir) < __time_to:
    #             __time_to = len(seq_parent_dir)
    #         # end if
            
    #         if self.dataset_type_charactersitic == 'sensor_st':
    #             logger.info(f'Case: `FileBackendSensorTimeSlicingDataset` dataset class.')
    #             __dataset = FileBackendSensorTimeSlicingDataset(seq_parent_dir, __time_from, __time_to)
    #             __dataset.run_files_validation()
    #         elif self.dataset_type_charactersitic == 'trajectory_st':
    #             logger.info(f'Case: `FileBackendTrajectoryTimeSlicingDataset` dataset class.')
    #             __dataset = FileBackendTrajectoryTimeSlicingDataset(seq_parent_dir, __time_from, __time_to)
    #             __dataset.run_files_validation()
    #         else:
    #             raise NotImplementedError('Undefiend case. Aborting.')
    #         # end if
            
    #         seq_dataset.append(__dataset)
    #     # end for
    #     assert len(seq_dataset) > 0, 'Generated TimeSlicing dataset is empty. Something wrong. Abort.'
        
    #     return seq_dataset
        
    
    
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
            elif self.dataset_type_charactersitic in ('sensor_st', 'trajectory_st'):
                # two cases; time_slicing and sample_based
                if self.dataset_type_algorithm == 'time_slicing':
                    logger.info(f'Case: dataset_type_backend={self.dataset_type_backend} \
                        and dataset_type_charactersitic={self.dataset_type_charactersitic} \
                            and dataset_type_algorithm={self.dataset_type_algorithm}')
                    __seq_dataset = self.__generate_file_backend_any_time_slicing_dataset()
                    generated_datasets = __seq_dataset
                elif self.dataset_type_algorithm == 'sample_based':
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