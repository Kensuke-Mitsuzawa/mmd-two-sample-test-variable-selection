import typing as ty
import logging
import copy
import dask
import dask.config
import gc
from pathlib import Path

import toml
import dacite

import numpy as np
import torch

from distributed import Client, LocalCluster

from ..logger_unit import handler

from ..datasets import (
    BaseDataset,
    FileBackendStaticDataset,
    FileBackendOneTimeLoadStaticDataset,
    SimpleDataset
)

from .module_configs import (
    CvSelectionConfigArgs,
    PossibleDataCharacteristic,
    LinearVariableSelectionConfigArgs,
)
from .interface_config_args import (
    InterfaceConfigArgs,
)
from .data_objects import BasicVariableSelectionResult, OutputObject

from .module_utils_wasserstein_independent import module_wasserstein_independent
from .module_linear_regression import module_linear_variable_selection
from .dataset_generater import DatasetGenerater, PossibleInputType
from .module_sample_based import module_mmd_sample_based

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# --------------------------------------------------------------------------- #



class Interface(object):
    def __init__(self, config_args: InterfaceConfigArgs):
        """I do not describe arguments here. Please refer to `InterfaceConfigArgs`.
        """
        # --------------------------------------------------------------------------- #
        # Attributes
        self.dask_cluster = None
        self.dask_client = None
        
        self.path_work_dir: ty.Optional[Path] = None
        self.path_ml_logger_dir: ty.Optional[Path] = None
        self.path_model_dir: ty.Optional[Path] = None
        
        self.detection_sample_based: ty.Optional[BasicVariableSelectionResult] = None
        # --------------------------------------------------------------------------- #        
        
        self.config_args = config_args
        self.__validate_approach_parameters()
        
    def __init_dask_client(self, cluster_mode: str = 'detection') -> ty.Tuple[ty.Optional[LocalCluster], ty.Optional[Client]]:
        assert cluster_mode in ('detection', ), f'Invalid cluster_mode: {cluster_mode}'
        
        # if cluster_mode == 'preprocessing':
        #     dask_config = self.config_args.resource_config_args.dask_config_preprocessing
        if cluster_mode == 'detection':
            dask_config = self.config_args.resource_config_args.dask_config_detection
        else:
            raise ValueError(f'Invalid cluster_mode: {cluster_mode}')
        # end if
        assert dask_config is not None, f'dask_config is not set.'
        
        if dask_config.dask_scheduler_host is None:
            dask_cluster = None
            dask_client = None
            return dask_cluster, dask_client
        else:            
            if dask_config.is_use_local_dask_cluster:
                logger.debug(f'Creating local dask cluster...')
                __distination = f'{dask_config.dask_scheduler_host}:{dask_config.dask_scheduler_port}'
                
                if self.config_args.detector_algorithm_config_args.mmd_cv_selection_args is not None and isinstance(self.config_args.detector_algorithm_config_args.mmd_cv_selection_args, CvSelectionConfigArgs):
                    is_use_dataloader_workers = self.config_args.detector_algorithm_config_args.mmd_cv_selection_args.dataloader_n_workers_train_dataloader > 0 or\
                        self.config_args.detector_algorithm_config_args.mmd_cv_selection_args.dataloader_n_workers_validation_dataloader > 0
                else:
                    is_use_dataloader_workers = False
                # end if
                logger.info(f'Dataloader requires subprocessing -> {is_use_dataloader_workers}')
                
                if is_use_dataloader_workers:
                    dask.config.set(distributed__worker__daemon=False)
                # end if
                
                if dask_config.dask_dashboard_address is None:
                    __dask_dashboard_address = ":8787"
                else:
                    __dask_dashboard_address = dask_config.dask_dashboard_address
                # end if
                
                dask_cluster = LocalCluster(
                    __distination,
                    n_workers=dask_config.dask_n_workers,
                    threads_per_worker=dask_config.dask_threads_per_worker,
                    dashboard_address=__dask_dashboard_address,)
                dask_client = Client(self.dask_cluster)
                logger.debug(f'Local dask cluster is created.')
            else:
                dask_cluster = None
                logger.debug(f'Connecting to dask cluster...')
                dask_scheduler_address = f'tcp://{dask_config.dask_scheduler_host}:{dask_config.dask_scheduler_port}'
                dask_client = Client(address=dask_scheduler_address)
                logger.debug(f'Connected to dask cluster.')
            # end if
            return dask_cluster, dask_client
        # end if
        
        
    # --------------------------------------------------------------------------- #
    # Private APIs for data management
    
    def __validate_approach_parameters(self):
        """Check arguments if the specified detector is valid for the combination of data representation and dataset type.
        """
        if self.config_args.approach_config_args.approach_data_representation == 'sample_based':
            # three possible data representation: static, sensro_st, trajectory_st
            assert self.config_args.data_config_args.dataset_type_charactersitic in PossibleDataCharacteristic
            if self.config_args.data_config_args.dataset_type_charactersitic == 'static':
                # all approach_variable_detector
                pass
            elif self.config_args.data_config_args.dataset_type_charactersitic == 'sensor_st':
                # all approach variable detector
                # TODO: working.
                raise NotImplementedError('This combination is still under development.\
                    I advice using approach_data_representation=`static` intead.')
            else:
                raise ValueError(f'Invalid combination of approach_data_representation and dataset_type_charactersitic')
        else:
            raise ValueError(f'Invalid approach_data_representation: {self.config_args.approach_config_args.approach_data_representation}')
        
    def __input_data_preparation(self, data_x: PossibleInputType, data_y: PossibleInputType) -> ty.Tuple[PossibleInputType, PossibleInputType]:
        """Prepare input data for the specified approach.
        
        When the input is already on RAM, do nothing.
        When the input is a file, just check if the key exists in the file.
        When the input is a directory, run rglob and listing up all files.
        """
        if self.config_args.data_config_args.dataset_type_backend == 'ram':
            assert isinstance(data_x, torch.Tensor), f'data_x must be torch.Tensor object. Are you giving in str'
            assert isinstance(data_y, torch.Tensor), f'data_y must be torch.Tensor object. Are you giving in str'            
            return data_x, data_y
        elif self.config_args.data_config_args.dataset_type_backend == 'flexible-file':
            # check if the key exists in the file.
            assert isinstance(data_x, Path), f'data_x must be Path object. Are you giving in str?'
            assert isinstance(data_y, Path), f'data_y must be Path object. Are you giving in str?'
            
            if data_x.is_file():
                assert data_y.is_file(), f'data_y must be a file. {data_y} is not a file.'
                return data_x, data_y
            else:            
                # comment: Case. where a directory path is given. So, I search files under the given directory.
                __list_file_x = list(data_x.rglob(f'*/{self.config_args.data_config_args.file_name_x}'))
                __list_file_y = list(data_x.rglob(f'*/{self.config_args.data_config_args.file_name_y}'))
                
                assert len(__list_file_x) > 0, f'No file found for {data_x}'
                assert len(__list_file_y) > 0, f'No file found for {data_y}'                
                # asserting that length of file-list same for x and y. 
                assert len(__list_file_x) == len(__list_file_y), f'Length of file-list is different for x and y.'
                
                seq_parent_dir_x = [__p for __p in __list_file_x]
                seq_parent_dir_y = [__p for __p in __list_file_y]                
                
                return seq_parent_dir_x, seq_parent_dir_y
            # end if

        elif self.config_args.data_config_args.dataset_type_backend == 'file':
            # check if the key exists in the file.
            assert isinstance(data_x, Path), f'data_x must be Path object. Are you giving in str?'
            assert isinstance(data_y, Path), f'data_y must be Path object. Are you giving in str?'
            input_is_file = data_x.is_file()
            if input_is_file:
                # comment: case. the input file x, y are a single file.
                assert data_y.is_file()
                # TODO: I want to open npz as well.
                assert self.config_args.data_config_args.key_name_array in torch.load(data_x)
                assert self.config_args.data_config_args.key_name_array in torch.load(data_y)
                return data_x, data_y
            else:
                # comment: Case. where a directory path is given. So, I search files under the given directory.
                __list_file_x = list(data_x.rglob(f'*/{self.config_args.data_config_args.file_name_x}'))
                __list_file_y = list(data_x.rglob(f'*/{self.config_args.data_config_args.file_name_y}'))
                
                assert len(__list_file_x) > 0, f'No file found for {data_x}'
                assert len(__list_file_y) > 0, f'No file found for {data_y}'                
                # asserting that length of file-list same for x and y. 
                assert len(__list_file_x) == len(__list_file_y), f'Length of file-list is different for x and y.'
                
                seq_parent_dir_x = [__p for __p in __list_file_x]
                seq_parent_dir_y = [__p for __p in __list_file_y]                
                
                return seq_parent_dir_x, seq_parent_dir_y
        else:
            raise ValueError(f'Invalid dataset_type_backend: {self.config_args.data_config_args.dataset_type_backend}')
        
    def __set_datasets(self) -> ty.Tuple[ty.List[BaseDataset], ty.Optional[ty.List[BaseDataset]]]:
        """Prepare datasets for the specified approach.
        Test-dataset can be None.
        
        Returns
        -------
        seq_train_dataset_obj: ty.List[BaseDataset]
            List of train dataset object.
        seq_test_dataset_obj: ty.Optional[ty.List[BaseDataset]]
            List of test dataset object.
            If test dataset is not given and ratio is -1, None is returned.
        """
        # data preparation. Calling DatasetGenerater interface.
        assert isinstance(self.config_args.resource_config_args.path_work_dir, Path), \
            f'path_work_dir must be Path object. Are you giving in str? Something bug?'
        path_data_dir = self.config_args.resource_config_args.path_work_dir / self.config_args.resource_config_args.dir_name_data

        logger.debug(f'Validating and initializaing datasets...')
        seq_train_dataset_obj: ty.List[BaseDataset]
        train_input_data_x, train_input_data_y = self.__input_data_preparation(
            self.config_args.data_config_args.data_x_train, 
            self.config_args.data_config_args.data_y_train)
        train_dataset_generater = DatasetGenerater(
            data_x=train_input_data_x,
            data_y=train_input_data_y,
            dataset_type_backend=self.config_args.data_config_args.dataset_type_backend,
            dataset_type_charactersitic=self.config_args.data_config_args.dataset_type_charactersitic,
            dataset_type_algorithm=self.config_args.approach_config_args.approach_data_representation,
            time_aggregation_per=self.config_args.data_config_args.time_aggregation_per,
            key_name_array=self.config_args.data_config_args.key_name_array,
            path_work_dir=path_data_dir)
        seq_train_dataset_obj = train_dataset_generater.get_dataset()
        logger.debug(f'Dataset is ready.')
        
        # generating test dataset.
        # case of being non-test dataset, you can split dataset from `train_dataset_obj`.
        seq_test_dataset_obj: ty.Optional[ty.List[BaseDataset]]
        if self.config_args.data_config_args.data_x_test is not None and self.config_args.data_config_args.data_y_test is not None:
            logger.debug(f'Validating and initializaing test datasets...')
            __test_input_data_x, __test_input_data_y = self.__input_data_preparation(
                self.config_args.data_config_args.data_x_test, 
                self.config_args.data_config_args.data_y_test)
            __test_dataset_generater = DatasetGenerater(
                data_x=__test_input_data_x,
                data_y=__test_input_data_y,
                dataset_type_backend=self.config_args.data_config_args.dataset_type_backend,
                dataset_type_charactersitic=self.config_args.data_config_args.dataset_type_charactersitic,
                dataset_type_algorithm=self.config_args.approach_config_args.approach_data_representation,
                time_aggregation_per=self.config_args.data_config_args.time_aggregation_per,
                key_name_array=self.config_args.data_config_args.key_name_array,
                path_work_dir=path_data_dir)
            seq_test_dataset_obj = __test_dataset_generater.get_dataset()
            logger.debug(f'Test Dataset is ready.')            
        elif self.config_args.data_config_args.ratio_train_test != -1:
            # splitting dataset into train and test.
            logger.info(f'Splitting dataset into train and test...')
            if self.config_args.approach_config_args.approach_data_representation == 'sample_based':
                # the list must be 1
                assert len(seq_train_dataset_obj) == 1
                train_dataset_obj = seq_train_dataset_obj[0]
                __test = train_dataset_obj.split_train_and_test(train_ratio=self.config_args.data_config_args.ratio_train_test)
                seq_test_dataset_obj = [__test.test_dataset]
                seq_train_dataset_obj = [__test.train_dataset]
            else:
                raise ValueError(f'Invalid approach_data_representation: {self.config_args.approach_config_args.approach_data_representation}')
        else:
            # no splitting.
            logger.info(f'No splitting dataset into train and test...')
            seq_test_dataset_obj = None
        # end if
        
        return seq_train_dataset_obj, seq_test_dataset_obj
    
    # --------------------------------------------------------------------------- #
    # Private APIs for variable selection
            

    def __run_variable_selection_sample_based(self, 
                                              seq_dataset_train: ty.List[BaseDataset], 
                                              seq_dataset_test: ty.Optional[ty.List[BaseDataset]],
                                              dask_client: ty.Optional[Client] = None
                                              ) -> BasicVariableSelectionResult:
        """Running detection task for sample-based data representation.
        """
        assert len(seq_dataset_train) == 1
        assert seq_dataset_test is None or (isinstance(seq_dataset_test, list) and len(seq_dataset_test) == 1)
        
        dataset_train = seq_dataset_train[0]
        dataset_test = seq_dataset_test[0] if seq_dataset_test is not None else None
        
        assert isinstance(dataset_train, (SimpleDataset, FileBackendStaticDataset, FileBackendOneTimeLoadStaticDataset)), \
            f'Invalid dataset_obj: {dataset_train.__class__.__name__}'
        
        if self.config_args.approach_config_args.approach_variable_detector == 'wasserstein_independence':
            selection_result = module_wasserstein_independent.main(dataset_train, dataset_test, self.config_args.resource_config_args, dask_client)
        elif self.config_args.approach_config_args.approach_variable_detector == 'linear_variable_selection':
            assert self.config_args.detector_algorithm_config_args.linear_variable_selection_args is not None
            assert isinstance(self.config_args.detector_algorithm_config_args.linear_variable_selection_args, LinearVariableSelectionConfigArgs)
            assert self.path_model_dir is not None, 'self.path_model_dir is not set.'
            selection_result = module_linear_variable_selection.main(
                dataset_train=dataset_train, 
                args=self.config_args.detector_algorithm_config_args.linear_variable_selection_args,
                path_work_dir=self.path_model_dir,
                dask_client=dask_client)
        elif self.config_args.approach_config_args.approach_variable_detector == 'interpretable_mmd':
            selection_result = module_mmd_sample_based.main(
                config_args=self.config_args,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                dask_client=dask_client)
        else:
            raise ValueError(f'Invalid approach_variable_detector: {self.config_args.approach_config_args.approach_variable_detector}')
    
        return selection_result

    
    # --------------------------------------------------------------------------- #

    @classmethod
    def from_toml(cls, toml_path: Path):
        assert toml_path.exists(), f'{toml_path} does not exist.'
        config_obj = toml.load(toml_path)
        logger.debug(f'config toml file from {toml_path}, validating...')
        args_obj = dacite.from_dict(InterfaceConfigArgs, config_obj)
        logger.debug(f'The config toml file is valid.')
        
        return cls(args_obj)
        
    def fit(self):        
        # making directory for saving objects.
        assert self.config_args.resource_config_args.path_work_dir is not None
        self.path_work_dir = Path(self.config_args.resource_config_args.path_work_dir)
        self.path_work_dir.mkdir(parents=True, exist_ok=True)
        
        self.path_ml_logger_dir = self.path_work_dir / self.config_args.resource_config_args.dir_name_ml_logger
        self.path_ml_logger_dir.mkdir(parents=True, exist_ok=True)
        
        self.path_model_dir = self.path_work_dir / self.config_args.resource_config_args.dir_name_model
        self.path_model_dir.mkdir(parents=True, exist_ok=True)

        # adding file-handler to a logger.
        self.path_model_logs = self.path_work_dir / self.config_args.resource_config_args.dir_name_logs
        self.path_model_logs.mkdir(parents=True, exist_ok=True)
        __logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        __fileHandler = logging.FileHandler(self.path_model_logs / "log.txt")
        __fileHandler.setFormatter(__logFormatter)
        logger.addHandler(__fileHandler)

        # comment: a list of dataset in `time_slicing` cases. 
        # for integration reasons, I use the list format.
        seq_train_dataset, seq_test_dataset = self.__set_datasets()
        
        if self.config_args.resource_config_args.dask_config_detection.distributed_mode == 'dask':
            dask_cluster, dask_client = self.__init_dask_client(cluster_mode='detection')
        else:
            dask_cluster, dask_client = None, None
        # end if        
        
        if self.config_args.approach_config_args.approach_data_representation == 'sample_based':
            __detection_sample_based = self.__run_variable_selection_sample_based(
                seq_train_dataset, 
                seq_test_dataset,
                dask_client=dask_client)
            self.detection_sample_based = __detection_sample_based
        else:
            raise ValueError(f'Invalid approach_data_representation: {self.config_args.approach_config_args.approach_data_representation}')
        # end if
        
        if dask_client is not None:
            dask_client.close()
            del dask_client
            gc.collect()
        # end if
        
        if dask_cluster is not None:
            dask_cluster.close()
            del dask_cluster
            gc.collect()
        # end if
        
    def get_result(self, output_mode: str = 'simple') -> OutputObject:
        assert output_mode in ['simple', 'verbose']
        assert self.detection_sample_based is not None, \
            'fit() is not called yet. Please call fit() before calling get_result().'
                
        if output_mode == 'simple':
            __detection: BasicVariableSelectionResult = copy.deepcopy(self.detection_sample_based)
            __detection.verbose_field = None
        else:
            __detection = self.detection_sample_based
        # end if
        detection_result_sample_based = __detection
         
        # TODO I need integration, reshaping output objects here. May be in `get_result()`.
        object_return = OutputObject(
            configurations=self.config_args,
            detection_result_sample_based=detection_result_sample_based)
        return object_return
