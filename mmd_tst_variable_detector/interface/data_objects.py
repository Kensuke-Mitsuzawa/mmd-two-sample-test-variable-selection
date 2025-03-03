import typing as ty
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch


from ..logger_unit import handler


from ..baselines.regression_based_variable_selection.tst_based_regression_tuner import TstBasedRegressionTunerResult
from ..detection_algorithm.cross_validation_detector.cross_validation_detector import CrossValidationTrainedParameter
from ..detection_algorithm.detection_algorithm_one import AlgorithmOneResult

from ..detection_algorithm.search_regularization_min_max.optuna_module.commons import RegularizationSearchParameters

from .dataset_generater import PossibleInputType

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



PossibleDataRepresentation = ('sample_based', 'time_slicing')
PossibleVariableDetector = ('wasserstein_independence', 'linear_variable_selection', 'interpretable_mmd')
PossibleInterpretableMMD = ('cv_selection', 'algorithm_one', '')
PossibleDataCharacteristic = ('static', 'sensor_st', 'trajectory_st')
PossibleDatasetBackend = ('ram', 'file', 'flexible-file')

# --------------------------------------------------------------------------- #
# Argument definition.
# These Argument classes are possible to be initialized from toml as well.


@dataclass
class DistributedConfigArgs:
    """
    Args
    --------
    dask_scheduler_host: ty.Optional[str]
        Host name of dask scheduler.
        If you use local dask cluster, you can set it to 'localhost'.
    dask_scheduler_port: ty.Optional[int]
        Port number of dask scheduler.
        If you use local dask cluster, you can set it to 8786.
    is_use_local_dask_cluster: bool
        Whether you use local dask cluster or not.
        If you use local dask cluster, you can set it to True.
    n_workers: int
        Number of workers.
        This parameter is used only when you use local dask cluster.
    threads_per_worker: int
        Number of threads per worker.
        This parameter is used only when you use local dask cluster.
    distributed_mode: str
        Distributed mode.
        Either of following choices,
        1. 'single': single machine.
        2. 'dask': dask distributed.        
    """
    distributed_mode: str = 'dask'
    
    dask_scheduler_host: ty.Optional[str] = '0.0.0.0'
    dask_scheduler_port: ty.Optional[int] = 8786
    dask_dashboard_address: ty.Optional[str] = ':8787'
    dask_n_workers: int = 4
    dask_threads_per_worker: int = 4
    
    is_use_local_dask_cluster: bool = True
    
    def __post_init__(self):
        assert self.distributed_mode in ['single', 'dask', 'joblib'], f'{self.distributed_mode} is not supported.'
    

@dataclass
class ResourceConfigArgs:
    """Configuration class for resource. Resource includes path to files and distributed computing configuration.
    
    Parameters
    ----------
    path_work_dir: ty.Union[str, Path]
        Path to working directory.
        All files are saved in this directory.
    dir_name_ml_logger: str
        Directory name for ml_logger.
        ml_logger is a tool for logging.
    dir_name_model: str
        Directory name for saving trained models.
    dir_name_data: str
        Directory name for saving data.
    dir_name_logs: str
        Directory name for saving logs.
    train_accelerator: str
        Train accelerator.
        Either of following choices,
        1. 'cpu': cpu.
        2. 'gpu': gpu. Not `cuda`. 
            According to the documentation: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html
    distributed_config_preprocessing: ty.Optional[DistributedConfigArgs]
        Configuration for dask cluster.
    distributed_config_detection: ty.Optional[DistributedConfigArgs]
        Configuration for dask cluster.
        
        
    Updates:
    --------
    `dask_config_preprocessing` and `dask_config_detection` will be removed in the future. At the commit: `217309c35803cf1731cf772ebb8c2d05d7910977`
    """
    
    path_work_dir: ty.Union[str, Path, None] = Path('/tmp/mmd_tst_variable_detector/interface')
    
    dir_name_ml_logger: str = 'mlruns'
    dir_name_model: str = 'model'  # directory for saving trained models.
    dir_name_data: str = 'data'  # directory for saving data.
    dir_name_logs: str = 'logs'  # directory for saving logs.
    
    train_accelerator: str = 'cpu'
    
    # Distributed backend configurations    
    distributed_config_preprocessing: DistributedConfigArgs = DistributedConfigArgs()
    distributed_config_detection: DistributedConfigArgs = DistributedConfigArgs()

    # demolish the following two fields.
    # dask_config_preprocessing: DistributedConfigArgs = DistributedConfigArgs()
    # dask_config_detection: DistributedConfigArgs = DistributedConfigArgs()
    # I hold these fileds for the version compatibility.
    dask_config_preprocessing: ty.Optional[DistributedConfigArgs] = None
    dask_config_detection: ty.Optional[DistributedConfigArgs] = None

    
    def __post_init__(self):        
        if self.path_work_dir is None:
            self.path_work_dir = Path('/tmp/mmd_tst_variable_detector/interface') / datetime.now().isoformat()
        
        if isinstance(self.path_work_dir, str):
            self.path_work_dir = Path(self.path_work_dir)
        # end if
        if not self.path_work_dir.exists():
            logger.debug(f'Creating working directory: {self.path_work_dir}')
            self.path_work_dir.mkdir(parents=True, exist_ok=True)
        # end if    
        
        # I have to set values to `distributed_config_***` becasue a lot of modules use these values.
        self.dask_config_preprocessing = self.distributed_config_preprocessing
        self.dask_config_detection = self.distributed_config_detection


@dataclass
class DataSetConfigArgs:
    """
    Parameters
    ----------
    data_x_train: DataPossibleArgumentType
        Data for SampleSet x.
        When configuration is from toml, either of following choices,
        1. path to the file where data is stored
        2. path to the directory where a set of files are stored.
    data_y_train: DataPossibleArgumentType
        Data for SampleSet y.
        The same discription as `data_x`.
    data_x_test: DataPossibleArgumentType
        Data for SampleSet x.
        The same discription as `data_x`.
        You can leave it blank if you do not have test data.
    data_y_test: DataPossibleArgumentType
        Data for SampleSet y.
        The same discription as `data_x`.
        You can leave it blank if you do not have test data.
    dataset_type_backend: str
        Backend of dataset.
        Either of following choices,
        1. 'ram': data is stored in RAM.
        2. 'file': data is stored in file.
        3. 'flexible-file': data is stored in file, and you can specify the number of data to be loaded.
    dataset_type_charactersitic: str
        Charactersitic of dataset.
        Either of following choices,
        1. 'static': data is static.
        2. 'sensor_st': data is sensor data and static.
        3. 'trajectory_st': data is trajectory data and static.
    file_name_x: str
        File name of data for SampleSet x.
        This parameter is used only when `dataset_type_backend` is 'file'.
    file_name_y: str
        File name of data for SampleSet y.
        The same discription as `file_name_x`.
    time_aggregation_per: int
        Time aggregation period.
        This parameter is used only when `dataset_type_charactersitic` is 'sensor_st' or 'trajectory_st'.
    time_slicing_per: int | str | ty.List[int]s
        Time slicing period.
        This parameter is used only when `dataset_type_charactersitic` is 'sensor_st' or 'trajectory_st'.
        3 possible types of parameters.
        When `int`, all time-buckets have a constant time periods.
        When `ty.List[int]`, each time-bucket has a different time periods as you speficiy.
    ratio_train_test: float
        ratio of splitting dataset into train and test.
        Not used when you give `data_x_test` and `data_y_test`.
        When `ratio_train_test = -1`, no splitting.
    """
    data_x_train: PossibleInputType
    data_y_train: PossibleInputType
    
    data_x_test: ty.Optional[PossibleInputType]
    data_y_test: ty.Optional[PossibleInputType]
    
    dataset_type_backend: str
    dataset_type_charactersitic: str
    
    is_value_between_timestamp: bool = False
    
    key_name_array: str = 'array'
    file_name_x: str = 'x.pt'
    file_name_y: str = 'y.pt'
    
    time_aggregation_per: int = 100
    time_slicing_per: ty.Union[int, ty.List[int]] = 100
    
    ratio_train_test: float = 0.8
    
    def __post_init__(self):        
        self.dataset_type_backend = self.dataset_type_backend.lower()
        self.dataset_type_charactersitic = self.dataset_type_charactersitic.lower()
        
        assert self.dataset_type_backend in PossibleDatasetBackend
        assert self.dataset_type_charactersitic in PossibleDataCharacteristic

        if isinstance(self.data_x_train, str):
            self.data_x_train = Path(self.data_x_train)
        # end if
        if isinstance(self.data_y_train, str):
            self.data_y_train = Path(self.data_y_train)
        # end if

        if self.data_x_test == '':
            assert self.data_y_test == '', 'data_x_test is not given, but data_y_test is given.'
            self.data_x_test = None
            self.data_y_test = None
        else:
            logger.info(f'`ratio_train_test`={self.ratio_train_test} is ignored. \
                I use `data_x_test` and `data_y_test` instead.')
            
            if isinstance(self.data_x_test, str):
                self.data_x_test = Path(self.data_x_test)
            # end if
            if isinstance(self.data_y_test, str):
                self.data_y_test = Path(self.data_y_test)
        # end if


@dataclass
class ApproachConfigArgs:
    """Configuration class for approach
    
    Parameters
    ----------
    approach_data_representation: str
        Data representation.
        Either of following choices,
        1. 'sample_based': data is sample based.
        2. 'time_slicing': data is time series data.
    approach_variable_detector: str
        Variable detector.
        Either of following choices,
        1. 'wasserstein_independence': Independent variable detection by Wasserstein, and permutation test by sliced-wasserstein.
        2. 'linear_variable_selection': Linear Variable Selection.
        3. 'interpretable_mmd': Interpretable MMD.
    approach_interpretable_mmd: str
        Interpretable MMD.
        Either of following choices,
        1. 'cv_selection': Cross Validation Selection.
        2. 'algorithm_one': Algorithm One in the paper.
    """
    approach_data_representation: str
    approach_variable_detector: str
    approach_interpretable_mmd: str
    
    def __post_init__(self):
        self.approach_data_representation = self.approach_data_representation.lower()
        self.approach_variable_detector = self.approach_variable_detector.lower()
        self.approach_interpretable_mmd = self.approach_interpretable_mmd.lower()
        assert self.approach_data_representation in PossibleDataRepresentation, f'{self.approach_data_representation} is not supported. Possible choise -> {PossibleDataRepresentation}'
        assert self.approach_variable_detector in PossibleVariableDetector, f'{self.approach_variable_detector} is not supported. Possible choise -> {PossibleVariableDetector}'
        assert self.approach_interpretable_mmd in PossibleInterpretableMMD, f'{self.approach_interpretable_mmd} is not supported. Possible choise -> {PossibleInterpretableMMD}'



@dataclass
class CvSelectionConfigArgs:
    setting_name: str = 'config_rapid'  # when custom, set the custom class to the field ``
    max_epoch: int = 9999
    batch_size: int = -1
    
    n_subsampling: int = 5
    
    # kernel length scale parameter strategy
    aggregation_kernel_length_scale: str = 'median'
    
    # lambda search parameter
    approach_regularization_parameter: str = 'param_searching'
    # search_max_concurrent_job: int = 3
    # search_n_search_iteration: int = 10
    # n_regularization_parameter: int = 6
    parameter_search_parameter : RegularizationSearchParameters = RegularizationSearchParameters(
        n_search_iteration=10,
        max_concurrent_job=3,
        n_regularization_parameter=6)

    # distance function used for permutation test
    test_distance_functions: ty.Union[ty.Tuple[str, ...], ty.List[str]] = ('sinkhorn', 'sliced_wasserstein')
    n_permutation_test: int = 500

    # dataloader parameter
    dataloader_n_workers_train_dataloader: int = 0
    dataloader_n_workers_validation_dataloader: int = 0
    dataloader_persistent_workers: bool = False

    # a custom field of setting for mmd_optimisation_config. The class `MmdOptimisationConfigTemplate`.
    # Note: I do not specity the type of this field, because it raises curculation importing.
    custom_mmd_variable_selection_config: ty.Any = None
    
    def __post_init__(self):
        self.setting_name = self.setting_name.lower()
        assert self.setting_name in ['config_rapid', 'config_tpami_draft', 'custom'], f'{self.setting_name} is not supported.'
        assert self.approach_regularization_parameter in ('param_searching', 'fixed_range'), f'{self.approach_regularization_parameter} is not supported.'
        if self.setting_name == 'custom':
            assert self.custom_mmd_variable_selection_config is not None, 'When setting_name is custom, you must give custom_mmd_variable_selection_config.'            
            assert self.is_subclass_of(self.custom_mmd_variable_selection_config.__class__, 'MmdOptimisationConfigTemplate'), f'{self.custom_mmd_variable_selection_config} must be a subclass of MmdOptimisationConfigTemplate.'
        # end if
        
    def is_subclass_of(self, cls, parent_class_name):
        """
        Recursively check if the class `cls` or any of its parent classes
        has the name `parent_class_name`.
        
        :param cls: The class to check.
        :param parent_class_name: The name of the parent class to look for.
        :return: True if a class with the given name is found, False otherwise.
        """
        if cls.__name__ == parent_class_name:
            return True
        for base in cls.__bases__:
            if self.is_subclass_of(base, parent_class_name):
                return True
        return False

@dataclass
class AlgorithmOneConfigArgs:
    setting_name: str = 'config_rapid'
    max_epoch: int = 9999
    batch_size: int = -1
    
    # kernel length scale parameter strategy
    aggregation_kernel_length_scale: str = 'median'
        
    # lambda search parameter
    approach_regularization_parameter: str = 'search_objective_based'
    
    # a ratio to split the dataset into train and dev
    train_dev_split_ratio: float = 0.8

    parameter_search_parameter : RegularizationSearchParameters = RegularizationSearchParameters(
        n_search_iteration=10,
        max_concurrent_job=3,
        n_regularization_parameter=6)
    
    # distance function used for permutation test
    test_distance_functions: ty.Union[ty.Tuple[str, ...], ty.List[str]] = ('sinkhorn', 'sliced_wasserstein')
    n_permutation_test: int = 500

    # dataloader parameter
    dataloader_n_workers_train_dataloader: int = 0
    dataloader_n_workers_validation_dataloader: int = 0
    dataloader_persistent_workers: bool = False

    # a custom field of setting for mmd_optimisation_config
    custom_mmd_variable_selection_config: ty.Any = None
    
    def __post_init__(self):
        self.setting_name = self.setting_name.lower()
        assert self.setting_name in ['config_rapid', 'config_tpami_draft', 'custom'], f'{self.setting_name} is not supported.'
        assert self.approach_regularization_parameter in ('search_objective_based', 'auto_min_max_range'), f'{self.approach_regularization_parameter} is not supported.'
        if self.setting_name == 'custom':
            assert self.custom_mmd_variable_selection_config is not None, 'When setting_name is custom, you must give custom_mmd_variable_selection_config.'
            assert self.is_subclass_of(self.custom_mmd_variable_selection_config.__class__, 'MmdOptimisationConfigTemplate'), f'{self.custom_mmd_variable_selection_config} must be a subclass of MmdOptimisationConfigTemplate.'
        # end if

    def is_subclass_of(self, cls, parent_class_name):
        """
        Recursively check if the class `cls` or any of its parent classes
        has the name `parent_class_name`.
        
        :param cls: The class to check.
        :param parent_class_name: The name of the parent class to look for.
        :return: True if a class with the given name is found, False otherwise.
        """
        if cls.__name__ == parent_class_name:
            return True
        for base in cls.__bases__:
            if self.is_subclass_of(base, parent_class_name):
                return True
        return False



@dataclass
class LinearVariableSelectionConfigArgs:
    # Configurations for Optuna Parameter Tuning
    n_trials: int = 100
    n_cv: int = 5
    concurrent_limit: int = 4
    
    score_function: str = 'error'
    
    def __post_init__(self):
        assert self.score_function in ('error', 'p_value')



@dataclass
class DetectorAlgorithmConfigArgs:
    mmd_cv_selection_args: ty.Optional[ty.Union[str, CvSelectionConfigArgs]] = None
    mmd_algorithm_one_args: ty.Optional[ty.Union[str, AlgorithmOneConfigArgs]] = None
    linear_variable_selection_args: ty.Optional[ty.Union[str, LinearVariableSelectionConfigArgs]] = None
        
    def __post_init__(self):
        if self.mmd_cv_selection_args == '':
            self.mmd_cv_selection_args = None
        if self.mmd_algorithm_one_args == '':
            self.mmd_algorithm_one_args = None 
        if self.linear_variable_selection_args == '':
            self.linear_variable_selection_args = None
        # end if


@dataclass
class InterfaceConfigArgs:
    """The class for configuration of `Interface`.
    
    Parameters
    ----------
    resource_config_args: ResourceConfigArgs
        Configuration for resource.
    approach_config_args: ApproachConfigArgs
        Configuration for approach.
    data_config_args: DataSetConfigArgs
        Configuration for dataset.
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
        Configuration for detector algorithm.
    """
    resource_config_args: ResourceConfigArgs
    approach_config_args: ApproachConfigArgs
    data_config_args: DataSetConfigArgs
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
    
    def __post_init__(self):
        # reset the distributed backend.
        if isinstance(self.detector_algorithm_config_args.mmd_algorithm_one_args, AlgorithmOneConfigArgs):
            self.detector_algorithm_config_args.mmd_algorithm_one_args.parameter_search_parameter.backend = self.resource_config_args.dask_config_detection.distributed_mode
        if isinstance(self.detector_algorithm_config_args.mmd_cv_selection_args, CvSelectionConfigArgs):
            self.detector_algorithm_config_args.mmd_cv_selection_args.parameter_search_parameter.backend = self.resource_config_args.dask_config_detection.distributed_mode
        # end if
    
# --------------------------------------------------------------------------- #
# Return Object definition.

import json

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, np.ndarray):
            return obj.tolist()        

@dataclass
class BasicVariableSelectionResult:
    """Basic result of variable selection.
    """
    weights: ty.Union[ty.List[int], np.ndarray]
    variables: ty.List[int]
    p_value: float
    verbose_field: ty.Optional[ty.Union[TstBasedRegressionTunerResult, CrossValidationTrainedParameter, AlgorithmOneResult]] = None
    n_sample_training: ty.Optional[int] = None
    n_sample_test: ty.Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.weights, list):
            self.weights = np.array(self.weights)
        # end if
        
    def as_dict(self):
        """Converting into a dictionary object, expect `verbose_field` field.
        """
        if isinstance(self.weights, torch.Tensor):
            self.weights = self.weights.detach().cpu().numpy()
        # end if
        assert isinstance(self.weights, np.ndarray)
        d_obj = dict(
            weights=self.weights.tolist(),
            variables=self.variables,
            p_value=self.p_value,
            n_sample_training=self.n_sample_training,
            n_sample_test=self.n_sample_test
        )
        return d_obj


@dataclass
class OutputObject:
    configurations: InterfaceConfigArgs
    detection_result_sample_based: ty.Optional[BasicVariableSelectionResult]
    detection_result_time_slicing: ty.Optional[ty.List[BasicVariableSelectionResult]]
    
    def as_json(self) -> str:
        if isinstance(self.detection_result_sample_based, BasicVariableSelectionResult):
            detection_result = asdict(self.detection_result_sample_based)
        elif isinstance(self.detection_result_time_slicing, list):
            detection_result =[asdict(__d) for __d in self.detection_result_time_slicing]
        else:
            raise ValueError(f'Unknown type case.')
        # end if
        
        dict_obj = {
            'configurations': asdict(self.configurations),
            'detection_result': detection_result
        }
        return json.dumps(dict_obj, cls=JSONEncoder)


