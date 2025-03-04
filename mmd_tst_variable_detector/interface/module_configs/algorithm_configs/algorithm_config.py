import typing as ty
from dataclasses import dataclass

import torch

from ....detection_algorithm.search_regularization_min_max.optuna_module.commons import RegularizationSearchParameters


@dataclass
class MmdEstimatorConfig:
    ard_weights_initial: ty.Union[str, torch.Tensor] = 'one'

    # Kernel Function Parameters
    kernel_class_name: str = 'QuadraticKernelGaussianKernel'
    aggregation_kernel_length_scale: str = 'median'
    is_dimension_median_heuristic: bool = True
    length_scale: ty.Union[str, torch.Tensor] = 'auto'

    # MMD Estimator Parameters
    mmd_class_name: str = 'QuadraticMmdEstimator'
    variance_term: str = 'liu_2020'
    biased: bool = False
    unit_diagonal: bool = False

    def __post_init__(self):
        assert self.kernel_class_name in ('QuadraticKernelGaussianKernel',), f'{self.kernel_class_name} is not supported.'
        assert self.aggregation_kernel_length_scale in ('median', 'mean'), f'{self.aggregation_kernel_length_scale} is not supported.'
        assert self.mmd_class_name in ('QuadraticMmdEstimator',), f'{self.mmd_class_name} is not supported.'
        assert self.variance_term in ('liu_2020', 'sutherland_2017'), f'{self.variance_term} is not supported.'
        
        if isinstance(self.length_scale, str):
            assert self.length_scale in ('auto', ), f'{self.length_scale} is not supported.'
        # end if
        if isinstance(self.ard_weights_initial, str):
            assert self.ard_weights_initial in ('one', 'wasserstein'), f'{self.ard_weights_initial} is not supported.' 
        # end if


@dataclass
class BaselineMmdConfigArgs:
    max_epoch: int = 9999
    batch_size: int = -1

    # distance function used for permutation test
    test_distance_functions: ty.Union[ty.Tuple[str, ...], ty.List[str]] = ('sinkhorn', 'sliced_wasserstein')
    n_permutation_test: int = 1000
    
    mmd_estimator_config: MmdEstimatorConfig = MmdEstimatorConfig()

    # dataloader parameter
    dataloader_n_workers_train_dataloader: int = 0
    dataloader_n_workers_validation_dataloader: int = 0
    dataloader_persistent_workers: bool = False
        
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
class CvSelectionConfigArgs:
    max_epoch: int = 9999
    batch_size: int = -1
    
    n_subsampling: int = 5
    
    mmd_estimator_config: MmdEstimatorConfig = MmdEstimatorConfig()
    
    # lambda search parameter
    approach_regularization_parameter: str = 'param_searching'

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

    
    def __post_init__(self):
        assert self.approach_regularization_parameter in ('param_searching', 'fixed_range'), f'{self.approach_regularization_parameter} is not supported.'
        
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
    max_epoch: int = 9999
    batch_size: int = -1
    
    mmd_estimator_config: MmdEstimatorConfig = MmdEstimatorConfig()
        
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
    
    def __post_init__(self):
        assert self.approach_regularization_parameter in ('search_objective_based', 'auto_min_max_range'), f'{self.approach_regularization_parameter} is not supported.'

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
