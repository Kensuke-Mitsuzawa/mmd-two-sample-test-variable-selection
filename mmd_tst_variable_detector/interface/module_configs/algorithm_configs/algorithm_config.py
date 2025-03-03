import typing as ty
from dataclasses import dataclass

from ....detection_algorithm.search_regularization_min_max.optuna_module.commons import RegularizationSearchParameters


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
        self.setting_name = self.setting_name.lower()
        assert self.setting_name in ['config_rapid', 'config_tpami_draft', 'custom'], f'{self.setting_name} is not supported.'
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
    
    def __post_init__(self):
        self.setting_name = self.setting_name.lower()
        assert self.setting_name in ['config_rapid', 'config_tpami_draft', 'custom'], f'{self.setting_name} is not supported.'
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
