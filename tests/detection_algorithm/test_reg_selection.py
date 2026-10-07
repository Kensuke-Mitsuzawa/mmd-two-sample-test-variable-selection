import shutil
import typing as ty

import pytest
import torch
import numpy as np
import pytorch_lightning as pl

import os
import toml
from pathlib import Path

import tempfile

from mmd_tst_variable_detector.datasets import (
    SimpleDataset
)
from mmd_tst_variable_detector.distance_module import L2Distance
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdDetector
from mmd_tst_variable_detector.detection_algorithm.reg_selection import (
    detection_reg_selection,
    RegSelectionIndividualResult,
    RegSelectionResult,
    RegSelectionVariableDetector,
    # Legacy aliases
    detection_algorithm_one,
    AlgorithmOneIndividualResult,
    AlgorithmOneResult,
    AlgorithmOneVariableDetector,
)
from mmd_tst_variable_detector.utils.permutation_test_runner import PermutationTest
from mmd_tst_variable_detector import (
    RegularizationParameter, 
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
    PostProcessLoggerHandler,
    RegularizationSearchParameters
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop

from tests import data_generator


torch.cuda.is_available = lambda : False


def test_reg_selection_aliases():
    """Verify legacy aliases for RegSelection symbols."""
    assert AlgorithmOneVariableDetector is RegSelectionVariableDetector
    assert AlgorithmOneResult is RegSelectionResult
    assert AlgorithmOneIndividualResult is RegSelectionIndividualResult
    assert detection_algorithm_one is detection_reg_selection
# end def


def test_reg_selection_min_max_param_range(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_reg_selection_min_max_param_range.__name__]

    # data sampling
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_test, __ = data_generator.test_data_xy_linear(sample_size=500)
    # dataset generation
    dataset_train = SimpleDataset(t_xy_train[0], t_xy_train[1])
    dataset_dev = SimpleDataset(t_xy_dev[0], t_xy_dev[1])
    dataset_test = SimpleDataset(t_xy_test[0], t_xy_test[1])
    
    # kernel function setup
    initial_ard = torch.ones(dataset_train.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset_train, batch_size=len(dataset_train))
    kernel.set_length_scale()

    # mmd estimator setup
    mmd_estimator = QuadraticMmdEstimator(kernel)

    # trainer setup
    parameter_base = InterpretableMmdTrainParameters(is_use_log=0)
    
    DefaultEarlyStoppingRule = ConvergenceEarlyStop()
    pl_trainer_args = PytorchLightningDefaultArguments(
        accelerator='cpu',
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=DefaultEarlyStoppingRule,
        max_epochs=config_obj['max_epochs'],        
    )
    
    seq_regularization_parameter = 'auto_min_max_range'
    
    temp_dir = Path('/tmp/test_reg_selection')
    __base_loggers = ['mlflow']
    __post_process_logger_handler = PostProcessLoggerHandler(
        __base_loggers,
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }})

    trained_model_obj = detection_reg_selection(
        mmd_estimator=mmd_estimator,
        pytorch_trainer_config=pl_trainer_args,
        base_training_parameter=parameter_base,
        dataset_training=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        test_distance_functions=('sliced_wasserstein', ),
        n_permutation_test=10,
        candidate_regularization_parameters=seq_regularization_parameter,
        post_process_handler=__post_process_logger_handler,
        regularization_search_parameter=RegularizationSearchParameters(
            n_search_iteration=1,
            max_concurrent_job=1,
            n_regularization_parameter=2
        )
    )
    
    assert isinstance(trained_model_obj, RegSelectionResult)
    assert isinstance(trained_model_obj.selected_model, RegSelectionIndividualResult)
    assert isinstance(trained_model_obj.trained_models, list)
    assert all([isinstance(_obj, RegSelectionIndividualResult) for _obj in trained_model_obj.trained_models])
    # Also verify legacy class instance check works
    assert isinstance(trained_model_obj, AlgorithmOneResult)

    shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)    
# end def


def test_reg_selection_search_objective_based(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_reg_selection_search_objective_based.__name__]

    # data sampling
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_test, __ = data_generator.test_data_xy_linear(sample_size=500)
    # dataset generation
    dataset_train = SimpleDataset(t_xy_train[0], t_xy_train[1])
    dataset_dev = SimpleDataset(t_xy_dev[0], t_xy_dev[1])
    dataset_test = SimpleDataset(t_xy_test[0], t_xy_test[1])
    
    # kernel function setup
    initial_ard = torch.ones(dataset_train.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset_train, batch_size=len(dataset_train))
    kernel.set_length_scale()

    # mmd estimator setup
    mmd_estimator = QuadraticMmdEstimator(kernel)

    # trainer setup
    parameter_base = InterpretableMmdTrainParameters(is_use_log=0)
    
    DefaultEarlyStoppingRule = ConvergenceEarlyStop()
    pl_trainer_args = PytorchLightningDefaultArguments(
        accelerator='cpu',
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=DefaultEarlyStoppingRule,
        max_epochs=config_obj['max_epochs'],        
    )
    
    seq_regularization_parameter = 'search_objective_based'
    
    temp_dir = Path(tempfile.mkdtemp()) / 'test_reg_selection_search_objective_based'
    temp_dir.mkdir(exist_ok=True, parents=True)
    __base_loggers = ['mlflow']
    __post_process_logger_handler = PostProcessLoggerHandler(
        __base_loggers,
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }})

    # Test via legacy alias function detection_algorithm_one
    trained_model_obj = detection_algorithm_one(
        mmd_estimator=mmd_estimator,
        pytorch_trainer_config=pl_trainer_args,
        base_training_parameter=parameter_base,
        dataset_training=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        test_distance_functions=('sliced_wasserstein', ),
        n_permutation_test=10,
        candidate_regularization_parameters=seq_regularization_parameter,
        post_process_handler=__post_process_logger_handler,
        regularization_search_parameter=RegularizationSearchParameters(
            n_search_iteration=2,
            max_concurrent_job=1,
            n_regularization_parameter=2
        )
    )
    
    assert isinstance(trained_model_obj, RegSelectionResult)
    assert isinstance(trained_model_obj.selected_model, RegSelectionIndividualResult)
    assert isinstance(trained_model_obj.trained_models, list)
    assert all([isinstance(_obj, RegSelectionIndividualResult) for _obj in trained_model_obj.trained_models])
    assert isinstance(trained_model_obj, AlgorithmOneResult)

    shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)
# end def
