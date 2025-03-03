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
from mmd_tst_variable_detector.detection_algorithm.detection_algorithm_one import detection_algorithm_one, AlgorithmOneIndividualResult, AlgorithmOneResult
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
# import data_generator




torch.cuda.is_available = lambda : False


def __gen_file_st_timeslicing_trajectory(path_tmp_save_root: Path,
                                         random_seed_x: int = 10,
                                         random_seed_y: int = 50) -> ty.List[Path]:
    
    __path_parent = path_tmp_save_root
    __path_parent.mkdir(parents=True, exist_ok=True)
    
    # random generator
    gen_x = np.random.default_rng(seed=random_seed_x)
    gen_y = np.random.default_rng(seed=random_seed_y)    

    # I create a trajectory array. The shape is (5, 300, 2).
    steps = (1.0 - 0.1) / 300
    __x = np.zeros((300, 5, 2))
    __y = np.zeros((300, 5, 2))
    # prob process
    for _i, _t in enumerate(np.arange(0.1, 1.0, steps)):
        __x[_i, :, :] = _t + gen_x.normal(0, 1.0, size=(5, 2))
        __y[_i, :, :] = 0.5 + gen_y.normal(0, 1.0, size=(5, 2))
    # end for
        
    seq_path_dir = []
    
    # time-slicing
    for _t in range(300):
        _x_t = torch.from_numpy(__x[_t, :, :]).clone()
        _y_t = torch.from_numpy(__y[_t, :, :]).clone()
        
        __path_t_dir = __path_parent / str(_t)
        __path_t_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({'array': _x_t}, (__path_t_dir / 'x.pt').as_posix())
        torch.save({'array': _y_t}, (__path_t_dir / 'y.pt').as_posix())
        
        seq_path_dir.append(__path_t_dir)
    # end for
    return seq_path_dir
    



def test_algorithm_one_min_max_param_range(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_algorithm_one_min_max_param_range.__name__]

    # data sampling
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_test, __ = data_generator.test_data_xy_linear(sample_size=500)
    # dataset generationg
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
    
    # permutation test runner
    from ot import sliced_wasserstein_distance
    permutation_test_runner = PermutationTest(
        func_distance=sliced_wasserstein_distance,
        n_permutation_test=10)
    
    DefaultEarlyStoppingRule = ConvergenceEarlyStop()
    pl_trainer_args = PytorchLightningDefaultArguments(
        accelerator='cpu',
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=DefaultEarlyStoppingRule,
        max_epochs=config_obj['max_epochs'],        
    )
    
    seq_regularization_parameter = 'auto_min_max_range'
    
    temp_dir = Path('/tmp/test_algorithm_one')
    __base_loggers = ['mlflow']
    __post_process_logger_handler = PostProcessLoggerHandler(
        __base_loggers,
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }})

    trained_model_obj = detection_algorithm_one(
        mmd_estimator=mmd_estimator,
        pytorch_trainer_config=pl_trainer_args,
        base_training_parameter=parameter_base,
        dataset_training=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        permutation_test_runner_base=permutation_test_runner,
        candidate_regularization_parameters=seq_regularization_parameter,
        post_process_handler=__post_process_logger_handler,
        regularization_search_parameter=RegularizationSearchParameters(
            n_search_iteration=1,
            max_concurrent_job=1,
            n_regularization_parameter=2
        )
    )
    
    assert isinstance(trained_model_obj, AlgorithmOneResult)
    assert isinstance(trained_model_obj.selected_model, AlgorithmOneIndividualResult)
    assert isinstance(trained_model_obj.trained_models, list)
    assert all([isinstance(_obj, AlgorithmOneIndividualResult) for _obj in trained_model_obj.trained_models])

    shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)    


def test_algorithm_one_search_objective_based(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_algorithm_one_search_objective_based.__name__]

    # data sampling
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=500)
    t_xy_test, __ = data_generator.test_data_xy_linear(sample_size=500)
    # dataset generationg
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
    
    # permutation test runner
    from ot import sliced_wasserstein_distance
    permutation_test_runner = PermutationTest(
        func_distance=sliced_wasserstein_distance,
        n_permutation_test=10)
    
    DefaultEarlyStoppingRule = ConvergenceEarlyStop()
    pl_trainer_args = PytorchLightningDefaultArguments(
        accelerator='cpu',
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=DefaultEarlyStoppingRule,
        max_epochs=config_obj['max_epochs'],        
    )
    
    seq_regularization_parameter = 'search_objective_based'
    
    temp_dir = Path(tempfile.mkdtemp()) / 'test_algorithm_one_search_objective_based'
    temp_dir.mkdir(exist_ok=True, parents=True)
    __base_loggers = ['mlflow']
    __post_process_logger_handler = PostProcessLoggerHandler(
        __base_loggers,
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }})

    trained_model_obj = detection_algorithm_one(
        mmd_estimator=mmd_estimator,
        pytorch_trainer_config=pl_trainer_args,
        base_training_parameter=parameter_base,
        dataset_training=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        permutation_test_runner_base=permutation_test_runner,
        candidate_regularization_parameters=seq_regularization_parameter,
        post_process_handler=__post_process_logger_handler,
        regularization_search_parameter=RegularizationSearchParameters(
            n_search_iteration=2,
            max_concurrent_job=1,
            n_regularization_parameter=2
        )
    )
    
    assert isinstance(trained_model_obj, AlgorithmOneResult)
    assert isinstance(trained_model_obj.selected_model, AlgorithmOneIndividualResult)
    assert isinstance(trained_model_obj.trained_models, list)
    assert all([isinstance(_obj, AlgorithmOneIndividualResult) for _obj in trained_model_obj.trained_models])

    shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)

