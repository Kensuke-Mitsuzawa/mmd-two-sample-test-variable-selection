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
from mmd_tst_variable_detector.detection_algorithm.baseline_mmd import baseline_mmd, BaselineMmdResult
from mmd_tst_variable_detector.utils.permutation_test_runner import PermutationTest
from mmd_tst_variable_detector import (
    
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
    PostProcessLoggerHandler,
    
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop


from tests import data_generator
# import data_generator




torch.cuda.is_available = lambda : False


def test_baseline_mmd(resource_path_root: Path):

    # data sampling
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=500, random_seed=1234)
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=500, random_seed=1111)
    t_xy_test, __ = data_generator.test_data_xy_linear(sample_size=500, random_seed=2222)
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
        n_permutation_test=500)
    
    DefaultEarlyStoppingRule = ConvergenceEarlyStop()
    pl_trainer_args = PytorchLightningDefaultArguments(
        accelerator='cpu',
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=DefaultEarlyStoppingRule,
        max_epochs=100,
    )
    
    
    temp_dir = Path(tempfile.mkdtemp()) / 'test_baseline_mmd'
    temp_dir.mkdir(exist_ok=True, parents=True)
    __base_loggers = ['mlflow']
    __post_process_logger_handler = PostProcessLoggerHandler(
        __base_loggers,
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }})

    trained_model_obj = baseline_mmd(
        mmd_estimator=mmd_estimator,
        pytorch_trainer_config=pl_trainer_args,
        base_training_parameter=parameter_base,
        dataset_training=dataset_train,
        dataset_dev=dataset_dev,
        dataset_test=dataset_test,
        permutation_test_runner=permutation_test_runner,
        post_process_handler=__post_process_logger_handler,
    )
    
    assert isinstance(trained_model_obj, BaselineMmdResult)

    shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)

