import shutil
from tempfile import mkdtemp
import time

import torch
import pytorch_lightning as pl

from distributed import Client, LocalCluster
import dask
import dask.config


import toml
from pathlib import Path

from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector import (
    CrossValidationAlgorithmParameter,
    CrossValidationTrainParameters,
    DistributedComputingParameter,
    InterpretableMmdTrainParameters,
    CrossValidationInterpretableVariableDetector,
)
from mmd_tst_variable_detector.detection_algorithm import PytorchLightningDefaultArguments
from mmd_tst_variable_detector.utils import PostProcessLoggerHandler
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.checkpoint_saver import CheckPointSaverStabilitySelection
from mmd_tst_variable_detector.detection_algorithm.search_regularization_min_max.optuna_module.commons import RegularizationSearchParameters

from tests import data_generator


torch.cuda.is_available = lambda : False


def test_all_auto(resource_path_root: Path):
    # config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_joblib_StabilitySelectionVariableTrainer.__name__]
    # if Path(config_obj["working_dir_checkpoint_network"]).exists():
    #     shutil.rmtree(config_obj["working_dir_checkpoint_network"])
    # # end if
    
    return True
    
    temp_dir = Path(mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    path_dir_mlflow = temp_dir / "mlruns"
    path_dir_mlflow.mkdir(parents=True, exist_ok=True)
    
    path_dir_checkpoint = temp_dir / "checkpoints"
    path_dir_checkpoint.mkdir(parents=True, exist_ok=True)
    
    search_param = RegularizationSearchParameters(
        n_regularization_parameter=2,
        n_search_iteration=2,
        max_concurrent_job=1)
    
    algorithm_param = CrossValidationAlgorithmParameter(
        candidate_regularization_parameter='auto',
        regularization_search_parameter=search_param)
    dist_param = DistributedComputingParameter(
        dask_scheduler_address=None)
    base_train_param = InterpretableMmdTrainParameters()

    ss_param = CrossValidationTrainParameters(
        algorithm_parameter=algorithm_param,
        base_training_parameter=base_train_param,
        distributed_parameter=dist_param,
        computation_backend='dask'
    )

    # Test re-loading and resume..
    t_xy, __ = data_generator.test_data_xy_linear(sample_size=500)
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened()[0])

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset, batch_size=len(my_dataset))
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel)    
    pl_argument = PytorchLightningDefaultArguments(
        max_epochs=10,
        accelerator='cpu'
    )

    # resume_checkpoint_saver = CheckPointSaverStabilitySelection(path_dir_checkpoint)

    post_process_logger_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={"mlflow": {"save_dir": path_dir_mlflow.as_posix(), "tracking_uri": "file:" + path_dir_mlflow.as_posix()}}
    )

    ss_trainer = CrossValidationInterpretableVariableDetector(
        # trainer_lightning=trainer_lightning,
        pytorch_trainer_config=pl_argument,
        training_parameter=ss_param,
        estimator=mmd_estimator,
        resume_checkpoint_saver=None,
        post_process_handler=post_process_logger_handler
    )
    ss_trainer.run_cv_detection(training_dataset=my_dataset, validation_dataset=my_dataset)

    shutil.rmtree(temp_dir)


def test_joblib_StabilitySelectionVariableTrainer(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_joblib_StabilitySelectionVariableTrainer.__name__]
    # if Path(config_obj["working_dir_checkpoint_network"]).exists():
    #     shutil.rmtree(config_obj["working_dir_checkpoint_network"])
    # # end if
    
    temp_dir = Path(mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    path_dir_mlflow = temp_dir / "mlruns"
    path_dir_mlflow.mkdir(parents=True, exist_ok=True)
    
    path_dir_checkpoint = temp_dir / "checkpoints"
    path_dir_checkpoint.mkdir(parents=True, exist_ok=True)
    
    algorithm_param = CrossValidationAlgorithmParameter(
        candidate_regularization_parameter=config_obj['candidate_regularization_parameter'],
        n_subsampling=config_obj['n_subsampling'])
    dist_param = DistributedComputingParameter(
        dask_scheduler_address=None,
        n_joblib=config_obj['n_joblib'])
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=config_obj['batch_size'],
    )

    ss_param = CrossValidationTrainParameters(
        algorithm_parameter=algorithm_param,
        base_training_parameter=base_train_param,
        distributed_parameter=dist_param,
        computation_backend='joblib'
    )

    # Test re-loading and resume..
    t_xy, __ = data_generator.test_data_xy_linear(sample_size=500)
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset, batch_size=len(my_dataset))
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel)

    # trainer_lightning = pl.Trainer(max_epochs=config_obj['max_epochs'], accelerator='cpu')
    
    pl_argument = PytorchLightningDefaultArguments(
        max_epochs=config_obj['max_epochs'],
        accelerator='cpu'
    )

    resume_checkpoint_saver = CheckPointSaverStabilitySelection(path_dir_checkpoint)

    post_process_logger_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={"mlflow": {"save_dir": path_dir_mlflow.as_posix(), "tracking_uri": "file:" + path_dir_mlflow.as_posix()}}
    )

    ss_trainer = CrossValidationInterpretableVariableDetector(
        # trainer_lightning=trainer_lightning,
        pytorch_trainer_config=pl_argument,
        training_parameter=ss_param,
        estimator=mmd_estimator,
        resume_checkpoint_saver=resume_checkpoint_saver,
        post_process_handler=post_process_logger_handler
    )
    ss_trainer.run_cv_detection(training_dataset=my_dataset, validation_dataset=my_dataset)

    shutil.rmtree(temp_dir)


def test_single_StabilitySelectionVariableTrainer(resource_path_root: Path):
    config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_joblib_StabilitySelectionVariableTrainer.__name__]
    # if Path(config_obj["working_dir_checkpoint_network"]).exists():
    #     shutil.rmtree(config_obj["working_dir_checkpoint_network"])

    temp_dir = Path(mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)

    path_mlflow = temp_dir / "mlruns"
    path_mlflow.mkdir(parents=True, exist_ok=True)
    
    path_checkpoint = temp_dir / "checkpoints"
    path_checkpoint.mkdir(parents=True, exist_ok=True)

    algorithm_param = CrossValidationAlgorithmParameter(
        candidate_regularization_parameter=config_obj['candidate_regularization_parameter'],
        n_subsampling=config_obj['n_subsampling'])
    dist_param = DistributedComputingParameter(
        dask_scheduler_address=None,
        n_joblib=config_obj['n_joblib'])
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=config_obj['batch_size'],
    )

    ss_param = CrossValidationTrainParameters(
        algorithm_parameter=algorithm_param,
        base_training_parameter=base_train_param,
        distributed_parameter=dist_param,
        computation_backend='single'
    )

    t_xy, __ = data_generator.test_data_xy_linear(sample_size=500)
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset, batch_size=len(my_dataset))
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel)

    # trainer_lightning = pl.Trainer(max_epochs=config_obj['max_epochs'], accelerator='cpu')
    pl_argument = PytorchLightningDefaultArguments(
        max_epochs=config_obj['max_epochs'],
        accelerator='cpu'
    )

    resume_checkpoint_saver = CheckPointSaverStabilitySelection(path_checkpoint)

    post_process_logger_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={"mlflow": {"save_dir": path_mlflow.as_posix(), 
                                  "tracking_uri": "file:" + path_mlflow.as_posix()}}
    )

    ss_trainer = CrossValidationInterpretableVariableDetector(
        # trainer_lightning=trainer_lightning,
        pytorch_trainer_config=pl_argument,
        training_parameter=ss_param,
        estimator=mmd_estimator,
        resume_checkpoint_saver=resume_checkpoint_saver,
        post_process_handler=post_process_logger_handler
    )
    ss_trainer.run_cv_detection(training_dataset=my_dataset, validation_dataset=my_dataset)

    shutil.rmtree(temp_dir)
    
    
def test_dask_CV_param_searching(resource_path_root: Path):

    dask.config.set(distributed__worker__daemon=False)
    
    temp_dir = Path(mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)

    path_mlflow = temp_dir / "mlruns"
    path_mlflow.mkdir(parents=True, exist_ok=True)

    # DO not use Dask in test.
    # cluster = LocalCluster(
    #     '127.0.0.1:9999',
    #     n_workers=2, 
    #     threads_per_worker=4)
    # dask_client = Client(cluster)    
    # dask_scheduler_address = '127.0.0.1:9999'
    # computation_backend='dask'
    
    dask_scheduler_address = None
    computation_backend='single'
    
    # parameter config manual
    candidate_regularization_parameter = 'auto'
    n_subsampling = 1
    batch_size = -1
    max_epochs = 100
    
    algorithm_param = CrossValidationAlgorithmParameter(
        approach_regularization_parameter='param_searching',
        candidate_regularization_parameter=candidate_regularization_parameter,
        n_subsampling=n_subsampling,
        regularization_search_parameter=RegularizationSearchParameters(
            n_regularization_parameter=1,
            max_concurrent_job=1,
            n_search_iteration=1),
        )
    dist_param = DistributedComputingParameter(
        dask_scheduler_address=dask_scheduler_address,
        n_joblib=0)
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=batch_size,
        n_workers_train_dataloader=1,
        n_workers_validation_dataloader=1,
        dataloader_persistent_workers=True)

    ss_param = CrossValidationTrainParameters(
        algorithm_parameter=algorithm_param,
        base_training_parameter=base_train_param,
        distributed_parameter=dist_param,
        computation_backend=computation_backend
    )

    t_xy, __ = data_generator.test_data_xy_linear(sample_size=500)
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])

    initial_ard = torch.ones(my_dataset.get_dimension_flattened())

    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(my_dataset)
    kernel.set_length_scale()

    mmd_estimator = QuadraticMmdEstimator(kernel)

    pl_argument = PytorchLightningDefaultArguments(
            max_epochs=max_epochs,
            accelerator='cpu'
        )

    # resume_checkpoint_saver = CheckPointSaverStabilitySelection(output_dir=config_obj["working_dir_checkpoint_network"])

    post_process_logger_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={"mlflow": {"save_dir": path_mlflow.as_posix(), 
                                  "tracking_uri": "file:" + path_mlflow.as_posix()}}
    )


    ss_trainer = CrossValidationInterpretableVariableDetector(
        pytorch_trainer_config=pl_argument,
        training_parameter=ss_param,
        estimator=mmd_estimator,
        post_process_handler=post_process_logger_handler
    )
    ss_trainer.run_cv_detection(
        training_dataset=my_dataset,
        validation_dataset=my_dataset)

    shutil.rmtree(temp_dir)    
