import numpy as np
import torch
import torch.utils.data

from distributed import LocalCluster

from mmd_tst_variable_detector import (
    QuadraticKernelGaussianKernel,
    QuadraticMmdEstimator,
    SimpleDataset,
    # CV related classes
    RegularizationParameter,
    InterpretableMmdTrainParameters,
    CrossValidationAlgorithmParameter,
    CrossValidationInterpretableVariableDetector,
    DistributedComputingParameter,
    CrossValidationTrainParameters,
    # training helper
    PytorchLightningDefaultArguments
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.assessment_helper.default_settings import lr_scheduler

import logzero
logger = logzero.logger


"""This example shows you how to run variable detection with CV-algorithm (Cross-Validation).
The CV-algorithm intends to select variables without knowing the regularization parameters.
"""


def example_cv_detector(n_cv_parameter: int = 2, max_epochs: int = 20):
    """Example of CV-alogrithm. Arguments are hyper-parameters.
    
    Args:
        n_cv_parameter: number of subsampling for CV algorithm.
        max_epochs: number of epochs for training.
    """

    # Preparation

    dimension = 20
    sample_size = 200

    noise_index = [0, 1]

    array_x = np.random.normal(size=(sample_size, dimension))
    array_y = np.random.normal(size=(sample_size, dimension))
    array_y[:, noise_index] = array_y[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    array_x_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev = np.random.normal(size=(sample_size, dimension))
    array_y_dev[:, noise_index] = array_y_dev[:, noise_index] + np.random.normal(loc=3, size=(sample_size, len(noise_index)))

    # Step-1: defining a dataset

    dataset_train = SimpleDataset(torch.tensor(array_x), torch.tensor(array_y))
    dataset_dev = SimpleDataset(torch.tensor(array_x_dev), torch.tensor(array_y_dev))    

    # Step-2: defining a kernel function

    # comment: In this example, I initialize ARD weights with all 1.0. However, this proper weights are supposed to be given.
    ard_weights = torch.ones(dimension)

    kernel_function = QuadraticKernelGaussianKernel(ard_weights=ard_weights)
    kernel_function.compute_length_scale_dataset(dataset_train)

    # Step-3: defining a MMD-estimator

    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel_function)

    # =========================================================================
    # Device & Execution Options
    # -------------------------------------------------------------------------
    # The dispatcher is resolved by create_task_dispatcher based on:
    #   (accelerator, dask_client / distributed_mode)
    #
    # Available modes:
    # 1. Single GPU (Active default below):
    #    - accelerator='gpu' (or 'cuda'), dask_client=None
    #    - Fast sequential execution on target GPU via SingleGpuTaskDispatcher.
    #
    # 2. Concurrent GPU via Dask:
    #    - accelerator='gpu', dask_client connected to DeviceSlotManager GPU cluster.
    #    - Multi-slot concurrent GPU tasks via ConcurrentGpuTaskDispatcher (NVIDIA MPS).
    #
    # 3. Single CPU:
    #    - accelerator='cpu', dask_client=None
    #    - Sequential in-process execution on CPU via SingleCpuTaskDispatcher.
    #
    # 4. Dask CPU:
    #    - accelerator='cpu', dask_client connected to LocalCluster.
    #    - Distributed execution across CPU workers via DaskCpuTaskDispatcher.
    # =========================================================================

    # Active configuration: GPU device (falls back to CPU if CUDA is not available)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    dask_client = None  # Single GPU mode: fast sequential execution on device

    # --- Example 1: Single CPU execution ---
    # accelerator = "cpu"
    # dask_client = None

    # --- Example 2: CPU with Dask distributed computing ---
    # from distributed import Client, LocalCluster
    # cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    # dask_client = Client(cluster)
    # accelerator = "cpu"

    # --- Example 3: Concurrent GPU with Dask (NVIDIA MPS multi-slot workers) ---
    # from mmd_tst_variable_detector.accelerator_optimizations.concurrent_gpu_modules import (
    #     DeviceSlotManager, GpuEnvironmentManager
    # )
    # gpu_mgr = GpuEnvironmentManager(enable_mps=True)
    # gpu_mgr.start_mps()
    # cluster, dask_client = DeviceSlotManager.create_gpu_cluster(n_gpus=1, k_slots_per_gpu=2)
    # accelerator = "gpu"

    # Step-4: defining a CV detector
    
    ## possible regularization parameters.
    candidate_reg_param = [
        RegularizationParameter(lambda_1=0.01, lambda_2=0.0),
        RegularizationParameter(lambda_1=0.05, lambda_2=0.0)
    ]
    
    cv_algorithm_param = CrossValidationAlgorithmParameter(
        n_subsampling=n_cv_parameter,
        candidate_regularization_parameter=candidate_reg_param
    )
    
    interpretable_mmd_train_param = InterpretableMmdTrainParameters(
        lr_scheduler=lr_scheduler,
        optimizer_args={'lr': 0.01},
    )
    
    dist_computing_param = DistributedComputingParameter(job_batch_size=2)
    
    cv_training_parameter = CrossValidationTrainParameters(
        algorithm_parameter=cv_algorithm_param,
        base_training_parameter=interpretable_mmd_train_param,
        distributed_parameter=dist_computing_param,
    )

    pytorch_trainer_config = PytorchLightningDefaultArguments(
        max_epochs=max_epochs,
        callbacks=[ConvergenceEarlyStop(ignore_epochs=10)],
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        accelerator=accelerator)

    cv_detector = CrossValidationInterpretableVariableDetector(
        training_parameter=cv_training_parameter,
        pytorch_trainer_config=pytorch_trainer_config,
        estimator=mmd_estimator,
        dask_client=dask_client,
    )
    cv_result = cv_detector.run_cv_detection(
        training_dataset=dataset_train,
        validation_dataset=dataset_dev
    )

    logger.info(f'Detected variables -> {cv_result.stable_s_hat}')
    logger.info(f'Weights array -> {cv_result.array_s_hat}')


def test_example():
    example_cv_detector(n_cv_parameter=2, max_epochs=20)


if __name__ == '__main__':
    example_cv_detector(n_cv_parameter=2, max_epochs=20)
    

