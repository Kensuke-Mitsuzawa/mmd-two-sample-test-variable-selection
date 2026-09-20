import pytest
import torch
import pytorch_lightning as pl
from distributed import Client, LocalCluster

from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.commons import (
    RegularizationParameter,
    InterpretableMmdTrainParameters,
)
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.commons import (
    AggregationKey,
    CrossValidationAlgorithmParameter,
    RequestDistributedFunction,
    SubLearnerTrainingResult,
)
from mmd_tst_variable_detector.accelerator_optimizations import (
    BaseTaskDispatcher,
    SingleCpuTaskDispatcher,
    SingleGpuTaskDispatcher,
    DaskCpuTaskDispatcher,
    ConcurrentGpuTaskDispatcher,
    create_task_dispatcher,
)
from tests import data_generator


def _is_cuda_functional() -> bool:
    """Check if CUDA is both available and capable of executing kernels in this PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, device="cuda") + 1
        return True
    except Exception:
        return False


def _create_dummy_task(task_idx: int = 0, accelerator: str = "cpu") -> RequestDistributedFunction:
    t_xy, __ = data_generator.test_data_xy_linear(
        dim_size=100,
        sample_size=300,
        ratio_dependent_variables=0.1,
        random_seed=42 + task_idx,
    )
    dataset = SimpleDataset(t_xy[0], t_xy[1])

    _d = dataset.get_dimension_flattened()
    dim_size = _d[0] if isinstance(_d, (tuple, list)) else int(_d)
    initial_ard = torch.ones(dim_size)
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
    kernel.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel)

    reg_param = RegularizationParameter(0.01, 0.0)
    task_id = AggregationKey(
        approach_regularization_parameter="fixed_range",
        trial_id_cross_validation=0,
        regularization=reg_param,
        job_id=task_idx,
    )

    trainer_pl = pl.Trainer(
        max_epochs=10,
        accelerator=accelerator,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )

    training_param = InterpretableMmdTrainParameters(
        regularization_parameter=reg_param,
        is_use_log=0,
        batch_size=-1,
        n_workers_train_dataloader=0,
        n_workers_validation_dataloader=0,
        dataloader_persistent_workers=False,
    )

    algo_param = CrossValidationAlgorithmParameter(
        n_permutation_test=10,
        ard_weight_selection_strategy="threshold",
        candidate_regularization_parameter=[reg_param],
    )

    return RequestDistributedFunction(
        task_id=task_id,
        training_parameter=training_param,
        dataset_train=dataset,
        dataset_val=dataset,
        trainer_lightning=trainer_pl,
        mmd_estimator=mmd_estimator,
        stability_algorithm_param=algo_param,
    )


def test_task_dispatcher_class_hierarchy():
    """Requirement: All specialized dispatchers must inherit from BaseTaskDispatcher."""
    dispatcher_single_cpu = create_task_dispatcher("cpu", "single")
    dispatcher_single_gpu = create_task_dispatcher("gpu", "single")
    dispatcher_dask_cpu = create_task_dispatcher("cpu", "dask")
    dispatcher_dask_gpu = create_task_dispatcher("gpu", "dask")

    assert isinstance(dispatcher_single_cpu, BaseTaskDispatcher)
    assert isinstance(dispatcher_single_gpu, BaseTaskDispatcher)
    assert isinstance(dispatcher_dask_cpu, BaseTaskDispatcher)
    assert isinstance(dispatcher_dask_gpu, BaseTaskDispatcher)

    assert isinstance(dispatcher_single_cpu, SingleCpuTaskDispatcher)
    assert isinstance(dispatcher_single_gpu, SingleGpuTaskDispatcher)
    assert isinstance(dispatcher_dask_cpu, DaskCpuTaskDispatcher)
    assert isinstance(dispatcher_dask_gpu, ConcurrentGpuTaskDispatcher)


def test_task_dispatcher_single_cpu():
    """Mode 1: Single CPU execution."""
    dispatcher = create_task_dispatcher(
        train_accelerator="cpu",
        distributed_mode="single",
        batch_size=2,
    )

    tasks = [_create_dummy_task(i, accelerator="cpu") for i in range(2)]
    results = dispatcher.dispatch(tasks)

    assert len(results) == 2
    for res in results:
        assert isinstance(res, SubLearnerTrainingResult)
        assert res.training_result is not None
        assert res.variable_detected is not None


def test_task_dispatcher_single_gpu():
    """Mode 2: Single GPU execution."""
    if not _is_cuda_functional():
        pytest.skip("CUDA device is either not present or not supported by current PyTorch binary")

    dispatcher = create_task_dispatcher(
        train_accelerator="gpu",
        distributed_mode="single",
        batch_size=2,
    )

    tasks = [_create_dummy_task(i, accelerator="gpu") for i in range(2)]
    results = dispatcher.dispatch(tasks)

    assert len(results) == 2
    for res in results:
        assert isinstance(res, SubLearnerTrainingResult)
        assert res.training_result is not None
        assert res.variable_detected is not None


def test_task_dispatcher_dask_cpu():
    """Mode 3: Dask CPU execution."""
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,
    )
    client = Client(cluster)

    try:
        dispatcher = create_task_dispatcher(
            train_accelerator="cpu",
            distributed_mode="dask",
            dask_client=client,
            batch_size=2,
        )

        tasks = [_create_dummy_task(i, accelerator="cpu") for i in range(2)]
        results = dispatcher.dispatch(tasks)

        assert len(results) == 2
        for res in results:
            assert isinstance(res, SubLearnerTrainingResult)
            assert res.training_result is not None
            assert res.variable_detected is not None
    finally:
        client.close()
        cluster.close()
