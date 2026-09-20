import pytest
import torch
import pytorch_lightning as pl

from mmd_tst_variable_detector.accelerator_optimizations import (
    SingleGpuTaskDispatcher,
    ConcurrentGpuTaskDispatcher,
    IncompatibleGpuArchitectureError,
    GpuEnvironmentManager,
    assert_device_compatibility,
    VramConsumptionEstimator,
    DeviceSlotManager,
    create_task_dispatcher,
)
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.commons import (
    AggregationKey,
    CrossValidationAlgorithmParameter,
    InterpretableMmdTrainParameters,
    RegularizationParameter,
    RequestDistributedFunction,
)
from tests import data_generator


def _is_cuda_functional() -> bool:
    """Check if CUDA is functional and compatible with the current PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        assert_device_compatibility(0)
        return True
    except (IncompatibleGpuArchitectureError, RuntimeError):
        return False


def _create_test_task(task_idx: int = 0, accelerator: str = "gpu") -> RequestDistributedFunction:
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
        ard_weight_selection_strategy="hist_based",
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


def test_device_slot_manager_worker_spec_generation():
    """Verify DeviceSlotManager generates correct CUDA_VISIBLE_DEVICES worker mappings."""
    worker_specs = DeviceSlotManager.build_worker_specs(
        n_gpus=2,
        k_slots_per_gpu=3,
        memory_limit="4GB",
    )

    assert len(worker_specs) == 6

    # GPU 0 workers
    for slot in range(3):
        w_name = f"gpu_0_slot_{slot}"
        assert w_name in worker_specs
        assert worker_specs[w_name]["options"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert worker_specs[w_name]["options"]["memory_limit"] == "4GB"
        assert worker_specs[w_name]["options"]["nthreads"] == 1

    # GPU 1 workers
    for slot in range(3):
        w_name = f"gpu_1_slot_{slot}"
        assert w_name in worker_specs
        assert worker_specs[w_name]["options"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert worker_specs[w_name]["options"]["memory_limit"] == "4GB"
        assert worker_specs[w_name]["options"]["nthreads"] == 1


def test_vram_estimator_slot_calculation():
    """Verify VramConsumptionEstimator formula for computing K slots."""
    estimator = VramConsumptionEstimator(device_id=0, safety_margin=0.80, max_slots_per_gpu=8)

    # 8GB free, 1GB peak -> floor(8 * 0.8 / 1) = 6 slots
    slots = estimator.calculate_k_slots(
        free_vram_bytes=8 * 1024**3,
        peak_vram_bytes=1 * 1024**3,
    )
    assert slots == 6

    # 8GB free, 5GB peak -> floor(8 * 0.8 / 5) = 1 slot
    slots_large = estimator.calculate_k_slots(
        free_vram_bytes=8 * 1024**3,
        peak_vram_bytes=5 * 1024**3,
    )
    assert slots_large == 1

    # Zero or negative protection
    assert estimator.calculate_k_slots(0, 1024) == 1
    assert estimator.calculate_k_slots(1024, 0) == 1


def test_gpu_environment_manager_lifecycle():
    """Test GpuEnvironmentManager detection and context manager."""
    mgr = GpuEnvironmentManager(enable_mps=True)
    count = mgr.get_gpu_count()
    assert count >= 0

    # Test MPS context manager cleanly handles entry and exit
    with GpuEnvironmentManager(enable_mps=False) as m:
        assert m.mps_active is False


from unittest import mock


def test_gpu_compatibility_check_passes_on_compatible_system():
    """Verify that assert_device_compatibility passes without error on a compatible GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on host.")
    assert_device_compatibility(0)


def test_gpu_compatibility_check_raises_on_gap():
    """Verify that assert_device_compatibility raises IncompatibleGpuArchitectureError
    when installed PyTorch does not support the target GPU architecture.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on host.")

    # Simulate an architecture gap (e.g. device is sm_61 but PyTorch only compiled for sm_80, sm_90)
    with mock.patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"]):
        with pytest.raises(IncompatibleGpuArchitectureError) as exc_info:
            assert_device_compatibility(0)

        err = exc_info.value
        assert "Execution is halted to prevent CUDA runtime crashes" in str(err)
        assert err.supported_architectures == ["sm_80", "sm_90"]


def test_single_gpu_and_concurrent_gpu_halt_on_incompatible_architecture():
    """Verify both SingleGpuTaskDispatcher and ConcurrentGpuTaskDispatcher halt
    with IncompatibleGpuArchitectureError when an architecture gap exists.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on host.")

    with mock.patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"]):
        single_dispatcher = create_task_dispatcher("gpu", "single")
        concurrent_dispatcher = create_task_dispatcher("gpu", "dask")

        task = _create_test_task(0, accelerator="gpu")

        with pytest.raises(IncompatibleGpuArchitectureError):
            single_dispatcher.dispatch([task])

        with pytest.raises(IncompatibleGpuArchitectureError):
            concurrent_dispatcher.dispatch([task])


def test_single_gpu_vs_concurrent_gpu_equivalence():
    """Verify that variable detection results from Single GPU and Concurrent GPU are identical.

    Requirement: Comparing variable detection using the single GPU and concurrent-GPU,
    the variable selection result must be same.
    """
    if not _is_cuda_functional():
        pytest.skip(
            "Functional CUDA execution is not available for this PyTorch/hardware combination. "
            "IncompatibleGpuArchitectureError properly halts execution as expected."
        )

    # In a fully compatible GPU environment:
    tasks_single = [_create_test_task(i, accelerator="gpu") for i in range(2)]
    tasks_concurrent = [_create_test_task(i, accelerator="gpu") for i in range(2)]

    single_dispatcher = SingleGpuTaskDispatcher(device_id=0, batch_size=2)
    concurrent_dispatcher = ConcurrentGpuTaskDispatcher(n_gpus=1, k_slots_per_gpu=2, batch_size=2)

    res_single = single_dispatcher.dispatch(tasks_single)
    res_concurrent = concurrent_dispatcher.dispatch(tasks_concurrent)

    assert len(res_single) == len(res_concurrent)
    for r_s, r_c in zip(res_single, res_concurrent):
        assert r_s.job_id == r_c.job_id
        assert r_s.variable_detected == r_c.variable_detected
        if r_s.training_result is not None and r_c.training_result is not None:
            torch.testing.assert_close(
                r_s.training_result.ard_weights_kernel_k,
                r_c.training_result.ard_weights_kernel_k,
                rtol=1e-3,
                atol=1e-3,
            )
