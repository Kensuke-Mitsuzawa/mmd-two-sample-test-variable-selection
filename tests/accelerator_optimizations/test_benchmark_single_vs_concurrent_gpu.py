import time
import typing as ty
import torch
import pytorch_lightning as pl

from mmd_tst_variable_detector.accelerator_optimizations import (
    SingleGpuTaskDispatcher,
    ConcurrentGpuTaskDispatcher,
    VramConsumptionEstimator,
    DeviceSlotManager,
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
    SubLearnerTrainingResult,
)
from tests import data_generator


def generate_benchmark_tasks(
    dim_size: int = 100,
    sample_size: int = 500,
    n_cv: int = 5,
    max_epochs: int = 30,
    random_seed: int = 42,
) -> ty.Tuple[ty.List[RequestDistributedFunction], ty.List[int]]:
    """Generate 5 CV tasks for 100-dimensional data across regularization values."""
    t_xy, ground_truth = data_generator.test_data_xy_linear(
        dim_size=dim_size,
        sample_size=sample_size,
        ratio_dependent_variables=0.1,
        random_seed=random_seed,
    )
    dataset = SimpleDataset(t_xy[0], t_xy[1])

    _d = dataset.get_dimension_flattened()
    dim_size = _d[0] if isinstance(_d, (tuple, list)) else int(_d)
    initial_ard = torch.ones(dim_size)
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset, batch_size=len(dataset))
    kernel.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel)

    # 5 CV splits x 2 regularizations = 10 tasks
    candidate_regularizations = [
        RegularizationParameter(0.01, 0.0),
        RegularizationParameter(0.05, 0.0),
    ]

    tasks = []
    job_idx = 0
    for cv_idx in range(n_cv):
        for reg in candidate_regularizations:
            task_id = AggregationKey(
                approach_regularization_parameter="fixed_range",
                trial_id_cross_validation=cv_idx,
                regularization=reg,
                job_id=job_idx,
            )
            trainer_pl = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="gpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )
            train_param = InterpretableMmdTrainParameters(
                regularization_parameter=reg,
                is_use_log=0,
                batch_size=-1,
                n_workers_train_dataloader=0,
                n_workers_validation_dataloader=0,
                dataloader_persistent_workers=False,
            )
            algo_param = CrossValidationAlgorithmParameter(
                n_permutation_test=10,
                ard_weight_selection_strategy="hist_based",
                candidate_regularization_parameter=[reg],
            )
            task = RequestDistributedFunction(
                task_id=task_id,
                training_parameter=train_param,
                dataset_train=dataset,
                dataset_val=dataset,
                trainer_lightning=trainer_pl,
                mmd_estimator=mmd_estimator,
                stability_algorithm_param=algo_param,
            )
            tasks.append(task)
            job_idx += 1

    return tasks, ground_truth


def run_benchmark(max_epochs: int = 30) -> ty.Dict[str, ty.Any]:
    """Execute comparative benchmark between Single GPU and Concurrent GPU."""
    assert torch.cuda.is_available(), "CUDA GPU is required for this benchmark."

    dim_size = 100
    n_cv = 5

    device_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)

    print("=" * 72)
    print("      MMD VARIABLE SELECTION BENCHMARK: SINGLE vs CONCURRENT GPU        ")
    print("=" * 72)
    print(f"Device:           {device_name}")
    print(f"VRAM Capacity:    Total: {total_mem / (1024**2):.1f} MB | Free: {free_mem / (1024**2):.1f} MB")
    print(f"Configuration:    {dim_size} dimensions | {n_cv} Cross-Validation folds | {max_epochs} epochs")

    # Generate benchmark tasks
    tasks_single, _ = generate_benchmark_tasks(dim_size=dim_size, n_cv=n_cv, max_epochs=max_epochs, random_seed=42)
    total_tasks = len(tasks_single)
    print(f"Total Workload:   {total_tasks} independent optimization tasks")
    print("-" * 72)

    # -------------------------------------------------------------
    # 1. Single GPU Benchmark (1 sequential task on GPU)
    # -------------------------------------------------------------
    print(f"[*] Executing Single GPU (sequential, 1 task on GPU)...", flush=True)
    single_dispatcher = SingleGpuTaskDispatcher(device_id=0, batch_size=total_tasks)

    t0_single = time.perf_counter()
    results_single: ty.List[SubLearnerTrainingResult] = single_dispatcher.dispatch(tasks_single)
    time_single = time.perf_counter() - t0_single
    print(f"    -> Done in {time_single:.2f} seconds")

    detected_single = [r.variable_detected for r in results_single]

    # Evaluate multiple concurrent slot counts
    slot_configs = [2, 4]
    benchmark_records = [
        {
            "mode": "Single GPU",
            "slots": 1,
            "time_seconds": time_single,
            "speedup": 1.0,
            "identical": True,
        }
    ]

    for k in slot_configs:
        print(f"[*] Executing Concurrent GPU ({k} concurrent tasks on GPU)...", flush=True)
        tasks_concurrent, _ = generate_benchmark_tasks(dim_size=dim_size, n_cv=n_cv, max_epochs=max_epochs, random_seed=42)
        concurrent_dispatcher = ConcurrentGpuTaskDispatcher(
            n_gpus=1,
            k_slots_per_gpu=k,
            batch_size=total_tasks,
            enable_mps=True,
        )

        t0_conc = time.perf_counter()
        results_conc = concurrent_dispatcher.dispatch(tasks_concurrent)
        time_conc = time.perf_counter() - t0_conc
        print(f"    -> Done in {time_conc:.2f} seconds")

        detected_conc = [r.variable_detected for r in results_conc]
        identical = detected_single == detected_conc

        benchmark_records.append({
            "mode": f"Concurrent GPU (K={k}, cold)",
            "slots": k,
            "time_seconds": time_conc,
            "speedup": time_single / time_conc if time_conc > 0 else 0.0,
            "identical": identical,
        })

    # Warm-start evaluation with pre-initialized cluster (K=4)
    print(f"[*] Executing Concurrent GPU (K=4, warm-start / pre-initialized cluster)...", flush=True)
    cluster, client = DeviceSlotManager.create_gpu_cluster(n_gpus=1, k_slots_per_gpu=4)
    try:
        tasks_concurrent_warm, _ = generate_benchmark_tasks(dim_size=dim_size, n_cv=n_cv, max_epochs=max_epochs, random_seed=42)
        warm_dispatcher = ConcurrentGpuTaskDispatcher(
            dask_client=client,
            batch_size=total_tasks,
        )
        t0_warm = time.perf_counter()
        results_warm = warm_dispatcher.dispatch(tasks_concurrent_warm)
        time_warm = time.perf_counter() - t0_warm
        print(f"    -> Done in {time_warm:.2f} seconds")
        detected_warm = [r.variable_detected for r in results_warm]
        benchmark_records.append({
            "mode": "Concurrent GPU (K=4, warm)",
            "slots": 4,
            "time_seconds": time_warm,
            "speedup": time_single / time_warm if time_warm > 0 else 0.0,
            "identical": detected_single == detected_warm,
        })
    finally:
        DeviceSlotManager.close_cluster(client=client, cluster=cluster)

    print("=" * 72)
    print(f"{'Mode':<30} | {'GPU Slots':<10} | {'Time (s)':<10} | {'Speedup':<9} | {'Equivalence':<11}")
    print("-" * 72)
    for r in benchmark_records:
        equiv_str = "Identical" if r["identical"] else "Mismatch"
        print(f"{r['mode']:<30} | {r['slots']:<10} | {r['time_seconds']:<10.2f} | {r['speedup']:<8.2f}x | {equiv_str:<11}")
    print("=" * 72)

    return {
        "device_name": device_name,
        "dim_size": dim_size,
        "n_cv": n_cv,
        "total_tasks": total_tasks,
        "records": benchmark_records,
    }


def test_benchmark_execution_speed():
    """Pytest hook for benchmark."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA GPU required for benchmark.")

    metrics = run_benchmark(max_epochs=15)
    for rec in metrics["records"]:
        assert rec["identical"] is True


if __name__ == "__main__":
    run_benchmark(max_epochs=30)
