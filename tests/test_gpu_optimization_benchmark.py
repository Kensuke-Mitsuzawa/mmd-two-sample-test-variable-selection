import time
import pytest
import torch
import pytorch_lightning as pl

from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
    InterpretableMmdDetector,
    LegacyInterpretableMmdDetector,
    InterpretableMmdTrainParameters,
)
from mmd_tst_variable_detector.detection_algorithm.commons import RegularizationParameter
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import (
    ConvergenceEarlyStop,
    LegacyConvergenceEarlyStop,
)
from mmd_tst_variable_detector.utils import evaluate_variable_detection
from tests import data_generator


def create_test_setup(sample_size: int = 400, dim_size: int = 10, seed: int = 42):
    t_xy, ground_truth = data_generator.test_data_xy_linear(
        sample_size=sample_size,
        dim_size=dim_size,
        ratio_dependent_variables=0.3,
        random_seed=seed,
    )
    dataset = SimpleDataset(t_xy[0], t_xy[1])
    initial_ard = torch.ones(dataset.get_dimension_flattened())
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard.clone())
    kernel.compute_length_scale_dataset(dataset)
    kernel.set_length_scale()
    return dataset, kernel, ground_truth


def test_variable_selection_equivalence_cpu():
    """Verify Legacy and New detector produce identical variable selection on CPU."""
    dataset, kernel, ground_truth = create_test_setup(sample_size=200, dim_size=8, seed=123)
    estimator_legacy = QuadraticMmdEstimator(kernel)
    estimator_fast = QuadraticMmdEstimator(kernel)

    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    torch.manual_seed(123)
    legacy_det = LegacyInterpretableMmdDetector(
        mmd_estimator=estimator_legacy,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_legacy = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer_legacy.fit(legacy_det)
    res_legacy = legacy_det.get_trained_variables()

    torch.manual_seed(123)
    fast_det = InterpretableMmdDetector(
        mmd_estimator=estimator_fast,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_fast = pl.Trainer(
        max_epochs=30,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer_fast.fit(fast_det)
    res_fast = fast_det.get_trained_variables()

    eval_legacy, vars_legacy = evaluate_variable_detection.evaluate_trained_variables(
        res_legacy.ard_weights_kernel_k, ground_truth
    )
    eval_fast, vars_fast = evaluate_variable_detection.evaluate_trained_variables(
        res_fast.ard_weights_kernel_k, ground_truth
    )

    assert vars_legacy == vars_fast, f"Variable mismatch on CPU: Legacy={vars_legacy}, Fast={vars_fast}"
    assert eval_legacy.f1 == eval_fast.f1, f"F1 mismatch: Legacy={eval_legacy.f1}, Fast={eval_fast.f1}"
    torch.testing.assert_close(res_legacy.ard_weights_kernel_k, res_fast.ard_weights_kernel_k, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark")
def test_variable_selection_equivalence_and_speedup_gpu():
    """Verify Legacy and New detector produce identical variable selection on GPU with measurable speedup."""
    torch.set_float32_matmul_precision('high')
    dataset, kernel, ground_truth = create_test_setup(sample_size=600, dim_size=12, seed=42)

    estimator_legacy = QuadraticMmdEstimator(kernel)
    estimator_fast = QuadraticMmdEstimator(kernel)

    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,  # full-batch as used by end users
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    # 1. Run Legacy on GPU
    torch.manual_seed(42)
    legacy_det = LegacyInterpretableMmdDetector(
        mmd_estimator=estimator_legacy,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_legacy = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    t0 = time.perf_counter()
    trainer_legacy.fit(legacy_det)
    torch.cuda.synchronize()
    time_legacy = time.perf_counter() - t0
    res_legacy = legacy_det.get_trained_variables()

    # 2. Run Fast on GPU
    torch.manual_seed(42)
    fast_det = InterpretableMmdDetector(
        mmd_estimator=estimator_fast,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_fast = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    t0 = time.perf_counter()
    trainer_fast.fit(fast_det)
    torch.cuda.synchronize()
    time_fast = time.perf_counter() - t0
    res_fast = fast_det.get_trained_variables()

    eval_legacy, vars_legacy = evaluate_variable_detection.evaluate_trained_variables(
        res_legacy.ard_weights_kernel_k, ground_truth
    )
    eval_fast, vars_fast = evaluate_variable_detection.evaluate_trained_variables(
        res_fast.ard_weights_kernel_k, ground_truth
    )

    print(f"\n[GPU Benchmark Results]")
    print(f"Legacy Execution Time: {time_legacy:.4f} s")
    print(f"Fast Execution Time:   {time_fast:.4f} s")
    print(f"Speedup Ratio:         {time_legacy / time_fast:.2f}x")
    print(f"Legacy Variables:      {vars_legacy} (F1: {eval_legacy.f1:.4f})")
    print(f"Fast Variables:        {vars_fast} (F1: {eval_fast.f1:.4f})")

    assert vars_legacy == vars_fast, f"Variable mismatch on GPU: Legacy={vars_legacy}, Fast={vars_fast}"
    assert eval_legacy.f1 == eval_fast.f1, f"F1 score mismatch on GPU: Legacy={eval_legacy.f1}, Fast={eval_fast.f1}"
    torch.testing.assert_close(res_legacy.ard_weights_kernel_k, res_fast.ard_weights_kernel_k, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU early stopping test")
def test_early_stopping_convergence_on_gpu():
    """Verify GPU ConvergenceEarlyStop activates and terminates optimization properly."""
    dataset, kernel, ground_truth = create_test_setup(sample_size=500, dim_size=8, seed=99)
    estimator = QuadraticMmdEstimator(kernel)

    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    early_stop = ConvergenceEarlyStop(
        ignore_epochs=40,
        check_span=25,
        threshold_convergence_ratio=0.005,
        is_noise_reduction=True,
    )

    detector = InterpretableMmdDetector(
        mmd_estimator=estimator,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(detector)
    assert trainer.current_epoch < 200, f"Early stopping did not activate: stopped at epoch {trainer.current_epoch}"
    print(f"\n[GPU Early Stopping Test] Successfully stopped early at epoch {trainer.current_epoch} / 200")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark")
def test_pure_pytorch_loop_equivalence_and_speedup():
    """Verify Pure PyTorch loop produces identical variable selection with massive speedup over Legacy."""
    from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer

    dataset, kernel, ground_truth = create_test_setup(sample_size=600, dim_size=12, seed=42)
    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    # 1. Legacy PyTorch Lightning
    torch.manual_seed(42)
    est_legacy = QuadraticMmdEstimator(kernel)
    legacy_det = LegacyInterpretableMmdDetector(
        mmd_estimator=est_legacy,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_legacy = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    t0 = time.perf_counter()
    trainer_legacy.fit(legacy_det)
    t_legacy = time.perf_counter() - t0
    res_legacy = legacy_det.get_trained_variables()
    eval_legacy, vars_legacy = evaluate_variable_detection.evaluate_trained_variables(
        res_legacy.ard_weights_kernel_k, ground_truth
    )

    # 2. Pure PyTorch Trainer
    torch.manual_seed(42)
    est_pure = QuadraticMmdEstimator(kernel)
    pure_det = InterpretableMmdDetector(
        mmd_estimator=est_pure,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_pure = PurePytorchTrainer(
        max_epochs=50,
        accelerator="gpu",
        check_val_every_n_epoch=1,
    )
    t0 = time.perf_counter()
    trainer_pure.fit(pure_det)
    t_pure = time.perf_counter() - t0
    res_pure = pure_det.get_trained_variables()
    eval_pure, vars_pure = evaluate_variable_detection.evaluate_trained_variables(
        res_pure.ard_weights_kernel_k, ground_truth
    )

    print(f"\n[Pure PyTorch Benchmark]")
    print(f"Legacy Lightning Time: {t_legacy:.4f} s | Variables: {vars_legacy} (F1: {eval_legacy.f1:.4f})")
    print(f"Pure PyTorch Time:     {t_pure:.4f} s | Variables: {vars_pure} (F1: {eval_pure.f1:.4f})")
    print(f"Pure PyTorch Speedup:  {t_legacy / t_pure:.2f}x")

    assert vars_legacy == vars_pure, f"Variable selection mismatch: Legacy={vars_legacy}, Pure={vars_pure}"
    assert eval_legacy.f1 == eval_pure.f1, f"F1 mismatch: {eval_legacy.f1} vs {eval_pure.f1}"
    torch.testing.assert_close(res_legacy.ard_weights_kernel_k, res_pure.ard_weights_kernel_k, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark")
def test_fused_triton_kernel_equivalence_and_speedup():
    """Verify Fused Triton Kernel produces identical variable selection and maximum throughput."""
    from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer
    from mmd_tst_variable_detector.kernels.fused_gaussian_kernel import (
        compute_fused_gaussian_kernel,
        FusedGaussianKernelFunction,
    )

    dataset, kernel, ground_truth = create_test_setup(sample_size=600, dim_size=12, seed=42)
    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    # 1. Baseline: Pure PyTorch Eager
    torch.manual_seed(42)
    est_eager = QuadraticMmdEstimator(kernel)
    eager_det = InterpretableMmdDetector(
        mmd_estimator=est_eager,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_eager = PurePytorchTrainer(max_epochs=50, accelerator="gpu")
    t0 = time.perf_counter()
    trainer_eager.fit(eager_det)
    t_eager = time.perf_counter() - t0
    res_eager = eager_det.get_trained_variables()
    eval_eager, vars_eager = evaluate_variable_detection.evaluate_trained_variables(
        res_eager.ard_weights_kernel_k, ground_truth
    )

    # 2. Fused Triton Kernel
    torch.manual_seed(42)
    initial_ard = torch.ones(dataset.get_dimension_flattened())
    fused_kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard.clone(), use_fused_kernel=True)
    fused_kernel.compute_length_scale_dataset(dataset)
    fused_kernel.set_length_scale()
    est_fused = QuadraticMmdEstimator(fused_kernel)

    # Warmup Triton kernel once to trigger JIT compilation
    warmup_det = InterpretableMmdDetector(
        mmd_estimator=est_fused,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    PurePytorchTrainer(max_epochs=1, accelerator="gpu", use_fused_kernel=True).fit(warmup_det)

    # Reset ARD weights for benchmark
    with torch.no_grad():
        est_fused.kernel_obj.ard_weights.copy_(initial_ard)
    # end with

    fused_det = InterpretableMmdDetector(
        mmd_estimator=est_fused,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )
    trainer_fused = PurePytorchTrainer(max_epochs=50, accelerator="gpu", use_fused_kernel=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer_fused.fit(fused_det)
    torch.cuda.synchronize()
    t_fused = time.perf_counter() - t0
    res_fused = fused_det.get_trained_variables()
    eval_fused, vars_fused = evaluate_variable_detection.evaluate_trained_variables(
        res_fused.ard_weights_kernel_k, ground_truth
    )

    print(f"\n[Fused Triton Kernel Benchmark]")
    print(f"Pure PyTorch Eager Time: {t_eager:.4f} s | Variables: {vars_eager} (F1: {eval_eager.f1:.4f})")
    print(f"Fused Triton Time:       {t_fused:.4f} s | Variables: {vars_fused} (F1: {eval_fused.f1:.4f})")
    print(f"Fused Triton Speedup vs Eager: {t_eager / t_fused:.2f}x")

    assert vars_eager == vars_fused, f"Variable mismatch: Eager={vars_eager}, Fused={vars_fused}"
    assert eval_eager.f1 == eval_fused.f1, f"F1 mismatch: {eval_eager.f1} vs {eval_fused.f1}"
    torch.testing.assert_close(res_eager.ard_weights_kernel_k, res_fused.ard_weights_kernel_k, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark")
def test_pure_pytorch_early_stopping():
    """Verify ConvergenceEarlyStop triggers correctly inside PurePytorchTrainer on GPU."""
    from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer

    dataset, kernel, ground_truth = create_test_setup(sample_size=500, dim_size=8, seed=99)
    estimator = QuadraticMmdEstimator(kernel)

    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    early_stop = ConvergenceEarlyStop(
        ignore_epochs=40,
        check_span=25,
        threshold_convergence_ratio=0.005,
        is_noise_reduction=True,
    )

    detector = InterpretableMmdDetector(
        mmd_estimator=estimator,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )

    trainer = PurePytorchTrainer(
        max_epochs=200,
        accelerator="gpu",
        callbacks=[early_stop],
    )

    trainer.fit(detector)
    assert trainer.current_epoch < 200, f"Early stop failed to activate: reached epoch {trainer.current_epoch}"
    print(f"\n[Pure PyTorch Early Stopping Test] Stopped early at epoch {trainer.current_epoch} / 200")

