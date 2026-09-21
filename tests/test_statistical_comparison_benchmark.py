import time
import pytest
import torch
import pytorch_lightning as pl

from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
    InterpretableMmdDetector,
    LegacyInterpretableMmdDetector,
    InterpretableMmdTrainParameters,
)
from mmd_tst_variable_detector.detection_algorithm.commons import RegularizationParameter
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer
from mmd_tst_variable_detector.utils import evaluate_variable_detection


DISTRIBUTION_P = {"type": "gaussian", "mu": 0.0, "sigma": 1.0}

DISTRIBUTIONS_Q = {
    "gaussian": {"type": "gaussian", "mu": 1.0, "sigma": 1.0},
    "laplace": {"type": "laplace", "mu": 1.0, "sigma": 1.0},
}

RANDOM_SEEDS = [101, 202, 303]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU benchmark")
@pytest.mark.parametrize("q_dist_name", ["gaussian", "laplace"])
@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_statistical_equivalence_and_speedup(q_dist_name: str, seed: int):
    """Statistical evaluation and comparison across interventions and seeds.

    Conditions:
      - Dimension = 20
      - Sample size = 200
      - Intervention (Distribution Q): gaussian or laplace
      - 2 ground truth variables among 20 (mixture_rate = 0.1)
      - max_epochs = 9999
      - Early stopping with TrajectoryNoiseReduction enabled
      - Comparison between Legacy and Native PyTorch + Fused Triton CUDA kernel
    """
    torch.set_float32_matmul_precision("high")
    q_conf = DISTRIBUTIONS_Q[q_dist_name]

    # Generate dataset with 2 ground truth variables among 20
    x_np, y_np, ground_truth = sampling_from_distribution(
        n_sample=200,
        dimension_size=20,
        mixture_rate=0.1,  # 0.1 * 20 = 2 variables
        distribution_conf_p=DISTRIBUTION_P,
        distribution_conf_q=q_conf,
        random_seed_x=seed,
        random_seed_y=seed + 1000,
        random_seed_noise=seed + 2000,
    )
    assert len(ground_truth) == 2, f"Expected 2 ground truth variables, got {len(ground_truth)}"

    dataset = SimpleDataset(
        torch.tensor(x_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32)
    )

    training_params = InterpretableMmdTrainParameters(
        batch_size=-1,
        regularization_parameter=RegularizationParameter(0.01, 0.0),
        optimizer_args={"lr": 0.01},
    )

    # 1. Run Legacy Implementation
    torch.manual_seed(seed)
    initial_ard = torch.ones(20)
    kernel_legacy = QuadraticKernelGaussianKernel(ard_weights=initial_ard.clone())
    kernel_legacy.compute_length_scale_dataset(dataset)
    kernel_legacy.set_length_scale()
    estimator_legacy = QuadraticMmdEstimator(kernel_legacy)

    legacy_det = LegacyInterpretableMmdDetector(
        mmd_estimator=estimator_legacy,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )

    es_legacy = ConvergenceEarlyStop(
        ignore_epochs=40,
        check_span=25,
        threshold_convergence_ratio=0.005,
        is_noise_reduction=True,
    )

    trainer_legacy = pl.Trainer(
        max_epochs=9999,
        accelerator="gpu",
        devices=1,
        callbacks=[es_legacy],
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )

    t0 = time.perf_counter()
    trainer_legacy.fit(legacy_det)
    t_legacy = time.perf_counter() - t0
    stopped_epoch_legacy = trainer_legacy.current_epoch
    res_legacy = legacy_det.get_trained_variables()
    eval_legacy, vars_legacy = evaluate_variable_detection.evaluate_trained_variables(
        res_legacy.ard_weights_kernel_k, ground_truth
    )

    # 2. Run Native PyTorch Runner & Fused Triton CUDA Kernel
    torch.manual_seed(seed)
    kernel_fused = QuadraticKernelGaussianKernel(
        ard_weights=initial_ard.clone(),
        use_fused_kernel=True,
    )
    kernel_fused.compute_length_scale_dataset(dataset)
    kernel_fused.set_length_scale()
    estimator_fused = QuadraticMmdEstimator(kernel_fused)

    fast_det = InterpretableMmdDetector(
        mmd_estimator=estimator_fused,
        training_parameter=training_params,
        dataset_train=dataset,
        dataset_validation=dataset,
    )

    es_fast = ConvergenceEarlyStop(
        ignore_epochs=40,
        check_span=25,
        threshold_convergence_ratio=0.005,
        is_noise_reduction=True,
    )

    trainer_fast = PurePytorchTrainer(
        max_epochs=9999,
        accelerator="gpu",
        callbacks=[es_fast],
        use_fused_kernel=True,
    )

    t0 = time.perf_counter()
    trainer_fast.fit(fast_det)
    t_fast = time.perf_counter() - t0
    stopped_epoch_fast = trainer_fast.current_epoch
    res_fast = fast_det.get_trained_variables()
    eval_fast, vars_fast = evaluate_variable_detection.evaluate_trained_variables(
        res_fast.ard_weights_kernel_k, ground_truth
    )

    speedup = t_legacy / t_fast

    print(f"\n[Statistical Test | Q: {q_dist_name.upper()} | Seed: {seed}]")
    print(f"Ground Truth Variables:   {ground_truth}")
    print(f"Legacy Result:            Epochs={stopped_epoch_legacy}, Time={t_legacy:.3f}s, Vars={vars_legacy}, F1={eval_legacy.f1:.4f}")
    print(f"Native+Fused Result:      Epochs={stopped_epoch_fast}, Time={t_fast:.3f}s, Vars={vars_fast}, F1={eval_fast.f1:.4f}")
    print(f"Speedup Ratio:            {speedup:.2f}x faster")

    # Assertions
    # A. Early stopping must trigger well before max_epochs (9999)
    assert stopped_epoch_legacy < 9999, f"Legacy failed to stop early: {stopped_epoch_legacy}"
    assert stopped_epoch_fast < 9999, f"Native+Fused failed to stop early: {stopped_epoch_fast}"

    # B. Variable selection results must match between Legacy and Native+Fused
    assert vars_legacy == vars_fast, f"Variable selection mismatch: Legacy={vars_legacy}, Fast={vars_fast}"
    assert eval_legacy.f1 == eval_fast.f1, f"F1 mismatch: Legacy={eval_legacy.f1}, Fast={eval_fast.f1}"

    # C. Both must successfully recall the intervention ground-truth variables
    assert eval_fast.recall >= 0.5, f"Recall too low ({eval_fast.recall}) for GT {ground_truth} vs detected {vars_fast}"
    assert eval_fast.f1 > 0.0, f"F1 is zero for GT {ground_truth} vs detected {vars_fast}"

    # D. Native + Fused must be significantly faster
    assert speedup > 1.0, f"Expected speedup > 1.0x, got {speedup:.2f}x"
