import time
import pickle
from datetime import datetime
from pathlib import Path
import typing as ty
import pytest
import torch
import numpy as np
import logging
from pydantic import BaseModel, Field, ConfigDict

from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector import (
    RegularizationParameter,
    InterpretableMmdTrainParameters,
    PytorchLightningDefaultArguments,
)
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop

# 1. Algorithm One
from mmd_tst_variable_detector.detection_algorithm.detection_algorithm_one import (
    AlgorithmOneVariableDetector,
    AlgorithmOneResult,
)

# 2. MMD-CV (Cross-Validation Detector)
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector import (
    CrossValidationInterpretableVariableDetector,
    CrossValidationTrainParameters,
    CrossValidationAlgorithmParameter,
    DistributedComputingParameter,
    CrossValidationTrainedParameter,
)
from mmd_tst_variable_detector.exceptions import SameDataException

# 3. Path Aggregation
from mmd_tst_variable_detector.detection_algorithm.path_aggregation_executor import (
    PathAggregationVariableDetector,
    PathAggregationDetectionResult,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


"""A benchmark script to compare 3 detection algorithms on the same dataet.

# How to run

`python -m pytest tests/test_benchmark_detection_algorithm.py`

# Way to check the result

The script writes out the comparison result under a directory `tmp/benchmark_result_*_seed_*/`.
"""



# ==============================================================================
# 1. Pydantic Parameter Configurations (Data Generation & Algorithms)
# ==============================================================================


class DataGenerationBenchmarkParameter(BaseModel):
    """Pydantic data object defining synthetic data generation settings."""

    n_sample: int = Field(default=200, description="Sample size for distributions P and Q.")
    dimension_size: int = Field(default=20, description="Total dimension size d.")
    mixture_rate: float = Field(
        default=0.1,
        description="Fraction of dimensions that are ground-truth discriminating features (0.1 * 20 = 2).",
    )
    distribution_p: ty.Dict[str, ty.Any] = Field(
        default_factory=lambda: {"type": "gaussian", "mu": 0.0, "sigma": 1.0},
        description="Baseline distribution P.",
    )
    distributions_q: ty.Dict[str, ty.Dict[str, ty.Any]] = Field(
        default_factory=lambda: {
            "gaussian": {"type": "gaussian", "mu": 1.0, "sigma": 1.0},
            "laplace": {"type": "laplace", "mu": 1.0, "sigma": 1.0},
        },
        description="Candidate target distributions Q for discrepancy testing.",
    )
    random_seeds: ty.List[int] = Field(
        default_factory=lambda: [101, 202, 303],
        description="Random seeds for benchmark reproducibility.",
    )
# end class


class SharedAlgorithmBenchmarkParameter(BaseModel):
    """Pydantic data object defining shared training and optimization settings."""

    lambda_grid: ty.List[float] = Field(
        default_factory=lambda: [round(float(val), 2) for val in np.arange(0.1, 1.05, 0.05)],
        description="Regularization grid Lambda: 0.1, 0.15, 0.2, ..., 1.0 (19 values).",
    )
    max_epochs: int = Field(default=9999, description="Maximum training epochs per task.")
    learning_rate: float = Field(default=0.01, description="Adam optimizer learning rate.")
    batch_size: int = Field(default=-1, description="Batch size (-1 for full-batch optimization).")
    accelerator: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Compute hardware accelerator: 'cpu' or 'cuda'.",
    )
# end class


class AlgorithmOneBenchmarkParameter(BaseModel):
    """Pydantic data object defining Algorithm One parameters."""

    train_ratio: float = Field(default=0.8, description="Train vs development split ratio.")
    is_p_value_filter: bool = Field(default=False, description="Whether to filter by permutation p-value.")
    n_permutation_test: int = Field(default=500, description="Number of permutations for p-value calculation.")
# end class


class MmdCvBenchmarkParameter(BaseModel):
    """Pydantic data object defining MMD-CV parameters."""

    n_subsampling: int = Field(default=5, description="Number of subsampling splits B.")
    ratio_subsampling: float = Field(default=0.8, description="Data subsampling ratio per split.")
    sampling_strategy: str = Field(default="random-splitting", description="Sampling strategy.")
    threshold_stability_score: float = Field(default=0.1, description="Stability selection threshold.")
    n_permutation_test: int = Field(default=500, description="Number of permutations for stability test.")
# end class


class PathAggregationBenchmarkParameter(BaseModel):
    """Pydantic data object defining Path Aggregation parameters."""

    threshold: float = Field(default=0.01, description="Cutoff threshold tau > 0 for aggregated importance.")
    selection_strategy: ty.Literal["threshold", "hist_based", "normalized_threshold"] = Field(
        default="hist_based",
        description="Selection approach: 'hist_based' (valley detection), 'threshold', or 'normalized_threshold'.",
    )
    weight_transformation: ty.Literal["identity", "bounded"] = Field(
        default="identity",
        description="Transformation rho: 'identity' (t) or 'bounded' (t / (1+t)).",
    )
    subsampling_splits_full: int = Field(default=1, description="Splits for full-sample path aggregation.")
    subsampling_splits_sub: int = Field(default=5, description="Splits for subsampled path aggregation.")
    subsampling_ratio: float = Field(default=0.8, description="Data fraction per subsample when splits > 1.")
    save_split_weights: bool = Field(
        default=True,
        description="Whether to store raw ARD weights for all subsampling splits (shape L x B x d).",
    )
# end class


class AlgorithmExecutionResult(BaseModel):
    """Pydantic data object containing results from executing an individual algorithm."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    detected_variables: ty.List[int] = Field(description="Detected variable indices.")
    execution_time_seconds: float = Field(description="Execution time in seconds.")
    raw_result: ty.Any = Field(description="Raw result object from algorithm execution.")
# end class


class BenchmarkMetricResult(BaseModel):
    """Pydantic data object for evaluation metrics."""

    precision: float = Field(description="Precision score.")
    recall: float = Field(description="Recall score.")
    f1: float = Field(description="F1 score.")
    num_detected: int = Field(description="Count of detected features.")
# end class


class BenchmarkAlgorithmResultEntry(BaseModel):
    """Pydantic data object for an individual algorithm benchmark result."""

    algorithm_name: str = Field(description="Name of the algorithm.")
    time_seconds: float = Field(description="Execution time in seconds.")
    num_detected: int = Field(description="Count of detected features.")
    precision: float = Field(description="Precision score.")
    recall: float = Field(description="Recall score.")
    f1_score: float = Field(description="F1 score.")
    detected_variables: ty.List[int] = Field(
        default_factory=list,
        description="Detected variable indices.",
    )
# end class


class BenchmarkExperimentOutput(BaseModel):
    """Container for benchmark experiment configuration and results."""

    timestamp: str = Field(description="ISO timestamp of the benchmark run.")
    q_distribution_name: str = Field(description="Target distribution Q name.")
    seed: int = Field(description="Random seed.")
    ground_truth_variables: ty.List[int] = Field(
        description="Ground truth discriminating features."
    )
    data_parameters: DataGenerationBenchmarkParameter = Field(
        description="Data generation parameters."
    )
    shared_algorithm_parameters: SharedAlgorithmBenchmarkParameter = Field(
        description="Shared algorithm parameters."
    )
    algorithm_one_parameters: AlgorithmOneBenchmarkParameter = Field(
        description="Algorithm One parameters."
    )
    mmd_cv_parameters: MmdCvBenchmarkParameter = Field(
        description="MMD-CV parameters."
    )
    path_aggregation_parameters: PathAggregationBenchmarkParameter = Field(
        description="Path Aggregation parameters."
    )
    results: ty.List[BenchmarkAlgorithmResultEntry] = Field(
        description="List of benchmark results across algorithms."
    )
# end class


# Default benchmark configuration instances
CONFIG_DATA = DataGenerationBenchmarkParameter()
CONFIG_SHARED_ALGO = SharedAlgorithmBenchmarkParameter()
CONFIG_ALGO_ONE = AlgorithmOneBenchmarkParameter()
CONFIG_MMD_CV = MmdCvBenchmarkParameter()
CONFIG_PATH_AGG = PathAggregationBenchmarkParameter()
DEFAULT_BENCHMARK_OUTPUT_DIR = Path("/tmp")


# ==============================================================================
# 2. Metric Computation, Data Setup & Persistence Functions
# ==============================================================================


def save_benchmark_run_artifacts(
    experiment_output: BenchmarkExperimentOutput,
    raw_algorithm_results: ty.Dict[str, ty.Any],
    output_base_dir: Path = DEFAULT_BENCHMARK_OUTPUT_DIR,
    file_format: ty.Literal["pt", "pkl"] = "pt",
) -> Path:
    """Save benchmark experiment output and raw algorithm result objects to a dedicated directory.

    Parameters
    ----------
    experiment_output : BenchmarkExperimentOutput
        Container storing configurations and summary results.
    raw_algorithm_results : Dict[str, Any]
        Dictionary mapping artifact file base name to raw algorithm result object.
    output_base_dir : Path
        Parent directory to create the run-specific output directory in.
    file_format : Literal["pt", "pkl"]
        Serialization format: 'pt' (torch.save) or 'pkl' (pickle).

    Returns
    -------
    Path
        Directory where all artifacts were written.
    """
    timestamp_str = int(time.time())
    run_dir_name = (
        f"benchmark_run_{experiment_output.q_distribution_name}_"
        f"seed_{experiment_output.seed}_{timestamp_str}"
    )
    run_dir = output_base_dir / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save summary JSON
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(experiment_output.model_dump_json(indent=2))
    # end with

    # 2. Save raw algorithm results (.pt or .pkl)
    for artifact_name, raw_obj in raw_algorithm_results.items():
        if raw_obj is None:
            continue
        # end if
        artifact_path = run_dir / f"{artifact_name}.{file_format}"
        if file_format == "pt":
            torch.save(raw_obj, artifact_path)
        elif file_format == "pkl":
            with open(artifact_path, "wb") as file_handle:
                pickle.dump(raw_obj, file_handle)
            # end with
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        # end if
    # end for

    print(f"\n[Benchmark Output] Artifact directory created at: {run_dir}")
    logger.info(f"Benchmark artifacts successfully saved to: {run_dir}")
    return run_dir
# end def


def write_benchmark_result_json(
    experiment_output: BenchmarkExperimentOutput,
    output_dir: Path = DEFAULT_BENCHMARK_OUTPUT_DIR,
) -> Path:
    """Write benchmark results to disk in JSON format and return the saved file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"benchmark_result_{experiment_output.q_distribution_name}_"
        f"seed_{experiment_output.seed}_{int(time.time())}.json"
    )
    saved_path = output_dir / filename
    with open(saved_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(experiment_output.model_dump_json(indent=2))
    # end with

    print(f"\n[Benchmark Output] Results successfully saved to: {saved_path}")
    logger.info(f"Benchmark results successfully saved to: {saved_path}")
    return saved_path
# end def


def compute_metrics_variable_detection(
    detected_indices: ty.List[int],
    ground_truth_indices: ty.List[int],
) -> BenchmarkMetricResult:
    """Compute Precision, Recall, and F1 score against ground-truth variables.

    Parameters
    ----------
    detected_indices : List[int]
        Indices identified by the detector.
    ground_truth_indices : List[int]
        Ground-truth informative variable indices.

    Returns
    -------
    BenchmarkMetricResult
        Pydantic data object with precision, recall, f1, and detected count.
    """
    set_detected = set(detected_indices)
    set_gt = set(ground_truth_indices)

    tp = len(set_detected & set_gt)
    fp = len(set_detected - set_gt)
    fn = len(set_gt - set_detected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return BenchmarkMetricResult(
        precision=precision,
        recall=recall,
        f1=f1,
        num_detected=len(set_detected),
    )
# end def


def build_dataset_and_estimator(
    q_dist_name: str,
    seed: int,
    data_parameter: DataGenerationBenchmarkParameter = CONFIG_DATA,
) -> ty.Tuple[SimpleDataset, ty.List[int], QuadraticMmdEstimator]:
    """Generate synthetic dataset and initialize MMD estimator with ARD product kernel."""
    q_conf = data_parameter.distributions_q[q_dist_name]

    x_np, y_np, ground_truth = sampling_from_distribution(
        n_sample=data_parameter.n_sample,
        dimension_size=data_parameter.dimension_size,
        mixture_rate=data_parameter.mixture_rate,
        distribution_conf_p=data_parameter.distribution_p,
        distribution_conf_q=q_conf,
        random_seed_x=seed,
        random_seed_y=seed + 1000,
        random_seed_noise=seed + 2000,
    )

    dataset = SimpleDataset(
        torch.tensor(x_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32),
    )

    initial_ard = torch.ones(data_parameter.dimension_size)
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset)
    kernel.set_length_scale()
    estimator = QuadraticMmdEstimator(kernel)

    return dataset, list(ground_truth), estimator
# end def


# ==============================================================================
# 3. Individual Algorithm Benchmark Runners
# ==============================================================================


def run_benchmark_algorithm_one(
    dataset: SimpleDataset,
    estimator: QuadraticMmdEstimator,
    algo_parameter: AlgorithmOneBenchmarkParameter = CONFIG_ALGO_ONE,
    shared_parameter: SharedAlgorithmBenchmarkParameter = CONFIG_SHARED_ALGO,
) -> AlgorithmExecutionResult:
    """Execute Algorithm One over candidate regularization parameters."""
    start_time = time.perf_counter()

    candidate_params = [
        RegularizationParameter(l_val, 0.0) for l_val in shared_parameter.lambda_grid
    ]
    early_stop_callback = ConvergenceEarlyStop(
        ignore_epochs=500,
        check_span=100,
        threshold_convergence_ratio=0.001,
        is_noise_reduction=True,
    )
    pl_config = PytorchLightningDefaultArguments(
        max_epochs=shared_parameter.max_epochs,
        accelerator=shared_parameter.accelerator,
        callbacks=[early_stop_callback],
    )
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=shared_parameter.batch_size,
        optimizer_args={"lr": shared_parameter.learning_rate},
        is_use_log=1,
    )

    return_split = dataset.split_train_and_test(
        train_ratio=algo_parameter.train_ratio,
        random_seed=42,
    )

    detector = AlgorithmOneVariableDetector(
        estimator=estimator,
        base_training_parameter=base_train_param,
        pytorch_trainer_config=pl_config,
        candidate_regularization_parameters=candidate_params,
        is_p_value_filter=algo_parameter.is_p_value_filter,
        n_permutation_test=algo_parameter.n_permutation_test,
    )

    result: AlgorithmOneResult = detector.run_detection(
        training_dataset=return_split.train_dataset,
        validation_dataset=return_split.test_dataset,
    )

    duration = time.perf_counter() - start_time
    detected_vars: ty.List[int] = []
    if result.selected_model is not None and result.selected_model.selected_variables is not None:
        detected_vars = list(result.selected_model.selected_variables)
    # end if

    return AlgorithmExecutionResult(
        detected_variables=detected_vars,
        execution_time_seconds=duration,
        raw_result=result,
    )
# end def


def run_benchmark_mmd_cv(
    dataset: SimpleDataset,
    estimator: QuadraticMmdEstimator,
    mmd_cv_parameter: MmdCvBenchmarkParameter = CONFIG_MMD_CV,
    shared_parameter: SharedAlgorithmBenchmarkParameter = CONFIG_SHARED_ALGO,
) -> AlgorithmExecutionResult:
    """Execute MMD-CV (Cross-Validation Detector) over candidate regularization parameters."""
    start_time = time.perf_counter()

    candidate_params = [
        RegularizationParameter(l_val, 0.0) for l_val in shared_parameter.lambda_grid
    ]
    early_stop_callback = ConvergenceEarlyStop(
        ignore_epochs=500,
        check_span=100,
        threshold_convergence_ratio=0.001,
        is_noise_reduction=True,
    )
    pl_config = PytorchLightningDefaultArguments(
        max_epochs=shared_parameter.max_epochs,
        accelerator=shared_parameter.accelerator,
        callbacks=[early_stop_callback],
    )
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=shared_parameter.batch_size,
        optimizer_args={"lr": shared_parameter.learning_rate},
        is_use_log=1,
    )

    cv_alg_param = CrossValidationAlgorithmParameter(
        approach_regularization_parameter="fixed_range",
        candidate_regularization_parameter=candidate_params,
        n_subsampling=mmd_cv_parameter.n_subsampling,
        sampling_strategy=mmd_cv_parameter.sampling_strategy,
        ratio_subsampling=mmd_cv_parameter.ratio_subsampling,
        threshold_stability_score=mmd_cv_parameter.threshold_stability_score,
        n_permutation_test=mmd_cv_parameter.n_permutation_test,
    )

    cv_train_params = CrossValidationTrainParameters(
        algorithm_parameter=cv_alg_param,
        base_training_parameter=base_train_param,
        distributed_parameter=DistributedComputingParameter(job_batch_size=1),
    )

    detector = CrossValidationInterpretableVariableDetector(
        estimator=estimator,
        training_parameter=cv_train_params,
        pytorch_trainer_config=pl_config,
    )

    detected_vars: ty.List[int] = []
    result: ty.Optional[CrossValidationTrainedParameter] = None
    try:
        result = detector.run_detection(training_dataset=dataset)
        if result.stable_s_hat is not None:
            detected_vars = list(result.stable_s_hat)
        # end if
    except SameDataException:
        detected_vars = []
    # end try

    duration = time.perf_counter() - start_time
    return AlgorithmExecutionResult(
        detected_variables=detected_vars,
        execution_time_seconds=duration,
        raw_result=result,
    )
# end def


def run_benchmark_path_aggregation(
    dataset: SimpleDataset,
    estimator: QuadraticMmdEstimator,
    subsampling_splits: int,
    path_agg_parameter: PathAggregationBenchmarkParameter = CONFIG_PATH_AGG,
    shared_parameter: SharedAlgorithmBenchmarkParameter = CONFIG_SHARED_ALGO,
) -> AlgorithmExecutionResult:
    """Execute Path Aggregation detector over candidate regularization parameters."""
    start_time = time.perf_counter()

    early_stop_callback = ConvergenceEarlyStop(
        ignore_epochs=500,
        check_span=100,
        threshold_convergence_ratio=0.001,
        is_noise_reduction=True,
    )
    pl_config = PytorchLightningDefaultArguments(
        max_epochs=shared_parameter.max_epochs,
        accelerator=shared_parameter.accelerator,
        callbacks=[early_stop_callback],
    )
    base_train_param = InterpretableMmdTrainParameters(
        batch_size=shared_parameter.batch_size,
        optimizer_args={"lr": shared_parameter.learning_rate},
        is_use_log=1,
    )

    detector = PathAggregationVariableDetector(
        estimator=estimator,
        regularization_grid=shared_parameter.lambda_grid,
        pytorch_trainer_config=pl_config,
        base_training_parameter=base_train_param,
        threshold=path_agg_parameter.threshold,
        selection_strategy=path_agg_parameter.selection_strategy,
        weight_transformation=path_agg_parameter.weight_transformation,
        subsampling_splits=subsampling_splits,
        subsampling_ratio=path_agg_parameter.subsampling_ratio,
        save_split_weights=path_agg_parameter.save_split_weights,
        train_accelerator=shared_parameter.accelerator,
        distributed_mode="single",
    )

    result: PathAggregationDetectionResult = detector.run_detection(training_dataset=dataset)
    duration = time.perf_counter() - start_time
    detected_vars = list(result.selected_variables)

    return AlgorithmExecutionResult(
        detected_variables=detected_vars,
        execution_time_seconds=duration,
        raw_result=result,
    )
# end def


# ==============================================================================
# 4. Pytest Test Case
# ==============================================================================


@pytest.mark.parametrize("q_dist_name", ["gaussian", "laplace"])
@pytest.mark.parametrize("seed", [101])
def test_benchmark_three_algorithms(q_dist_name: str, seed: int):
    """Run benchmark comparing Algorithm One, MMD-CV, and Path Aggregation."""
    grid_size = len(CONFIG_SHARED_ALGO.lambda_grid)
    print(f"\n=======================================================")
    print(f"BENCHMARK: Q={q_dist_name.upper()} | Seed={seed} | Grid Size={grid_size}")
    print(f"Lambda Grid: {CONFIG_SHARED_ALGO.lambda_grid}")
    print(f"=======================================================")

    dataset, ground_truth, estimator = build_dataset_and_estimator(
        q_dist_name=q_dist_name,
        seed=seed,
        data_parameter=CONFIG_DATA,
    )
    print(f"Ground Truth Variables ({CONFIG_DATA.mixture_rate * CONFIG_DATA.dimension_size:.0f} of {CONFIG_DATA.dimension_size}): {ground_truth}")

    # 1. Algorithm One
    res_a1 = run_benchmark_algorithm_one(
        dataset=dataset,
        estimator=estimator,
        algo_parameter=CONFIG_ALGO_ONE,
        shared_parameter=CONFIG_SHARED_ALGO,
    )
    metrics_a1 = compute_metrics_variable_detection(res_a1.detected_variables, ground_truth)

    # 2. MMD-CV (Cross-Validation Detector with B=5 splits)
    res_cv = run_benchmark_mmd_cv(
        dataset=dataset,
        estimator=estimator,
        mmd_cv_parameter=CONFIG_MMD_CV,
        shared_parameter=CONFIG_SHARED_ALGO,
    )
    metrics_cv = compute_metrics_variable_detection(res_cv.detected_variables, ground_truth)

    # 3. Path Aggregation (Full sample, B=1)
    res_pa_full = run_benchmark_path_aggregation(
        dataset=dataset,
        estimator=estimator,
        subsampling_splits=CONFIG_PATH_AGG.subsampling_splits_full,
        path_agg_parameter=CONFIG_PATH_AGG,
        shared_parameter=CONFIG_SHARED_ALGO,
    )
    metrics_pa_full = compute_metrics_variable_detection(res_pa_full.detected_variables, ground_truth)

    # 4. Path Aggregation (With Subsampling, B=5)
    res_pa_sub = run_benchmark_path_aggregation(
        dataset=dataset,
        estimator=estimator,
        subsampling_splits=CONFIG_PATH_AGG.subsampling_splits_sub,
        path_agg_parameter=CONFIG_PATH_AGG,
        shared_parameter=CONFIG_SHARED_ALGO,
    )
    metrics_pa_sub = compute_metrics_variable_detection(res_pa_sub.detected_variables, ground_truth)

    # Comparative results table
    print(f"\n{'Algorithm':<28} | {'Time (s)':<10} | {'Detected':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 90)
    print(f"{'Algorithm One':<28} | {res_a1.execution_time_seconds:<10.2f} | {metrics_a1.num_detected:<10} | {metrics_a1.precision:<10.3f} | {metrics_a1.recall:<10.3f} | {metrics_a1.f1:<10.3f}")
    print(f"{'MMD-CV (B=5)':<28} | {res_cv.execution_time_seconds:<10.2f} | {metrics_cv.num_detected:<10} | {metrics_cv.precision:<10.3f} | {metrics_cv.recall:<10.3f} | {metrics_cv.f1:<10.3f}")
    print(f"{'Path Aggregation (B=1)':<28} | {res_pa_full.execution_time_seconds:<10.2f} | {metrics_pa_full.num_detected:<10} | {metrics_pa_full.precision:<10.3f} | {metrics_pa_full.recall:<10.3f} | {metrics_pa_full.f1:<10.3f}")
    print(f"{'Path Aggregation (B=5)':<28} | {res_pa_sub.execution_time_seconds:<10.2f} | {metrics_pa_sub.num_detected:<10} | {metrics_pa_sub.precision:<10.3f} | {metrics_pa_sub.recall:<10.3f} | {metrics_pa_sub.f1:<10.3f}")
    print("=" * 90)

    # 5. Serialize results to disk in dedicated run directory
    results_entries = [
        BenchmarkAlgorithmResultEntry(
            algorithm_name="Algorithm One",
            time_seconds=res_a1.execution_time_seconds,
            num_detected=metrics_a1.num_detected,
            precision=metrics_a1.precision,
            recall=metrics_a1.recall,
            f1_score=metrics_a1.f1,
            detected_variables=res_a1.detected_variables,
        ),
        BenchmarkAlgorithmResultEntry(
            algorithm_name="MMD-CV (B=5)",
            time_seconds=res_cv.execution_time_seconds,
            num_detected=metrics_cv.num_detected,
            precision=metrics_cv.precision,
            recall=metrics_cv.recall,
            f1_score=metrics_cv.f1,
            detected_variables=res_cv.detected_variables,
        ),
        BenchmarkAlgorithmResultEntry(
            algorithm_name="Path Aggregation (B=1)",
            time_seconds=res_pa_full.execution_time_seconds,
            num_detected=metrics_pa_full.num_detected,
            precision=metrics_pa_full.precision,
            recall=metrics_pa_full.recall,
            f1_score=metrics_pa_full.f1,
            detected_variables=res_pa_full.detected_variables,
        ),
        BenchmarkAlgorithmResultEntry(
            algorithm_name="Path Aggregation (B=5)",
            time_seconds=res_pa_sub.execution_time_seconds,
            num_detected=metrics_pa_sub.num_detected,
            precision=metrics_pa_sub.precision,
            recall=metrics_pa_sub.recall,
            f1_score=metrics_pa_sub.f1,
            detected_variables=res_pa_sub.detected_variables,
        ),
    ]

    experiment_output = BenchmarkExperimentOutput(
        timestamp=datetime.now().isoformat(),
        q_distribution_name=q_dist_name,
        seed=seed,
        ground_truth_variables=ground_truth,
        data_parameters=CONFIG_DATA,
        shared_algorithm_parameters=CONFIG_SHARED_ALGO,
        algorithm_one_parameters=CONFIG_ALGO_ONE,
        mmd_cv_parameters=CONFIG_MMD_CV,
        path_aggregation_parameters=CONFIG_PATH_AGG,
        results=results_entries,
    )

    b_full = CONFIG_PATH_AGG.subsampling_splits_full
    b_sub = CONFIG_PATH_AGG.subsampling_splits_sub

    raw_results_dict = {
        "algorithm_one_result": res_a1.raw_result,
        "mmd_cv_result": res_cv.raw_result,
        f"path_aggregation_b{b_full}_result": res_pa_full.raw_result,
        f"path_aggregation_b{b_sub}_result": res_pa_sub.raw_result,
    }

    saved_run_dir = save_benchmark_run_artifacts(
        experiment_output=experiment_output,
        raw_algorithm_results=raw_results_dict,
        output_base_dir=DEFAULT_BENCHMARK_OUTPUT_DIR,
        file_format="pt",
    )

    # Validity assertions
    assert saved_run_dir.exists(), f"Saved run directory does not exist: {saved_run_dir}"
    assert (saved_run_dir / "summary.json").exists(), "summary.json does not exist"
    assert (saved_run_dir / "algorithm_one_result.pt").exists(), "algorithm_one_result.pt does not exist"
    assert (saved_run_dir / "mmd_cv_result.pt").exists(), "mmd_cv_result.pt does not exist"
    assert (saved_run_dir / f"path_aggregation_b{b_full}_result.pt").exists(), f"path_aggregation_b{b_full}_result.pt does not exist"
    assert (saved_run_dir / f"path_aggregation_b{b_sub}_result.pt").exists(), f"path_aggregation_b{b_sub}_result.pt does not exist"
    assert res_pa_full.execution_time_seconds > 0.0
    assert res_cv.execution_time_seconds > 0.0
    assert res_a1.execution_time_seconds > 0.0
# end def
