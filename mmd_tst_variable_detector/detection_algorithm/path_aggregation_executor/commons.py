import typing as ty
from pydantic import BaseModel, Field, ConfigDict


class PathAggregationAlgorithmParameter(BaseModel):
    """Configuration parameters for the path_aggregation algorithm."""

    regularization_grid: ty.List[float] = Field(
        ...,
        description="Grid of L1 regularization parameters lambda in ascending order.",
    )
    threshold: float = Field(
        default=0.01,
        gt=0.0,
        description="Selection threshold tau > 0 for aggregated importance scores.",
    )
    weight_transformation: ty.Literal["identity", "bounded"] = Field(
        default="identity",
        description="Transformation function rho(t): 'identity' (t) or 'bounded' (t / (1+t)).",
    )
    path_weights: ty.Optional[ty.List[float]] = Field(
        default=None,
        description="Non-negative path weights w_lambda summing to 1.0. Defaults to uniform weights.",
    )
    subsampling_splits: int = Field(
        default=1,
        ge=1,
        description="Number of subsampling splits B. If 1, operates on the full dataset without splitting.",
    )
    subsampling_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description="Data fraction to retain per subsample split when subsampling_splits > 1.",
    )
    random_seed: ty.Optional[int] = Field(
        default=42,
        description="Random seed for reproducible subsampling.",
    )
# end class


class PathAggregationTaskRequest(BaseModel):
    """Payload representing an individual regularized optimization task along the path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(description="Unique task identifier, e.g., 'lambda_0.010000-split_0'.")
    lambda_val: float = Field(description="L1 regularization parameter lambda.")
    split_index: int = Field(default=0, description="Subsample split index b (0 if no subsampling).")
    estimator: ty.Any = Field(description="Instance of BaseMmdEstimator with ARD product kernel.")
    dataset_train: ty.Any = Field(description="Training dataset containing sample groups X and Y.")
    dataset_validation: ty.Optional[ty.Any] = Field(
        default=None,
        description="Optional validation dataset (defaults to dataset_train if None).",
    )
    training_parameter: ty.Any = Field(description="InterpretableMmdTrainParameters for the learner.")
    pytorch_trainer_config: ty.Any = Field(description="PytorchLightningDefaultArguments configuration.")
# end class


class PathAggregationTaskResult(BaseModel):
    """Result of an individual regularized optimization task."""

    task_id: str = Field(description="Unique task identifier.")
    lambda_val: float = Field(description="L1 regularization parameter lambda used.")
    split_index: int = Field(default=0, description="Split index b.")
    is_success: bool = Field(description="Whether optimization converged without non-recoverable error.")
    trained_ard_weights: ty.List[float] = Field(description="Optimized ARD weight vector a_lambda as a float list.")
    loss_final: float = Field(default=0.0, description="Final training loss value.")
    execution_time_seconds: float = Field(default=0.0, description="Wall-clock execution duration in seconds.")
# end class


class PathAggregationDetectionResult(BaseModel):
    """Final output container for the path_aggregation algorithm."""

    selected_variables: ty.List[int] = Field(
        description="0-indexed list of variable indices identified where Pi_hat_j > tau."
    )
    aggregated_scores: ty.List[float] = Field(
        description="Length-d vector of aggregated variable importance scores Pi_hat_j."
    )
    regularization_path_weights: ty.List[ty.List[float]] = Field(
        description="Matrix of shape (L, d) containing optimized weights along the regularization path."
    )
    regularization_grid: ty.List[float] = Field(
        description="The regularization grid Lambda = {lambda_1, ..., lambda_L} evaluated."
    )
    execution_metadata: ty.Dict[str, ty.Any] = Field(
        default_factory=dict,
        description="Diagnostic statistics (execution time, convergence flags, task results).",
    )
# end class
