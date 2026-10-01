import typing as ty
import torch
from pydantic import BaseModel, Field

from .commons import PathAggregationAlgorithmParameter, PathAggregationTaskResult


class AggregatedPathScores(BaseModel):
    """Container for aggregated importance scores and path weights."""

    aggregated_scores: ty.List[float] = Field(
        description="Aggregated importance score vector Pi_hat."
    )
    regularization_path_weights: ty.List[ty.List[float]] = Field(
        description="Matrix of shape (L, d) storing average ARD weights along the regularization path."
    )
    split_path_weights: ty.Optional[ty.List[ty.List[ty.List[float]]]] = Field(
        default=None,
        description="Nested list of shape (L, B, d) storing individual split ARD weights for each lambda.",
    )
# end class


def transform_score_identity(tensor_val: torch.Tensor) -> torch.Tensor:
    """Identity score transformation rho(t) = t.

    Parameters
    ----------
    tensor_val : torch.Tensor
        Non-negative weight tensor.

    Returns
    -------
    torch.Tensor
        Transformed weight tensor.
    """
    return tensor_val
# end def


def transform_score_bounded(tensor_val: torch.Tensor) -> torch.Tensor:
    """Bounded score transformation rho(t) = t / (1 + t).

    Parameters
    ----------
    tensor_val : torch.Tensor
        Non-negative weight tensor.

    Returns
    -------
    torch.Tensor
        Transformed bounded weight tensor in [0, 1).
    """
    return torch.div(tensor_val, torch.add(tensor_val, 1.0))
# end def


class PathAggregator(object):
    """Aggregates regularized weights along the regularization path and performs selection.

    Computes:
        Pi_hat_j = sum_{lambda in Lambda} w_lambda * rho(a_{lambda, j})
    and selects variables:
        S_hat = { j : Pi_hat_j > tau }
    """

    def __init__(self, parameters: PathAggregationAlgorithmParameter) -> None:
        self.parameters = parameters
        if parameters.weight_transformation == "bounded":
            self.rho_fn = transform_score_bounded
        else:
            self.rho_fn = transform_score_identity
        # end if
    # end def

    def aggregate_path_weights(
        self,
        task_results: ty.List[PathAggregationTaskResult],
        dimension_size: int,
    ) -> AggregatedPathScores:
        """Aggregate weights across the regularization path and subsampling splits.

        Parameters
        ----------
        task_results : List[PathAggregationTaskResult]
            Completed task results across all lambda values and splits.
        dimension_size : int
            Number of feature dimensions d.

        Returns
        -------
        AggregatedPathScores
            Aggregated scores vector Pi_hat and path weights matrix.
        """
        grid = self.parameters.regularization_grid
        length_grid = len(grid)

        # Normalize path weights w_lambda
        if self.parameters.path_weights is not None:
            sum_weights = sum(self.parameters.path_weights)
            normalized_weights = [w / sum_weights for w in self.parameters.path_weights]
        else:
            normalized_weights = [1.0 / length_grid] * length_grid
        # end if

        # Group task results by lambda value
        dict_lambda_to_index = {val: idx for idx, val in enumerate(grid)}
        dict_index_to_weights: ty.Dict[int, ty.List[ty.List[float]]] = {
            idx: [] for idx in range(length_grid)
        }

        for result in task_results:
            if not result.is_success:
                continue
            # end if
            target_idx = dict_lambda_to_index.get(result.lambda_val)
            if target_idx is not None:
                dict_index_to_weights[target_idx].append(result.trained_ard_weights)
            # end if
        # end for

        # Average weights across splits for each lambda
        matrix_path_weights: ty.List[ty.List[float]] = []
        for l_idx in range(length_grid):
            split_weights = dict_index_to_weights[l_idx]
            if not split_weights:
                matrix_path_weights.append([0.0] * dimension_size)
            else:
                tensor_splits = torch.tensor(split_weights, dtype=torch.float32)
                tensor_mean = torch.mean(tensor_splits, dim=0)
                matrix_path_weights.append(tensor_mean.tolist())
            # end if
        # end for

        # Compute Pi_hat_j = sum_{lambda} w_lambda * rho(a_{lambda, j})
        tensor_scores = torch.zeros(dimension_size, dtype=torch.float32)
        for l_idx in range(length_grid):
            w_lambda = normalized_weights[l_idx]
            tensor_a = torch.tensor(matrix_path_weights[l_idx], dtype=torch.float32)
            tensor_scores += w_lambda * self.rho_fn(tensor_a)
        # end for

        # Collect individual split weights if requested
        matrix_split_weights: ty.Optional[ty.List[ty.List[ty.List[float]]]] = None
        if getattr(self.parameters, "save_split_weights", True):
            matrix_split_weights = [
                dict_index_to_weights[l_idx] for l_idx in range(length_grid)
            ]
        # end if

        return AggregatedPathScores(
            aggregated_scores=tensor_scores.tolist(),
            regularization_path_weights=matrix_path_weights,
            split_path_weights=matrix_split_weights,
        )
    # end def

    def select_variables(self, aggregated_scores: ty.List[float]) -> ty.List[int]:
        """Identify selected variable indices exceeding threshold tau or via histogram-based valley detection.

        Parameters
        ----------
        aggregated_scores : List[float]
            Aggregated importance score vector Pi_hat.

        Returns
        -------
        List[int]
            0-indexed selected variable indices.
        """
        strategy = getattr(self.parameters, "selection_strategy", "threshold")
        if strategy == "hist_based":
            from ...utils.variable_detection import detect_variables
            tensor_scores = torch.tensor(aggregated_scores, dtype=torch.float32)
            return detect_variables(
                variable_weights=tensor_scores,
                variable_detection_approach="hist_based",
            )
        elif strategy == "normalized_threshold":
            from ...utils.variable_detection import detect_variables
            tensor_scores = torch.tensor(aggregated_scores, dtype=torch.float32)
            return detect_variables(
                variable_weights=tensor_scores,
                variable_detection_approach="threshold",
                threshold_weights=self.parameters.threshold,
                is_normalize_ard_weights=True,
            )
        else:
            threshold = self.parameters.threshold
            selected_indices = [
                idx for idx, score in enumerate(aggregated_scores) if score > threshold
            ]
            return selected_indices
        # end if
    # end def
# end class
