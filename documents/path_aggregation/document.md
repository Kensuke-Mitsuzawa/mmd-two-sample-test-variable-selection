# `path_aggregation` Algorithm Specification

This document specifies the **`path_aggregation`** algorithm for two-sample test variable selection, based on the procedure described in [simplified_mmd_variable_selection.pdf](file:///root/mmd-two-sample-test-variable-selection/documents/path_aggregation/simplified_mmd_variable_selection.pdf).

---

## 1. Core Mathematical Flow

### 1.1 Problem Setting
Let $P$ and $Q$ be two Borel probability measures on $\mathbb{R}^d$. The goal is to identify a sparse subset of informative features $S^\star \subseteq \{1, \dots, d\}$ whose joint marginal distributions capture the discrepancy between $P$ and $Q$.

Given samples:
- $X = \{x^{(1)}, \dots, x^{(n_x)}\} \sim P$
- $Y = \{y^{(1)}, \dots, y^{(n_y)}\} \sim Q$

We use an Automatic Relevance Determination (ARD) product kernel:
$$k_a(x, x') = \prod_{j=1}^d \kappa(x_j, x'_j; a_j)$$
where $a = (a_1, \dots, a_d) \in \mathbb{R}^d_+$ is a vector of non-negative ARD feature weights, and $\kappa$ is a base 1D kernel (such as a 1D Gaussian kernel $\kappa(u, v; a_j) = \exp(-a_j (u - v)^2)$).

### 1.2 Empirical Test-Power Criterion
Let $\hat{J}_n(a)$ denote the empirical Maximum Mean Discrepancy (MMD) test-power criterion evaluated on the data:
$$\hat{J}_n(a) = \frac{\widehat{\mathrm{MMD}}^2(X, Y; a)}{\hat{\sigma}_{H_1}(a) + \varepsilon}$$
where $\widehat{\mathrm{MMD}}^2(X, Y; a)$ is the empirical unbiased/U-statistic MMD estimator, $\hat{\sigma}_{H_1}(a)$ is the estimated standard deviation of the test statistic under the alternative hypothesis $H_1: P \neq Q$, and $\varepsilon > 0$ is a small numerical regularizer. Larger values of $\hat{J}_n(a)$ correspond to higher estimated test power.

### 1.3 Regularized Optimization Along a Path
Instead of tuning a single regularization parameter via validation splits, a finite grid of regularization parameters $\Lambda$ is defined:
$$\Lambda = \{\lambda_1, \dots, \lambda_L\}, \quad 0 < \lambda_1 < \lambda_2 < \dots < \lambda_L$$

For each regularization strength $\lambda \in \Lambda$, the optimal ARD weight vector $\hat{a}_\lambda$ is obtained by solving:
$$\hat{a}_\lambda \in \arg\max_{a \in \mathbb{R}^d_+} \left\{ \log \hat{J}_n(a) - \lambda \|a\|_1 \right\}$$
- The $\ell_1$ penalty $-\lambda \|a\|_1 = -\lambda \sum_{j=1}^d a_j$ forces non-informative weights towards zero.
- Large $\lambda$ values yield sparse solutions capturing only the strongest discrepancies.
- Small $\lambda$ values allow weaker discriminating features to enter the active set.
- All solutions $\{\hat{a}_\lambda\}_{\lambda \in \Lambda}$ along the regularization path are retained.

### 1.4 Path Aggregation
For each variable $j \in \{1, \dots, d\}$, an aggregated importance score $\hat{\Pi}_j$ is computed by integrating feature weights over the regularization path:
$$\hat{\Pi}_j = \sum_{\lambda \in \Lambda} w_\lambda \rho\left(\hat{a}_{\lambda, j}\right)$$
where:
- $w_\lambda \ge 0$ with $\sum_{\lambda \in \Lambda} w_\lambda = 1$. The standard default is uniform weighting:
  $$w_\lambda = \frac{1}{|\Lambda|}$$
- $\rho: \mathbb{R}_+ \to \mathbb{R}_+$ is a monotonic score transformation satisfying:
  $$\rho(0) = 0 \quad \text{and} \quad \rho(t) > 0 \;\; \forall t > 0$$
  Supported transformations:
  - **Identity**: $\rho(t) = t$
  - **Bounded**: $\rho(t) = \frac{t}{1 + t}$

### 1.5 Final Variable Selection
Given a final aggregation threshold $\tau > 0$, the set of selected variables $\hat{S}$ is:
$$\hat{S} = \left\{ j \in \{1, \dots, d\} : \hat{\Pi}_j > \tau \right\}$$

### 1.6 Optional Stability Enhancement (Subsampling Layer)
If additional finite-sample stability is desired, repeated subsampling can be applied over $B$ splits ($b = 1, \dots, B$):
1. Subsample $X^{(b)} \subset X$ and $Y^{(b)} \subset Y$.
2. Compute the regularized weights $\hat{a}_\lambda^{(b)}$ for each $\lambda \in \Lambda$ on split $b$.
3. Compute the multi-split aggregated score:
   $$\hat{\Pi}_j = \frac{1}{B} \sum_{b=1}^B \sum_{\lambda \in \Lambda} w_\lambda \rho\left(\hat{a}_{\lambda, j}^{(b)}\right)$$

### 1.7 Population Theory Justification
Let $a_\lambda^\star \in \arg\max_{a \in \mathbb{R}^d_+} \{ \log J(a) - \lambda \|a\|_1 \}$ and $\Pi_j = \sum_{\lambda \in \Lambda} w_\lambda \rho(a_{\lambda, j}^\star)$.
The population theory requires two conditions:
1. **Irrelevant-variable exclusion**: If $j \notin S^\star$, then $a_{\lambda, j}^\star = 0$ for all $\lambda \in \Lambda$.
2. **Path coverage**: If $j \in S^\star$, then $a_{\lambda, j}^\star > 0$ for at least one $\lambda \in \Lambda$ with $w_\lambda > 0$.

Under these conditions:
$$\Pi_j > 0 \iff j \in S^\star$$
Path aggregation is therefore the direct finite-sample analogue of the population support-recovery argument.

---

## 2. Inputs and Outputs

### 2.1 Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `training_dataset` | `BaseDataset` | Dataset containing two sample groups $X \in \mathbb{R}^{n_x \times d}$ and $Y \in \mathbb{R}^{n_y \times d}$. |
| `regularization_grid` ($\Lambda$) | `Sequence[float]` or `RegularizationGridConfig` | Grid of regularization values $\Lambda = \{\lambda_1, \dots, \lambda_L\}$ in ascending order. |
| `threshold` ($\tau$) | `float` | Threshold $\tau > 0$ used to select features from aggregated scores $\hat{\Pi}$. |
| `weight_transformation` ($\rho$) | `Literal["identity", "bounded"]` or `Callable[[Tensor], Tensor]` | Transformation function $\rho(t)$ applied to weights before path summation (default: `"identity"`). |
| `path_weights` ($w_\lambda$) | `Optional[Sequence[float]]` | Non-negative path weights summing to 1. Defaults to uniform weights $w_\lambda = 1 / |\Lambda|$. |
| `subsampling_splits` ($B$) | `int` | Number of repeated subsampling splits $B \ge 1$ (default: `1`, meaning full data without subsampling). |
| `subsampling_ratio` ($r$) | `float` | Fraction of data used per subsample split when $B > 1$ (default: `0.8`). |
| `optimizer_config` | `PytorchLightningDefaultArguments` or optimizer settings | Optimization parameters for solving $\hat{a}_\lambda$ (learning rate, epochs, convergence criteria). |

### 2.2 Outputs

The algorithm returns a structured result (e.g. a Pydantic `BaseModel`):

| Output Field | Type | Description |
| :--- | :--- | :--- |
| `selected_variables` ($\hat{S}$) | `List[int]` | 0-indexed list of selected variable indices where $\hat{\Pi}_j > \tau$. |
| `aggregated_scores` ($\hat{\Pi}$) | `List[float]` / `Tensor` | Length-$d$ vector of aggregated importance scores $\hat{\Pi}_j$. |
| `regularization_path_weights` ($\hat{\mathbf{A}}$) | `List[List[float]]` / `Tensor` | Matrix of shape $(L, d)$ storing optimized weights $\hat{a}_\lambda$ for each $\lambda \in \Lambda$ (averaged across $B$ splits if $B > 1$). |
| `regularization_grid` ($\Lambda$) | `List[float]` | The regularization values $\{\lambda_1, \dots, \lambda_L\}$ used during execution. |
| `execution_metadata` | `Dict[str, Any]` | Diagnostic statistics including optimization loss history, convergence flags, and execution runtimes. |

---

## 3. Parameters of the Algorithm

The algorithm parameters are organized into two groups:

### 3.1 Statistical Parameters

1. **`regularization_grid` ($\Lambda$)**:
   - **Type**: `List[float]`
   - **Meaning**: Finite sequence of penalty parameters $0 < \lambda_1 < \dots < \lambda_L$.
   - **Default generation**: Logarithmically spaced grid spanning from strong regularization (e.g. $\lambda_{\max} \approx 10.0$) down to weak regularization (e.g. $\lambda_{\min} \approx 10^{-4}$), typically with $L \in [10, 30]$ points.
2. **`threshold` ($\tau$)**:
   - **Type**: `float`
   - **Meaning**: Cutoff value for aggregated importance.
   - **Default**: Configurable positive value (e.g. $\tau = 10^{-3}$ or $\tau = 10^{-2}$). Controls tolerance to finite-sample noise.
3. **`weight_transformation` ($\rho$)**:
   - **Type**: `str` or `Callable`
   - **Options**:
     - `"identity"`: $\rho(t) = t$
     - `"bounded"`: $\rho(t) = \frac{t}{1 + t}$
   - **Default**: `"identity"`.
4. **`path_weights` ($w_\lambda$)**:
   - **Type**: `Optional[List[float]]`
   - **Meaning**: Relative weight of each $\lambda \in \Lambda$.
   - **Default**: Uniform $w_\lambda = \frac{1}{|\Lambda|}$.
5. **`subsampling_splits` ($B$)**:
   - **Type**: `int`
   - **Meaning**: Number of subsampling iterations.
   - **Default**: `1` (Full-data path aggregation).
6. **`subsampling_ratio` ($r$)**:
   - **Type**: `float`
   - **Meaning**: Subsample size fraction when $B > 1$.
   - **Default**: `0.8`.

### 3.2 Optimization & Engine Parameters

1. **`learning_rate`**: Learning rate for optimizing ARD weights $a$ (default: `1e-2` or `5e-3`).
2. **`max_epochs` / `max_iterations`**: Maximum optimization steps per $\lambda \in \Lambda$ (default: `200` to `500`).
3. **`non_negativity_enforcement`**: Method used to maintain $a \ge 0$ (e.g., projection $a \leftarrow \max(a, 0)$, softplus parameterization $a = \operatorname{softplus}(\theta)$, or proximal operator).
4. **`sparse_optimizer`**: Optional proximal-gradient / iterative soft-thresholding operator to achieve exact zero weights.

---

## 4. Pseudo Code

### 4.1 Basic Algorithm (Full-Data Path Aggregation)

```python
"""
Algorithm: path_aggregation (Basic Formulation)
"""

def run_path_aggregation(
    X: Tensor,                  # Sample matrix (n_x, d)
    Y: Tensor,                  # Sample matrix (n_y, d)
    estimator: BaseMmdEstimator,# MMD test power estimator
    lambda_grid: List[float],   # Regularization grid [lambda_1, ..., lambda_L]
    tau: float,                 # Selection threshold > 0
    rho_fn: Callable = lambda t: t,  # Transformation function rho(t)
    path_weights: Optional[List[float]] = None,
) -> PathAggregationResult:
    d = X.shape[1]
    L = len(lambda_grid)
    
    # 1. Initialize path weights
    if path_weights is None:
        w = [1.0 / L] * L
    else:
        w = path_weights
    # end if

    path_weights_matrix = []  # Shape: (L, d)

    # 2. Regularized optimization along path
    for lambda_val in lambda_grid:
        # Solve: max_{a >= 0} { log J_n(a) - lambda_val * ||a||_1 }
        a_hat_lambda = optimize_regularized_mmd(
            X=X,
            Y=Y,
            estimator=estimator,
            regularization_weight=lambda_val,
        )
        path_weights_matrix.append(a_hat_lambda)
    # end for

    # 3. Path aggregation
    pi_hat = [0.0] * d
    for j in range(d):
        aggregated_val = 0.0
        for l_idx, lambda_val in enumerate(lambda_grid):
            a_val = path_weights_matrix[l_idx][j]
            aggregated_val += w[l_idx] * rho_fn(a_val)
        # end for
        pi_hat[j] = aggregated_val
    # end for

    # 4. Final variable selection
    selected_indices = []
    for j in range(d):
        if pi_hat[j] > tau:
            selected_indices.append(j)
        # end if
    # end for

    return PathAggregationResult(
        selected_variables=selected_indices,
        aggregated_scores=pi_hat,
        regularization_path_weights=path_weights_matrix,
        regularization_grid=lambda_grid,
    )
# end def
```

### 4.2 Algorithm with Optional Stability Subsampling ($B > 1$)

```python
"""
Algorithm: path_aggregation (With Stability Subsampling)
"""

def run_path_aggregation_with_subsampling(
    X: Tensor,
    Y: Tensor,
    estimator: BaseMmdEstimator,
    lambda_grid: List[float],
    tau: float,
    B: int = 1,                 # Number of subsampling splits
    subsample_ratio: float = 0.8,
    rho_fn: Callable = lambda t: t,
    path_weights: Optional[List[float]] = None,
) -> PathAggregationResult:
    d = X.shape[1]
    L = len(lambda_grid)
    
    if path_weights is None:
        w = [1.0 / L] * L
    else:
        w = path_weights
    # end if

    # If B == 1, execute full-data path aggregation
    if B <= 1:
        return run_path_aggregation(
            X=X,
            Y=Y,
            estimator=estimator,
            lambda_grid=lambda_grid,
            tau=tau,
            rho_fn=rho_fn,
            path_weights=w,
        )
    # end if

    pi_hat = [0.0] * d
    accumulated_path_weights = [[0.0] * d for _ in range(L)]

    for b in range(B):
        # Draw subsamples of size r * n
        X_sub = draw_subsample(X, ratio=subsample_ratio)
        Y_sub = draw_subsample(Y, ratio=subsample_ratio)

        for l_idx, lambda_val in enumerate(lambda_grid):
            a_hat_lambda_b = optimize_regularized_mmd(
                X=X_sub,
                Y=Y_sub,
                estimator=estimator,
                regularization_weight=lambda_val,
            )
            for j in range(d):
                accumulated_path_weights[l_idx][j] += a_hat_lambda_b[j] / B
                pi_hat[j] += (w[l_idx] * rho_fn(a_hat_lambda_b[j])) / B
            # end for
        # end for
    # end for

    # Thresholding
    selected_indices = [j for j in range(d) if pi_hat[j] > tau]

    return PathAggregationResult(
        selected_variables=selected_indices,
        aggregated_scores=pi_hat,
        regularization_path_weights=accumulated_path_weights,
        regularization_grid=lambda_grid,
    )
# end def
```
