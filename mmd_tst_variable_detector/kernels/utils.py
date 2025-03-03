import torch

# @torch.jit.script
def distance_over_3rd_reshape_same_data(x: torch.Tensor
                                        ) -> torch.Tensor:
    """Distance matrix only when x and y have the same number of data.
    Less computation thanks to a triangular matrix.
    Args:
        x: (Sample, N, M)
        device_obj:
    Return:
         A 2nd tensor (Sample, Sample) of L2-norm distance.
    """
    x_samples = x.shape[0]

    if len(x.shape) == 1:
        x_reshaped = x.reshape((x_samples, 1))
    elif len(x.shape) == 2:
        shape_operator = x.shape[1]
        x_reshaped = x.reshape((x_samples, shape_operator))
    elif len(x.shape) == 3:
        shape_operator = x.shape[1] * x.shape[2]
        x_reshaped = x.reshape((x_samples, shape_operator))
    else:
        raise NotImplementedError()

    xx_torch = torch.pdist(x_reshaped)
    tri_torch = torch.zeros((x_samples, x_samples), device=xx_torch.device)
    target_indices = torch.triu_indices(tri_torch.shape[0], tri_torch.shape[1], offset=1)
    xx_torch = xx_torch.to(dtype=tri_torch.dtype)
    tri_torch[target_indices[0], target_indices[1]] = xx_torch
    # copy upper triangular elements
    index_lower = torch.tril_indices(tri_torch.shape[0], tri_torch.shape[1])
    tri_torch[index_lower[0], index_lower[1]] = tri_torch.T[index_lower[0], index_lower[1]]

    return tri_torch


# @torch.jit.script
def distance_over_3rd_reshape_xy_data(x: torch.Tensor,
                                      y: torch.Tensor) -> torch.Tensor:
    """Distance matrix only when x and y have the same number of data.
    Less computation thanks to a triangular matrix.
    Args:
        x: (Sample, N, M)
        y: (Sample, N, M)
        power
    Return:
         A 2nd tensor (Sample, Sample) of 2-norm distance.
    """
    # assert x.shape == y.shape  # comment out 2023/02/28. Is that correct?
    # n_samples = x.shape[0]

    if len(x.shape) == 1:
        x_reshaped = x.reshape((x.shape[0], 1))
        y_reshaped = y.reshape((y.shape[0], 1))
    elif len(x.shape) == 2:
        shape_operator = x.shape[1]
        x_reshaped = x.reshape((x.shape[0], shape_operator))
        y_reshaped = y.reshape((y.shape[0], shape_operator))
    elif len(x.shape) == 3:
        shape_operator = x.shape[1] * x.shape[2]
        x_reshaped = x.reshape((x.shape[0], shape_operator))
        y_reshaped = y.reshape((y.shape[0], shape_operator))
    else:
        raise NotImplementedError()

    xy_torch = torch.cdist(x_reshaped, y_reshaped)
    return xy_torch