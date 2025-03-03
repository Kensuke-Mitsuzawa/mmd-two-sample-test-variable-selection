import typing
import torch
from torch.jit import script
from torch.nn.functional import pad


"""
Trial module that I attempted to compile with torch.jit.script.
But, there are a lot of errors, and I don't know how to fix them.
I leave this scipt for the moment.
"""


def zero_pad(a: torch.Tensor, pad_width, value=0):
    # pad_width is an array of ndim tuples indicating how many 0 before and after
    # we need to add. We first need to make it compliant with torch syntax, that
    # starts with the last dim, then second last, etc.
    how_pad = []
    for tupl in pad_width[::-1]:
        for element in tupl:
            how_pad.append(element)
        # end for
    # end for
    # how_pad = tuple([element for tupl in pad_width[::-1] for element in tupl])
    return pad(a, how_pad, value=value)


def quantile_function(qs: torch.Tensor, cws: torch.Tensor, xs: torch.Tensor):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    n = xs.shape[0]    
    # this is to ensure the best performance for torch searchsorted
    # and avoid a warning related to non-contiguous arrays
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()

    idx = torch.searchsorted(cws, qs).T
    return torch.take_along_dim(xs, torch.clip(idx, 0, n - 1), dim=0)


def wasserstein_1d(u_values: torch.Tensor, 
                   v_values: torch.Tensor, 
                   u_weights=None, 
                   v_weights=None, 
                   p: int = 1, 
                   require_sort=True):
    r"""
    Computes the 1 dimensional OT loss [15] between two (batched) empirical
    distributions

    .. math:
        OT_{loss} = \int_0^1 |cdf_u^{-1}(q) - cdf_v^{-1}(q)|^p dq

    It is formally the p-Wasserstein distance raised to the power p.
    We do so in a vectorized way by first building the individual quantile functions then integrating them.

    This function should be preferred to `emd_1d` whenever the backend is
    different to numpy, and when gradients over
    either sample positions or weights are required.

    Parameters
    ----------
    u_values: array-like, shape (n, ...)
        locations of the first empirical distribution
    v_values: array-like, shape (m, ...)
        locations of the second empirical distribution
    u_weights: array-like, shape (n, ...), optional
        weights of the first empirical distribution, if None then uniform weights are used
    v_weights: array-like, shape (m, ...), optional
        weights of the second empirical distribution, if None then uniform weights are used
    p: int, optional
        order of the ground metric used, should be at least 1 (see [2, Chap. 2], default is 1
    require_sort: bool, optional
        sort the distributions atoms locations, if False we will consider they have been sorted prior to being passed to
        the function, default is True

    Returns
    -------
    cost: float/array-like, shape (...)
        the batched EMD

    References
    ----------
    .. [15] Peyré, G., & Cuturi, M. (2018). Computational Optimal Transport.

    """

    assert p >= 1, f"The OT loss is only valid for p>=1, {p} was given"

    # if u_weights is not None and v_weights is not None:
    #     nx = get_backend(u_values, v_values, u_weights, v_weights)
    # else:
    #     nx = get_backend(u_values, v_values)

    n = u_values.shape[0]
    m = v_values.shape[0]

    if u_weights is None:
        u_weights = torch.full(u_values.shape, 1. / n, dtype=u_values.dtype)
    elif u_weights.ndim != u_values.ndim:
        u_weights = torch.repeat_interleave(u_weights[..., None], u_values.shape[-1], -1)
    if v_weights is None:
        v_weights = torch.full(v_values.shape, 1. / m, dtype=v_values.dtype)
    elif v_weights.ndim != v_values.ndim:
        v_weights = torch.repeat_interleave(v_weights[..., None], v_values.shape[-1], -1)

    if require_sort:
        u_sorter = torch.argsort(u_values, 0)
        u_values = torch.take_along_dim(u_values, u_sorter, 0)

        v_sorter = torch.argsort(v_values, 0)
        v_values = torch.take_along_dim(v_values, v_sorter, 0)

        u_weights = torch.take_along_dim(u_weights, u_sorter, 0)
        v_weights = torch.take_along_dim(v_weights, v_sorter, 0)

    u_cumweights = torch.cumsum(u_weights, 0)
    v_cumweights = torch.cumsum(v_weights, 0)

    qs_obj = torch.sort(torch.concatenate((u_cumweights, v_cumweights), 0), 0)
    qs = qs_obj.values
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    
    # pad_width = [(1, 0)] + (qs.dim() - 1) * [(0, 0)]
    # pad_width = [(1, 0), (0, 0)]
    # how_pad = []
    # for tupl in pad_width[::-1]:
    #     for element in tupl:
    #         how_pad.append(element)
    #     # end for
    # # end for
    how_pad = [0, 0, 0, 1]
    qs = pad(qs, how_pad, value=0.0)
    # qs = zero_pad(a=qs, pad_width=[(1, 0)] + (qs.ndim - 1) * [(0, 0)])
    delta = qs[1:, ...] - qs[:-1, ...]
    diff_quantiles = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * diff_quantiles, dim=0)
    return torch.sum(delta * torch.pow(diff_quantiles, p), dim=0)


def get_random_projections(d: int, n_projections: int, seed: typing.Optional[int]=None, backend=None, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`

    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """
    rng_ = torch.Generator("cpu")    
    if seed is not None:
        rng_.manual_seed(seed)
    # end if
    
    projections = torch.randn(d, n_projections, generator=rng_)
    projections = projections / torch.sqrt(torch.sum(projections**2, 0, keepdims=True))
    return projections


@script
def sliced_wasserstein_distance(X_s: torch.Tensor, 
                                X_t: torch.Tensor, 
                                a: typing.Optional[torch.Tensor] = None, 
                                b: typing.Optional[torch.Tensor] = None, 
                                n_projections: int = 50, 
                                p: int = 2,
                                projections=None, 
                                seed: typing.Optional[int] = None):
    r"""
    Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance

    .. math::
        \mathcal{SWD}_p(\mu, \nu) = \underset{\theta \sim \mathcal{U}(\mathbb{S}^{d-1})}{\mathbb{E}}\left(\mathcal{W}_p^p(\theta_\# \mu, \theta_\# \nu)\right)^{\frac{1}{p}}


    where :

    - :math:`\theta_\# \mu` stands for the pushforwards of the projection :math:`X \in \mathbb{R}^d \mapsto \langle \theta, X \rangle`


    Parameters
    ----------
    X_s : ndarray, shape (n_samples_a, dim)
        samples in the source domain
    X_t : ndarray, shape (n_samples_b, dim)
        samples in the target domain
    a : ndarray, shape (n_samples_a,), optional
        samples weights in the source domain
    b : ndarray, shape (n_samples_b,), optional
        samples weights in the target domain
    n_projections : int, optional
        Number of projections used for the Monte-Carlo approximation
    p: float, optional =
        Power p used for computing the sliced Wasserstein
    projections: shape (dim, n_projections), optional
        Projection matrix (n_projections and seed are not used in this case)
    seed: int or RandomState or None, optional
        Seed used for random number generator
    log: bool, optional
        if True, sliced_wasserstein_distance returns the projections used and their associated EMD.

    Returns
    -------
    cost: float
        Sliced Wasserstein Cost
    log : dict, optional
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> n_samples_a = 20
    >>> X = np.random.normal(0., 1., (n_samples_a, 5))
    >>> sliced_wasserstein_distance(X, X, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0

    References
    ----------

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters of measures." Journal of Mathematical Imaging and Vision 51.1 (2015): 22-45
    """


    # X_s, X_t = list_to_array(X_s, X_t)

    # if a is not None and b is not None and projections is None:
    #     nx = get_backend(X_s, X_t, a, b)
    # elif a is not None and b is not None and projections is not None:
    #     nx = get_backend(X_s, X_t, a, b, projections)
    # elif a is None and b is None and projections is not None:
    #     nx = get_backend(X_s, X_t, projections)
    # else:
    #     nx = get_backend(X_s, X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = torch.full((n,), 1 / n, dtype=X_s.dtype)
    if b is None:
        b = torch.full((m,), 1 / m, dtype=X_s.dtype)

    d = X_s.shape[1]

    if projections is None:
        projections = get_random_projections(d, n_projections, seed, type_as=X_s)
    else:
        n_projections = projections.shape[1]

    projections = projections.type_as(X_s)
    X_s_projections = torch.matmul(X_s, projections)
    X_t_projections = torch.matmul(X_t, projections)

    projected_emd = wasserstein_1d(X_s_projections, X_t_projections, a, b, p=p)

    res = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
    return res
