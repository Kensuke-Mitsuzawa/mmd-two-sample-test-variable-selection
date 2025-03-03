import typing
import math
import random

import numpy



def __generate_noised_sampling(
        n_sample: int,
        dimension_size: int,
        mixture_rate: float,
        distribution_conf_p: typing.Dict,
        distribution_conf_q: typing.Dict,
        index_replace: typing.Optional[typing.List[int]] = None,
        random_seed_x: typing.Optional[int] = None,
        random_seed_y: typing.Optional[int] = None,
        random_seed_noise: typing.Optional[int] = None,        
        ) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.List[int]]:
    if random_seed_x is None:
        random_seed_x = random.randint(0, 100)
    # end if
    if random_seed_y is None:
        random_seed_y = random_seed_x + 100
    # end if
    if random_seed_noise is None:
        random_seed_noise = random_seed_x + 200
    # end if

    rng_x = numpy.random.default_rng(random_seed_x)
    rng_y = numpy.random.default_rng(random_seed_y)
    rng_noise = numpy.random.default_rng(random_seed_noise)

    x = rng_x.normal(loc=distribution_conf_p['mu'], scale=distribution_conf_p['sigma'],
                            size=(n_sample, dimension_size))
    y = rng_y.normal(loc=distribution_conf_p['mu'], scale=distribution_conf_p['sigma'],
                            size=(n_sample, dimension_size))

    if index_replace is None:
        n_replace_dimension = math.ceil(mixture_rate * dimension_size)
        assert n_replace_dimension is not None
        index_replace = list(rng_noise.choice(range(0, dimension_size), size=n_replace_dimension, replace=False))
    else:
        n_replace_dimension = len(index_replace)
        # end if
    # end if

    if distribution_conf_q['type'] == 'gaussian':
        y_replace = rng_noise.normal(
            loc=distribution_conf_q['mu'], scale=distribution_conf_q['sigma'], size=(n_sample, n_replace_dimension)
        )
    elif distribution_conf_q['type'] == 'laplace':
        y_replace = rng_noise.laplace(
            loc=distribution_conf_q['mu'], scale=distribution_conf_q['sigma'], size=(n_sample, n_replace_dimension)
        )
    elif distribution_conf_q['type'] == 'copy_gaussian':
        __y_value = rng_noise.normal(loc=distribution_conf_q['mu'], scale=distribution_conf_q['sigma'], size=(n_sample, 1))
        __array_duplicated = numpy.repeat(__y_value, n_replace_dimension, axis=1)
        y_replace = __array_duplicated
    else:
        raise NotImplementedError()
    # end if

    for i, ind in enumerate(index_replace):
        y[:, ind] = y_replace[:, i]
    # end for
    assert not numpy.array_equal(x, y)

    return x, y, index_replace


def __generate_noised_sampling_redundant(        
        n_sample: int,
        dimension_size: int,
        mixture_rate: float,
        distribution_conf_p: typing.Dict,
        distribution_conf_q: typing.Dict,
        index_replace: typing.Optional[typing.List[int]] = None,
        random_seed_x: typing.Optional[int] = None,
        random_seed_y: typing.Optional[int] = None,
        random_seed_noise: typing.Optional[int] = None
        ) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.List[int]]:
    """X_s and Y_s are dimensions where distributions P, Q differ. Other dimensions are all euqally zero values.
    """
    set_x = numpy.zeros((n_sample, dimension_size))
    set_y = numpy.zeros((n_sample, dimension_size))

    if random_seed_x is None:
        random_seed_x = random.randint(0, 100)
    # end if
    if random_seed_y is None:
        random_seed_y = random_seed_x + 100
    # end if
    if random_seed_noise is None:
        random_seed_noise = random_seed_x + 200
    # end if

    rng_x = numpy.random.default_rng(random_seed_x)
    rng_y = numpy.random.default_rng(random_seed_y)
    rng_noise = random.Random(random_seed_noise)

    if index_replace is None:
        n_replace_dimension = math.ceil(mixture_rate * dimension_size)
        index_replace = rng_noise.sample(range(0, dimension_size), k=n_replace_dimension)
    else:
        n_replace_dimension = len(index_replace)
    # end if

    x_s = rng_x.normal(loc=distribution_conf_p['mu'], scale=distribution_conf_p['sigma'],
                            size=(n_sample, n_replace_dimension))
    y_s = rng_y.normal(loc=distribution_conf_q['mu'], scale=distribution_conf_q['sigma'],
                            size=(n_sample, n_replace_dimension))

    for i, __ind_replace in enumerate(index_replace):
        set_x[:, __ind_replace] = x_s[:, i]
        set_y[:, __ind_replace] = y_s[:, i]
    # end for

    return set_x, set_y, index_replace


def sampling_from_distribution(
        n_sample: int,
        dimension_size: int,
        mixture_rate: float,
        distribution_conf_p: typing.Dict,
        distribution_conf_q: typing.Dict,
        index_replace: typing.Optional[typing.List[int]] = None,        
        generation_mode: str = 'noise',
        random_seed_x: typing.Optional[int] = None,
        random_seed_y: typing.Optional[int] = None,
        random_seed_noise: typing.Optional[int] = None        
        ) -> typing.Tuple[numpy.ndarray, numpy.ndarray, typing.List[int]]:
    """
    :math:`X = \{x_1,...,x_n\}, Y = \{y_1,...,y_n\}, x_n, y_m \in \mathbb{R}^{d}`.
    Args:
        n_sample: sample
        dimension_size: dimension size
        mixture_rate:
        distribution_conf_p: {'type', 'mu', 'sigma'}
        distribution_conf_q: {'type', 'mu', 'sigma'}
        index_replace:
        random_seed_x: seed value for sampling X datasets.
        random_seed_y: seed value for sampling Y datasets.
        random_seed_noise: seed value for sampling noise datasets.
    Returns: (ndarray of sample X, ndarray of sample Y, list of index where the distribution Q is mixed)
    """
    assert generation_mode in ['noise', 'noise_redundant']

    if generation_mode == 'noise':
        return __generate_noised_sampling(
            n_sample=n_sample,
            dimension_size=dimension_size,
            mixture_rate=mixture_rate,
            distribution_conf_p=distribution_conf_p,
            distribution_conf_q=distribution_conf_q,
            index_replace=index_replace,
            random_seed_x=random_seed_x,
            random_seed_y=random_seed_y,
            random_seed_noise=random_seed_noise)
    elif generation_mode == 'noise_redundant':
        return __generate_noised_sampling_redundant(
            n_sample=n_sample,
            dimension_size=dimension_size,
            mixture_rate=mixture_rate,
            distribution_conf_p=distribution_conf_p,
            distribution_conf_q=distribution_conf_q,
            index_replace=index_replace,
            random_seed_x=random_seed_x,
            random_seed_y=random_seed_y,
            random_seed_noise=random_seed_noise)
    else:
        raise NotImplementedError()