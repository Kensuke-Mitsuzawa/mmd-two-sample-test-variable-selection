import typing as t
import torch
import numpy as np
import random


def sample_gaussian(x, sample: int, random_seed: int = 1234) -> np.ndarray:
    random_gen = np.random.default_rng(random_seed)
    return random_gen.normal(loc=0, scale=1, size=(sample,))


def test_data_xy_linear(dim_size: int = 20,
                        sample_size: int = 1000,
                        ratio_dependent_variables: float = 0.1,
                        random_seed: int = 1234,
                        func_noise_function: t.Callable[[float, int], np.ndarray] = sample_gaussian
                        ) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.List[int]]:
    """
    :param dim_size:
    :param sample_size:
    :param ratio_dependent_variables:
    :return: (data samples, ground-truth)
    """
    random_gen = np.random.default_rng(random_seed)
    # dimension size
    sample_x = random_gen.normal(loc=10, scale=1, size=(sample_size, dim_size))
    sample_y = random_gen.normal(loc=10, scale=1, size=(sample_size, dim_size))

    # the number of dimensions to be replaced
    n_replaced = int(dim_size * ratio_dependent_variables)
    dim_ground_truth = random_gen.choice(a=range(0, dim_size), size=n_replaced, replace=False)

    # transformation equation
    for dim_replace in dim_ground_truth:
        y_value = func_noise_function(0.0, sample_size)
        sample_y[:, dim_replace] = y_value
    # end for

    x_tensor = torch.tensor(sample_x)
    y_tensor = torch.tensor(sample_y)
    return (x_tensor, y_tensor), dim_ground_truth



def test_data_discrete_category(dim_size: int = 20,
                                sample_size: int = 1000,
                                ratio_dependent_variables: float = 0.1,
                                num_max_category: int = 5
                                ) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.List[int]]:
    """
    :param dim_size:
    :param sample_size:
    :param ratio_dependent_variables:
    :return: (data samples, ground-truth)
    """
    # dimension size
    sample_x = np.zeros(shape=(sample_size, dim_size))
    sample_y = np.zeros(shape=(sample_size, dim_size))

    # discrete value sampling
    category_labels_population = range(0, num_max_category)
    weights_distribution = np.random.dirichlet(np.ones(num_max_category), size=1)[0]
    for __d in range(0, dim_size):
        sample_x[:, __d] = np.random.choice(category_labels_population, sample_size, p=weights_distribution)
        sample_y[:, __d] = np.random.choice(category_labels_population, sample_size, p=weights_distribution)
    # end for

    # the number of dimensions to be replaced
    n_replaced = int(dim_size * ratio_dependent_variables)
    dim_ground_truth = random.sample(range(0, dim_size), k=n_replaced)

    # transformation equation
    # noise is from poisson distribution
    for dim_replace in dim_ground_truth:
        __poisson_sample = np.random.poisson(num_max_category, sample_size)
        __poisson_sample[__poisson_sample > num_max_category] = 0
        sample_y[:, dim_replace] = __poisson_sample
    # end for

    x_tensor = torch.tensor(sample_x)
    y_tensor = torch.tensor(sample_y)
    return (x_tensor, y_tensor), dim_ground_truth

