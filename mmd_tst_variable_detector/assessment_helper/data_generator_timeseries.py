import numpy
import numpy as np
import typing
import copy
import torch
import random

def generate_random_gaussian_vector(length):
    # Generate random values from a Gaussian distribution
    random_values = np.random.normal(loc=0.001, scale=1.0, size=length)

    # Clip the values to ensure they are within the range [0, 1]
    random_values = np.clip(random_values, 0.001, 1)

    # Normalize the vector so that the sum is 1.0
    normalized_vector = random_values / np.sum(random_values)

    return normalized_vector


# end def


def generate_weight_matrix(spatio_space_size: int, time_space_size: int) -> np.ndarray:
    w_array = np.zeros(shape=(spatio_space_size, spatio_space_size, (time_space_size - 1)))

    for _time in range(0, time_space_size - 1):
        for _space_from in range(0, spatio_space_size):
            _weight_time_from_space = generate_random_gaussian_vector(spatio_space_size)
            w_array[_space_from, :, _time] = _weight_time_from_space
        # end for
    # end for
    return w_array
# end def


def generate_time_zero_vector_random_gaussian_vector(total_amount: int, spatio_space: int) -> np.ndarray:
    """
    The function generates a vector at the time=0.

    - Gaussian(10, 1) for [0, 10] space-index.
    - Gaussian (0, 1) for [11,] space-index.

    Args:
        total_amount (int): _description_
        spatio_space (int): _description_

    Returns:
        np.ndarray: _description_
    """
    # Generate random values from a Gaussian distribution
    # random_values = np.random.normal(loc=0.0, scale=1.0, size=spatio_space)
    initial_vector = np.zeros(shape=(spatio_space,))

    if spatio_space <= 5:
        initial_vector[0:2] = np.random.normal(loc=10.0, scale=1.0, size=2)

        _size_rest = spatio_space - 2
        initial_vector[2:] = np.random.normal(loc=0.0, scale=1.0, size=_size_rest)
    elif spatio_space >= 10:
        initial_vector[0:10] = np.random.normal(loc=10.0, scale=1.0, size=10)

        _size_rest = spatio_space - 10
        initial_vector[10:] = np.random.normal(loc=0.0, scale=1.0, size=_size_rest)
    else:
        raise NotImplementedError()
    # end if

    return initial_vector


# end def


def modify_transition_weight_matrix(
        transition_weight_matrix: np.ndarray,
        spatio_target_index: typing.List[int],
        spatio_zero_index: typing.List[int],
        noise_start: int,
        noise_end: int) -> np.ndarray:
    """
    Summary: move the transition weights at zero in spatio_zero_index into target in spatio_target_index.
    Note: spatio_zero_index and spatio_target_index are SET. The two have the same length.

    Pseudo code:
    updating W[s_from, target, t] = W[s_from, target: t] + W[s_from, zero, t]
    updating W[s_from, zero, t] = 0.0
    """

    transition_weight_q = copy.deepcopy(transition_weight_matrix)
    dict_correspond_zero2target = {_zero: spatio_target_index[_i] for _i, _zero in enumerate(spatio_zero_index)}

    assert noise_start < transition_weight_q.shape[2]
    assert noise_end < transition_weight_q.shape[2]

    for _time in range(noise_start, noise_end):
        for _zero, _target in dict_correspond_zero2target.items():
            transition_weight_q[:, _target, _time] = transition_weight_q[:, _target, _time] + transition_weight_q[:,
                                                                                              _zero, _time]
            transition_weight_q[:, _zero, _time] = 0.0
        # end for
    # end for

    # for s_from in range(0, transition_weight_q.shape[0]):
    #     for _time in range(transition_weight_q.shape[2]):
    #         _diff_expected = abs(1.0 - sum(transition_weight_q[s_from, :, _time]))
    #         assert 0.0 <= _diff_expected <= 0.1, f"Expected-SUM == 1.0, Actual-SUM {sum(transition_weight_q[s_from, :, _time])}"
    return transition_weight_q


def generate_sample_p(spatio_space: int, time_space: int, n_total_agent: int,
                      transition_weight_matrix: np.ndarray) -> np.ndarray:
    """Generate a matrix following a distribution of weight.

    Return: A pseudo traffic matrix in the space of $$S \times T$$.
    """
    # サンプルごとに異なる値になる。初期値。初期値はfloat。ガウス分布からサンプル。
    p_time_zero = generate_time_zero_vector_random_gaussian_vector(n_total_agent, spatio_space)

    # TODO 関数分離
    assert len(p_time_zero) == spatio_space

    n_agent_total = sum(p_time_zero)
    sample_matrix = np.zeros(shape=(spatio_space, time_space))
    # set the vector at time=0
    sample_matrix[:, 0] = p_time_zero

    for _time in range(1, time_space):
        for _s_to in range(0, spatio_space):
            _sum_from_to = 0.0
            for _s_from in range(0, spatio_space):
                # print(_time, _s_from, _s_to, sample_matrix[_s_from, (_time - 1)], transition_weight_matrix[_s_from, _s_to, (_time - 1)])
                _value_from_to = sample_matrix[_s_from, (_time - 1)] * transition_weight_matrix[
                    _s_from, _s_to, (_time - 1)]
                _sum_from_to += _value_from_to
            # end for
            sample_matrix[_s_to, _time] = copy.deepcopy(_sum_from_to)
        # end for
        assert round(sum(sample_matrix[:, _time])) == round(
            n_agent_total), f"{sum(sample_matrix[:, _time])} != {n_agent_total}"
    # end for

    return sample_matrix



def matrix_to_vector(z: torch.Tensor, dim_vector: int) -> torch.Tensor:
    """a helper function to convert the 3d tensor to the 2d tensor.

    :return:
    """
    sample_size = z.shape[0]
    z_converted = torch.reshape(z, (sample_size, dim_vector))
    return z_converted


def main_generate_sumo_pseudo_data(
        sample_size: int,
        spatio_space: int,
        time_space: int,
        noise_start: int,
        noise_end: int,
        n_noise_spatio: float,
        n_agent_total: int = 300) -> typing.Tuple[typing.Dict, typing.Dict]:
    set_spatio_space = list(range(spatio_space))
    assert (n_noise_spatio * 2) < len(set_spatio_space)
    spatio_target_index = random.sample(set_spatio_space, k=n_noise_spatio)  # spatio-space set with noise
    spatio_zero_index = random.sample(set(set_spatio_space) - set(spatio_target_index),
                                      k=n_noise_spatio)  # spatio-space set with 0.0

    assert len(set(spatio_target_index).intersection(set(spatio_zero_index))) == 0

    # A transition 3d-array (S, S, T). The 3d-array corresponds to the SUMO road-map definition.
    transition_weight_matrix_p = generate_weight_matrix(spatio_space, time_space)
    # A transition 3d-array (S, S, T). The 3d-array corresponds to the perturbed SUMO road-map definition.
    transition_weight_matrix_q = modify_transition_weight_matrix(
        transition_weight_matrix=transition_weight_matrix_p,
        spatio_target_index=spatio_target_index,
        spatio_zero_index=spatio_zero_index,
        noise_start=noise_start,
        noise_end=noise_end)

    train_x = np.zeros((sample_size, spatio_space, time_space))
    train_y = np.zeros((sample_size, spatio_space, time_space))

    dev_x = np.zeros((sample_size, spatio_space, time_space))
    dev_y = np.zeros((sample_size, spatio_space, time_space))

    test_x = np.zeros((sample_size, spatio_space, time_space))
    test_y = np.zeros((sample_size, spatio_space, time_space))

    # spatio_noise_index = random.sample(range(spatio_space), k=n_noise_spatio)

    for i in range(sample_size):
        # generating sample X from P.
        sample_x_train = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_p)
        sample_x_dev = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_p)
        sample_x_test = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_p)

        # generating samples from Q.
        sample_y_train = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_q)
        sample_y_dev = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_q)
        sample_y_test = generate_sample_p(
            spatio_space=spatio_space,
            time_space=time_space,
            n_total_agent=n_agent_total,
            transition_weight_matrix=transition_weight_matrix_q)

        train_x[i] = sample_x_train
        train_y[i] = sample_y_train

        dev_x[i] = sample_x_dev
        dev_y[i] = sample_y_dev

        test_x[i] = sample_x_test
        test_y[i] = sample_y_test
    # end for

    assert np.isnan(train_x).sum() == 0
    assert np.isnan(train_y).sum() == 0
    assert np.isnan(dev_x).sum() == 0
    assert np.isnan(dev_y).sum() == 0
    assert np.isnan(test_x).sum() == 0
    assert np.isnan(test_y).sum() == 0

    ground_truth_info = {
        "spatio_target_index": spatio_target_index,
        "spatio_zero_index": spatio_zero_index,
        "noise_start": noise_start,
        "noise_end": noise_end
    }

    data_matrix = {
        "train_x": train_x,
        "train_y": train_y,
        "dev_x": dev_x,
        "dev_y": dev_y,
        "test_x": test_x,
        "test_y": test_y
    }

    return data_matrix, ground_truth_info

