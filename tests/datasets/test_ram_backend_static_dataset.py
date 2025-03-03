import typing

import numpy as np
import torch

from mmd_tst_variable_detector.datasets import SimpleDataset


def test_vector_input():
    x = np.random.normal(size=(100, 5))
    y = np.random.normal(size=(100, 5))

    dataset = SimpleDataset(x, y)
    x, y = dataset.__getitem__(0)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert dataset.__len__() == 100
    assert dataset.data_format_x == 'vector' and dataset.data_format_y == 'vector'

    x = np.random.normal(size=(100, 5))
    y = np.random.normal(size=(100, 5))

    dataset_b = SimpleDataset(x, y)
    dataset_c = dataset.merge_new_dataset(dataset_b)
    assert isinstance(dataset_c, SimpleDataset)

    __random_sample_id = dataset.get_random_data_id(1)
    assert isinstance(__random_sample_id, list)

    __id, sub_sample_dataset = dataset.get_subsample_dataset(10)
    assert isinstance(__id, list)
    assert isinstance(sub_sample_dataset, SimpleDataset)

    __id, sub_sample_dataset = dataset.get_bootstrap_dataset()
    assert isinstance(__id, tuple)
    assert isinstance(sub_sample_dataset, SimpleDataset)

    sub_selected_variable_dataset = dataset.get_selected_variables_dataset([1, 2])
    assert isinstance(sub_selected_variable_dataset, SimpleDataset)


def test_matrix_input():
    x = np.random.normal(size=(100, 5, 5))
    y = np.random.normal(size=(100, 5, 5))

    dataset = SimpleDataset(x, y)
    x, y = dataset.__getitem__(0)
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert len(x.shape) == 1 and len(y.shape) == 1

    dim_x = dataset.get_dimension_flattened()
    assert dim_x == 25

    assert len(x) == dim_x

    assert dataset.__len__() == 100
    assert dataset.data_format_x == 'matrix' and dataset.data_format_y == 'matrix'

    x = np.random.normal(size=(100, 5, 5))
    y = np.random.normal(size=(100, 5, 5))

    dataset_b = SimpleDataset(x, y)
    dataset_c = dataset.merge_new_dataset(dataset_b)
    assert isinstance(dataset_c, SimpleDataset)

    __random_sample_id = dataset.get_random_data_id(1)
    assert isinstance(__random_sample_id, list)

    __id, sub_sample_dataset = dataset.get_subsample_dataset(10)
    assert isinstance(__id, list)
    assert isinstance(sub_sample_dataset, SimpleDataset)

    __id, sub_sample_dataset = dataset.get_bootstrap_dataset()
    assert isinstance(__id, tuple)
    assert isinstance(sub_sample_dataset, SimpleDataset)

    sub_selected_variable_dataset = dataset.get_selected_variables_dataset([1, 2])
    assert isinstance(sub_selected_variable_dataset, SimpleDataset)

    assert len(sub_selected_variable_dataset) == 100
    x, y = sub_selected_variable_dataset.__getitem__(1)
    assert len(x) == 2 and len(y) == 2
