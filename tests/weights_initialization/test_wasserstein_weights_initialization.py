from mmd_tst_variable_detector.weights_initialization.wasserstein_weights_initialization import main

import numpy as np

from distributed import Client, LocalCluster

from mmd_tst_variable_detector.datasets import SimpleDataset

from tests.data_generator import test_data_xy_linear


def test_main():
    array_xy, ground_truth = test_data_xy_linear()
    
    dataset = SimpleDataset(array_xy[0], array_xy[1])
    
    weights_normal = main(dataset, dask_client=None, distributed_backend='joblib')
    assert isinstance(weights_normal, np.ndarray)

    index_max = np.argmax(weights_normal)
    assert index_max in ground_truth, f'index_max={index_max} not in ground_truth={ground_truth}'

    cluster = LocalCluster()
    client = cluster.get_client()
    weights_normal_dask = main(dataset, dask_client=client, distributed_backend='dask')
    
    index_max_dask = np.argmax(weights_normal_dask)
    assert index_max_dask in ground_truth, f'index_max={index_max_dask} not in ground_truth={ground_truth}'
