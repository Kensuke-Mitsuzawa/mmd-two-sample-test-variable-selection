from pathlib import Path
import toml
import functools
import logging

import numpy as np
import torch
import pytorch_lightning as pl

from mmd_tst_variable_detector.utils.permutation_test_runner import PermutationTest
from mmd_tst_variable_detector.datasets import SimpleDataset
from ot import sliced_wasserstein_distance


torch.cuda.is_available = lambda : False


logger = logging.getLogger(f'test.{__name__}')


def test_permutation_test_runner(resource_path_root: Path):
    x = torch.from_numpy(np.random.normal(loc=0, scale=1, size=(500, 20)))
    y = torch.from_numpy(np.random.normal(loc=5, scale=1, size=(500, 20)))
    dataset = SimpleDataset(x, y)
    
    test_runner_full_batch = PermutationTest(batch_size=-1)
    p_value, stats = test_runner_full_batch.run_test(dataset=dataset)
    
    logger.debug(f'p_value: {p_value}')

    test_runner_full_batch = PermutationTest(batch_size=100)
    p_value_batch, stats_batch = test_runner_full_batch.run_test(dataset=dataset)
    
    logger.debug(f'p_value: {p_value_batch}')
    # print(p_value, p_value_batch)
    
    feature_weights = torch.tensor(np.random.normal(loc=0, scale=1, size=(20,)))
    test_runner_full_batch_with_weights = PermutationTest(batch_size=-1)
    p_value_batch_weight, stats_batch_weight = test_runner_full_batch_with_weights.run_test(dataset=dataset, featre_weights=feature_weights)
    logger.debug(f'p_value: {p_value_batch_weight}')
    

if __name__ == '__main__':
    p = Path('../testresources')
    test_permutation_test_runner(p)
