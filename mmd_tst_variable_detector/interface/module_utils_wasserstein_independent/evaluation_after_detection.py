import typing as ty
import logging
import functools

import ot
import geomloss
import numpy as np
import torch
import torch.utils.data

from ...datasets import BaseDataset
from ...datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ...utils import PermutationTest
from ...logger_unit import handler

from ...detection_algorithm.utils.permutation_tests import (
    distance_sinkhorn_custom, 
    distance_sliced_wasserstein_custom,
    tune_sliced_wasserstein)


logger = logging.getLogger(__name__)
logger.addHandler(handler)


def permutation_tests(
    dataset_test: BaseDataset,
    seq_variables: ty.List[int],
    distance_functions: ty.Tuple[str,...] = ('sliced_wasserstein',),
    ) -> ty.List[float]:
    """Public function. Running the Permutation-Test. Various Permutation-Test.
    """
    if isinstance(dataset_test, FileBackendOneTimeLoadStaticDataset):
        dataset_test = dataset_test.generate_dataset_on_ram()
    # end if
    
    # -----------------------------------------------------------------------------------------
    __seq_permutation_test = []
    
    __dataset: ty.Optional[BaseDataset] = None
    
    for __d_function in distance_functions:
        # do variable selection first. applying dataset to the se
        if __d_function == 'sliced_wasserstein':
            __dataset = dataset_test.get_selected_variables_dataset(seq_variables)
            logger.debug('I am executing a parameter tuning of sliced-wasserstein...')
            __n_projection_tune = tune_sliced_wasserstein(dataset=dataset_test)
            logger.debug(f'Done. Selected n-projection -> {__n_projection_tune}')                
            __d_module = functools.partial(distance_sliced_wasserstein_custom, n_projections=__n_projection_tune)
        elif __d_function == 'sinkhorn':
            __dataset = dataset_test.get_selected_variables_dataset(seq_variables)
            __geomloss_module = geomloss.SamplesLoss(loss='sinkhorn', blur=0.01)
            __d_module = functools.partial(distance_sinkhorn_custom, geomloss_module=__geomloss_module)
        else:
            raise NotImplementedError()
        # end if
        
        assert __dataset is not None
        
        # check if the given dataset is totally same or not.
        __t_xy = [__dataset.__getitem__(__i) for __i in range(len(__dataset))]
        __avg_diff = np.sum([torch.count_nonzero(t[0] - t[1]) for t in __t_xy])
        
        if __avg_diff == 0.0:
            logger.debug('X and Y are totally same. Check your data if you feel suspicious.')
            __p_value = 99.99
        else:
            __test_runner = PermutationTest(
                func_distance=__d_module,
                n_permutation_test=1000)
            __p_value, __stats_permutation_test = __test_runner.run_test(__dataset)
        # end if
        __seq_permutation_test.append(__p_value)
    # end for

    return __seq_permutation_test