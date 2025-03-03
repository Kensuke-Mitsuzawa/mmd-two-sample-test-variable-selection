import typing as ty
import logging
import functools

import ot
import geomloss
import numpy as np
import torch
import torch.utils.data

from distributed import Client

from ..datasets import BaseDataset
from ..datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset 
from ..weights_initialization import weights_initialization
from ..utils import (
    detect_variables,
)

from ..logger_unit import handler

from .data_objects import BasicVariableSelectionResult
from .module_utils_wasserstein_independent import evaluation_after_detection

logger = logging.getLogger(__name__)
logger.addHandler(handler)



# def __tune_sliced_wasserstein(dataset: BaseDataset,
#                               n_projection_min: int = 50,
#                               n_projection_max: ty.Optional[int] = None,
#                               ratio_projection_max: float = 0.8,
#                               iteration: int = 10) -> int:
#     """Private function.
#     Sliced Wasserstein requires "n_projection" as a hyperparameter.
#     This function seeks the most stable "n_projection".

#     Parameters
#     -----------
#     dataset (TimeSliceHdf5Dataset): _description_
#     n_projection_min (int, optional): _description_. Defaults to 50.
#     n_projection_max (ty.Optional[int], optional): _description_. Defaults to None.
#     ratio_projection_max (float, optional): _description_. Defaults to 0.5.
#     iteration (int, optional): _description_. Defaults to 10.

#     Returns
#     -----------
#     int: `n_projection` parameter where the variance of `iteration` is the smallest.
#     """
#     if n_projection_max is None:
#         dim_x, __ = dataset.get_dimension_flattened()
#         if dim_x > n_projection_min:
#             n_projection_max = int(ratio_projection_max * dim_x)
#         else:
#             n_projection_max = 100
#         # end if
#     # end if
    
#     # comment: better to do with more intelligent way/////
#     # Bayesian Based search. Objective value is variance. Solve the min problem.
#     dict_n_projection2values: ty.Dict[int, ty.List[float]] = {}

#     __loader = torch.utils.data.DataLoader(dataset)
#     __seq_pair = [t for t in __loader]
#     __x = torch.concatenate([t[0] for t in __seq_pair])
#     __y = torch.concatenate([t[1] for t in __seq_pair])
    
#     for __nprojection in range(n_projection_min, n_projection_max, 10):
#         __value_stack = []
#         logger.debug(f'Selecting the SlicedWasserstein parameter with {__nprojection}...')
#         for __i_iteration in range(iteration):
#             __distance = ot.sliced_wasserstein_distance(__x.numpy(), __y.numpy(), n_projections=__nprojection)
#             __value_stack.append(__distance)
#         # end for
#         dict_n_projection2values[__nprojection] = __value_stack
#     # end for

#     # select a n_projection where the variance is the smallest.
#     __seq_variance = [(k, np.var(seq_v)) for k, seq_v in dict_n_projection2values.items()]
#     __t_min_variance = sorted(__seq_variance, key=lambda t: t[1])[0]

#     return __t_min_variance[0]



# def __wrapper_pot_sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_projections: int) -> np.ndarray:
#     return ot.sliced_wasserstein_distance(x.numpy(), y.numpy(), n_projections=n_projections)


# def __permutation_tests(
#     dataset_test: BaseDataset,
#     seq_variables: ty.List[int],
#     distance_functions: ty.Tuple[str,...] = ('sliced_wasserstein',),
#     ) -> ty.List[float]:
#     """Private function. Running the Permutation-Test. Various Permutation-Test.
#     """
#     # -----------------------------------------------------------------------------------------
#     __seq_permutation_test = []
    
#     __dataset: ty.Optional[BaseDataset] = None
    
#     for __d_function in distance_functions:
#         # do variable selection first. applying dataset to the se
#         if __d_function == 'sliced_wasserstein':
#             __dataset = dataset_test.get_selected_variables_dataset(seq_variables)
#             logger.debug('I am executing a parameter tuning of sliced-wasserstein...')
#             __n_projection_tune = __tune_sliced_wasserstein(dataset=dataset_test)
#             logger.debug(f'Done. Selected n-projection -> {__n_projection_tune}')                
#             __d_module = functools.partial(__wrapper_pot_sliced_wasserstein, n_projections=__n_projection_tune)
#         elif __d_function == 'sinkhorn':
#             __dataset = dataset_test.get_selected_variables_dataset(seq_variables)
#             __d_module = geomloss.SamplesLoss(loss='sinkhorn', blur=0.01)
#         else:
#             raise NotImplementedError()
#         # end if
        
#         assert __dataset is not None
        
#         # check if the given dataset is totally same or not.
#         __t_xy = [__dataset.__getitem__(__i) for __i in range(len(__dataset))]
#         __avg_diff = np.sum([torch.count_nonzero(t[0] - t[1]) for t in __t_xy])
        
#         if __avg_diff == 0.0:
#             logger.debug('X and Y are totally same. Check your data if you feel suspicious.')
#             __p_value = 99.99
#         else:
#             __test_runner = PermutationTest(
#                 func_distance=__d_module,
#                 n_permutation_test=1000)
#             __p_value, __stats_permutation_test = __test_runner.run_test(__dataset)
#         # end if
#         __seq_permutation_test.append(__p_value)
#     # end for

#     return __seq_permutation_test



def main(dataset_obj_train: BaseDataset, 
         dataset_obj_test: ty.Optional[BaseDataset], 
         resource_config: "ResourceConfigArgs",  # I can not import due to circular import.
         dask_client: ty.Optional[Client] = None) -> BasicVariableSelectionResult:    
    initial_weights = weights_initialization(
        dataset_obj_train, 
        approach_name='wasserstein',
        distributed_backend=resource_config.dask_config_detection.distributed_mode,
        dask_client=dask_client)
    variables = detect_variables(initial_weights)

    if dataset_obj_test is None:
        p_value_max = np.nan
        n_sample_test = -1
    else:
        seq_p_values = evaluation_after_detection.permutation_tests(
            dataset_test=dataset_obj_test, 
            seq_variables=variables)
        # comment: if max(p_value) is too small, then we can say that the selected variables reject plausiblly H0.
        p_value_max = np.max(seq_p_values)
        n_sample_test = len(dataset_obj_test)
    # end if
    
    n_sample_training = len(dataset_obj_train)

    return BasicVariableSelectionResult(
        weights=initial_weights,
        variables=variables,
        p_value=p_value_max,
        n_sample_training=n_sample_training,
        n_sample_test=n_sample_test)
