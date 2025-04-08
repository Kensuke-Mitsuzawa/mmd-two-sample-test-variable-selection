import typing as ty
import functools
import logging
import dataclasses
from collections import OrderedDict

import ot
import geomloss

from distributed import Client

import numpy as np
import torch
import torch.utils.data

from .. import InterpretableMmdTrainResult
from ..commons import ArgumentParameters
from ...kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from ...mmd_estimator import QuadraticMmdEstimator
from ...distance_module import L2Distance
from ...datasets import BaseDataset
from ...utils import (
    detect_variables,
    PermutationTest
)
from ... import logger_unit



logger = logging.getLogger(f'{__package__}.{__name__}')

POSSIBLE_DISTANCE_FUNCTIONS = ('mmd_ard', 'sliced_wasserstein', 'sinkhorn')


@dataclasses.dataclass
class _TwoSampleTestResultContainer:
    variable_selection_approach: str
    algorithm: str
    p_value: float
    binary_flag: str
    selected_variables: ty.List[int]
    

def distance_sinkhorn_custom(x: np.ndarray, y: np.ndarray, geomloss_module) -> float:
    def __zero_or_distance(target_index: int) -> torch.Tensor:
        if torch.count_nonzero(torch_x[:, :, target_index] - torch_y[:, :, target_index]) > 0:
            __distance_target = geomloss_module(torch_x[:, :, target_index], torch_y[:, :, target_index])
        else:
            __distance_target = torch.tensor([0])
        # end if
        return __distance_target
    # end if
    
    n_dim_data = len(x.shape)
    t_shape_dim = x.shape
    
    torch_x = torch.from_numpy(x)
    torch_y = torch.from_numpy(y)
    
    if n_dim_data == 2:
        d_value = geomloss_module(torch_x, torch_y)
    elif n_dim_data == 3:
        if t_shape_dim[-1] == 2:
            __distance_one = __zero_or_distance(0)
            __distance_two = __zero_or_distance(1)
            d_value = __distance_one + __distance_two
        elif t_shape_dim[-1] == 3:
            __distance_one = __zero_or_distance(0)
            __distance_two = __zero_or_distance(1)
            __distance_three = __zero_or_distance(2)
            d_value = __distance_one + __distance_two + __distance_three
        else:
            raise ValueError(f'Undefined case. Impossible to compute for the shape: {t_shape_dim}')
    else:
        raise ValueError()
    # end if
    return d_value.numpy()



def distance_sliced_wasserstein_custom(x: np.ndarray, y: np.ndarray, n_projections: int = 50) -> float:
    n_dim_data = len(x.shape)
    t_shape_dim = x.shape
    
    if n_dim_data == 2:
        d_value = ot.sliced_wasserstein_distance(x, y, n_projections=n_projections)
    elif n_dim_data == 3:
        if t_shape_dim[-1] == 2:
            __distance_one = ot.sliced_wasserstein_distance(x[:, :, 0], y[:, :, 0], n_projections=n_projections)
            __distance_two = ot.sliced_wasserstein_distance(x[:, :, 1], y[:, :, 1], n_projections=n_projections)
            d_value = __distance_one + __distance_two
        elif t_shape_dim[-1] == 3:
            __distance_one = ot.sliced_wasserstein_distance(x[:, :, 0], y[:, :, 0], n_projections=n_projections)
            __distance_two = ot.sliced_wasserstein_distance(x[:, :, 1], y[:, :, 1], n_projections=n_projections)
            __distance_three = ot.sliced_wasserstein_distance(x[:, :, 2], y[:, :, 2], n_projections=n_projections)
            d_value = __distance_one + __distance_two + __distance_three
        else:
            raise ValueError(f'Undefined case. Impossible to compute for the shape: {t_shape_dim}')
    else:
        raise ValueError()
    # end if
    return d_value

    
def tune_sliced_wasserstein(dataset: BaseDataset,
                              n_projection_min: ty.Optional[int] = None,
                              n_projection_max: ty.Optional[int] = None,
                              ratio_projection_max: float = 0.8,
                              iteration: int = 10) -> int:
    """Private function.
    Sliced Wasserstein requires "n_projection" as a hyperparameter.
    This function seeks the most stable "n_projection".

    Parameters
    -----------
    dataset (TimeSliceHdf5Dataset): _description_
    n_projection_min (int, optional): _description_. Defaults to half of the dim.
    n_projection_max (ty.Optional[int], optional): _description_. Defaults to None.
    ratio_projection_max (float, optional): _description_. Defaults to 0.5.
    iteration (int, optional): _description_. Defaults to 10.

    Returns
    -----------
    int: `n_projection` parameter where the variance of `iteration` is the smallest.
    """
    if n_projection_min is None:
        dim_x = dataset.get_dimension_flattened()
        n_projection_min = int(dim_x / 2)
    # end if
    
    if n_projection_max is None:
        dim_x = dataset.get_dimension_flattened()
        if dim_x > n_projection_min:
            n_projection_max = int(ratio_projection_max * dim_x)
            if n_projection_max < 50:
                n_projection_max = 50
            # end if
        else:
            n_projection_max = 100
        # end if
    # end if
    
    # comment: better to do with more intelligent way/////
    # Bayesian Based search. Objective value is variance. Solve the min problem.
    dict_n_projection2values: ty.Dict[int, ty.List[float]] = {}

    with torch.no_grad():
        __loader = torch.utils.data.DataLoader(dataset)
        __seq_pair = [t for t in __loader]
        __x = np.concatenate([t[0].numpy() for t in __seq_pair])
        __y = np.concatenate([t[1].numpy() for t in __seq_pair])
    # end with
    
    if (n_projection_max - n_projection_min) < 10:
        range_value = 10
    else:
        range_value = int((n_projection_max - n_projection_min) / 10)
    # end if
    
    for __nprojection in range(n_projection_min, n_projection_max, range_value):
        __value_stack = []
        logger.debug(f'Selecting the SlicedWasserstein parameter with {__nprojection}...')
        for __i_iteration in range(iteration):
            __distance = distance_sliced_wasserstein_custom(__x, __y, n_projections=__nprojection)
            __value_stack.append(__distance)
        # end for
        dict_n_projection2values[__nprojection] = __value_stack
    # end for

    # select a n_projection where the variance is the smallest.
    __seq_variance = [(k, np.var(seq_v)) for k, seq_v in dict_n_projection2values.items()]
    __t_min_variance = sorted(__seq_variance, key=lambda t: t[1])[0]

    return __t_min_variance[0]


class ObtainedMmdEstimatorSet(ty.NamedTuple):
    is_mmd_hard_possible: bool
    seq_variables: ty.List[int]
    mmd_estimator_parameters: ty.Optional[ty.Union[ty.Dict, OrderedDict, QuadraticMmdEstimator]]
    mmd_estimator_arguments: ty.Optional[ArgumentParameters]


def __get_mmd_estimator(interpretable_mmd_result,
                        variable_selection_approach: str) -> ObtainedMmdEstimatorSet:
    is_mmd_hard_possible: bool = False
    
    # extract ard weights and selected variables.
    if isinstance(interpretable_mmd_result, InterpretableMmdTrainResult):
        # output from baseline
        # trained_parameters = interpretable_mmd_trainer_done.get_trained_variables()
        seq_variables = detect_variables(interpretable_mmd_result.ard_weights_kernel_k)
        # ard_weights = interpretable_mmd_result.ard_weights_kernel_k.numpy()
        mmd_estimator_parameters = interpretable_mmd_result.mmd_estimator
        mmd_estimator_arguments = interpretable_mmd_result.mmd_estimator_hyperparameters
        is_mmd_hard_possible = False
    elif interpretable_mmd_result.__class__.__name__ == "AlgorithmOneResult":
        # output from model_selection
        # I save only the best model. Do I need all model results...? I do not think so....
        assert interpretable_mmd_result.selected_model is not None
        selected_estimator = interpretable_mmd_result.selected_model
        assert selected_estimator.interpretable_mmd_train_result is not None
        seq_variables = selected_estimator.selected_variables
        # ard_weights = selected_estimator.trained_ard_weights
        mmd_estimator_parameters = selected_estimator.interpretable_mmd_train_result.mmd_estimator
        mmd_estimator_arguments = selected_estimator.interpretable_mmd_train_result.mmd_estimator_hyperparameters
        is_mmd_hard_possible = False
    elif interpretable_mmd_result.__class__.__name__ == "CrossValidationTrainedParameter":
        seq_variables = interpretable_mmd_result.stable_s_hat
        ard_weights = interpretable_mmd_result.array_s_hat
        # assert interpretable_mmd_result.variable_detection_postprocess_soft is not None
        if variable_selection_approach == 'soft':
            if interpretable_mmd_result.variable_detection_postprocess_soft is None:
                mmd_estimator_parameters = None
                mmd_estimator_arguments = None
            else:
                mmd_estimator_parameters = interpretable_mmd_result.variable_detection_postprocess_soft.mmd_estimator
                mmd_estimator_arguments = interpretable_mmd_result.variable_detection_postprocess_soft.mmd_estimator_hyperparameters
            # end if
        elif variable_selection_approach == 'hard':
            if interpretable_mmd_result.variable_detection_postprocess_hard is None:
                # comment: there is a case that `variable_detection_postprocess_hard` is not available.
                # many reasons such as `variable_detection_postprocess_hard` failed to to be optimized.
                is_mmd_hard_possible = False
                mmd_estimator_parameters = None
                mmd_estimator_arguments = None
            else:
                is_mmd_hard_possible = True
                mmd_estimator_parameters = interpretable_mmd_result.variable_detection_postprocess_hard.mmd_estimator
                mmd_estimator_arguments = interpretable_mmd_result.variable_detection_postprocess_hard.mmd_estimator_hyperparameters                                
            # end if
        else:
            raise ValueError()
    elif interpretable_mmd_result.__class__.__name__ == "BaselineMmdResult":
        _interpretable_mmd_train_result = interpretable_mmd_result.interpretable_mmd_train_result
        assert isinstance(_interpretable_mmd_train_result, InterpretableMmdTrainResult)
        # output from baseline
        # trained_parameters = interpretable_mmd_trainer_done.get_trained_variables()
        seq_variables = detect_variables(_interpretable_mmd_train_result.ard_weights_kernel_k)
        # ard_weights = interpretable_mmd_result.ard_weights_kernel_k.numpy()
        mmd_estimator_parameters = _interpretable_mmd_train_result.mmd_estimator
        mmd_estimator_arguments = _interpretable_mmd_train_result.mmd_estimator_hyperparameters
        is_mmd_hard_possible = False
    else:
        raise NotImplementedError(f'The given type is not implemented yet: {type(interpretable_mmd_result)}')
    # end if

    return ObtainedMmdEstimatorSet(is_mmd_hard_possible, seq_variables, mmd_estimator_parameters, mmd_estimator_arguments)
    


def __restore_mmd_estimator(mmd_estimator_set: ObtainedMmdEstimatorSet):
    assert mmd_estimator_set.mmd_estimator_parameters is not None
    if isinstance(mmd_estimator_set.mmd_estimator_parameters, QuadraticMmdEstimator):
        __mmd_estimator = mmd_estimator_set.mmd_estimator_parameters
    else:
        # loading parameters from the dictionary.
        assert mmd_estimator_set.mmd_estimator_arguments is not None
        assert mmd_estimator_set.mmd_estimator_arguments.kernel_object_arguments is not None
        __args_kernel = mmd_estimator_set.mmd_estimator_arguments.kernel_object_arguments
        __args_mmd_estimator = mmd_estimator_set.mmd_estimator_arguments.mmd_object_arguments
        
        if mmd_estimator_set.mmd_estimator_arguments.distance_class_name == 'L2Distance':
            d_module = L2Distance(**mmd_estimator_set.mmd_estimator_arguments.distance_object_arguments)
        else:
            raise NotImplementedError(f'{mmd_estimator_set.mmd_estimator_arguments.distance_class_name} is not implemented yet.')
        # end if
        
        __kernel = QuadraticKernelGaussianKernel(**__args_kernel, distance_module=d_module)
        __mmd_estimator = QuadraticMmdEstimator(__kernel, **__args_mmd_estimator)                
        __mmd_estimator.load_state_dict(mmd_estimator_set.mmd_estimator_parameters)
    # end if
    return __mmd_estimator
    
    
def __func_mmd_ard_estimator(x: np.ndarray, y: np.ndarray, mmd_estimator: QuadraticMmdEstimator) -> float:
    torch_x = torch.from_numpy(x)
    torch_y = torch.from_numpy(y)
    
    mmd_obj = mmd_estimator.forward(torch_x, torch_y)
    return mmd_obj.mmd.item()



def permutation_tests(
    dataset_test: BaseDataset,
    variable_selection_approach: str,
    interpretable_mmd_result: ty.Optional["InterpretableMmdTrainResult"],
    n_permutation_test: int = 500,
    distance_functions: ty.Tuple[str,...] = ('mmd_ard', 'sliced_wasserstein',),
    dask_client: ty.Optional[Client] = None,
    is_tune_sliced_wasserstein: bool = False
    ) -> ty.List[_TwoSampleTestResultContainer]:
    """Private function. Running the Permutation-Test. Various Permutation-Test.

    Possible `distance_functions` options are either    
        - mmd-ard: Use ARD weights as a distance function.
        - mmd-variable: Use selected variable as a distance function.

    
    Parameters
    -------------
    variable_selection_approach: str. 'soft' or 'hard'.
        'hard' does select variables by discritizing. 'soft' does a multiplication operation of weight * data. 
    """
    assert all(__f_name in POSSIBLE_DISTANCE_FUNCTIONS for __f_name in distance_functions),\
        f'Undefined distance function: {distance_functions}'
    
    # -----------------------------------------------------------------------------------------
    # Comment: I do not like closure. But for convenience, I define closure here.
    
    
    # -----------------------------------------------------------------------------------------
    __mmd_estimator_set = __get_mmd_estimator(interpretable_mmd_result, variable_selection_approach)
    
    __seq_permutation_test = []
    
    __featre_weights: ty.Optional[np.ndarray] = None
    
    for __d_function in distance_functions:
        if variable_selection_approach == 'soft':
            assert __mmd_estimator_set.mmd_estimator_parameters is not None
            __mmd_estimator = __restore_mmd_estimator(__mmd_estimator_set)
            __featre_weights = __mmd_estimator.kernel_obj.ard_weights.detach().numpy()
            __dataset = dataset_test
            
            if __d_function == 'mmd_ard':
                assert __mmd_estimator_set.mmd_estimator_arguments is not None
                __d_module = functools.partial(__func_mmd_ard_estimator, mmd_estimator=__mmd_estimator)
                # __featre_weights = __mmd_estimator.kernel_obj.ard_weights.numpy()
            elif __d_function == 'sliced_wasserstein':
                if is_tune_sliced_wasserstein:
                    __n_projection_tune = tune_sliced_wasserstein(dataset=dataset_test)
                else:
                    __n_projection_tune = int(__dataset.get_dimension_flattened() / 2)
                # end if
                __d_module = functools.partial(distance_sliced_wasserstein_custom, n_projections=__n_projection_tune)
            elif __d_function == 'sinkhorn':
                __d_sinkhorn_module = geomloss.SamplesLoss(loss='sinkhorn', blur=0.01)
                __d_module = functools.partial(distance_sinkhorn_custom, geomloss_module=__d_sinkhorn_module)
            else:
                raise NotImplementedError()
        elif variable_selection_approach == 'hard':
            # do variable selection first. applying dataset to the se
            if __d_function == 'mmd_ard':
                if __mmd_estimator_set.is_mmd_hard_possible is False:
                    logger.info(f'I can not execute mmd_ard in hard because {type(interpretable_mmd_result)} does not have it.')
                    continue
                else:
                    __mmd_estimator = __restore_mmd_estimator(__mmd_estimator_set)
                    __dataset = dataset_test
                    __d_module = functools.partial(__func_mmd_ard_estimator, mmd_estimator=__mmd_estimator)
            elif __d_function == 'sliced_wasserstein':
                __dataset = dataset_test.get_selected_variables_dataset(tuple(__mmd_estimator_set.seq_variables))
                if is_tune_sliced_wasserstein:
                    logger.debug('I am executing a parameter tuning of sliced-wasserstein...')
                    __n_projection_tune = tune_sliced_wasserstein(dataset=__dataset)
                    logger.debug(f'Done. Selected n-projection -> {__n_projection_tune}')
                else:
                    __n_projection_tune = len(__mmd_estimator_set.seq_variables)
                # end if
                __d_module = functools.partial(distance_sliced_wasserstein_custom, n_projections=__n_projection_tune)
            elif __d_function == 'sinkhorn':
                __dataset = dataset_test.get_selected_variables_dataset(tuple(__mmd_estimator_set.seq_variables))
                __d_sinkhorn_module = geomloss.SamplesLoss(loss='sinkhorn', blur=0.01)
                __d_module = functools.partial(distance_sinkhorn_custom, geomloss_module=__d_sinkhorn_module)
            else:
                raise NotImplementedError()
            # end if
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
                n_permutation_test=n_permutation_test,
                dask_client=dask_client)
            if variable_selection_approach == 'soft':
                assert __featre_weights is not None
                __featre_weights_tensor = torch.from_numpy(__featre_weights)
                __p_value, __stats_permutation_test = __test_runner.run_test(__dataset, featre_weights=__featre_weights_tensor)
            elif variable_selection_approach == 'hard':
                __p_value, __stats_permutation_test = __test_runner.run_test(__dataset)
            else:
                raise ValueError()
        # end if
        
        result_container = _TwoSampleTestResultContainer(
            variable_selection_approach=variable_selection_approach,
            algorithm=__d_function,
            p_value=__p_value,
            binary_flag='h0' if __p_value > 0.05 else 'h1',
            selected_variables=__mmd_estimator_set.seq_variables)
        __seq_permutation_test.append(result_container)
    # end for

    return __seq_permutation_test