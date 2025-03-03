import typing as ty
import copy
import numpy as np
import logging
from frozendict import frozendict
from dataclasses import asdict
from copy import deepcopy

import pytorch_lightning as pl

from ...mmd_estimator import BaseMmdEstimator
from ...datasets.base import BaseDataset
from ..interpretable_mmd_detector import (
    InterpretableMmdDetector,
    RegularizationParameter,
    InterpretableMmdTrainParameters,
    InterpretableMmdTrainResult
)
from ...utils.post_process_logger import PostProcessLoggerHandler
from ..pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ...utils.variable_detection import detect_variables
from .optuna_module.commons import SelectionResult
from ...logger_unit import handler 
logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


def log_post_process(post_process_handler: PostProcessLoggerHandler, 
                     results_one_batch: InterpretableMmdTrainResult,
                     n_iteration: int,
                     search_mode: str = 'tpami-heuristic-search',
                     cv_detection_experiment_name: str = 'heuristic-search') -> None:
    """Private API. Logging post-process results."""
    
    __run_name = f'{search_mode}-trial-{n_iteration}'
    __loggers = post_process_handler.initialize_logger(run_name=__run_name, 
                                                        group_name=cv_detection_experiment_name)
    post_process_handler.log(loggers=__loggers, target_object=results_one_batch)
    # end for


def __is_end_condition(
    records_selected_variables_previous: ty.Sequence[ty.Sequence[int]],
    selected_variables_current: ty.Sequence[int],
    patience: int = 2,
    rules: ty.Tuple[str, ...] = ("one",),
) -> ty.Tuple[bool, str]:
    """End condition, one of these three cases below,
    1. (same_number) selected_variables_{previous} and selected_variables_{current} are same, where selected_variables_{previous} is a one record before the patience period
    2. (one) |selected_variables| == 1
    3. (less_number) |selected_variables_{previous}| < |selected_variables_{current}|, this rule is optional by default.

    :arg
        :records_selected_variables_previous: The selected variables in the previous iteration.
        :selected_variables_current: The selected variables in the current iteration.
        :patience: The number of iterations to wait before stopping the search. Starting with 0. 0 is no-patience, immediately check.
    """
    if len(records_selected_variables_previous) == 0:
        return False, ""

    # first rule
    if "same_number" in rules:
        if (
            len(records_selected_variables_previous) > patience
            and len(selected_variables_current) > 0
        ):
            # checking the previous records
            previous_record_before_patience = records_selected_variables_previous[
                -(patience + 1)
            ]
            if set(previous_record_before_patience) == set(selected_variables_current):
                return True, "same_number"
    # second rule
    if "one" in rules:
        if len(selected_variables_current) == 1:
            return True, "one"

    if "less_number" in rules:
        # third rule
        if len(records_selected_variables_previous[-1]) < len(
            selected_variables_current
        ):
            return True, "less_number"

    return False, ""


def __next_regularization(
    ratio_regularization_up: ty.Union[str, float], regularization_parameter: float
) -> float:
    if ratio_regularization_up == "auto":
        if 0 < regularization_parameter < 0.1:
            regularization_next = regularization_parameter * 5.0
        elif 0.1 <= regularization_parameter < 1.0:
            regularization_next = regularization_parameter * 2.0
        elif 1.0 <= regularization_parameter:
            regularization_next = regularization_parameter + 0.5
        elif regularization_parameter == 0.0:
            regularization_next = 0.0
        else:
            raise NotImplementedError()
    else:
        regularization_next = regularization_parameter * ratio_regularization_up
    # end if
    return regularization_next


def __procedure_max_try(seq_stacks: ty.List[ty.Dict],
                        max_try_iteration: int,
                        initial_regularization_search: RegularizationParameter,
                        n_regularization_parameter: int,
                        strategy_max_try_reach: str = "smallest") -> SelectionResult:
    """Procedure for max try reached."""
    if strategy_max_try_reach == "exception":
        raise RuntimeError(f"Max try reached: {max_try_iteration}")
    # end if

    # selection criteria: smallest number of selected variables & largest regularization parameter
    func_select_smallest_variable = lambda d: len(d["selected_variables"])

    selected_parameter = sorted(seq_stacks, key=func_select_smallest_variable)
    n_variables = len(selected_parameter[0]['selected_variables'])
    func_filter = lambda d: len(d["selected_variables"]) == n_variables
    seq_filtered_parameter = filter(func_filter, selected_parameter)

    func_sort_regularization_parameter = lambda d: d["regularization_parameter"].lambda_1 \
        if d["regularization_parameter"].lambda_2 == 0.0 else d["regularization_parameter"].lambda_2

    selected_parameter_reg: ty.Dict = sorted(seq_filtered_parameter, key=func_sort_regularization_parameter)[0]

    # parameter generation
    result_reg_params = __generate_hyper_parameters(
        regularization_parameter_last=selected_parameter_reg['regularization_parameter'],
        n_regularization_parameter=n_regularization_parameter,
        initial_regularization_search=initial_regularization_search,
    )

    ard_weights = [(d["regularization_parameter"], d["ard_weights"]) for d in seq_stacks]
    selected_variables = [
        (d["regularization_parameter"], d["selected_variables"]) for d in seq_stacks
    ]
    res_obj = SelectionResult(
        regularization_parameters=result_reg_params,
        selected_variables=selected_variables,
        dict_regularization2model_parameter=frozendict(
            {
                result_obj['regularization_parameter']: result_obj['mmd_estimator'] 
                for result_obj in seq_stacks
            }),
    )
    logger.info(f"Max try reached: {max_try_iteration}. Select a parameter yielding the smallest variables. {selected_parameter_reg['regularization_parameter']}")
    return res_obj


def __generate_hyper_parameters(regularization_parameter_last: RegularizationParameter,
                                n_regularization_parameter: int,
                                initial_regularization_search: RegularizationParameter
                                ) -> ty.List[RegularizationParameter]:
    
    if regularization_parameter_last.lambda_1 != 0.0:
        reg_step_l1 = regularization_parameter_last.lambda_1 / (
            n_regularization_parameter + 1
        )
        if regularization_parameter_last == initial_regularization_search:
            # setting the starting point 0.0
            regularization_parameters_l1 = np.arange(
                start=(0.0 + reg_step_l1), 
                stop=(regularization_parameter_last.lambda_1 + reg_step_l1), 
                step=reg_step_l1,
            )
        else:
            regularization_parameters_l1 = np.arange(
                start=initial_regularization_search.lambda_1, 
                stop=(regularization_parameter_last.lambda_1 + reg_step_l1),
                step=reg_step_l1,
            )
        # end if
    else:
        regularization_parameters_l1 = None
    # end if

    if regularization_parameter_last.lambda_2 != 0.0:
        reg_step_l2 = regularization_parameter_last.lambda_2 / (
            n_regularization_parameter + 1
        )
        if regularization_parameter_last == initial_regularization_search:
            regularization_parameters_l2 = np.arange(
                start=(0.0 + reg_step_l2), 
                stop=(regularization_parameter_last.lambda_2 + reg_step_l2), 
                step=reg_step_l2,
            )
        else:
            regularization_parameters_l2 = np.arange(
                start=initial_regularization_search.lambda_2,
                stop=(regularization_parameter_last.lambda_2 + reg_step_l2),
                step=reg_step_l2,
            )
        # end if
    else:
        regularization_parameters_l2 = None
    # end if
    
    if regularization_parameters_l1 is None:
        regularization_parameters_l1 = np.zeros(len(regularization_parameters_l2))
    if regularization_parameters_l2 is None:
        regularization_parameters_l2 = np.zeros(len(regularization_parameters_l1))

    assert len(regularization_parameters_l1) == len(regularization_parameters_l2)
    result_reg_params = [
        RegularizationParameter(*t)
        for t in list(
            zip(
                regularization_parameters_l1.tolist(),
                regularization_parameters_l2.tolist(),
            )
        )
    ]

    return result_reg_params


def run_parameter_space_search(
        dataset: BaseDataset,
        mmd_estimator: BaseMmdEstimator,
        training_parameter: InterpretableMmdTrainParameters,
        pytorch_trainer_config: PytorchLightningDefaultArguments,
        n_regularization_parameter: int = 4,
        ratio_regularization_up: ty.Union[str, float] = "auto",
        initial_regularization_search: RegularizationParameter = RegularizationParameter(0.01, 0.0),
        max_try: int = 20,
        patience: int = 3,
        rules: ty.Tuple[str, ...] = ("one", "same_number"),
        strategy_max_try_reach: str = "smallest",
        variable_trainer: ty.Optional[InterpretableMmdDetector] = None,
        variable_detection_approach: str = "hist_based",
        post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,) -> SelectionResult:
    """A util-function to get a set of regularization parameters for ARD weights optimization.
    This function obtains the upper bound of the parameter where detected-variables does not change.

    :param dataset:
    :param mmd_estimator:
    :param training_parameter:
    :param pytorch_trainer:
    :param n_regularization_parameter: The number of regularization parameters. The obtained parameters "(n_regularization_parameter + 2)"
    :param ratio_regularization_up: The ratio of the regularization parameter for the next search. 'auto' selects in heuristics.
    :param initial_regularization_search: The initial regularization parameter.
    :param max_try: The maximum number of tries.
    :param patience: the number of iterations to wait before stopping the search.
    :param rules: [rule-name], "one", "same_number", "less_number"
    :param strategy_max_try_reach: Procedure when iteration reaches the max iteration. "smallest": set a parameter having the smallest variables. "exception": raise an exception.
    :return: a list of data container.
    """
    # Need proper implementation
    stacks = []

    assert len(rules) > 0, "rules must be one or more."
    assert strategy_max_try_reach in ('smallest', 'exception')

    regularization_parameters = initial_regularization_search
    records_regularization_parameters_previous = []
    __records_variables_previous = []
    __variables = [0] * 10  # dummy initialization
    __i = 0

    # if ratio_regularization_up == "auto":
    #     __ratio_regularization_up = 1.0
    # else:
    #     __ratio_regularization_up = 10.0
    # # end if

    while True:
        __copy_base_estimator = deepcopy(mmd_estimator)
        __training_parameter = copy.deepcopy(training_parameter)
        __training_parameter.regularization_parameter = regularization_parameters
        if variable_trainer is None:
            trainer_obj = InterpretableMmdDetector(
                mmd_estimator=__copy_base_estimator,
                training_parameter=__training_parameter,
                dataset_train=dataset,
                dataset_validation=dataset,
            )
        else:
            trainer_obj = copy.deepcopy(variable_trainer)
            trainer_obj.training_parameter.regularization_parameter = (
                regularization_parameters
            )
        # end if

        trainer_pl = pl.Trainer(**asdict(pytorch_trainer_config))
        trainer_pl.fit(trainer_obj)

        # getting variables
        __trained_result = trainer_obj.get_trained_variables()
        # __ard_normalized = __trained_result.ard_weights_kernel_k / torch.max(
        #     __trained_result.ard_weights_kernel_k
        # )
        # __variables = torch.where(__ard_normalized > 0.1)[0].tolist()
        __variables = detect_variables(
            variable_detection_approach=variable_detection_approach,
            variable_weights=__trained_result.ard_weights_kernel_k,
        )

        stacks.append(
            {
                "regularization_parameter": regularization_parameters,
                "ard_weights": __trained_result.ard_weights_kernel_k,
                "selected_variables": __variables,
                "mmd_estimator": __trained_result.mmd_estimator
            }
        )
        _epoch_used = __trained_result.trajectory_record_training[-1].epoch
        logger.info(f"Lambda-Upper-Bound-Search. Lambda={regularization_parameters}, variables -> {__variables}. Necessary-epoch -> {_epoch_used}")

        # logging to the ML logger
        if post_process_handler is not None:
            log_post_process(post_process_handler, __trained_result, n_iteration=__i)
        # end if

        if __i == max_try:
            return __procedure_max_try(
                seq_stacks=stacks,
                max_try_iteration=max_try,
                initial_regularization_search=initial_regularization_search,
                n_regularization_parameter=n_regularization_parameter,
                strategy_max_try_reach=strategy_max_try_reach)
        # end if

        # updating to the next iteration
        __l1_next = __next_regularization(
            ratio_regularization_up, regularization_parameters.lambda_1
        )
        __l2_next = __next_regularization(
            ratio_regularization_up, regularization_parameters.lambda_2
        )

        records_regularization_parameters_previous.append(
            copy.deepcopy(regularization_parameters)
        )
        regularization_parameters = RegularizationParameter(
            lambda_1=__l1_next, lambda_2=__l2_next
        )
        __records_variables_previous.append(__variables)

        # end condition
        is_stop, rule_name = __is_end_condition(
            __records_variables_previous, __variables, patience, rules=rules
        )
        if is_stop:
            break
        else:
            __i += 1
    # end while

    if rule_name == "same_number":
        if len(records_regularization_parameters_previous) > patience:
            regularization_parameter_last = records_regularization_parameters_previous[
                -(patience + 1)
            ]
        else:
            regularization_parameter_last = records_regularization_parameters_previous[
                -1
            ]
    else:
        regularization_parameter_last = records_regularization_parameters_previous[-1]
    # end if

    result_reg_params = __generate_hyper_parameters(
        regularization_parameter_last,
        n_regularization_parameter,
        initial_regularization_search
    )

    selected_variables = [
        (d["regularization_parameter"], d["selected_variables"]) for d in stacks
    ]

    dict_reg2model_parameter = frozendict(
            {
                result_obj['regularization_parameter']: result_obj['mmd_estimator'] 
                for result_obj in stacks
            })


    res_obj = SelectionResult(
        regularization_parameters=result_reg_params,
        selected_variables=selected_variables,
        dict_regularization2model_parameter=dict_reg2model_parameter,
        regularization_upper_searched=RegularizationParameter(lambda_1=initial_regularization_search.lambda_1, lambda_2=0.0),
        regularization_lower_searched=None
    )

    return res_obj



heuristic_search_regularization_min_max = run_parameter_space_search