import typing as ty
import optuna
import numpy as np
import functools
import json
import pickle
from pathlib import Path
from tempfile import mkdtemp
import logging
import optuna

from sklearn.linear_model import (
    ARDRegression,
    Ridge,
    LogisticRegression
)
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from distributed import Client, LocalCluster

import dataclasses

from ...datasets import BaseDataset
from ...utils.permutation_test_runner import PermutationTest
from .data_models import (
    AcceptableClass,
    CandidateModelContainer,
    # default parameter search range
    BaseSearchParameter,
    SearchParameterARDRegression,
    SearchParameterRidge,
    SearchParameterLogisticRegression,
    SearchParameterSVR,
)
from .variable_detector_regression import RegressionBasedVariableDetector
from .data_models import TrainedResultRegressionBasedVariableDetector, AcceptableClass

from ...logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


@dataclasses.dataclass
class TstBasedRegressionTunerResult(object):
    selected_trial: TrainedResultRegressionBasedVariableDetector
    evaluated_trials: ty.List[TrainedResultRegressionBasedVariableDetector]


def func_wrapper_function_optuna(trial: optuna.Trial,
                                 path_work_dir: Path,
                                 dataset_train: BaseDataset,
                                 dataset_test: BaseDataset,
                                 search_space_sk_model: ty.List[str],
                                 dict_class2search_parameter: ty.Dict[str, BaseSearchParameter],
                                 permutation_test_runner: ty.List[PermutationTest],
                                 batch_size: int = -1,
                                 p_value_select: str = 'hard',
                                 score_function_mode: str = 'p_value'):
    assert score_function_mode in ('error', 'p_value'), f'Unknown score_function_mode = {score_function_mode}'
    # TODO if statement of choosing a sk-model.
    sk_model = trial.suggest_categorical("sk_model", search_space_sk_model)
    
    parameter_obj = dict_class2search_parameter[sk_model]
    
    if sk_model == 'ARDRegression':
        assert isinstance(parameter_obj, SearchParameterARDRegression)
        __sk_model = ARDRegression(
            alpha_1=trial.suggest_float('alpha_1', parameter_obj.alpha_1[0], parameter_obj.alpha_1[1]),
            alpha_2=trial.suggest_float('alpha_2', parameter_obj.alpha_2[0], parameter_obj.alpha_2[1]),
            lambda_1=trial.suggest_float('lambda_1', parameter_obj.lambda_1[0], parameter_obj.lambda_1[1]),
            lambda_2=trial.suggest_float('lambda_2', parameter_obj.lambda_1[0], parameter_obj.lambda_2[1])
        )
    elif sk_model == 'Ridge':
        assert isinstance(parameter_obj, SearchParameterRidge)
        __sk_model = Ridge(alpha=trial.suggest_float('alpha', parameter_obj.alpha[0], parameter_obj.alpha[1]))
    elif sk_model == 'LogisticRegression':
        assert isinstance(parameter_obj, SearchParameterLogisticRegression)
        __type_penalty = trial.suggest_categorical('penalty', parameter_obj.penalty)
        if __type_penalty == 'elasticnet':
            __l1_ratio = trial.suggest_float('l1_ratio', parameter_obj.l1_ratio[0], parameter_obj.l1_ratio[1])
        else:
            __l1_ratio = None
        # end if
        
        __sk_model = LogisticRegression(
            penalty=__type_penalty,
            C=trial.suggest_float('C', parameter_obj.C[0], parameter_obj.C[1]),
            l1_ratio=__l1_ratio,
            solver='saga'
        )
    elif sk_model == 'SVR':
        assert isinstance(parameter_obj, SearchParameterSVR)
        __sk_model = SVR(
            kernel='linear',
            degree=trial.suggest_int('degree', parameter_obj.degree[0], parameter_obj.degree[1]),
            coef0=trial.suggest_float('coef0', parameter_obj.coef0[0], parameter_obj.coef0[1]),
            C=trial.suggest_float('C', parameter_obj.C[0], parameter_obj.C[1]))
    else:
        raise NotImplementedError(f'Not implemented class name: {sk_model}')
    # end if        
    
    detector = RegressionBasedVariableDetector(
        regression_module=__sk_model,
        permutation_test_runners=permutation_test_runner,
        batch_size=batch_size
    )
    detection_result = detector.run_variable_detection(
        dataset_train=dataset_train,
        dataset_test=dataset_test)
    
    path_result = (path_work_dir / f'{trial.number}.pickle')
    with path_result.open('wb') as f:
        pickle.dump(detection_result, f)
    # end with
    
    if score_function_mode == 'p_value':        
        if p_value_select == 'hard':
            return detection_result.p_value_hard_max
        elif p_value_select == 'soft':
            return detection_result.p_value_soft_max
        else:
            raise ValueError(f'Unknown p_value_select = {p_value_select}')
    elif score_function_mode == 'error':
        assert detection_result.error_regression is not None
        return -1 * detection_result.error_regression
    else:
        raise ValueError(f'Unknown score_function_mode = {score_function_mode}')
        

def func_dask_weapper_function_optuna(trial: optuna.Trial,
                                      path_work_dir: Path,
                                      dataset_train: BaseDataset,
                                      dataset_test: BaseDataset,
                                      search_space_sk_model: ty.List[str],
                                      dict_class2search_parameter: ty.Dict[str, BaseSearchParameter],
                                      permutation_test_runner: ty.List[PermutationTest],
                                      batch_size: int = -1,
                                      p_value_select: str = 'hard',
                                      score_mode: str = 'p_value') -> ty.Tuple[optuna.Trial, float, TrainedResultRegressionBasedVariableDetector]:
    score_value = func_wrapper_function_optuna(
        trial=trial,
        path_work_dir=path_work_dir,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        search_space_sk_model=search_space_sk_model,
        dict_class2search_parameter=dict_class2search_parameter,
        permutation_test_runner=permutation_test_runner,
        batch_size=batch_size,
        p_value_select=p_value_select,
        score_function_mode=score_mode)
    with open(path_work_dir / f"{trial.number}.pickle", "rb") as fin:
        best_clf = pickle.load(fin)
    # end with
    
    return trial, score_value, best_clf



def select_best_opt_result(study: optuna.Study, 
                           seq_trials: ty.List[ty.Tuple[int, TrainedResultRegressionBasedVariableDetector]],
                           score_mode: str = 'p_value'
                           ) -> TrainedResultRegressionBasedVariableDetector:
    """
    there are possibilities that multiple trials have the same p-value.
    Hence, I sort results by multiple keys.
    first-key: p-value of "hard" variable selection.
    second-key: p-value of "soft" variable selection.
    3rd-key: the number of selected variables.
    
    """
    if score_mode == 'error':
        score_value_best = max([_opt_results[1].error_regression for _opt_results in seq_trials if _opt_results[1].error_regression is not None])
        trials_select = [__t_trial for __t_trial in seq_trials if __t_trial[1].error_regression == score_value_best]
    elif score_mode == 'p_value':
        p_value_best = min([_opt_result[1].p_value_hard_max for _opt_result in seq_trials])
        # selection by p-value of hard selection.
        trials_select = [__t_trial for __t_trial in seq_trials if __t_trial[1].p_value_hard_max == p_value_best]
        # selection by p-value of soft selection.
        # possibiliy that p-value-soft-selection is -1. 
        __trials_p_max_soft_selection = [__t_trial for __t_trial in trials_select if __t_trial[1].p_value_soft_max != -1]    
        p_value_best = study.best_value
        if len(__trials_p_max_soft_selection) == 0:
            # do nothing.
            trials_select = trials_select
        else:
            __best_p_value_soft_selection = min([__t_trial[1].p_value_soft_max for __t_trial in __trials_p_max_soft_selection])
            trials_select = [__t_trial for __t_trial in trials_select if __t_trial[1].p_value_soft_max == __best_p_value_soft_selection]
        # end if
    # end if
    
    # selection by the number of selected variables.
    trials_select = sorted(trials_select, key=lambda x: len(x[1].selected_variable_indices), reverse=True)
    
    selected_trial = trials_select[0][1]
    return selected_trial


def __tst_based_regression_tuner_no_cv(dataset_all: BaseDataset,
                                    path_opt_result: Path,
                                    search_space_sk_model: ty.List[str],
                                    dict_class2search_parameter: ty.Dict[str, BaseSearchParameter],
                                    permutationt_test_runner: ty.List[PermutationTest],
                                    score_mode: str,
                                    n_trials: int,
                                    concurrent_limit: int,
                                    study: optuna.Study,
                                    dask_client: ty.Optional[Client] = None,
                                    ratio_validation: float = 1 / 5
                                    ) -> TstBasedRegressionTunerResult:
    # Manual Cross-Validation
    seq_sample_id = list(range(dataset_all.__len__()))
    random_gen = np.random.default_rng(0)
    random_gen.shuffle(seq_sample_id)
    n_validation = int(ratio_validation * len(seq_sample_id))
    
    train_sample_id = seq_sample_id[:n_validation]
    test_sample_id = seq_sample_id[n_validation:]
    
    # stack to save
    stack_opt_result = []
    
    __, dataset_train = dataset_all.get_subsample_dataset(sample_ids=train_sample_id)
    __,dataset_test = dataset_all.get_subsample_dataset(sample_ids=test_sample_id)
    
    __func_dask_task = functools.partial(
        func_dask_weapper_function_optuna,
        path_work_dir=path_opt_result,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        search_space_sk_model=search_space_sk_model,
        dict_class2search_parameter=dict_class2search_parameter,
        permutation_test_runner=permutationt_test_runner,
        score_mode=score_mode
    )
    
    current_trials = 0
    while current_trials < n_trials:
        logger.debug(f'current_trials = {current_trials}')

        __seq_trial_stack = [study.ask() for __i in range(concurrent_limit)]
        if dask_client is None:
            task_return = [__func_dask_task(__trial) for __trial in __seq_trial_stack]
        else:
            logger.debug(f'Executing tasks with Dask...')    
            task_queue = dask_client.map(__func_dask_task, __seq_trial_stack)
            task_return = dask_client.gather(task_queue)
            logger.debug(f'Finished executing tasks with Dask.')
        # end if
        
        for __t_return in task_return:
            __trial = __t_return[0]
            __eval_score = __t_return[1]
            __eval_result = __t_return[2]
            
            study.tell(__trial, __eval_score)
            
            stack_opt_result.append([__trial.number, __eval_result])
        # end for
        current_trials += concurrent_limit
    # end while
    
    selected_trial = select_best_opt_result(study, stack_opt_result)
    all_trials = [__trial[1] for __trial in stack_opt_result]
    
    result = TstBasedRegressionTunerResult(
            selected_trial=selected_trial,
            evaluated_trials=all_trials)
    return result
    


def __tst_based_regression_tuner_cv(dataset_all: BaseDataset,
                                    n_cv: int,
                                    path_opt_result: Path,
                                    search_space_sk_model: ty.List[str],
                                    dict_class2search_parameter: ty.Dict[str, BaseSearchParameter],
                                    permutationt_test_runner: ty.List[PermutationTest],
                                    score_mode: str,
                                    n_trials: int,
                                    concurrent_limit: int,
                                    study: optuna.Study,
                                    dask_client: ty.Optional[Client] = None,
                                    ) -> TstBasedRegressionTunerResult:
    # Manual Cross-Validation
    seq_sample_id = range(dataset_all.__len__())
    kf_splitter = KFold(n_splits=n_cv)
    pair_xy_train_test_sample_id = list(kf_splitter.split(seq_sample_id))
    
    # stack to save
    stack_opt_result = []
    
    for __i, (__train_sample_id, __test_sample_id) in enumerate(pair_xy_train_test_sample_id):
        logger.debug(f'Executing CV = {__i}...')
        __, dataset_train = dataset_all.get_subsample_dataset(sample_ids=__train_sample_id)
        __,dataset_test = dataset_all.get_subsample_dataset(sample_ids=__test_sample_id)
        
        __func_dask_task = functools.partial(
            func_dask_weapper_function_optuna,
            path_work_dir=path_opt_result,
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            search_space_sk_model=search_space_sk_model,
            dict_class2search_parameter=dict_class2search_parameter,
            permutation_test_runner=permutationt_test_runner,
            score_mode=score_mode
        )
        
        current_trials = 0
        while current_trials < n_trials:
            logger.debug(f'current_trials = {current_trials}')

            __seq_trial_stack = [study.ask() for __i in range(concurrent_limit)]
            if dask_client is None:
                task_return = [__func_dask_task(__trial) for __trial in __seq_trial_stack]
            else:
                logger.debug(f'Executing tasks with Dask...')    
                task_queue = dask_client.map(__func_dask_task, __seq_trial_stack)
                task_return = dask_client.gather(task_queue)
                logger.debug(f'Finished executing tasks with Dask.')
            # end if
            
            for __t_return in task_return:
                __trial = __t_return[0]
                __eval_score = __t_return[1]
                __eval_result = __t_return[2]
                
                study.tell(__trial, __eval_score)
                
                stack_opt_result.append([__trial.number, __eval_result])
            # end for
            current_trials += concurrent_limit
        # end while
    # end for
    
    selected_trial = select_best_opt_result(study, stack_opt_result)
    all_trials = [__trial[1] for __trial in stack_opt_result]
    
    result = TstBasedRegressionTunerResult(
            selected_trial=selected_trial,
            evaluated_trials=all_trials)
    return result


def tst_based_regression_tuner(dataset_train: BaseDataset,
                               candidate_sklearn_models: CandidateModelContainer,
                               search_parameter_objects: ty.Optional[ty.List[BaseSearchParameter]] = None,
                               dataset_test: ty.Optional[BaseDataset] = None,
                               permutationt_test_runner: ty.Optional[ty.List[PermutationTest]] = None,
                               path_work_dir: ty.Optional[Path] = None,
                               dask_client: ty.Optional[Client] = None,
                               n_cv: int = 5,
                               n_trials: int = 20,
                               concurrent_limit: int = 1,
                               score_mode: str = 'p_value',
                               is_use_cross_validation: bool = True) -> TstBasedRegressionTunerResult:
    """
    Parameters
    -----------
    dataset_train: BaseDataset
        training dataset.
    candidate_sklearn_models: CandidateModelContainer
        candidate models.
    search_parameter_objects: ty.Optional[ty.List[BaseSearchParameter]]
        search space of hyper-parameters.
    dataset_test: ty.Optional[BaseDataset]
        test dataset.
    permutationt_test_runner: ty.Optional[ty.List[PermutationTest]]
        permutation test runner.
    path_work_dir: ty.Optional[Path]
        path to save results.
    dask_client: ty.Optional[Client]
        dask client.
    n_cv: int
        number of cross-validation.
    n_trials: int
        number of trials.
    concurrent_limit: int
        number of concurrent trials.
    """
    
    if path_work_dir is None:
        path_work_dir =  Path(mkdtemp()) / 'tst_based_regression_tuner'
        path_work_dir.mkdir(parents=True, exist_ok=True)
    # end if
    
    # merging two dataset into one.
    if dataset_test is None:
        dataset_all = dataset_train
    else:
        logger.debug(f'Merging two datasets: {dataset_train} and {dataset_test}')
        dataset_all = dataset_train.merge_new_dataset(dataset_test)
    # end if
    
    assert path_work_dir.exists(), f'path_work_dir does not exist: {path_work_dir}'
    # -------------------------------------------------------------------
    # get all regression models
    seq_class_names = candidate_sklearn_models.get_model_class_names()

    # if search_parameter_objects is None, use default search space.
    if search_parameter_objects is None:
        # deleted SearchParameterARDRegression(), SearchParameterRidge(), SearchParameterSVR()
        # LogisticRegression is enough for normal usage.
        search_parameter_objects = [
            SearchParameterLogisticRegression(),
        ]
    # end if
    
    dict_class2search_parameter = {
        __param_obj.class_name: __param_obj for __param_obj in search_parameter_objects
    }
    # definition of scikit-learn model
    search_space_sk_model = seq_class_names
    # -------------------------------------------------------------------
    dataset_test = dataset_train
    
    if permutationt_test_runner is None:
        permutationt_test_runner = [PermutationTest(n_permutation_test=500)]
    # end if
    
    # -------------------------------------------------------------------
    path_storage_backend_db = f'sqlite:///{path_work_dir / "optuna.sqlite3"}'
    study = optuna.create_study(storage=path_storage_backend_db, direction="minimize")
    
    path_opt_result = path_work_dir / 'optuna'
    path_opt_result.mkdir(parents=True, exist_ok=True)
        
    logger.debug(f'concurrent_limit = {concurrent_limit}')
    # assert dask_client is not None, 'dask_client is None'
    
    if is_use_cross_validation:
        result = __tst_based_regression_tuner_cv(
            dataset_all=dataset_all,
            n_cv=n_cv,
            path_opt_result=path_opt_result,
            search_space_sk_model=search_space_sk_model,
            dict_class2search_parameter=dict_class2search_parameter,
            permutationt_test_runner=permutationt_test_runner,
            score_mode=score_mode,
            n_trials=n_trials,
            concurrent_limit=concurrent_limit,
            study=study,
            dask_client=dask_client)
    else:
        result = __tst_based_regression_tuner_no_cv(
            dataset_all=dataset_all,
            path_opt_result=path_opt_result,
            search_space_sk_model=search_space_sk_model,
            dict_class2search_parameter=dict_class2search_parameter,
            permutationt_test_runner=permutationt_test_runner,
            score_mode=score_mode,
            n_trials=n_trials,
            concurrent_limit=concurrent_limit,
            study=study,
            dask_client=dask_client)
    # end if
    
    return result