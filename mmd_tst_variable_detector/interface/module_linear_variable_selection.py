from pathlib import Path
import typing as ty
import logging
import timeit

from sklearn.linear_model import LogisticRegression, Ridge, ARDRegression
from sklearn.svm import SVR

from distributed import Client

from ..baselines.regression_based_variable_selection import tst_based_regression_tuner
from ..baselines.regression_based_variable_selection.data_models import CandidateModelContainer
from ..utils.evaluate_variable_detection import evaluate_trained_variables
from ..utils import PostProcessLoggerHandler

from ..datasets import BaseDataset
from ..datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset

from ..logger_unit import handler

from .data_objects import (
    BasicVariableSelectionResult,
    LinearVariableSelectionConfigArgs
)


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


def main(dataset_train: BaseDataset,
         args: LinearVariableSelectionConfigArgs, 
         path_work_dir: Path,
         exp_name: str = 'linear_variable_selection',
         dask_client: ty.Optional[Client] = None) -> BasicVariableSelectionResult:
    
    if isinstance(dataset_train, FileBackendOneTimeLoadStaticDataset):
        dataset_train = dataset_train.generate_dataset_on_ram()
    # end if
    
    start_wall_time = timeit.default_timer()
    
    __path_dir_mlflow_log = path_work_dir
    __path_dir_mlflow_log.mkdir(parents=True, exist_ok=True)
    
    __logger_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config=dict(
            mlflow=dict(
                save_dir=__path_dir_mlflow_log.as_posix(), 
                tracking_uri=f'file://{__path_dir_mlflow_log.as_posix()}'))
        )
    
    __candidate_sk_models = CandidateModelContainer(
        model_candidate_id=exp_name,
        regression_models=[LogisticRegression()]
    )
    
    __tst_result = tst_based_regression_tuner(
        dataset_train=dataset_train,
        candidate_sklearn_models=__candidate_sk_models,
        dask_client=dask_client,
        path_work_dir=path_work_dir,
        n_cv=args.n_cv,
        n_trials=args.n_trials,
        concurrent_limit=args.concurrent_limit,
        dataset_test=None  # comment: making it automatic
    )
    
    # logging to mlflow
    for __eval_result in __tst_result.evaluated_trials:
        __run_name = str(__eval_result.regression_model)
        __loggers = __logger_handler.initialize_logger(group_name=exp_name, 
                                                       run_name=__run_name)
        __logger_handler.log(__loggers, target_object=__eval_result)
    # end for
    
    __selected_model = __tst_result.selected_trial
    logger.debug(f'Selected model: {__selected_model.regression_model} with p-value {__selected_model.p_value_hard_max}')
    __weights = __selected_model.weight_vector
    assert __weights is not None, f'No weights are selected. Abort.'
    
    assert __selected_model.weight_vector is not None, f'No weight vector is selected. Abort.'
    assert __selected_model.selected_variable_indices is not None, f'No selected variable indices are selected. Abort.'
    
    n_sample_training = len(dataset_train)
    
    return BasicVariableSelectionResult(
        weights=__selected_model.weight_vector,
        variables=__selected_model.selected_variable_indices,
        p_value=__selected_model.p_value_hard_max,
        verbose_field=__tst_result,
        n_sample_training=n_sample_training,
    )