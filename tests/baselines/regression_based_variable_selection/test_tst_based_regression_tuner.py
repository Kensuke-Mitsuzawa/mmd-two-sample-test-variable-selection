import logging
import functools
from pathlib import Path
from tempfile import mkdtemp
import shutil

import torch

from distributed import Client, LocalCluster

from sklearn.linear_model import LogisticRegression, Ridge, ARDRegression

from mmd_tst_variable_detector.baselines.regression_based_variable_selection.tst_based_regression_tuner import (
    tst_based_regression_tuner
)
from mmd_tst_variable_detector.baselines.regression_based_variable_selection.data_models import (
    CandidateModelContainer
)
from mmd_tst_variable_detector import (
    SimpleDataset,
    PermutationTest
)
from mmd_tst_variable_detector.utils import evaluate_variable_detection

from ...data_generator import test_data_xy_linear


logger = logging.getLogger(f'test.{__name__}')


def test_optuna_search_tst_regression():
    (x_tensor, y_tensor), dim_ground_truth = test_data_xy_linear(dim_size=20,
                                                                 sample_size=1000,
                                                                 ratio_dependent_variables=0.1)
    
    x_train = x_tensor[:800]
    y_train = y_tensor[:800]
    dataset_obj_train = SimpleDataset(x_train, y_train)

    x_dev = x_tensor[800:]
    y_dev = y_tensor[800:]
    dataset_obj_dev = SimpleDataset(x_dev, y_dev)
    
    candidate_models = CandidateModelContainer(
        model_candidate_id='test',
        regression_models=[LogisticRegression()]
    )
    
    # dask_cluster = LocalCluster()
    # dask_client = dask_cluster.get_client()
    
    permutation_runer = [PermutationTest(n_permutation_test=10)]
    
    path_work_dir = Path(mkdtemp()) / 'test_tst_based_regression_tuner'
    path_work_dir.mkdir(parents=True, exist_ok=True)
    
    res = tst_based_regression_tuner(
        dataset_train=dataset_obj_train,
        candidate_sklearn_models=candidate_models,
        dask_client=None,
        permutationt_test_runner=permutation_runer,
        path_work_dir=path_work_dir,
        n_cv=2,
        n_trials=2,
    )
    best_selection = res.selected_trial
    
    eval_res, index_detection = evaluate_variable_detection.evaluate_trained_variables(
        ard_weights=torch.tensor(best_selection.weight_vector), 
        ground_truth_index=dim_ground_truth)
    
    logger.debug(f'eval_res: {eval_res}')
    
    shutil.rmtree(path_work_dir)
