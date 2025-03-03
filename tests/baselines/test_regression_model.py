import numpy as np
import typing as ty

from mmd_tst_variable_detector.baselines.regression_model import get_baseline_regression_model
from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution


def test_run_cross_validation_grid():
    x, y, index_replace = sampling_from_distribution(
        n_sample=500,
        dimension_size=20,
        mixture_rate=0.1,
        distribution_conf_p={'type': 'gaussian', 'mu': 0, 'sigma': 1},
        distribution_conf_q={'type': 'gaussian', 'mu': 5.0, 'sigma': 1},
    )

    x_train = x[:300]
    y_train = y[:300]
    x_test = x[300:]
    y_test = y[300:]

    result = get_baseline_regression_model(train_x=x_train, train_y=y_train, dev_x=x_test, dev_y=y_test)

    commons = set(index_replace).intersection(set(result.variable_ranking))
    precision = len(commons) / len(index_replace)
    recall = len(commons) / len(result.variable_ranking)
    f_score = (2 * precision * recall) / (precision + recall)
    # print(f_score)
