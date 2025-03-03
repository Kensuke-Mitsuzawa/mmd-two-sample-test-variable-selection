import typing

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

DEFAULT_PARAMETER = [
    {
        'C': np.logspace(-4, 4, 20),
        'max_iter': [100, 1000, 2500, 5000]
     }
]


class LogisticRegressionBasedSelection(typing.NamedTuple):
    coefficient_vector: np.ndarray
    variable_ranking: typing.List[int]
    normalized_coefficient_vector: np.ndarray


def __run_cross_validation_grid(model: LogisticRegression,
                              x: np.ndarray,
                              y: np.ndarray,
                              cv_split: int = 5,
                              parameters = DEFAULT_PARAMETER
                              ) -> LogisticRegression:
    grid_search = GridSearchCV(model, parameters, cv=cv_split)
    grid_search.fit(x, y)
    return grid_search.best_estimator_

def get_baseline_regression_model(train_x: np.ndarray,
                                  train_y: np.ndarray,
                                  dev_x: np.ndarray,
                                  dev_y: np.ndarray,
                                  penalty: str = 'l1',
                                  param_selection: str = 'GridSearchCV'
                                  ) -> LogisticRegressionBasedSelection:
    if param_selection == 'GridSearchCV':
        x = np.concatenate([train_x, dev_x])
        y = np.concatenate([train_y, dev_y])
        label_vector = np.zeros(shape=(len(x) + len(y),))
        for index_x in range(len(x)):
            label_vector[index_x] = 1
        # end for
        input_vector = np.concatenate([x, y])
        clf = LogisticRegression(penalty=penalty, solver='liblinear')

        estimator = __run_cross_validation_grid(clf, input_vector, label_vector)
    else:
        y_vector = np.zeros(shape=(len(train_x) + len(train_y),))
        for index_x in range(len(train_x)):
            y_vector[index_x] = 1
        # end for
        x_vector = np.concatenate([train_x, train_y])

        clf = LogisticRegression(penalty=penalty)
        estimator = clf.fit(x_vector, y_vector)
    # end if
    coef_vector = estimator.coef_

    v = coef_vector
    normalized_vector = (v - v.min()) / (v.max() - v.min())

    return LogisticRegressionBasedSelection(
        coefficient_vector=coef_vector,
        variable_ranking=np.argsort(-coef_vector).tolist()[0],
        normalized_coefficient_vector=normalized_vector
    )


