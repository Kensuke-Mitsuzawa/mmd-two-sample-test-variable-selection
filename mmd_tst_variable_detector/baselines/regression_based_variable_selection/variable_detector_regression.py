import typing as ty
import logging
from dataclasses import dataclass
import time
import timeit

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

from ...datasets import BaseDataset
from ...logger_unit import handler

import torch.utils.data
from torch.utils.data import DataLoader

from .data_models import TrainedResultRegressionBasedVariableDetector, AcceptableClass

from ...utils import detect_variables
from ...utils.permutation_test_runner import PermutationTest


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


"""
class module that executes variable detection by regression based methods.
"""



class RegressionBasedVariableDetector(object):
    def __init__(self,
                 regression_module: AcceptableClass,
                 permutation_test_runners: ty.List[PermutationTest],
                 batch_size: int = -1,
                 is_shuffle: bool = False,
                 is_run_soft_permutation_test: bool = False,
                 preprocessing_module: ty.Optional[AcceptableClass] = None,  # comment: place-holder. I will use it in the future.
                 ) -> None:
        self.regression_module = regression_module
        
        assert hasattr(regression_module, "fit")
        
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.is_run_soft_permutation_test = is_run_soft_permutation_test
        
        if len(permutation_test_runners) == 0:
            logger.info("permutation_test_runners is empty. I use the default runers.")
            self.permutation_test_runners = [PermutationTest()]
        else:
            self.permutation_test_runners = permutation_test_runners
        
        self.variable_detection_result: ty.Optional[TrainedResultRegressionBasedVariableDetector] = None  
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self        
        
    def _post_process(self, coef_vector: np.ndarray) -> ty.Tuple[ty.Optional[np.ndarray], ty.List[int]]:
        is_all_zero = np.all(coef_vector == 0.0)
        if is_all_zero:
            logger.warning(f"With the {self.regression_module}. All coefficients are zero. Returning None.")
            normalized_vector = None
            seq_selected_variable = []
        else:
            # Power2 when Ridge or Lasso.
            # Ridge or Lasso represent importance by minus and plus.
            coef_vector = coef_vector ** 2
            normalized_vector = (coef_vector - coef_vector.min()) / (coef_vector.max() - coef_vector.min())
            # I think need to post-process per model.
            seq_selected_variable = detect_variables(variable_weights=normalized_vector)
        # end if
        
        return normalized_vector, seq_selected_variable
    
    def _run_permutation_test(self, 
                              datsaet_test: BaseDataset,
                              weight_vector: np.ndarray, 
                              seq_selected_variable: ty.List[int]
                              ) -> ty.Tuple[ty.List[float], ty.List[float]]:
        p_values_soft = []
        p_values_hard = []
                        
        for __permutation_runner in self.permutation_test_runners:
            dataset_test_select = datsaet_test.get_selected_variables_dataset(seq_selected_variable)
            p_dev, __ = __permutation_runner.run_test(dataset_test_select)
            p_values_hard.append(p_dev)

            if self.is_run_soft_permutation_test:
                p_dev, __ = __permutation_runner.run_test(
                    dataset=datsaet_test,
                    featre_weights=torch.tensor(weight_vector))
                p_values_soft.append(p_dev)
            # end if            
        # end for
        
        return p_values_soft, p_values_hard
    
    # def score(self, X: np.ndarray, y: np.ndarray, p_value_mode: str = 'hard') -> float:
    #     """Scikit-learn's standard API.
        
    #     Args:
    #         X: sample X.
    #         y: sample Y.
    #         p_value_mode: 'hard' or 'soft'. 'hard' means the p-value of the selected variables.
    #     Returns:
    #         score, which is a p-value.
    #     """
    #     assert p_value_mode in ['hard', 'soft']
    #     assert self.variable_detection_result is not None
        
    #     if p_value_mode == 'hard':
    #         p_value = self.variable_detection_result.p_value_hard_max
    #     elif p_value_mode == 'soft':
    #         p_value = self.variable_detection_result.p_value_soft_max
    #     else:
    #         raise ValueError(f'Unknown p_value_mode = {p_value_mode}')
    #     # end if
        
    #     return p_value

        
    # def partial_fit(self, 
    #                 batch_x: np.ndarray, 
    #                 batch_y: np.ndarray, 
    #                 dataset_test: BaseDataset):
    #     feature_vector = np.concatenate([batch_x, batch_y])
    #     label_vector = np.zeros(shape=(feature_vector.shape[0]),)
    #     for index_x in range(len(batch_x)):
    #         label_vector[index_x] = 1
    #     # end for            
        
    #     self.regression_module.fit(batch_x, batch_y)

    #     coef_vector = self.regression_module.coef_
    #     assert isinstance(coef_vector, np.ndarray)

    #     if isinstance(self.regression_module, (LogisticRegression, SVR)):
    #         coef_vector = coef_vector[0]
    #     # end if
        
    #     normalized_vector, variables = self._post_process(coef_vector)
        
    #     if normalized_vector is None:
    #         p_soft, p_hard = None, None
    #         p_value_soft_max, p_value_hard_max = -1, -1
    #     else:
    #         p_soft, p_hard = self._run_permutation_test(datsaet_test=dataset_test,
    #                                                     weight_vector=normalized_vector,
    #                                                     seq_selected_variable=variables)
    #         p_value_soft_max = np.max(p_soft)
    #         p_value_hard_max = np.max(p_hard)
    #     # end if
        
    #     res = TrainedResultRegressionBasedVariableDetector(
    #         regression_model=self.regression_module,
    #         p_value_soft_max=p_value_soft_max,
    #         p_value_hard_max=p_value_hard_max,
    #         weight_vector=normalized_vector,
    #         selected_variable_indices=variables,
    #         seq_p_value_soft=p_soft,
    #         seq_p_value_hard=p_hard)
    #     self.variable_detection_result = res
            
        
    def run_variable_detection(self, 
            dataset_train: BaseDataset,
            dataset_test: BaseDataset) -> TrainedResultRegressionBasedVariableDetector:
        """Construction a regression model, label = model(input). 
        Input: x or y. the label is x=1, y=0.
        """
        if self.batch_size == -1:
            batch_size = len(dataset_train) * 2
        else:
            batch_size = self.batch_size
        # end if
        
        start_cpu_time = time.process_time()
        start_wall_time = timeit.default_timer()

        data_loader = DataLoader(dataset_train, 
                                 batch_size=batch_size, 
                                 shuffle=self.is_shuffle)
        
        for __batch_xy in data_loader:
            __x, __y = __batch_xy
            __x_np = __x.numpy()
            __y_np = __y.numpy()
            
            __feature_vector = np.concatenate([__x_np, __y_np])
            __label_vector = np.zeros(shape=(__feature_vector.shape[0]),)
            for index_x in range(len(__x_np)):
                __label_vector[index_x] = 1
            # end for            
            
            self.regression_module.fit(__feature_vector, __label_vector)
        # end for
        
        assert hasattr(self.regression_module, 'score'), f"Regression model must have the score method. {self.regression_module}"
        
        seq_tuple_xy = [dataset_test.__getitem__(__i_sample) for __i_sample in range(len(dataset_test))]
        test_array_var_x = np.stack([__t[0].numpy() for __t in seq_tuple_xy])
        test_array_var_y = np.stack([__t[1].numpy() for __t in seq_tuple_xy])        
        test_array = np.concatenate([test_array_var_x, test_array_var_y])
        test_label = np.array([1] * len(test_array_var_x) + [0] * len(test_array_var_y))
        _score_test = self.regression_module.score(test_array, test_label)
        assert isinstance(_score_test, float), f"Regression model must return a float. {_score_test}"
        
            
        coef_vector = self.regression_module.coef_
        assert isinstance(coef_vector, np.ndarray)

        if isinstance(self.regression_module, (LogisticRegression, SVR)):
            coef_vector = coef_vector[0]
        # end if

        normalized_vector, variables = self._post_process(coef_vector)
        if normalized_vector is None:
            p_soft, p_hard = None, None
            # comment: I put 1.0 for p-value when I encounter exceptions. p=1.0 is the worst case.
            p_value_soft_max, p_value_hard_max = 1.0, 1.0
        else:
            p_soft, p_hard = self._run_permutation_test(datsaet_test=dataset_test,
                                                        weight_vector=normalized_vector,
                                                        seq_selected_variable=variables)
            p_value_hard_max = np.max(p_hard)
            if self.is_run_soft_permutation_test:
                p_value_soft_max = np.max(p_soft)
            else:
                # comment: I put 1.0 for p-value when I encounter exceptions. p=1.0 is the worst case.
                p_value_soft_max = 1.0
        # end if

        end_cpu_time = time.process_time()
        end_wall_time = timeit.default_timer()

        exec_time_cpu = end_cpu_time - start_cpu_time
        exec_time_wallclock = end_wall_time - start_wall_time

        res = TrainedResultRegressionBasedVariableDetector(
            regression_model=self.regression_module,
            p_value_soft_max=p_value_soft_max,
            p_value_hard_max=p_value_hard_max,
            weight_vector=normalized_vector,
            selected_variable_indices=variables,
            seq_p_value_soft=p_soft,
            seq_p_value_hard=p_hard,
            execution_time_wallclock=exec_time_wallclock,
            execution_time_cpu=exec_time_cpu,
            error_regression=_score_test)
        self.variable_detection_result = res
        return res