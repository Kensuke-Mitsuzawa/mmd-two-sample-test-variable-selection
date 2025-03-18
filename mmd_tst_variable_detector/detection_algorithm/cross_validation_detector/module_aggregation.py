import typing as ty
import itertools
import logging

import torch

from .commons import (
    CrossValidationAggregatedResult,
    SubEstimatorResultContainer,
    CrossValidationTrainParameters,
    AggregationResultContainer,
    STABILITY_SCORE_BASE,
    WEIGHTING_MODE,
    PRE_FILTERING_ESTIMATORS
)

from .module_utils import get_frequency_tensor
from ...exceptions import SameDataException
from ...utils import detect_variables
from ...utils.post_process_logger import PostProcessLoggerHandler
from ...logger_unit import handler 


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class LogicException(Exception):
    pass


class PostAggregatorMmdAGG(object):
    def __init__(self,
                 training_parameter: CrossValidationTrainParameters,
                 post_process_handler: ty.Optional[PostProcessLoggerHandler] = None,
                 cv_detection_experiment_name: ty.Optional[str] = None,
                 minimum_guard_test_power_val: float = 0.0001) -> None:
        """
        Args:
            minimum_guard_test_power_val: A minimum value of test power. If the test power (validation) is <0.0, the value is replaced with this value.
        """
        self.training_parameter = training_parameter
        self.post_process_handler = post_process_handler
        
        if cv_detection_experiment_name is None:
            self.cv_detection_experiment_name = 'post_aggregation_mmd_agg'
        else:
            self.cv_detection_experiment_name = cv_detection_experiment_name
        # end if

        self.minimum_guard_test_power_val = minimum_guard_test_power_val
        
    def __weight_score(self, array_value: torch.Tensor, weighting_mode: str, p_value: float, test_power_val: float) -> torch.Tensor:
        if test_power_val < 0.0:
            test_power_val = self.minimum_guard_test_power_val
        # end if

        if weighting_mode == 'plane':
            return array_value
        elif weighting_mode == 'p_value':
            return array_value * p_value
        elif weighting_mode == 'test_power':
            return array_value * test_power_val
        elif weighting_mode == 'p_value_filter':
            return torch.zeros(len(array_value)) if p_value > 0.05 else array_value
        elif weighting_mode == 'p_value_filter_test_power':
            return torch.zeros(len(array_value)) if p_value > 0.05 else array_value * test_power_val
        elif weighting_mode == 'p_value_min_test_power':
            # idea: bigger test_power and smaller p_value -> better 
            return array_value * (test_power_val * (1 - p_value))
        else:
            raise NotImplementedError()
    
    @staticmethod
    def __get_aggregation_code_name(weighting_mode: str, 
                                    score_value: str,
                                    filtering_mode: str,
                                    filtering_parameter: ty.Union[int, float]) -> str:
        """Private method. Getting the aggregation code name."""
        return f'{weighting_mode}-{score_value}-{filtering_mode}-{filtering_parameter}'
    
    def __filter_estimators(self, 
                            seq_trained_variables: ty.List[SubEstimatorResultContainer],
                            pre_filtering: str,
                            pre_filtering_parameter: ty.Union[int, float]) -> ty.List[SubEstimatorResultContainer]:
        """Selecting good estimators based on a score: `p_value_min_test_power`.
        The cocept is described in https://github.com/Kensuke-Mitsuzawa/mmd-tst-variable-detector/issues/394
        """
        assert pre_filtering in PRE_FILTERING_ESTIMATORS, f'pre_filtering is not in {PRE_FILTERING_ESTIMATORS}'
        assert pre_filtering_parameter is not None, 'pre_filtering_parameter is None.'
        # sort estimators by the value of 'p_value_min_test_power'
        __seq_estimator_score = []
        for __estimator_result in seq_trained_variables:
            if __estimator_result.training_result is None:
                __score = 0.0
            else:
                assert __estimator_result.p_value_selected is not None, 'p_value_selected is None.'
                # the scoring strategy: `p_value_min_test_power`
                __score = __estimator_result.training_result.trajectory_record_validation[-1].ratio * (1 - __estimator_result.p_value_selected)
            # end if
            __seq_estimator_score.append([__estimator_result, __score])
        # end for
        sorted_seq_estimator_score = sorted(__seq_estimator_score, key=lambda x: x[1], reverse=True)
        
        if pre_filtering == 'ranking_top_k':
            if isinstance(pre_filtering_parameter, float):
                __k_selection = int(len(sorted_seq_estimator_score) * pre_filtering_parameter)
                logger.warning(f'pre_filtering_parameter is set as float. Skip it')
                raise LogicException('pre_filtering_parameter is set as float. Skip it')
            else:
                __k_selection = pre_filtering_parameter
            # end if
            seq_selected_estimators = sorted_seq_estimator_score[:__k_selection]
        elif pre_filtering == 'ranking_top_ratio':
            assert isinstance(pre_filtering_parameter, float), 'pre_filtering_parameter is not float.'
            __k_selection = int(len(sorted_seq_estimator_score) * pre_filtering_parameter)
            seq_selected_estimators = sorted_seq_estimator_score[:__k_selection]
        else:
            raise NotImplementedError(f'pre_filtering is not implemented. {pre_filtering}')
        # end if
        
        seq_return = [__t[0] for __t in seq_selected_estimators]
        return seq_return

    def __get_all_possible_aggregations(self, 
                                        seq_trained_variables: ty.List[SubEstimatorResultContainer]
                                        ) -> ty.List[AggregationResultContainer]:
        """Private method. Executing all possible aggregation modes."""
        seq_result_container = []

        iter_possible_combination = itertools.product(STABILITY_SCORE_BASE, WEIGHTING_MODE, PRE_FILTERING_ESTIMATORS)
        for __score_value, __weighting_mode, __pre_filtering in iter_possible_combination:
            try:
                cv_aggregated = self.get_stability_score(
                    seq_trained_variables=seq_trained_variables,
                    weighting_mode=__weighting_mode,
                    score_value=__score_value,
                    pre_filtering=__pre_filtering,
                    pre_filtering_parameter=self.training_parameter.algorithm_parameter.pre_filtering_parameter
                    )
                __stability_score_matrix = cv_aggregated.stability_score_matrix.detach().cpu().numpy() if cv_aggregated.stability_score_matrix is not None else None
                __array_s_hat = cv_aggregated.array_s_hat.detach().cpu().numpy() if cv_aggregated.array_s_hat is not None else None
                __stable_s_hat = cv_aggregated.stable_s_hat if cv_aggregated.stable_s_hat is not None else []
                __cv_agg_container = AggregationResultContainer(
                    weighting_name=__weighting_mode,
                    stability_score_base=__score_value,
                    stability_score_matrix=__stability_score_matrix,
                    array_s_hat=__array_s_hat,
                    stable_s_hat=__stable_s_hat,
                    filtering_estimators=__pre_filtering,
                    filtering_parameter=self.training_parameter.algorithm_parameter.pre_filtering_parameter)
                seq_result_container.append(__cv_agg_container)
                # posting the aggregated cv result to loggers
                if self.post_process_handler is not None:
                    __agg_code_name = self.__get_aggregation_code_name(
                        weighting_mode=__weighting_mode, 
                        score_value=__score_value,
                        filtering_mode=__pre_filtering,
                        filtering_parameter=self.training_parameter.algorithm_parameter.pre_filtering_parameter)
                    __loggers = self.post_process_handler.initialize_logger(
                        run_name=__agg_code_name, 
                        group_name=self.cv_detection_experiment_name)
                    self.post_process_handler.log(loggers=__loggers, target_object=cv_aggregated)
            # end if                
            except (SameDataException, LogicException):
                pass
            except Exception as e:
                logger.error(f'Error during CV-detection. Exception message -> {e}')
            # end try
        # end for
        return seq_result_container

    def __get_stable_s_hat(self,
                           stability_score: torch.Tensor,
                           threshold: float,
                           strategy_stability_score: str = 'mean',
                           is_normalize: bool = True
                           ) -> ty.Tuple[ty.List[int], torch.Tensor]:
        """Private method.

        Get stable_s_hat that shows us the best lambda condition dimension wise.
        
        Parameters
        --------------
        stability_score: Tensor of (|lambda-candidate|, |dimension|). Values represent probabilities.
        threshold: A threshold against the probability.
        strategy_stability_score: an operation on the stability score matrix. 'max' or 'mean'.
        is_normalize: True, the S.S. is normalized by (S.S.) / max(S.S.). The values range is [0.0, 1.0].
        
        Returns
        --------------
        (`stable_s_hat`, `stability_score_agg`) `stable_s_hat` indicates indices of dimensions above the threshold.
        A list of index number or a list of a tuple of (index, column).
        """
        assert strategy_stability_score in self.training_parameter.algorithm_parameter.strategy_stability_score

        if strategy_stability_score == 'max':
            stability_score_agg = torch.max(stability_score, dim=0)
        elif strategy_stability_score == 'mean':
            stability_score_agg = torch.mean(stability_score, dim=0)
        else:
            raise NotImplementedError()
        # end if

        if is_normalize:
            stability_score_agg = stability_score_agg / torch.max(stability_score_agg)
        # end if
        # stable_s_hat = [k for k, score in enumerate(stability_score_agg) if score > threshold]
        if len(stability_score_agg) < 100:
            hist_bins = 100
        else:
            hist_bins = int(len(stability_score_agg) / 2)
        # end if
        stable_s_hat = detect_variables(stability_score_agg, hist_bins=hist_bins)

        # end if

        return stable_s_hat, stability_score_agg
    
    @staticmethod
    def __get_ard_weights_size(seq_trained_variables: ty.List[SubEstimatorResultContainer]) -> int:
        """Private method. Getting the size of ARD weights."""
        seq_size = []
        for __sub_learner in seq_trained_variables:
            if __sub_learner.training_result is not None:
                seq_size.append(len(__sub_learner.training_result.ard_weights_kernel_k))
            # end if
        # end for
        assert len(set(seq_size)) == 1, 'The size of ARD weights is not consistent.'
        return seq_size[0]
        
    def _compute_stability_score(self,
                                 seq_trained_variables: ty.List[SubEstimatorResultContainer],
                                 weighting_mode: str,
                                 score_value: str,
                                 ) -> ty.Tuple[torch.Tensor, ty.List[SubEstimatorResultContainer], ty.List[str]]:
        """Private method.

        Computing Stability Score matrix. The matrix size is (|Lambda|, Dimension).

        Parameters
        --------------
        
        Returns
        --------------
        (stability score matrix, `ty.List[SubEstimatorResultContainer]`, lambda_parameter_list_label)
        """
        size_ard_weights = self.__get_ard_weights_size(seq_trained_variables)
        keys_aggregation_lambda = list(set([__key_obj.job_id.regularization for __key_obj in seq_trained_variables]))
        stability_score = torch.zeros((len(keys_aggregation_lambda), size_ard_weights))

        func_sort = lambda training_result: training_result.job_id.regularization
        iter_g_obj = [
            (__trial_id, list(training_result))
            for __trial_id, training_result
            in itertools.groupby(sorted(seq_trained_variables, key=func_sort), key=func_sort)]
        # a list to save the lambda label for the torch.Tensor.
        seq_lambda_parameter_list_label = []

        for _lambda_index, _tuple_agg in enumerate(iter_g_obj):
            _lambda_label = str(_tuple_agg[0])
            seq_results: ty.List[SubEstimatorResultContainer] = _tuple_agg[1]
            n_subsampling = len(seq_results)

            # array shape: (N, D).
            stability_variable = torch.zeros((len(seq_results), size_ard_weights))

            # ---------------------------------------------------------
            # region: Aggregation per regularization parameter

            for i_sub_learner, result_container in enumerate(seq_results):
                if result_container.training_result is None:
                    # if the optimization is failed, the result is None.
                    continue

                if score_value == 'ard':
                    __array_value = torch.pow(result_container.training_result.ard_weights_kernel_k, 2) / torch.max(torch.pow(result_container.training_result.ard_weights_kernel_k, 2))
                elif score_value == 'variable':
                    __array_value = get_frequency_tensor(torch.pow(result_container.training_result.ard_weights_kernel_k, 2))
                else:
                    raise NotImplementedError()
                # end if

                # weighting
                test_power_validation = result_container.training_result.trajectory_record_validation[-1].ratio
                assert isinstance(result_container.p_value_selected, float)
                __array_value = self.__weight_score(__array_value,
                                                    weighting_mode,
                                                    result_container.p_value_selected,
                                                    test_power_validation)
                stability_variable[i_sub_learner, :] = __array_value
            # endregion: end for
            # ---------------------------------------------------------

            score_at_reg = torch.sum(stability_variable, axis=0) / n_subsampling  # type: ignore

            # 3. saving the record
            stability_score[_lambda_index, :] = score_at_reg
            
            # adding lambda label.
            seq_lambda_parameter_list_label.append(_lambda_label)
        # endregion: end for
        return stability_score, seq_trained_variables, seq_lambda_parameter_list_label    
    
    def get_stability_score(
            self,
            seq_trained_variables: ty.List[SubEstimatorResultContainer],
            weighting_mode: str,
            score_value: str,
            pre_filtering: str,
            pre_filtering_parameter: ty.Optional[ty.Union[int, float]] = None
            ) -> CrossValidationAggregatedResult:
        """Public method.

        Executing post-processing after the ARD weights optimization based on sub-sampled dataset.
        The post-processing continues as follows,
        1. Getting the Stability score.
        2. purgeing inappropriate regularization parameters.
        3. Getting Stable S, which is a set of predicted coordinates.
        4. ARD weights optimization only with the predicted coordinates.
        Use this method when you already have ARD weights optimization results based on sub-sampled dataset,
        and when you wanna recalculate Stability scores.
        
        Parameters
        --------------
        seq_trained_variables: list of ARD weights optimization results generated based on the sub-sampled dataset.
        weighting_mode: If None, `CrossValidationAlgorithmParameter.aggregation_mode` is used.
        score_value: If None, `CrossValidationAlgorithmParameter.score_value` is used.
        """

        # comment: removing lambda parameters and its result where the training is failed with the Exception.
        __stack_sub_learner_training = [
            __learner_obj for __learner_obj in seq_trained_variables
            if __learner_obj.training_result is not None]
        # Do nothing when input is empty.
        if len(__stack_sub_learner_training) == 0:
            return CrossValidationAggregatedResult(
                stable_s_hat=[],
                array_s_hat=None,
                stability_score_matrix=None,
                learner_training_log=None,
                lambda_labels=[])
        # end if
        
        if pre_filtering != 'off':
            assert pre_filtering_parameter is not None, 'pre_filtering_parameter is None.'
            __stack_sub_learner_training = self.__filter_estimators(
                seq_trained_variables=__stack_sub_learner_training,
                pre_filtering=pre_filtering,
                pre_filtering_parameter=pre_filtering_parameter)
        # end if

        # computing the Stability Score
        stability_score_original, learner_training_log, seq_lambda_labels = self._compute_stability_score(
            seq_trained_variables=__stack_sub_learner_training,
            weighting_mode=weighting_mode,
            score_value=score_value
        )
        
        # 
        if torch.count_nonzero(stability_score_original) == 0:
            logger.error('All values in stability_score_original are zero. Could be P=Q.')
            raise SameDataException('stability_score_original (CV matrix score) is all zero. This is because P=Q.')
        # end if
        
        # getting the Stable-S that is a set of detected coordinates
        stable_s_hat, stability_score_agg = self.__get_stable_s_hat(
            stability_score=stability_score_original,
            threshold=self.training_parameter.algorithm_parameter.threshold_stability_score,
            strategy_stability_score=self.training_parameter.algorithm_parameter.strategy_stability_score,
            is_normalize=self.training_parameter.algorithm_parameter.is_normalize_agg_stability_score
        )
        return CrossValidationAggregatedResult(
            stable_s_hat=stable_s_hat,
            array_s_hat=stability_score_agg,
            stability_score_matrix=stability_score_original,
            learner_training_log=learner_training_log,
            lambda_labels=seq_lambda_labels)
    
    # ------------------- Public methods -------------------
    
    def fit(self):
        return self
    
    def fit_transform(self, 
                      seq_trained_variables: ty.List[SubEstimatorResultContainer],
                      ) -> ty.Tuple[CrossValidationAggregatedResult, ty.Optional[ty.List[AggregationResultContainer]]]:
        """Public method.
        
        Raises
        --------------
        `SameDataException`: When the input data is the same
        """
        self.seq_trained_variables = seq_trained_variables
        cv_aggregated = self.get_stability_score(
            seq_trained_variables=seq_trained_variables,
            weighting_mode=self.training_parameter.algorithm_parameter.weighting_mode,
            score_value=self.training_parameter.algorithm_parameter.stability_score_base,
            pre_filtering=self.training_parameter.algorithm_parameter.pre_filtering_trained_estimator,
            pre_filtering_parameter=self.training_parameter.algorithm_parameter.pre_filtering_parameter)
        # posting the aggregated cv result to loggers
        if self.post_process_handler is not None:
            __agg_code_name = self.__get_aggregation_code_name(
                weighting_mode=self.training_parameter.algorithm_parameter.weighting_mode, 
                score_value=self.training_parameter.algorithm_parameter.stability_score_base,
                filtering_mode=self.training_parameter.algorithm_parameter.pre_filtering_trained_estimator,
                filtering_parameter=self.training_parameter.algorithm_parameter.pre_filtering_parameter)
            __loggers = self.post_process_handler.initialize_logger(
                run_name=__agg_code_name, 
                group_name=self.cv_detection_experiment_name)
            self.post_process_handler.log(loggers=__loggers, target_object=cv_aggregated)
        # end if

        # attempting all possible weigting modes.
        if self.training_parameter.algorithm_parameter.is_attempt_all_weighting:
            seq_agg_containers = self.__get_all_possible_aggregations(seq_trained_variables=seq_trained_variables)
        else:
            seq_agg_containers = None
        # end if
                
        return cv_aggregated, seq_agg_containers