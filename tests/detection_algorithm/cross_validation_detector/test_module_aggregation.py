import numpy as np
import torch
from tempfile import mkdtemp
from pathlib import Path

from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.module_aggregation import PostAggregatorMmdAGG
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.commons import (
    CrossValidationTrainParameters,
    SubEstimatorResultContainer,
    AggregationKey,
    InterpretableMmdTrainParameters,
    InterpretableMmdTrainResult,
    RegularizationParameter,
    DistributedComputingParameter,
    CrossValidationAlgorithmParameter
)
from mmd_tst_variable_detector.detection_algorithm.commons import (
    TrainingStatistics,
    TrajectoryRecord)
from mmd_tst_variable_detector.utils.post_process_logger import PostProcessLoggerHandler



def test_module_aggregation():
    n_estimators = 10
    np_random_gen = np.random.default_rng(0)
    seq_random_seed_sequence = [np_random_gen.integers(0, 1000) for __i in range(n_estimators)]
    
    temp_dir = Path(mkdtemp())
    
    seq_sub_estimator_result = [
        SubEstimatorResultContainer(
            job_id=AggregationKey(
                approach_regularization_parameter='fixed_range', 
                trial_id_cross_validation=__i, 
                job_id=__i,
                regularization=RegularizationParameter(0.1, 0.0)),
            training_result=InterpretableMmdTrainResult(
                ard_weights_kernel_k=torch.from_numpy(np.random.default_rng(__random_seed_val).uniform(0, 1, (10,))),
                mmd_estimator={},
                training_stats=TrainingStatistics(
                    global_step=1000,
                    nan_frequency=0,
                    nan_ratio=0.0),
                trajectory_record_training=[TrajectoryRecord(epoch=1000, mmd=0.1, var=0.1, ratio=0.1, loss=0.1)],
                trajectory_record_validation=[TrajectoryRecord(epoch=1000, mmd=0.1, var=0.1, ratio=0.1, loss=0.1)],
                training_parameter=InterpretableMmdTrainParameters(),
                training_configurations=None),
            training_parameter=InterpretableMmdTrainParameters(),
            p_value_selected=np.random.default_rng(__random_seed_val).uniform(0, 1),
            variable_detected=[1,2,3],
            ard_weight_selected_binary=None,
            execution_time_wallclock=np.random.default_rng(__random_seed_val).uniform(0, 10),
            execution_time_cpu=np.random.default_rng(__random_seed_val).uniform(0, 10),
            epoch=1000)
        for __i, __random_seed_val in enumerate(seq_random_seed_sequence)
    ]
    
    cv_param = CrossValidationTrainParameters(
        algorithm_parameter=CrossValidationAlgorithmParameter(
            is_attempt_all_weighting=True,
            pre_filtering_trained_estimator='ranking_top_ratio'),
        base_training_parameter=InterpretableMmdTrainParameters(),
        distributed_parameter=DistributedComputingParameter()
    )

    post_log_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }}
    )
    
    post_aggregator = PostAggregatorMmdAGG(
        training_parameter=cv_param,
        post_process_handler=post_log_handler,
    )
    cv_aggregated, seq_agg_containers = post_aggregator.fit_transform(seq_sub_estimator_result)
    
    assert cv_aggregated is not None
    assert seq_agg_containers is not None
    
