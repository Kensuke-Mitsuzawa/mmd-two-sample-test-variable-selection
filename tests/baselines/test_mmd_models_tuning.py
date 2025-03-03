import torch

from mmd_tst_variable_detector.baselines import mmd_models_tuning

from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdTrainResult, InterpretableMmdTrainParameters, TrajectoryRecord, TrainingStatistics
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel

def test_mmd_models_tuning():
    dim_size = 20
    dummy_ard = torch.normal(mean=0, std=1.0, size=[dim_size])
    dummy_kernel = QuadraticKernelGaussianKernel(ard_weights=dummy_ard)
    dummy_mmd_estimator = QuadraticMmdEstimator(kernel_obj=dummy_kernel)

    seq_optimized_mmd_results = [
        InterpretableMmdTrainResult(ard_weights_kernel_k=dummy_ard,
                       mmd_estimator=dummy_mmd_estimator,
                       trajectory_record_training=[TrajectoryRecord(epoch=100, mmd=0, var=1, ratio=0.0, loss=-1.0)],
                       trajectory_record_validation=[TrajectoryRecord(epoch=100, mmd=0, var=1, ratio=0.0, loss=-1.0)],
                       training_stats=TrainingStatistics(global_step=100, nan_frequency=-1, nan_ratio=-1),
                       training_parameter=InterpretableMmdTrainParameters()
                       ),
        InterpretableMmdTrainResult(ard_weights_kernel_k=dummy_ard,
                       mmd_estimator=dummy_mmd_estimator,
                       trajectory_record_training=[TrajectoryRecord(epoch=100, mmd=100, var=1, ratio=0.0, loss=-1.0)],
                       trajectory_record_validation=[TrajectoryRecord(epoch=100, mmd=100, var=1, ratio=0.0, loss=-1.0)],
                       training_stats=TrainingStatistics(global_step=100, nan_frequency=-1, nan_ratio=-1),
                       training_parameter=InterpretableMmdTrainParameters())
    ]
    sorted_optimized_result = mmd_models_tuning.tune_trained_mmd_models(seq_optimized_mmd_results)
    assert len(sorted_optimized_result) == 2
    assert sorted_optimized_result[0].trajectory_record_training[-1].mmd == 100
