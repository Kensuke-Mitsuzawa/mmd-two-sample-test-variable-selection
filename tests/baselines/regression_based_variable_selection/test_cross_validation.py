# import logging
# import functools

# import torch
# import geomloss
# import numpy as np

# from distributed import Client, LocalCluster

# from sklearn.linear_model import LogisticRegression, Ridge, ARDRegression

# from mmd_tst_variable_detector.baselines.regression_based_variable_selection import algorithm_one
# from mmd_tst_variable_detector.baselines.regression_based_variable_selection.cross_validation import (
#     RegressionCrossValidationInterpretableVariableDetector,
#     RegressionCrossValidationTrainParameters,
#     RegressionCrossValidationAlgorithmParameter,
# )
# from mmd_tst_variable_detector import (
#     SimpleDataset,
#     PermutationTest,
#     DistributedComputingParameter    
# )

# from ...data_generator import test_data_xy_linear


# logger = logging.getLogger(f'test.{__name__}')


# def __func_distance_sinkhorn(x: torch.Tensor, y: torch.Tensor, geomloss_obj) -> torch.Tensor:
#     return geomloss_obj(x, y)


# def test_cross_validation():
#     (x_tensor, y_tensor), dim_ground_truth = test_data_xy_linear(dim_size=20,
#                                                                  sample_size=1000,
#                                                                  ratio_dependent_variables=0.1)
    
#     x_train = x_tensor[:800]
#     y_train = y_tensor[:800]
#     dataset_obj_train = SimpleDataset(x_train, y_train)

#     x_dev = x_tensor[800:]
#     y_dev = y_tensor[800:]
#     dataset_obj_dev = SimpleDataset(x_dev, y_dev)
    
#     candidate_models = algorithm_one.CandidateModelContainer(
#         model_candidate_id='test',
#         regression_models=[
#             LogisticRegression(C=1.0), 
#             LogisticRegression(C=0.5)
#             ]
#     )
        
#     func_distance = functools.partial(__func_distance_sinkhorn,
#                                       geomloss_obj=geomloss.SamplesLoss(loss='sinkhorn'))
    
#     permutation_test_runner = PermutationTest(func_distance=func_distance)
    
#     # launching Dask cluster at Local
#     local_cluster = LocalCluster(n_workers=3, threads_per_worker=10)
#     dask_client = local_cluster.get_client()
    
#     alg_param = RegressionCrossValidationAlgorithmParameter(n_subsampling=2)
#     dist_param = DistributedComputingParameter(dask_scheduler_address=local_cluster.scheduler_address)
    
#     training_parameter = RegressionCrossValidationTrainParameters(
#         algorithm_parameter=alg_param,
#         dist_parameter=dist_param,
#         computation_backend='dask'
#     )
    
#     runner = RegressionCrossValidationInterpretableVariableDetector(
#         training_parameter=training_parameter,
#         candidate_model_container=candidate_models,
#         training_dataset=dataset_obj_train,
#         validation_dataset=dataset_obj_dev,
#         seq_permutation_runner=[permutation_test_runner],
#     )
#     result = runner.run_cv_detection()
    
#     assert isinstance(result.stable_s_hat, list)
#     assert isinstance(result.array_s_hat, np.ndarray)
#     assert isinstance(result.stability_score_matrix, np.ndarray)
#     logger.debug(f'result: {result.stable_s_hat}')
