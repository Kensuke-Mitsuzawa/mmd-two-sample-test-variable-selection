# import logging
# import functools

# import torch
# import geomloss

# from distributed import Client, LocalCluster

# from sklearn.linear_model import LogisticRegression, Ridge, ARDRegression

# from mmd_tst_variable_detector.baselines.regression_based_variable_selection import algorithm_one
# from mmd_tst_variable_detector import (
#     SimpleDataset,
#     PermutationTest
# )

# from ...data_generator import test_data_xy_linear


# logger = logging.getLogger(f'test.{__name__}')


# def __func_distance_sinkhorn(x: torch.Tensor, y: torch.Tensor, geomloss_obj) -> torch.Tensor:
#     return geomloss_obj(x, y)


# def test_algorithm_one():
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
#         regression_models=[LogisticRegression(), Ridge(), ARDRegression()]
#     )
    
#     local_cluster = LocalCluster(n_workers=2, threads_per_worker=10)
#     dask_client = local_cluster.get_client()
    
#     func_distance = functools.partial(__func_distance_sinkhorn,
#                                       geomloss_obj=geomloss.SamplesLoss(loss='sinkhorn'))
    
#     permutation_test_runner = PermutationTest(func_distance=func_distance)
    
#     algorithm_one_result = algorithm_one.detection_algorithm_one(
#         dataset_training=dataset_obj_train,
#         dataset_dev=dataset_obj_dev,
#         candidate_models=candidate_models,
#         dask_client=dask_client,
#         seq_permutation_test_runner_base=[permutation_test_runner])
#     assert algorithm_one_result.selected_model is not None
    
#     logger.debug(f'p-value of algorithm_one_result.selected_model: {algorithm_one_result.selected_model.seq_p_value_dev}')