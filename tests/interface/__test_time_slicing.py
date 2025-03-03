# from pathlib import Path
# import typing as ty
# import os

# import logging

# import numpy as np
# import torch

# import shutil

# import dacite
# import toml

# from mmd_tst_variable_detector.interface import (
#     interface, 
#     InterfaceConfigArgs, 
#     DistributedConfigArgs, 
#     AlgorithmOneConfigArgs,
#     CvSelectionConfigArgs,
#     RegularizationSearchParameters)
# from mmd_tst_variable_detector.logger_unit import handler
# from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution


# logger = logging.getLogger(f'{__package__}.{__name__}')
# logger.addHandler(handler)



# def __gen_file_st_timeslicing_trajectory(path_tmp_save_root: Path) -> ty.Tuple[Path, Path]:
    
#     __path_parent = path_tmp_save_root
#     __path_parent.mkdir(parents=True, exist_ok=True)
        
#     __x = np.random.random(size=(10, 100, 2))
#     __y = np.random.random(size=(10, 100, 2))
    
#     torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
#     torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
#     # end for
#     return __path_parent / 'x.pt', __path_parent / 'y.pt'




# def __gen_file_st_sensor(path_tmp_save_root: Path) -> ty.Tuple[Path, Path]:
    
#     __path_parent = path_tmp_save_root
#     __path_parent.mkdir(parents=True, exist_ok=True)
        
#     __x = np.random.random(size=(10, 200))
#     __y = np.random.random(size=(10, 200))
    
#     torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
#     torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
#     # end for
#     return __path_parent / 'x.pt', __path_parent / 'y.pt'



# def test_file_backend_timeseries_dataset_timeslicing_sensor_st_interpretable_mmd(resource_path_root: Path) -> None:
#     test_approach_variable_detector = 'interpretable_mmd'
    
#     path_data_train = Path('/tmp/test_data/train')
#     path_data_train.mkdir(parents=True, exist_ok=True)
#     path_train_x, path_train_y = __gen_file_st_sensor(path_data_train)

#     path_data_test = Path('/tmp/test_data/test')
#     path_test_x, path_test_y = __gen_file_st_sensor(path_data_test)

#     path_toml = resource_path_root / 'test_interface_config.toml'
#     __config_obj = toml.load(path_toml)
        
#     __config_obj['approach_config_args']['approach_data_representation'] = 'time_slicing'
#     __config_obj['approach_config_args']['approach_variable_detector'] = 'interpretable_mmd'
#     __config_obj['approach_config_args']['approach_interpretable_mmd'] = 'algorithm_one'
    
#     __config_obj['data_config_args']['data_x_train'] = path_train_x.as_posix()
#     __config_obj['data_config_args']['data_y_train'] = path_train_y.as_posix()
    
#     __config_obj['data_config_args']['data_x_test'] = path_test_x.as_posix()
#     __config_obj['data_config_args']['data_y_test'] = path_test_y.as_posix()
    
#     __config_obj['data_config_args']['dataset_type_backend'] = 'file'
#     __config_obj['data_config_args']['dataset_type_charactersitic'] = 'sensor_st'
#     __config_obj['data_config_args']['time_slicing_per'] = 100
        
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     config_args = dacite.from_dict(data_class=InterfaceConfigArgs, data=__config_obj)
#     config_args.resource_config_args.path_work_dir = Path(f'/tmp/mmd-tst-variable-detector/interface/test_file_backend_timeseries_dataset_timeslicing_sensor_st/{test_approach_variable_detector}')
#     config_args.detector_algorithm_config_args.mmd_algorithm_one_args = AlgorithmOneConfigArgs(
#         max_epoch=10,
#         parameter_search_parameter=RegularizationSearchParameters(
#             n_regularization_parameter=1,
#             n_search_iteration=1,
#             max_concurrent_job=1,
#         ),
#         test_distance_functions=('sinkhorn',))
#     config_args.resource_config_args.dask_config_preprocessing = DistributedConfigArgs(
#         distributed_mode='single')
#     config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
#         distributed_mode='single')
#     config_args.data_config_args.time_slicing_per = 100
#     # Create the interface
#     interface_instance = interface.Interface(config_args=config_args)
#     interface_instance.fit()
#     res_object = interface_instance.get_result()
    
#     assert isinstance(res_object.detection_result_time_slicing, list)
#     assert len(res_object.detection_result_time_slicing) >= 2, f'This test expects 2 time-slicing outputs, but got {len(res_object.detection_result_time_slicing)}'
#     assert isinstance(res_object.detection_result_time_slicing[0].weights, np.ndarray)
#     assert len(res_object.detection_result_time_slicing[0].weights) == 10, f'The weight array must be 20 length.'
#     assert isinstance(res_object.detection_result_time_slicing[0].variables, list)
#     assert isinstance(res_object.detection_result_time_slicing[0].p_value, float)
#     logger.debug(f'p_value={res_object.detection_result_time_slicing[0].p_value}')
#     # getting result
#     shutil.rmtree(config_args.resource_config_args.path_work_dir)

#     shutil.rmtree(path_data_train.parent)


# def test_file_backend_timeseries_dataset_timeslicing_sensor_st_wasserstein_independence(resource_path_root: Path) -> None:
#     test_approach_variable_detector = 'wasserstein_independence' 
    
#     path_data_train = Path('/tmp/test_data/train')
#     path_data_train.mkdir(parents=True, exist_ok=True)
#     path_train_x, path_train_y = __gen_file_st_sensor(path_data_train)

#     path_data_test = Path('/tmp/test_data/test')
#     path_test_x, path_test_y = __gen_file_st_sensor(path_data_test)

#     path_toml = resource_path_root / 'test_interface_config.toml'
#     __config_obj = toml.load(path_toml)
        
#     __config_obj['approach_config_args']['approach_data_representation'] = 'time_slicing'
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     __config_obj['approach_config_args']['approach_interpretable_mmd'] = 'algorithm_one'
    
#     __config_obj['data_config_args']['data_x_train'] = path_train_x.as_posix()
#     __config_obj['data_config_args']['data_y_train'] = path_train_y.as_posix()
    
#     __config_obj['data_config_args']['data_x_test'] = path_test_x.as_posix()
#     __config_obj['data_config_args']['data_y_test'] = path_test_y.as_posix()
    
#     __config_obj['data_config_args']['dataset_type_backend'] = 'file'
#     __config_obj['data_config_args']['dataset_type_charactersitic'] = 'sensor_st'
#     __config_obj['data_config_args']['time_slicing_per'] = 100
    
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     config_args = dacite.from_dict(data_class=InterfaceConfigArgs, data=__config_obj)
#     config_args.resource_config_args.path_work_dir = Path(f'/tmp/mmd-tst-variable-detector/interface/test_file_backend_timeseries_dataset_timeslicing_sensor_st/{test_approach_variable_detector}')
#     config_args.resource_config_args.dask_config_preprocessing = DistributedConfigArgs(
#         distributed_mode='single')
#     config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
#         distributed_mode='single')    
#     config_args.data_config_args.time_slicing_per = 100
    
#     # Create the interface
#     interface_instance = interface.Interface(config_args=config_args)
#     interface_instance.fit()
#     res_object = interface_instance.get_result()
    
#     assert isinstance(res_object.detection_result_time_slicing, list)
#     assert len(res_object.detection_result_time_slicing) >= 2, f'This test expects 2 time-slicing outputs, but got {len(res_object.detection_result_time_slicing)}'
#     assert isinstance(res_object.detection_result_time_slicing[0].weights, np.ndarray)
#     assert len(res_object.detection_result_time_slicing[0].weights) == 10, f'The weight array must be 20 length.'
#     assert isinstance(res_object.detection_result_time_slicing[0].variables, list)
#     assert isinstance(res_object.detection_result_time_slicing[0].p_value, float)
#     logger.debug(f'p_value={res_object.detection_result_time_slicing[0].p_value}')
#     # getting result
#     shutil.rmtree(config_args.resource_config_args.path_work_dir)

#     shutil.rmtree(path_data_train.parent)


# def test_file_backend_timeseries_dataset_timeslicing_trajectory_interpretable_mmd(resource_path_root: Path) -> None:
#     test_approach_variable_detector = 'interpretable_mmd'
    
#     path_data_train = Path('/tmp/test_data/train')
#     path_data_train.mkdir(parents=True, exist_ok=True)
#     path_train_x, path_train_y = __gen_file_st_timeslicing_trajectory(path_data_train)

#     path_data_test = Path('/tmp/test_data/test')
#     path_test_x, path_test_y = __gen_file_st_timeslicing_trajectory(path_data_test)

#     path_toml = resource_path_root / 'test_interface_config.toml'
#     __config_obj = toml.load(path_toml)
#     __config_obj['approach_config_args']['approach_data_representation'] = 'time_slicing'
#     __config_obj['approach_config_args']['approach_variable_detector'] = 'interpretable_mmd'
#     __config_obj['approach_config_args']['approach_interpretable_mmd'] = 'algorithm_one'
    
#     __config_obj['data_config_args']['data_x_train'] = path_train_x.as_posix()
#     __config_obj['data_config_args']['data_y_train'] = path_train_y.as_posix()
    
#     __config_obj['data_config_args']['data_x_test'] = path_test_x.as_posix()
#     __config_obj['data_config_args']['data_y_test'] = path_test_y.as_posix()
    
#     __config_obj['data_config_args']['dataset_type_backend'] = 'file'
#     __config_obj['data_config_args']['dataset_type_charactersitic'] = 'trajectory_st'
#     __config_obj['data_config_args']['time_slicing_per'] = 50
            
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     config_args = dacite.from_dict(data_class=InterfaceConfigArgs, data=__config_obj)
#     config_args.detector_algorithm_config_args.mmd_algorithm_one_args = AlgorithmOneConfigArgs(
#         max_epoch=10,
#         parameter_search_parameter=RegularizationSearchParameters(
#             n_regularization_parameter=1,
#             n_search_iteration=1,
#             max_concurrent_job=1,            
#         ),
#         test_distance_functions=('sinkhorn',))
#     config_args.resource_config_args.dask_config_preprocessing = DistributedConfigArgs(
#         distributed_mode='single')
#     config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
#         distributed_mode='single')    

#     # Create the interface
#     interface_instance = interface.Interface(config_args=config_args)
#     interface_instance.fit()
#     res_object = interface_instance.get_result()
    
#     assert isinstance(res_object.detection_result_time_slicing, list)
#     assert len(res_object.detection_result_time_slicing) >= 2, f'This test expects 2 time-slicing outputs, but got {len(res_object.detection_result_time_slicing)}'
#     assert isinstance(res_object.detection_result_time_slicing[0].weights, np.ndarray)
#     assert len(res_object.detection_result_time_slicing[0].weights) == 10, f'The weight array must be 20 length.'
#     assert isinstance(res_object.detection_result_time_slicing[0].variables, list)
#     assert isinstance(res_object.detection_result_time_slicing[0].p_value, float)
#     logger.debug(f'p_value={res_object.detection_result_time_slicing[0].p_value}')
#     # getting result
#     # end for
    
#     shutil.rmtree(path_data_train.parent)
#     shutil.rmtree(config_args.resource_config_args.path_work_dir.as_posix())
    

# def test_file_backend_timeseries_dataset_timeslicing_trajectory_wasserstein_independence(resource_path_root: Path) -> None:
#     test_approach_variable_detector = 'wasserstein_independence'
    
#     path_data_train = Path('/tmp/test_data/train')
#     path_data_train.mkdir(parents=True, exist_ok=True)
#     path_train_x, path_train_y = __gen_file_st_timeslicing_trajectory(path_data_train)

#     path_data_test = Path('/tmp/test_data/test')
#     path_test_x, path_test_y = __gen_file_st_timeslicing_trajectory(path_data_test)

#     path_toml = resource_path_root / 'test_interface_config.toml'
#     __config_obj = toml.load(path_toml)
#     __config_obj['approach_config_args']['approach_data_representation'] = 'time_slicing'
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     __config_obj['approach_config_args']['approach_interpretable_mmd'] = 'cv_selection'
    
#     __config_obj['data_config_args']['data_x_train'] = path_train_x.as_posix()
#     __config_obj['data_config_args']['data_y_train'] = path_train_y.as_posix()
    
#     __config_obj['data_config_args']['data_x_test'] = path_test_x.as_posix()
#     __config_obj['data_config_args']['data_y_test'] = path_test_y.as_posix()
    
#     __config_obj['data_config_args']['dataset_type_backend'] = 'file'
#     __config_obj['data_config_args']['dataset_type_charactersitic'] = 'trajectory_st'
    
#     __config_obj['approach_config_args']['approach_variable_detector'] = test_approach_variable_detector
#     config_args = dacite.from_dict(data_class=InterfaceConfigArgs, data=__config_obj)
#     config_args.data_config_args.time_slicing_per = 50
#     config_args.detector_algorithm_config_args.mmd_cv_selection_args = CvSelectionConfigArgs(
#         max_epoch=10,
#         n_subsampling=1,
#         parameter_search_parameter=RegularizationSearchParameters(
#             n_regularization_parameter=1,
#             n_search_iteration=1,
#             max_concurrent_job=1,            
#         ),
#         test_distance_functions=('sinkhorn',))
    

#     config_args.resource_config_args.dask_config_preprocessing = DistributedConfigArgs(
#         distributed_mode='single')
#     config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
#         distributed_mode='single')

#     # Create the interface
#     interface_instance = interface.Interface(config_args=config_args)
#     interface_instance.fit()
#     res_object = interface_instance.get_result()
    
#     assert isinstance(res_object.detection_result_time_slicing, list)
#     assert len(res_object.detection_result_time_slicing) >= 2, f'This test expects 2 time-slicing outputs, but got {len(res_object.detection_result_time_slicing)}'    
#     assert isinstance(res_object.detection_result_time_slicing[0].weights, np.ndarray)
#     assert len(res_object.detection_result_time_slicing[0].weights) == 10, f'The weight array must be 20 length.'
#     assert isinstance(res_object.detection_result_time_slicing[0].variables, list)
#     assert isinstance(res_object.detection_result_time_slicing[0].p_value, float)
#     logger.debug(f'p_value={res_object.detection_result_time_slicing[0].p_value}')
#     # getting result
#     # end for
    
#     shutil.rmtree(path_data_train.parent)
#     shutil.rmtree(config_args.resource_config_args.path_work_dir.as_posix())


# if __name__ == '__main__':
#     # for debugging speed
#     import logging
#     logging.basicConfig(level=logging.DEBUG)
    
#     path_resource_root = Path('../testresources')
    
#     # test_file_backend_timeseries_dataset_timeslicing_sensor_st_interpretable_mmd(path_resource_root)
#     # test_file_backend_timeseries_dataset_timeslicing_trajectory_interpretable_mmd(path_resource_root)