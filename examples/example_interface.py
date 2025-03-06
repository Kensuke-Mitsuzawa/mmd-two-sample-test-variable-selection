from mmd_tst_variable_detector import Interface
from mmd_tst_variable_detector import  InterfaceConfigArgs
from mmd_tst_variable_detector import (
    InterfaceConfigArgs,
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DetectorAlgorithmConfigArgs,
    CvSelectionConfigArgs,
    BasicVariableSelectionResult,
    OutputObject,
    DistributedConfigArgs,
    RegularizationSearchParameters
)


import shutil

import typing as ty

import json

import torch
import numpy as np

from pathlib import Path

import logzero
logger = logzero.logger


"""This is an example of High-Level API"""


def generate_sample_data_static_data_matrix(
        random_seed: int = 42,
        n_samples: int = 100,
        ) -> ty.Tuple[torch.Tensor, torch.Tensor]:
    """This function generates 10 files. Each file contains 2D array of (32, 32).
    You can interpret each array as `image data`, which is noramlly (32, 32) or (64, 64).
    """
    seq_x = []
    seq_y = []
    
    random_generator = np.random.Generator(np.random.PCG64(random_seed))
    noised_pixel = [(5, 5), (10, 10), (20, 20)]

    for i in range(n_samples):        
        __x = np.random.normal(loc=1, size=(32, 32))
        __y_base = np.random.normal(loc=1, size=(32, 32))
        
        for t_noise_pixel in noised_pixel:
            __y_base[t_noise_pixel] = random_generator.normal(loc=1, scale=10)
        # end for
        
        seq_x.append(__x)
        seq_y.append(__y_base)
        
        noised_vector = np.ravel_multi_index([[t[0] for t in noised_pixel], [t[1] for t in noised_pixel]], (32, 32))
    # end for
    logger.debug(f'noised_vector={noised_vector}')

    torch_x = torch.tensor(seq_x, dtype=torch.float32)
    torch_y = torch.tensor(seq_y, dtype=torch.float32)
    return torch_x, torch_y


def get_api_configurations(tensor_x_train: torch.Tensor, 
                           tensor_y_train: torch.Tensor, 
                           tensor_x_test: torch.Tensor, 
                           tensor_y_test: torch.Tensor,
                           path_work_dir: Path,
                           train_accelerator: str = 'cpu'
                           ) -> InterfaceConfigArgs:
    """This function sets configuration objects for the interface.
    You can define the configuration objects in a toml file.
    When you use the toml file, you can use `dacite` to load the configurations.
    For example,
    `dacite.from_dict(data=toml.load(path_config_toml), data_class=InterfaceConfigArgs)`
    """
    data_config_args = DataSetConfigArgs(
            data_x_train=tensor_x_train,
            data_y_train=tensor_y_train,
            data_x_test=tensor_x_test,
            data_y_test=tensor_y_test,
            dataset_type_backend='ram',
            dataset_type_charactersitic='static')

    # configuration about distributed-computing.
    # you can choose dask when you want to use distributed computing.
    distributed_config = DistributedConfigArgs(
        distributed_mode='single',
        dask_scheduler_host=None)

    # parameters for regularisation search.
    parameter_search_parameter = RegularizationSearchParameters(
        n_regularization_parameter=3,
        n_search_iteration=5,
        max_concurrent_job=2)

    interface_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                train_accelerator=train_accelerator,
                path_work_dir=path_work_dir,
                distributed_config_detection=distributed_config),  #comment: 8 threads is best choice.
            approach_config_args=ApproachConfigArgs(
                approach_data_representation='sample_based',
                approach_variable_detector='interpretable_mmd',
                approach_interpretable_mmd='cv_selection'),
            data_config_args=data_config_args,
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                mmd_cv_selection_args=CvSelectionConfigArgs(
                    max_epoch=9999,
                    parameter_search_parameter=parameter_search_parameter,
                    n_subsampling=3,
                ))
        )

    return interface_args


def example():
    path_work_dir = Path('/tmp/mmd_tst_variable_detector')

    # # generating example data. Of course, you have your own data normally.
    # path_train_data = __generate_sample_data_file_static_data_matrix(path_tmp_save_root=config_args.data_config_args.data_x_train)
    # # Test means "Two-Sample-Test" in this context. You can leave it None. 
    # path_test_data = __generate_sample_data_file_static_data_matrix(path_tmp_save_root=config_args.data_config_args.data_x_test)
    tensor_x_train, tensor_y_train = generate_sample_data_static_data_matrix(random_seed=42)
    tensor_x_test, tensor_y_test = generate_sample_data_static_data_matrix(random_seed=24)

    # You have two ways to set the config.
    # 1. loading your configurations from a toml file (this example).
    # 2. setting parameters directly to `InterfaceConfigArgs`.
    interface_configs = get_api_configurations(
        tensor_x_train, 
        tensor_y_train, 
        tensor_x_test, 
        tensor_y_test, 
        path_work_dir, 
        train_accelerator='auto')
    
    # Create the interface
    interface_instance = Interface(config_args=interface_configs)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert res_object.detection_result_sample_based is not None
    var_weights: np.ndarray = res_object.detection_result_sample_based.weights
    detected_variables: ty.List[int] = res_object.detection_result_sample_based.variables
    p_value: float = res_object.detection_result_sample_based.p_value
    
    # var_weights is flattened 1D array. When you wanna 2D array, you can reshape it.
    weights_pixel_form = np.reshape(var_weights, (32, 32))
    noise_detection_pixel = list(zip(*np.unravel_index(detected_variables, (32, 32))))
    
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    logger.debug(f'detected pixels={noise_detection_pixel}')
    
    # if you wanna save in json
    f_out = Path('/tmp/result.json')
    with f_out.open('w') as f:
        f.write(res_object.as_json())
    

if __name__ == '__main__':
    example()
