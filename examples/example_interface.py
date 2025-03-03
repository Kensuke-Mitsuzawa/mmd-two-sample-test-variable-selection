from mmd_tst_variable_detector import Interface
from mmd_tst_variable_detector import  InterfaceConfigArgs

import shutil
import toml
import dacite

import typing as ty

import json

import torch
import numpy as np

from pathlib import Path

import logzero
logger = logzero.logger


"""This is an example of High-Level API"""


def __generate_sample_data_file_static_data_matrix(path_tmp_save_root: Path) -> ty.List[Path]:
    """This function generates 10 files. Each file contains 2D array of (32, 32).
    You can interpret each array as `image data`, which is noramlly (32, 32) or (64, 64).
    """
    seq_path_parent = []
    
    # generating 100 files.
    for i in range(100):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent.append(__path_parent)
        
        __x = np.random.normal(loc=1, size=(32, 32))
        __y_base = np.random.normal(loc=1, size=(32, 32))
        
        noised_pixel = [(5, 5), (10, 10), (20, 20)]
        for t_noise_pixel in noised_pixel:
            __y_base[t_noise_pixel] = np.random.normal(loc=1, scale=10)
        # end for
        
        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y_base)}, (__path_parent / 'y.pt').as_posix())
        
        noised_vector = np.ravel_multi_index([[t[0] for t in noised_pixel], [t[1] for t in noised_pixel]], (32, 32))
    # end for
    logger.debug(f'noised_vector={noised_vector}')
    return seq_path_parent


def example():
    path_config_toml = './config_example.toml'
    
    # You have two ways to set the config.
    # 1. loading your configurations from a toml file (this example).
    # 2. setting parameters directly to `InterfaceConfigArgs`.
    
    config_args = dacite.from_dict(
        data=toml.load(path_config_toml),
        data_class=InterfaceConfigArgs)
    
    # generating example data. Of course, you have your own data normally.
    path_train_data = __generate_sample_data_file_static_data_matrix(path_tmp_save_root=config_args.data_config_args.data_x_train)
    # Test means "Two-Sample-Test" in this context. You can leave it None. 
    path_test_data = __generate_sample_data_file_static_data_matrix(path_tmp_save_root=config_args.data_config_args.data_x_test)
    
    # Create the interface
    interface_instance = Interface(config_args=config_args)
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
    
    shutil.rmtree(config_args.data_config_args.data_x_train)
    shutil.rmtree(config_args.data_config_args.data_x_test)
    
    # if you wanna save in json
    f_out = Path('/tmp/result.json')
    with f_out.open('w') as f:
        f.write(res_object.as_json())
    

if __name__ == '__main__':
    example()
