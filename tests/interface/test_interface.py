from pathlib import Path
import typing as ty

import logging

import numpy as np
import torch

import shutil

from tempfile import mkdtemp

import dacite
import toml

from mmd_tst_variable_detector.interface import interface, InterfaceConfigArgs
from mmd_tst_variable_detector.logger_unit import handler
from mmd_tst_variable_detector import (
    BasicVariableSelectionResult,
    RegularizationSearchParameters
)
from mmd_tst_variable_detector.interface.module_configs import (
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DistributedConfigArgs,
    CvSelectionConfigArgs,
    LinearVariableSelectionConfigArgs,
    BaselineMmdConfigArgs
)
from mmd_tst_variable_detector.interface.interface_config_args import DetectorAlgorithmConfigArgs
from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


def __gen_file_static_data_vector(path_tmp_save_root: Path) -> ty.List[Path]:
    seq_path_parent = []
    
    # generating 10 files.
    for i in range(10):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent.append(__path_parent)
        
        __x = np.random.normal(loc=1, size=(20,))
        __y = np.random.normal(loc=10, size=(20,))
        
        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
    # end for
    return seq_path_parent


def __gen_file_static_data_matrix(path_tmp_save_root: Path) -> ty.List[Path]:
    seq_path_parent = []
    
    # generating 10 files.
    for i in range(10):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent.append(__path_parent)
        
        __x = np.random.normal(loc=1, size=(5, 5))
        __y = np.random.normal(loc=5, size=(5, 5))
        
        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
    # end for
    return seq_path_parent




def test_file_backend_timeseries_dataset_sample_based(resource_path_root: Path) -> None:
    # pass: not ready yet.
    pass



def test_ram_backend_static_dataset_sample_based_linear_variable_selection(resource_path_root: Path) -> None:
    test_approach_variable_detector = 'linear_variable_selection'
    
    __sample_x, __sample_y, ground_truth = sampling_from_distribution(
        n_sample=100,
        dimension_size=5,
        mixture_rate=0.1,
        distribution_conf_p={'type': 'gaussian', 'mu': 0.0, 'sigma': 1.0},
        distribution_conf_q={'type': 'gaussian', 'mu': 5.0, 'sigma': 1.0}
    )
    
    sample_x, sample_y = torch.from_numpy(__sample_x), torch.from_numpy(__sample_y)
        
    config_args = InterfaceConfigArgs(
        resource_config_args=ResourceConfigArgs(
            distributed_config_detection=DistributedConfigArgs(distributed_mode='single')
        ),
        approach_config_args=ApproachConfigArgs(
            approach_data_representation='sample_based',
            approach_variable_detector=test_approach_variable_detector,
            approach_interpretable_mmd='cv_selection'),
        data_config_args=DataSetConfigArgs(
            data_x_train=sample_x,
            data_y_train=sample_y,
            data_x_test=None,
            data_y_test=None,
            dataset_type_backend='ram',
            dataset_type_charactersitic='static',),
        detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
            mmd_cv_selection_args=CvSelectionConfigArgs(max_epoch=200, 
                                                        n_subsampling=1,
                                                        parameter_search_parameter=RegularizationSearchParameters(
                                                            n_search_iteration=1,
                                                            n_regularization_parameter=1,),
                                                        dataloader_n_workers_train_dataloader=1,
                                                        dataloader_n_workers_validation_dataloader=1,
                                                        dataloader_persistent_workers=True),
            linear_variable_selection_args=LinearVariableSelectionConfigArgs(n_trials=2,
                                                                                n_cv=2,
                                                                                concurrent_limit=1),
        )
    )

    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    assert isinstance(config_args.resource_config_args.path_work_dir, Path)
    if config_args.resource_config_args.path_work_dir.exists():
        shutil.rmtree(config_args.resource_config_args.path_work_dir)


def test_ram_backend_static_dataset_sample_based_interpretable_mmd_cv(resource_path_root: Path) -> None:
    test_approach_variable_detector = 'interpretable_mmd'
    
    __sample_x, __sample_y, ground_truth = sampling_from_distribution(
        n_sample=10,
        dimension_size=5,
        mixture_rate=0.1,
        distribution_conf_p={'type': 'gaussian', 'mu': 0.0, 'sigma': 1.0},
        distribution_conf_q={'type': 'gaussian', 'mu': 5.0, 'sigma': 1.0}
    )
    
    sample_x, sample_y = torch.from_numpy(__sample_x), torch.from_numpy(__sample_y)
    
    config_args = InterfaceConfigArgs(
        resource_config_args=ResourceConfigArgs(
            distributed_config_detection=DistributedConfigArgs(distributed_mode='single'),
        ),
        approach_config_args=ApproachConfigArgs(
            approach_data_representation='sample_based',
            approach_variable_detector=test_approach_variable_detector,
            approach_interpretable_mmd='cv_selection'),
        data_config_args=DataSetConfigArgs(
            data_x_train=sample_x,
            data_y_train=sample_y,
            data_x_test=None,
            data_y_test=None,
            dataset_type_backend='ram',
            dataset_type_charactersitic='static',),
        detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
            mmd_cv_selection_args=CvSelectionConfigArgs(max_epoch=50,
                                                        parameter_search_parameter=RegularizationSearchParameters(
                                                            n_search_iteration=1,
                                                            n_regularization_parameter=1,),
                                                        n_subsampling=1,
                                                        n_permutation_test=10),
            linear_variable_selection_args=LinearVariableSelectionConfigArgs(
                n_trials=2,
                n_cv=2,
                concurrent_limit=1),
        )
    )

    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    assert isinstance(config_args.resource_config_args.path_work_dir, Path)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)


def test_ram_backend_static_dataset_sample_based_interpretable_mmd_baseline_mmd(resource_path_root: Path) -> None:
    test_approach_variable_detector = 'interpretable_mmd'
    
    __sample_x, __sample_y, ground_truth = sampling_from_distribution(
        n_sample=10,
        dimension_size=5,
        mixture_rate=0.1,
        distribution_conf_p={'type': 'gaussian', 'mu': 0.0, 'sigma': 1.0},
        distribution_conf_q={'type': 'gaussian', 'mu': 5.0, 'sigma': 1.0}
    )
    
    sample_x, sample_y = torch.from_numpy(__sample_x), torch.from_numpy(__sample_y)
    
    path_work_dir = Path(mkdtemp())
    path_work_dir.mkdir(parents=True, exist_ok=True)

    config_args = InterfaceConfigArgs(
        resource_config_args=ResourceConfigArgs(
            path_work_dir=path_work_dir,
            distributed_config_detection=DistributedConfigArgs(distributed_mode='single'),
        ),
        approach_config_args=ApproachConfigArgs(
            approach_data_representation='sample_based',
            approach_variable_detector=test_approach_variable_detector,
            approach_interpretable_mmd='baseline_mmd'),
        data_config_args=DataSetConfigArgs(
            data_x_train=sample_x,
            data_y_train=sample_y,
            data_x_test=None,
            data_y_test=None,
            dataset_type_backend='ram',
            dataset_type_charactersitic='static',),
        detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
            mmd_baseline_args=BaselineMmdConfigArgs(
                max_epoch=50,
            )
        )
    )

    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    shutil.rmtree(path_work_dir)



def test_ram_backend_static_dataset_sample_based_wasserstein_independence(resource_path_root: Path) -> None:
    test_approach_variable_detector = 'wasserstein_independence'

    __sample_x, __sample_y, ground_truth = sampling_from_distribution(
        n_sample=10,
        dimension_size=5,
        mixture_rate=0.1,
        distribution_conf_p={'type': 'gaussian', 'mu': 0.0, 'sigma': 1.0},
        distribution_conf_q={'type': 'gaussian', 'mu': 5.0, 'sigma': 1.0}
    )
    
    sample_x, sample_y = torch.from_numpy(__sample_x), torch.from_numpy(__sample_y)
        
    config_args = InterfaceConfigArgs(
        resource_config_args=ResourceConfigArgs(
            distributed_config_detection=DistributedConfigArgs(distributed_mode='single'),
        ),
        approach_config_args=ApproachConfigArgs(
            approach_data_representation='sample_based',
            approach_variable_detector=test_approach_variable_detector,
            approach_interpretable_mmd='cv_selection'),
        data_config_args=DataSetConfigArgs(
            data_x_train=sample_x,
            data_y_train=sample_y,
            data_x_test=None,
            data_y_test=None,
            dataset_type_backend='ram',
            dataset_type_charactersitic='static',),
        detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
            mmd_cv_selection_args=CvSelectionConfigArgs(max_epoch=10, 
                                                        parameter_search_parameter=RegularizationSearchParameters(
                                                            n_search_iteration=2,
                                                            n_regularization_parameter=1,),
                                                        n_subsampling=2,
                                                        dataloader_n_workers_train_dataloader=1,
                                                        dataloader_n_workers_validation_dataloader=1,
                                                        dataloader_persistent_workers=True),
            linear_variable_selection_args=LinearVariableSelectionConfigArgs(n_trials=2,
                                                                                n_cv=2,
                                                                                concurrent_limit=1),
        )
    )

    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    assert isinstance(config_args.resource_config_args.path_work_dir, Path)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)


def __set_config_test_file_backend_static_dataset_sample_based(resource_path_root: Path,
                                                               path_data_train: Path,
                                                               path_data_test: Path):
    path_toml = resource_path_root / 'test_interface_config.toml'
    __config_obj = toml.load(path_toml)
    __config_obj['approach_config_args']['approach_data_representation'] = 'sample_based'
    __config_obj['approach_config_args']['approach_variable_detector'] = 'interpretable_mmd'
    __config_obj['approach_config_args']['approach_interpretable_mmd'] = 'cv_selection'
    
    __config_obj['data_config_args']['dataset_type_backend'] = 'file'
    __config_obj['data_config_args']['dataset_type_charactersitic'] = 'static'
    __config_obj['approach_config_args']['approach_data_representation'] = 'sample_based'

    __config_obj['data_config_args']['data_x_train'] = path_data_train
    __config_obj['data_config_args']['data_y_train'] = path_data_train

    config_args = dacite.from_dict(data_class=InterfaceConfigArgs, data=__config_obj)
    
    config_args.data_config_args.data_x_test = path_data_test
    config_args.data_config_args.data_y_test = path_data_test
    

    mmd_cv_selection_args = CvSelectionConfigArgs(
        max_epoch=50,
        parameter_search_parameter=RegularizationSearchParameters(
            n_search_iteration=1,
            n_regularization_parameter=1,),
        n_subsampling=1,
        n_permutation_test=10
    )
    
    linear_variable_selection_args = LinearVariableSelectionConfigArgs(
        n_trials=2
    )

    config_args.detector_algorithm_config_args.mmd_cv_selection_args = mmd_cv_selection_args
    config_args.detector_algorithm_config_args.linear_variable_selection_args = linear_variable_selection_args

    return config_args


def test_file_backend_static_dataset_sample_based_wasserstein_independence(resource_path_root: Path) -> None:
    path_data_train = Path('/tmp/test_data/train')
    path_data_train.mkdir(parents=True, exist_ok=True)
    seq_input_train = __gen_file_static_data_vector(path_data_train)

    path_data_test = Path('/tmp/test_data/test')
    seq_input_test = __gen_file_static_data_vector(path_data_test)
        
    config_args = __set_config_test_file_backend_static_dataset_sample_based(
        resource_path_root=resource_path_root,
        path_data_train=path_data_train,
        path_data_test=path_data_test
    )

    
    config_args.approach_config_args.approach_variable_detector = 'wasserstein_independence'
    config_args.resource_config_args.path_work_dir = Path(mkdtemp())

    config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
        distributed_mode='single')
        
    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    
    shutil.rmtree(path_data_train.parent)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)
        


def test_file_backend_static_dataset_sample_based_linear_variable_selection(resource_path_root: Path) -> None:
    path_data_train = Path('/tmp/test_data/train')
    path_data_train.mkdir(parents=True, exist_ok=True)
    seq_input_train = __gen_file_static_data_vector(path_data_train)

    path_data_test = Path('/tmp/test_data/test')
    seq_input_test = __gen_file_static_data_vector(path_data_test)
        
    config_args = __set_config_test_file_backend_static_dataset_sample_based(
        resource_path_root=resource_path_root,
        path_data_train=path_data_train,
        path_data_test=path_data_test
    )

    
    config_args.approach_config_args.approach_variable_detector = 'linear_variable_selection'
    config_args.resource_config_args.path_work_dir = Path(mkdtemp())
    
    config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
        distributed_mode='single')
    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    
    shutil.rmtree(path_data_train.parent)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)
    
    
def test_file_backend_static_dataset_sample_based_interpretable_mmd(resource_path_root: Path) -> None:
    path_data_train = Path('/tmp/test_data/train')
    path_data_train.mkdir(parents=True, exist_ok=True)
    seq_input_train = __gen_file_static_data_vector(path_data_train)

    path_data_test = Path('/tmp/test_data/test')
    seq_input_test = __gen_file_static_data_vector(path_data_test)
        
    config_args = __set_config_test_file_backend_static_dataset_sample_based(
        resource_path_root=resource_path_root,
        path_data_train=path_data_train,
        path_data_test=path_data_test
    )

    
    config_args.approach_config_args.approach_variable_detector = 'interpretable_mmd'
    config_args.resource_config_args.path_work_dir = Path(mkdtemp())
    
    config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
        distributed_mode='single')
    
    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    
    shutil.rmtree(path_data_train.parent)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)
    
    
def test_flexible_file_backend_static_dataset_sample_based_wasserstein_independence(resource_path_root: Path) -> None:
    path_data_train = Path('/tmp/test_data/train')
    path_data_train.mkdir(parents=True, exist_ok=True)
    seq_input_train = __gen_file_static_data_vector(path_data_train)

    path_data_test = Path('/tmp/test_data/test')
    seq_input_test = __gen_file_static_data_vector(path_data_test)
        
    config_args = __set_config_test_file_backend_static_dataset_sample_based(
        resource_path_root=resource_path_root,
        path_data_train=path_data_train,
        path_data_test=path_data_test
    )
    
    config_args.data_config_args.dataset_type_backend = 'flexible-file'
    config_args.approach_config_args.approach_variable_detector = 'wasserstein_independence'
    config_args.resource_config_args.path_work_dir = Path(mkdtemp())
    
    config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
        distributed_mode='single')
    
    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    
    shutil.rmtree(path_data_train.parent)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)


def test_flexible_file_backend_static_dataset_sample_based_interpretable_mmd(resource_path_root: Path) -> None:
    path_data_train = Path('/tmp/test_data/train')
    path_data_train.mkdir(parents=True, exist_ok=True)
    seq_input_train = __gen_file_static_data_vector(path_data_train)

    path_data_test = Path('/tmp/test_data/test')
    seq_input_test = __gen_file_static_data_vector(path_data_test)
        
    config_args = __set_config_test_file_backend_static_dataset_sample_based(
        resource_path_root=resource_path_root,
        path_data_train=path_data_train,
        path_data_test=path_data_test
    )
    
    config_args.data_config_args.dataset_type_backend = 'flexible-file'
    config_args.approach_config_args.approach_variable_detector = 'interpretable_mmd'
    config_args.resource_config_args.path_work_dir = Path(mkdtemp())
    
    config_args.detector_algorithm_config_args
    
    config_args.resource_config_args.dask_config_detection = DistributedConfigArgs(
        distributed_mode='single')
    
    # Create the interface
    interface_instance = interface.Interface(config_args=config_args)
    interface_instance.fit()
    res_object = interface_instance.get_result()
    
    assert isinstance(res_object.detection_result_sample_based, BasicVariableSelectionResult)
    assert isinstance(res_object.detection_result_sample_based.weights, np.ndarray)
    assert isinstance(res_object.detection_result_sample_based.variables, list)
    assert isinstance(res_object.detection_result_sample_based.p_value, float)
    logger.debug(f'p_value={res_object.detection_result_sample_based.p_value}')
    # getting result
    
    shutil.rmtree(path_data_train.parent)
    shutil.rmtree(config_args.resource_config_args.path_work_dir)


if __name__ == '__main__':
    # for debugging speed
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    path_resource_root = Path('../testresources')
