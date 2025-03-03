from pathlib import Path
import typing as ty
import logging
import timeit

import numpy as np
import torch

from distributed import Client

from ...detection_algorithm.utils import permutation_tests

from ...utils import PostProcessLoggerHandler
# datasets
from ...datasets import BaseDataset
from ...datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset

# CV selection modules
from ...mmd_estimator import QuadraticMmdEstimator
from ...kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from ...distance_module import L2Distance
from ...detection_algorithm.cross_validation_detector import (
    CrossValidationInterpretableVariableDetector,
    CrossValidationTrainedParameter,
    # InterpretableMmdTrainParameters,
    # DistributedComputingParameter,
    # CrossValidationAlgorithmParameter,
    # CrossValidationTrainParameters,
)
# from ...detection_algorithm.search_regularization_min_max import RegularizationSearchParameters
# from ...weights_initialization.weights_initialization import weights_initialization

# Algorithm one
from ...detection_algorithm import detection_algorithm_one, AlgorithmOneResult

from ...assessment_helper import default_settings

from .. import data_objects
from .. import module_mmd_config

from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


def __main_run(dataset_train: BaseDataset,
               training_conf_args: data_objects.InterfaceConfigArgs,
               path_dir_model: Path,
               path_dir_ml_logger: Path,
               dask_client: ty.Optional[Client] = None,
               seed_root_random: int = 42,
               ) -> ty.Union[AlgorithmOneResult, CrossValidationTrainedParameter]:
    mmd_config_args = module_mmd_config.get_mmd_config_args(training_conf_args)

    # comment: for the RAM memory management reasons, I define another variable `dataset_train_for_preprocess`.
    # The content is totally same as `dataset_train`.
    # if isinstance(dataset_train, FileBackendOneTimeLoadStaticDataset):
    if dataset_train.is_dataset_on_ram():
        dataset_train_for_preprocess = dataset_train.generate_dataset_on_ram()
    else:
        dataset_train_for_preprocess = dataset_train
    # end if
    
    # get initial weights.
    initial_value = module_mmd_config.get_initial_weights_before_search(dataset=dataset_train_for_preprocess, dask_client=dask_client)
    # initialization of Kernel instance.
    # initialization of MMD instance.
    kernel = QuadraticKernelGaussianKernel.from_dataset(
        dataset_train_for_preprocess, 
        ard_weights=torch.from_numpy(initial_value),
        heuristic_operation=mmd_config_args.aggregation_kernel_length_scale)
    mmd_estimator = QuadraticMmdEstimator(kernel)
        
    # comment: optuna search exists already in CV intergace.
    # No need to execute it manually.
    if dask_client is not None:
        dask_scheduler_address = dask_client.scheduler.address if dask_client is not None else None
    else:
        dask_scheduler_address = ''
    # end if
    
    if mmd_config_args.setting_name == 'config_rapid':
        __class_config = module_mmd_config.ConfigRapid()
        cv_train_param, pl_param = __class_config.get_configs(path_dir_model, dask_scheduler_address, training_conf_args)
    elif mmd_config_args.setting_name == 'config_tpami_draft':
        __class_config = module_mmd_config.ConfigTPamiDraft()
        cv_train_param, pl_param = __class_config.get_configs(path_dir_model, dask_scheduler_address, training_conf_args)
    elif mmd_config_args.setting_name == 'custom':
        assert mmd_config_args.custom_mmd_variable_selection_config is not None, 'custom_mmd_variable_selection_config is not given.'
        assert isinstance(mmd_config_args.custom_mmd_variable_selection_config, module_mmd_config.MmdOptimisationConfigTemplate), 'Invalid Class objects. Type Error.'
        cv_train_param, pl_param = mmd_config_args.custom_mmd_variable_selection_config.get_configs(path_dir_model, dask_scheduler_address, training_conf_args)    # type: ignore
    else:
        raise ValueError(f'Unexpected setting name: {mmd_config_args.setting_name}')
    # end if
    
    post_process_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={
            "mlflow": {
                "save_dir": path_dir_ml_logger.as_posix(), 
                "tracking_uri": f"file://{path_dir_ml_logger.as_posix()}" 
            }})
    
    
    if isinstance(mmd_config_args, data_objects.CvSelectionConfigArgs):
        detector = CrossValidationInterpretableVariableDetector(
            pytorch_trainer_config=pl_param,
            training_parameter=cv_train_param,
            estimator=mmd_estimator,
            post_process_handler=post_process_handler,
            seed_root_random=seed_root_random)
        res = detector.run_cv_detection(dataset_train)
    elif isinstance(mmd_config_args, data_objects.AlgorithmOneConfigArgs):
        # comment: I take `cv_config` for common.
        logger.debug('Running Algorithm One...')
        # spliting dataset into train/dev
        
        data_split_obj = dataset_train.split_train_and_test(
            train_ratio=mmd_config_args.train_dev_split_ratio,
            random_seed=seed_root_random)
        __data_train = data_split_obj.train_dataset
        __data_dev = data_split_obj.test_dataset
        
        __regularization_search_parameter = mmd_config_args.parameter_search_parameter
        res = detection_algorithm_one(
            dataset_training=__data_train,
            dataset_dev=__data_dev,
            mmd_estimator=mmd_estimator,
            base_training_parameter=cv_train_param.base_training_parameter,
            pytorch_trainer_config=pl_param,
            post_process_handler=post_process_handler,
            regularization_search_parameter=__regularization_search_parameter,
            candidate_regularization_parameters=mmd_config_args.approach_regularization_parameter,
            test_distance_functions=tuple(mmd_config_args.test_distance_functions),
            n_permutation_test=mmd_config_args.n_permutation_test,
            dask_client=dask_client)
    else:
        raise ValueError(f'Unexpected mmd_config_args type: {type(mmd_config_args)}')
    # end if
    
    return res


def main(config_args: data_objects.InterfaceConfigArgs,
         dataset_train: BaseDataset,
         dask_client: ty.Optional[Client] = None, 
         dataset_test: ty.Optional[BaseDataset] = None,) -> data_objects.BasicVariableSelectionResult:
    assert (config_args.detector_algorithm_config_args.mmd_cv_selection_args is not None) or (config_args.detector_algorithm_config_args.mmd_algorithm_one_args is not None),\
        'Either of "mmd_cv_selection_args" or "mmd_algorithm_one_args" must be given.'
    if config_args.detector_algorithm_config_args.mmd_cv_selection_args is not None:
        assert isinstance(config_args.detector_algorithm_config_args.mmd_cv_selection_args, data_objects.CvSelectionConfigArgs), 'mmd_cv_selection_args is not given.'
    if config_args.detector_algorithm_config_args.mmd_algorithm_one_args is not None and config_args.detector_algorithm_config_args.mmd_algorithm_one_args != '':
        assert isinstance(config_args.detector_algorithm_config_args.mmd_algorithm_one_args, data_objects.AlgorithmOneConfigArgs), 'mmd_algorithm_one_args is not given.'
    # end if
    
    path_resource_root = Path(config_args.resource_config_args.path_work_dir)
    path_dir_model = path_resource_root / config_args.resource_config_args.dir_name_model
    path_dir_ml_logger = path_resource_root / config_args.resource_config_args.dir_name_ml_logger
    
    res = __main_run(
        dataset_train, 
        training_conf_args=config_args,
        path_dir_model=path_dir_model,
        path_dir_ml_logger=path_dir_ml_logger,
        dask_client=dask_client)

    if dataset_test is not None:
        if isinstance(dataset_test, FileBackendOneTimeLoadStaticDataset):
            dataset_test = dataset_test.generate_dataset_on_ram()
        # end if
        
        seq_tst_result = permutation_tests.permutation_tests(
            dataset_test=dataset_test,
            variable_selection_approach='hard',
            interpretable_mmd_result=res)
        __p_max = np.max([__tst_obj.p_value for __tst_obj in seq_tst_result])
        n_sample_test = len(dataset_test)
    else:
        __p_max = np.nan
        n_sample_test = -1
    # end if
    
    n_sample_training = len(dataset_train)
    
    if isinstance(res, CrossValidationTrainedParameter):
        assert isinstance(res.array_s_hat, np.ndarray)
        weights = res.array_s_hat
        variables = res.stable_s_hat
    elif isinstance(res, AlgorithmOneResult):
        assert res.selected_model is not None
        tensor_weights = res.selected_model.interpretable_mmd_train_result.ard_weights_kernel_k
        weights = tensor_weights.cpu().numpy()
        variables = res.selected_model.selected_variables
    else:
        raise ValueError(f'Unexpected type: {type(res)}')
    # end if
    

    return data_objects.BasicVariableSelectionResult(
        weights=weights,
        variables=variables,
        p_value=__p_max,
        verbose_field=res,
        n_sample_training=n_sample_training,
        n_sample_test=n_sample_test)