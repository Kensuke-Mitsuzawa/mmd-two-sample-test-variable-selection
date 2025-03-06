from pathlib import Path
import typing as ty
import logging

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
)
from ..module_configs import ApproachConfigArgs
from ..module_configs.algorithm_configs.algorithm_config import (
    CvSelectionConfigArgs,
    AlgorithmOneConfigArgs,
    BaselineMmdConfigArgs
)
from ..interface_config_args import (
    DetectorAlgorithmConfigArgs,
    InterfaceConfigArgs
)
# Algorithm one
from ...detection_algorithm import detection_algorithm_one, AlgorithmOneResult
# mmd baseline
from ...detection_algorithm.baseline_mmd import BaselineMmdResult, baseline_mmd

from ...assessment_helper import default_settings

from .. import data_objects
from ..module_configs.algorithm_configs import module_optimisation_config

from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



def get_mmd_config_args(approach_config_args: ApproachConfigArgs,
                        detector_algorithm_config_args: DetectorAlgorithmConfigArgs
                        ) -> ty.Union[CvSelectionConfigArgs, AlgorithmOneConfigArgs]:

    if approach_config_args.approach_interpretable_mmd == 'cv_selection':
        mmd_config_args: CvSelectionConfigArgs = detector_algorithm_config_args.mmd_cv_selection_args

        assert mmd_config_args is not None, 'mmd_cv_selection_args is not given.'
        assert isinstance(mmd_config_args, CvSelectionConfigArgs), 'mmd_cv_selection_args is not given.'
    elif approach_config_args.approach_interpretable_mmd == 'algorithm_one':
        mmd_config_args: AlgorithmOneConfigArgs = detector_algorithm_config_args.mmd_algorithm_one_args

        assert mmd_config_args is not None, 'mmd_algorithm_one_args is not given.'
        assert isinstance(mmd_config_args, AlgorithmOneConfigArgs), 'mmd_algorithm_one_args is not given.'
    elif approach_config_args.approach_interpretable_mmd == 'baseline_mmd':
        mmd_config_args: BaselineMmdConfigArgs = detector_algorithm_config_args.mmd_baseline_args

        assert mmd_config_args is not None, 'mmd_baseline_args is not given.'
        assert isinstance(mmd_config_args, BaselineMmdConfigArgs), 'mmd_baseline_args is not given.'
    else:
        raise ValueError(f'config_args.approach_config_args.approach_interpretable_mmd: {approach_config_args.approach_interpretable_mmd}')
    # end if

    return mmd_config_args



def computing_l1_distance_d_dim(input_dataset: BaseDataset, index_dimension: int) -> ty.Tuple[int, float]:    
    tensor_x, tensor_y = input_dataset.get_samples_at_dimension_d(index_dimension)
    
    l1_distance = torch.abs(tensor_x - tensor_y).sum()
    
    return index_dimension, l1_distance.item()


def get_initial_weights_before_search(dataset: BaseDataset, 
                                      dask_client: ty.Optional[Client] = None,
                                      strategy: str = 'all_one') -> np.ndarray:
    """Private. TODO This function must be integrated to the interior of CV-interface.
    A core of this experiment. It initializes a set of weights before the Optuna Parameter search.
    I want to know if this initial search contributes to speed performance/epoch size while keeping the detection quality.
    """
    assert strategy in ['all_one', 'one', 'wasserstein'], f'Unexpected strategy: {strategy}'
    if strategy == 'waterstein':
        raise NotImplementedError('For the moment, I do not confirm validness of this strategy strategy == "waterstein". Stop the execution here.')
        initial_value = weights_initialization(dataset, approach_name='wasserstein', dask_client=dask_client)
    elif strategy == 'all_one' or strategy == 'one':
        x_dimension_x = dataset.get_dimension_flattened()
        # end if
        
        if dataset.is_dataset_on_ram():
            dataset = dataset.generate_dataset_on_ram()
        # end if
        
        is_tensor_ram = dataset.is_dataset_on_ram() and hasattr(dataset, f'_{dataset.__class__.__name__}__tensor_x') and hasattr(dataset, f'_{dataset.__class__.__name__}__tensor_y')

        results = [computing_l1_distance_d_dim(dataset, __index_d) for __index_d in range(x_dimension_x)]
        # find dimensions where all 0.0 (L1). If L1 is 0.0 at a dimension d, then I set 0.0 to the initial value.
        initial_value = np.ones(x_dimension_x)
        for __index_d, __l1_distance in results:
            if __l1_distance == 0.0:
                initial_value[__index_d] = 0.0
            # end if
        # end for
    else:
        raise NotImplementedError(f'Unexpected strategy: {strategy}')
    # end if

    return initial_value



def __main_run(dataset_train: BaseDataset,
               training_conf_args: InterfaceConfigArgs,
               path_dir_model: Path,
               path_dir_ml_logger: Path,
               dask_client: ty.Optional[Client] = None,
               seed_root_random: int = 42,
               ) -> ty.Union[AlgorithmOneResult, CrossValidationTrainedParameter, BaselineMmdResult]:
    mmd_config_args = get_mmd_config_args(
        approach_config_args=training_conf_args.approach_config_args,
        detector_algorithm_config_args=training_conf_args.detector_algorithm_config_args,
    )

    # comment: for the RAM memory management reasons, I define another variable `dataset_train_for_preprocess`.
    # The content is totally same as `dataset_train`.
    if dataset_train.is_dataset_on_ram():
        dataset_train_for_preprocess = dataset_train.generate_dataset_on_ram()
    else:
        dataset_train_for_preprocess = dataset_train
    # end if
    
    mmd_estimator_config = mmd_config_args.mmd_estimator_config

    if isinstance(mmd_estimator_config.ard_weights_initial, str):
        # get initial weights.
        initial_value = get_initial_weights_before_search(
            dataset=dataset_train_for_preprocess, 
            dask_client=dask_client,
            strategy=mmd_estimator_config.ard_weights_initial)
    else:
        assert isinstance(mmd_estimator_config.ard_weights_initial, torch.Tensor), 'Unexpected type.'
        initial_value = mmd_estimator_config.ard_weights_initial.numpy()
    # end if
    
    # initialization of Kernel instance.
    __length_scale_given = mmd_estimator_config.length_scale if isinstance(mmd_estimator_config.length_scale, torch.Tensor) else None
    kernel = QuadraticKernelGaussianKernel.from_dataset(
        dataset_train_for_preprocess, 
        ard_weights=torch.from_numpy(initial_value),
        heuristic_operation=mmd_estimator_config.aggregation_kernel_length_scale,
        is_dimension_median_heuristic=mmd_estimator_config.is_dimension_median_heuristic,
        bandwidth=__length_scale_given)
    # initialization of MMD instance.
    mmd_estimator = QuadraticMmdEstimator(
        kernel_obj=kernel,
        unit_diagonal=mmd_estimator_config.unit_diagonal,
        biased=mmd_estimator_config.biased,
        variance_term=mmd_estimator_config.variance_term)
        
    # comment: optuna search exists already in CV intergace.
    # No need to execute it manually.
    if dask_client is not None:
        dask_scheduler_address = dask_client.scheduler.address if dask_client is not None else None
    else:
        dask_scheduler_address = ''
    # end if

    # Obtaining the optimiser configuration.
    cv_train_param, pl_param = training_conf_args.detector_algorithm_config_args.mmd_optimiser_configs.get_configs(
        path_work_dir=path_dir_model, 
        dask_scheduler_address=dask_scheduler_address, 
        algorithm_config=mmd_config_args,
        resource_config_args=training_conf_args.resource_config_args)
    
    # setting MLFlow logger
    post_process_handler = PostProcessLoggerHandler(
        loggers=['mlflow'],
        logger2config={
            "mlflow": {
                "save_dir": path_dir_ml_logger.as_posix(), 
                "tracking_uri": f"file://{path_dir_ml_logger.as_posix()}" 
            }})
    
    
    if isinstance(mmd_config_args, CvSelectionConfigArgs):
        detector = CrossValidationInterpretableVariableDetector(
            pytorch_trainer_config=pl_param,
            training_parameter=cv_train_param,
            estimator=mmd_estimator,
            post_process_handler=post_process_handler,
            seed_root_random=seed_root_random)
        res = detector.run_cv_detection(dataset_train)
    elif isinstance(mmd_config_args, AlgorithmOneConfigArgs):
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
    elif isinstance(mmd_config_args, BaselineMmdConfigArgs):
        res = baseline_mmd(
            mmd_estimator=mmd_estimator,
            pytorch_trainer_config=pl_param,
            base_training_parameter=cv_train_param.base_training_parameter,
            dataset_training=dataset_train,
            dataset_test=None,
            path_work_dir=path_dir_model,
            post_process_handler=post_process_handler)
    else:
        raise ValueError(f'Unexpected mmd_config_args type: {type(mmd_config_args)}')
    # end if
    
    return res


def main(config_args: InterfaceConfigArgs,
         dataset_train: BaseDataset,
         dask_client: ty.Optional[Client] = None, 
         dataset_test: ty.Optional[BaseDataset] = None,) -> data_objects.BasicVariableSelectionResult:
    assert (config_args.detector_algorithm_config_args.mmd_cv_selection_args is not None) or \
        (config_args.detector_algorithm_config_args.mmd_algorithm_one_args is not None) or \
        (config_args.detector_algorithm_config_args.mmd_baseline_args is not None), 'Either of "mmd_cv_selection_args" or "mmd_algorithm_one_args" must be given.'
    if config_args.detector_algorithm_config_args.mmd_cv_selection_args is not None:
        assert isinstance(config_args.detector_algorithm_config_args.mmd_cv_selection_args, CvSelectionConfigArgs), 'mmd_cv_selection_args is not given.'
    if config_args.detector_algorithm_config_args.mmd_algorithm_one_args is not None and config_args.detector_algorithm_config_args.mmd_algorithm_one_args != '':
        assert isinstance(config_args.detector_algorithm_config_args.mmd_algorithm_one_args, AlgorithmOneConfigArgs), 'mmd_algorithm_one_args is not given.'
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
        assert res.selected_model.interpretable_mmd_train_result is not None
        tensor_weights = res.selected_model.interpretable_mmd_train_result.ard_weights_kernel_k
        weights = tensor_weights.cpu().numpy()
        variables = res.selected_model.selected_variables
    elif isinstance(res, BaselineMmdResult):
        assert res.selected_variables is not None
        assert res.interpretable_mmd_train_result is not None
        tensor_weights = res.interpretable_mmd_train_result.ard_weights_kernel_k
        weights = tensor_weights.cpu().numpy()
        variables = res.selected_variables
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