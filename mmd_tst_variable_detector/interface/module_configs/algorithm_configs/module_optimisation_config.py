from abc import ABC, abstractmethod
from pathlib import Path
import typing as ty
import logging
import numpy as np
import torch

from distributed import Client


# CV selection modules
from ....detection_algorithm.cross_validation_detector import (
    InterpretableMmdTrainParameters,
    DistributedComputingParameter,
    CrossValidationAlgorithmParameter,
    CrossValidationTrainParameters,
)
from ....datasets import (
    BaseDataset
)
from ....detection_algorithm.search_regularization_min_max import RegularizationSearchParameters
from ....detection_algorithm.early_stoppings import (
    ConvergenceEarlyStop,
    VariableEarlyStopping,
    # ArdWeightsEarlyStopping
)
from ....detection_algorithm.pytorch_lightning_trainer import PytorchLightningDefaultArguments
from ....weights_initialization.weights_initialization import weights_initialization

from ....assessment_helper import default_settings

from .. import (
    ApproachConfigArgs,
    AlgorithmOneConfigArgs,
    CvSelectionConfigArgs,
    ResourceConfigArgs
)

from ....logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class MmdOptimisationConfigTemplate(ABC):    
    @abstractmethod
    def get_configs(self,
                    path_work_dir: Path,
                    dask_scheduler_address: ty.Optional[str],
                    algorithm_config: ty.Union[CvSelectionConfigArgs, AlgorithmOneConfigArgs],
                    resource_config_args: ResourceConfigArgs
                    ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
        raise NotImplementedError('This method must be implemented in the derived class.')
# enc class


class ConfigTPamiDraft(MmdOptimisationConfigTemplate):
    def get_configs(self,
                    path_work_dir: Path,
                    dask_scheduler_address: ty.Optional[str],
                    algorithm_config: ty.Union[CvSelectionConfigArgs, AlgorithmOneConfigArgs],
                    resource_config_args: ResourceConfigArgs
                    ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
        """The training configuration that I used in study-71.
        The configuration is based on the TPAMI draft (ver-1): https://arxiv.org/pdf/2311.01537#page=4.76
        """
        logger.warning('The configuration is based on the TPAMI draft (ver-1): https://arxiv.org/pdf/2311.01537#page=4.76')
        logger.warning('I forcely update parameters.')

        early_stopper = ConvergenceEarlyStop(
            check_span=100,  #  set in the paper draft. 
            ignore_epochs=500,
            threshold_convergence_ratio=0.001  # set in the paper draft.
        )

        if resource_config_args.train_accelerator == 'cuda':
            n_device = 1
        else:
            n_device = 'auto'
        # end if

        pl_trainer_config = PytorchLightningDefaultArguments(
            accelerator=resource_config_args.train_accelerator,
            devices=n_device,
            max_epochs=algorithm_config.max_epoch,
            callbacks=[early_stopper],
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=None,
            default_root_dir=path_work_dir
        )

        base_training_parameter = InterpretableMmdTrainParameters(
            is_use_log=algorithm_config.batch_size,
            lr_scheduler=default_settings.lr_scheduler,
            lr_scheduler_monitor_on='train_loss',
            optimizer_args={"lr": 0.01}  # set in the paper draft.
        )

        reg_param_search_param = RegularizationSearchParameters(
            search_strategy='heuristic',
            n_search_iteration=20,
            n_regularization_parameter=6,
        )
        
        algorithm_parameter = CrossValidationAlgorithmParameter(
            n_subsampling=10,  # set in the paper draft.
            sampling_strategy='cross-validation',
            ratio_subsampling=0.8,
            weighting_mode='p_value_min_test_power',
            ard_weight_selection_strategy='hist_based',
            permutation_test_metric='sliced_wasserstein',
            n_permutation_test=1000,  # Not defined in the paper.
            regularization_search_parameter=reg_param_search_param,
            approach_regularization_parameter='fixed_range',
            pre_filtering_trained_estimator='off'
        )
        distributed_parameter = DistributedComputingParameter(
            dask_scheduler_address=dask_scheduler_address
        )
        
        training_parameter = CrossValidationTrainParameters(
            algorithm_parameter=algorithm_parameter,
            base_training_parameter=base_training_parameter,
            distributed_parameter=distributed_parameter,
            computation_backend = 'dask')
        
        # NOTE: I forcely update the search strategy. It's always fixed to 'heuristic'.
        algorithm_config.parameter_search_parameter.search_strategy = 'heuristic'

        return training_parameter, pl_trainer_config


# class ConfigTPamiDraftVer2(MmdOptimisationConfigTemplate):
#     def get_configs(self,
#                     path_work_dir: Path,
#                     dask_scheduler_address: ty.Optional[str],
#                     config_args: data_objects.InterfaceConfigArgs
#                     ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
#         """The training configuration that I used in study-71.
#         The configuration is based on the TPAMI draft (ver-2): NOT YET.
#         """
#         logger.warning('The configuration is based on the TPAMI draft (ver-2): ')
#         logger.warning('I forcely update parameters.')

#         training_conf_args = self.get_mmd_config_args(config_args)

#         early_stopper = ConvergenceEarlyStop(
#             check_span=100,  #  set in the paper draft. 
#             ignore_epochs=500,
#             threshold_convergence_ratio=0.001  # set in the paper draft.
#         )
        

#         pl_trainer_config = PytorchLightningDefaultArguments(
#             max_epochs=training_conf_args.max_epoch,
#             callbacks=[early_stopper],
#             enable_checkpointing=False,
#             enable_progress_bar=True,
#             enable_model_summary=False,
#             logger=None,
#             default_root_dir=path_work_dir
#         )

#         base_training_parameter = InterpretableMmdTrainParameters(
#             is_use_log=training_conf_args.batch_size,
#             lr_scheduler=default_settings.lr_scheduler,
#             lr_scheduler_monitor_on='train_loss',
#             optimizer_args={"lr": 0.01}  # set in the paper draft.
#         )

#         reg_param_search_param = RegularizationSearchParameters(
#             search_strategy='heuristic',
#             n_search_iteration=20,
#             n_regularization_parameter=6,
#         )
        
#         if isinstance(training_conf_args, data_objects.CvSelectionConfigArgs):
#             algorithm_parameter = CrossValidationAlgorithmParameter(
#                 n_subsampling=training_conf_args.n_subsampling,  # set in the paper draft.
#                 sampling_strategy='cross-validation',
#                 ratio_subsampling=0.8,
#                 weighting_mode='p_value_filter_test_power',
#                 ard_weight_selection_strategy='hist_based',
#                 permutation_test_metric='sliced_wasserstein',
#                 n_permutation_test=1000,  # Not defined in the paper.
#                 regularization_search_parameter=reg_param_search_param,
#                 approach_regularization_parameter='fixed_range'
#             )
#             distributed_parameter = DistributedComputingParameter(
#                 dask_scheduler_address=dask_scheduler_address
#             )
            
#             training_parameter = CrossValidationTrainParameters(
#                 algorithm_parameter=algorithm_parameter,
#                 base_training_parameter=base_training_parameter,
#                 distributed_parameter=distributed_parameter,
#                 computation_backend = 'dask')
#         elif isinstance(training_conf_args, data_objects.AlgorithmOneConfigArgs):            # I use the same configuration as the CV-selection.
#             algorithm_parameter = CrossValidationAlgorithmParameter(
#                 candidate_regularization_parameter='auto',
#                 n_subsampling=1,
#                 regularization_search_parameter=training_conf_args.parameter_search_parameter)
#             distributed_parameter = DistributedComputingParameter(dask_scheduler_address=dask_scheduler_address)
            
#             training_parameter = CrossValidationTrainParameters(
#                 algorithm_parameter,
#                 base_training_parameter,
#                 distributed_parameter,
#                 computation_backend=config_args.resource_config_args.dask_config_detection.distributed_mode)
#         else:
#             raise ValueError(f'Invalid type of training_conf_args: {type(training_conf_args)}')
#         # end if
        
#         # NOTE: I forcely update the search strategy. It's always fixed to 'heuristic'.
#         training_conf_args.parameter_search_parameter.search_strategy = 'heuristic'

#         return training_parameter, pl_trainer_config



class ConfigRapid(MmdOptimisationConfigTemplate):
    def get_configs(self,
                    path_work_dir: Path,
                    dask_scheduler_address: ty.Optional[str],
                    algorithm_config: ty.Union[CvSelectionConfigArgs, AlgorithmOneConfigArgs],
                    resource_config_args: ResourceConfigArgs
                    ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
        """The training configuration that I used in study-71.
        
        References
        ----------
        https://github.com/Kensuke-Mitsuzawa/mmd-tst-variable-detector/issues/71
        """        
        # TODO Reset this config. Must be longer
        seq_early_stopper = [
            ConvergenceEarlyStop(check_span=100, ignore_epochs=200),
            VariableEarlyStopping(ignore_epochs=600),
        ]
        
        if resource_config_args.train_accelerator == 'cuda':
            n_device = 1
        else:
            n_device = 'auto'
        # end if
        
        pl_trainer_config = PytorchLightningDefaultArguments(
            max_epochs=algorithm_config.max_epoch,
            callbacks=seq_early_stopper,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=None,
            default_root_dir=path_work_dir,
            devices=n_device)
        
        base_training_parameter = InterpretableMmdTrainParameters(
            is_use_log=algorithm_config.batch_size,
            lr_scheduler=default_settings.lr_scheduler,
            lr_scheduler_monitor_on='train_loss',
            optimizer_args={"lr": 0.01},
            n_workers_train_dataloader=algorithm_config.dataloader_n_workers_train_dataloader,
            n_workers_validation_dataloader=algorithm_config.dataloader_n_workers_validation_dataloader,
            dataloader_persistent_workers=algorithm_config.dataloader_persistent_workers,
            limit_steps_early_stop_negative_mmd=3000)  # comment: 3000 for tmp setting
        
        if isinstance(algorithm_config, CvSelectionConfigArgs):
            algorithm_parameter = CrossValidationAlgorithmParameter(
                candidate_regularization_parameter='auto',
                approach_regularization_parameter=algorithm_config.approach_regularization_parameter,
                n_subsampling=algorithm_config.n_subsampling,
                n_permutation_test=algorithm_config.n_permutation_test,
                regularization_search_parameter=algorithm_config.parameter_search_parameter,
                weighting_mode='p_value_min_test_power',
                stability_score_base='ard',
                permutation_test_metric='sliced_wasserstein',
                pre_filtering_trained_estimator='off')
            distributed_parameter = DistributedComputingParameter(dask_scheduler_address=dask_scheduler_address)
            
            # is_use_dask = 'dask' if dask_scheduler_address is not None else 'single'
            
            training_parameter = CrossValidationTrainParameters(
                algorithm_parameter,
                base_training_parameter,
                distributed_parameter,
                computation_backend=resource_config_args.dask_config_detection.distributed_mode)
        elif isinstance(algorithm_config, AlgorithmOneConfigArgs):
            # I use the same configuration as the CV-selection.
            algorithm_parameter = CrossValidationAlgorithmParameter(
                candidate_regularization_parameter='auto',
                n_subsampling=1,
                regularization_search_parameter=algorithm_config.parameter_search_parameter)
            distributed_parameter = DistributedComputingParameter(dask_scheduler_address=dask_scheduler_address)
            
            training_parameter = CrossValidationTrainParameters(
                algorithm_parameter,
                base_training_parameter,
                distributed_parameter,
                computation_backend=resource_config_args.dask_config_detection.distributed_mode)
        else:
            raise ValueError(f'Invalid type of algorithm_config: {type(algorithm_config)}')
        # end if
        
        return training_parameter, pl_trainer_config
    


# def get_config_tpami_draft(path_work_dir: Path,
#                            dask_scheduler_address: ty.Optional[str],
#                            config_args: data_objects.InterfaceConfigArgs
#                            ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
#     """The training configuration that I used in study-71.
#     The configuration is based on the TPAMI draft (ver-1): https://arxiv.org/pdf/2311.01537#page=4.76
#     """
#     training_conf_args = get_mmd_config_args(config_args)

#     early_stopper = ConvergenceEarlyStop(
#         check_span=100,  #  set in the paper draft. 
#         ignore_epochs=500,
#         threshold_convergence_ratio=0.001  # set in the paper draft.
#     )
    

#     pl_trainer_config = PytorchLightningDefaultArguments(
#         max_epochs=training_conf_args.max_epoch,
#         callbacks=[early_stopper],
#         enable_checkpointing=False,
#         enable_progress_bar=True,
#         enable_model_summary=False,
#         logger=None,
#         default_root_dir=path_work_dir
#     )

#     base_training_parameter = InterpretableMmdTrainParameters(
#         is_use_log=training_conf_args.batch_size,
#         lr_scheduler=default_settings.lr_scheduler,
#         lr_scheduler_monitor_on='train_loss',
#         optimizer_args={"lr": 0.01}  # set in the paper draft.
#     )

#     reg_param_search_param = RegularizationSearchParameters(
#         search_strategy='heuristic',
#         n_search_iteration=20,
#         n_regularization_parameter=6,
#     )
    
#     algorithm_parameter = CrossValidationAlgorithmParameter(
#         n_subsampling=10,  # set in the paper draft.
#         sampling_strategy='cross-validation',
#         ratio_subsampling=0.8,
#         weighting_mode='p_value_min_test_power',
#         ard_weight_selection_strategy='hist_based',
#         permutation_test_metric='sliced_wasserstein',
#         n_permutation_test=1000,  # Not defined in the paper.
#         regularization_search_parameter=reg_param_search_param
#     )
#     distributed_parameter = DistributedComputingParameter(
#         dask_scheduler_address=dask_scheduler_address
#     )
    
#     training_parameter = CrossValidationTrainParameters(
#         algorithm_parameter=algorithm_parameter,
#         base_training_parameter=base_training_parameter,
#         distributed_parameter=distributed_parameter,
#         computation_backend = 'dask')

#     return training_parameter, pl_trainer_config


# def get_config_rapid(path_work_dir: Path,
#                      dask_scheduler_address: ty.Optional[str],
#                      config_args: data_objects.InterfaceConfigArgs
#                      ) -> ty.Tuple[CrossValidationTrainParameters, PytorchLightningDefaultArguments]:
#     """The training configuration that I used in study-71.
    
#     References
#     ----------
#     https://github.com/Kensuke-Mitsuzawa/mmd-tst-variable-detector/issues/71
#     """
#     training_conf_args = get_mmd_config_args(config_args)
        
#     # TODO early stopping ignore epochs
#     # early_stopper = ConvergenceEarlyStop(
#     #     check_span=100,
#     #     ignore_epochs=200 + 10)
    
#     # early_stopper = ArdWeightsEarlyStopping()
    
#     seq_early_stopper = [
#         ConvergenceEarlyStop(check_span=100, ignore_epochs=200),
#         VariableEarlyStopping(ignore_epochs=600),
#     ]
    
#     if config_args.resource_config_args.train_accelerator == 'cuda':
#         n_device = 1
#     else:
#         n_device = 'auto'
#     # end if    
    
#     pl_trainer_config = PytorchLightningDefaultArguments(
#         max_epochs=training_conf_args.max_epoch,
#         callbacks=seq_early_stopper,
#         enable_checkpointing=False,
#         enable_progress_bar=True,
#         enable_model_summary=False,
#         logger=None,
#         default_root_dir=path_work_dir,
#         devices=n_device)
    
#     base_training_parameter = InterpretableMmdTrainParameters(
#         is_use_log=training_conf_args.batch_size,
#         lr_scheduler=default_settings.lr_scheduler,
#         lr_scheduler_monitor_on='train_loss',
#         optimizer_args={"lr": 0.01},
#         n_workers_train_dataloader=training_conf_args.dataloader_n_workers_train_dataloader,
#         n_workers_validation_dataloader=training_conf_args.dataloader_n_workers_validation_dataloader,
#         dataloader_persistent_workers=training_conf_args.dataloader_persistent_workers,
#         limit_steps_early_stop_negative_mmd=3000)  # comment: 3000 for tmp setting
    
#     if isinstance(training_conf_args, data_objects.CvSelectionConfigArgs):    
#         algorithm_parameter = CrossValidationAlgorithmParameter(
#             candidate_regularization_parameter='auto',
#             approach_regularization_parameter=training_conf_args.approach_regularization_parameter,
#             n_subsampling=training_conf_args.n_subsampling,
#             n_permutation_test=training_conf_args.n_permutation_test,
#             regularization_search_parameter=training_conf_args.parameter_search_parameter)
#         distributed_parameter = DistributedComputingParameter(dask_scheduler_address=dask_scheduler_address)
        
#         # is_use_dask = 'dask' if dask_scheduler_address is not None else 'single'
        
#         training_parameter = CrossValidationTrainParameters(
#             algorithm_parameter,
#             base_training_parameter,
#             distributed_parameter,
#             computation_backend=config_args.resource_config_args.dask_config_detection.distributed_mode)
#     elif isinstance(training_conf_args, data_objects.AlgorithmOneConfigArgs):
#         # I use the same configuration as the CV-selection.
#         algorithm_parameter = CrossValidationAlgorithmParameter(
#             candidate_regularization_parameter='auto',
#             n_subsampling=1,
#             regularization_search_parameter=training_conf_args.parameter_search_parameter)
#         distributed_parameter = DistributedComputingParameter(dask_scheduler_address=dask_scheduler_address)
        
#         training_parameter = CrossValidationTrainParameters(
#             algorithm_parameter,
#             base_training_parameter,
#             distributed_parameter,
#             computation_backend=config_args.resource_config_args.dask_config_detection.distributed_mode)
#     else:
#         raise ValueError(f'Invalid type of training_conf_args: {type(training_conf_args)}')
#     # end if
    
#     return training_parameter, pl_trainer_config



# -----------------------------------------------------------------------------------------------------


# def get_mmd_config_args(approach_config_args: ApproachConfigArgs,
#                         detector_algorithm_config_args: DetectorAlgorithmConfigArgs
#                         ) -> ty.Union[CvSelectionConfigArgs, AlgorithmOneConfigArgs]:

#     if approach_config_args.approach_interpretable_mmd == 'cv_selection':
#         mmd_config_args: CvSelectionConfigArgs = detector_algorithm_config_args.mmd_cv_selection_args

#         assert mmd_config_args is not None, 'mmd_cv_selection_args is not given.'
#         assert isinstance(mmd_config_args, CvSelectionConfigArgs), 'mmd_cv_selection_args is not given.'
#     elif approach_config_args.approach_interpretable_mmd == 'algorithm_one':
#         mmd_config_args: AlgorithmOneConfigArgs = detector_algorithm_config_args.mmd_algorithm_one_args

#         assert mmd_config_args is not None, 'mmd_algorithm_one_args is not given.'
#         assert isinstance(mmd_config_args, AlgorithmOneConfigArgs), 'mmd_algorithm_one_args is not given.'
#     else:
#         raise ValueError(f'config_args.approach_config_args.approach_interpretable_mmd: {approach_config_args.approach_interpretable_mmd}')
#     # end if

#     return mmd_config_args



# def computing_l1_distance_d_dim(input_dataset: BaseDataset, index_dimension: int) -> ty.Tuple[int, float]:    
#     tensor_x, tensor_y = input_dataset.get_samples_at_dimension_d(index_dimension)
    
#     l1_distance = torch.abs(tensor_x - tensor_y).sum()
    
#     return index_dimension, l1_distance.item()


# def get_initial_weights_before_search(dataset: BaseDataset, 
#                                       dask_client: ty.Optional[Client] = None,
#                                       strategy: str = 'all_one') -> np.ndarray:
#     """Private. TODO This function must be integrated to the interior of CV-interface.
#     A core of this experiment. It initializes a set of weights before the Optuna Parameter search.
#     I want to know if this initial search contributes to speed performance/epoch size while keeping the detection quality.
#     """
#     assert strategy in ['all_one', 'wasserstein'], f'Unexpected strategy: {strategy}'
#     if strategy == 'waterstein':
#         raise NotImplementedError('For the moment, I do not confirm validness of this strategy strategy == "waterstein". Stop the execution here.')
#         initial_value = weights_initialization(dataset, approach_name='wasserstein', dask_client=dask_client)
#     elif strategy == 'all_one':
#         x_dimension_x = dataset.get_dimension_flattened()
#         # end if
        
#         if dataset.is_dataset_on_ram():
#             dataset = dataset.generate_dataset_on_ram()
#         # end if
        
#         is_tensor_ram = dataset.is_dataset_on_ram() and hasattr(dataset, f'_{dataset.__class__.__name__}__tensor_x') and hasattr(dataset, f'_{dataset.__class__.__name__}__tensor_y')

#         results = [computing_l1_distance_d_dim(dataset, __index_d) for __index_d in range(x_dimension_x)]
#         # find dimensions where all 0.0 (L1). If L1 is 0.0 at a dimension d, then I set 0.0 to the initial value.
#         initial_value = np.ones(x_dimension_x)
#         for __index_d, __l1_distance in results:
#             if __l1_distance == 0.0:
#                 initial_value[__index_d] = 0.0
#             # end if
#         # end for
#     else:
#         raise NotImplementedError(f'Unexpected strategy: {strategy}')
#     # end if

#     return initial_value
