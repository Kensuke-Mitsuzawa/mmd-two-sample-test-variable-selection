import typing as ty
from pathlib import Path
from tempfile import mkdtemp
import shutil
import json
import dataclasses
from dataclasses import asdict
import tempfile
import logging

import plotly.express as px
import matplotlib.pyplot as plt

import xarray as xr
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
import pytorch_lightning.loggers

# from ..detection_algorithm.commons import (
#     RegularizationParameter,
#     InterpretableMmdTrainResult)
# from ..detection_algorithm.commons import SubLearnerTrainingResult


msg_logger = logging.getLogger(f'{__package__}.{__name__}')



def function_overwrite_logger_mmd_estimator(
        base_logger: Logger,
        regularization_parameter: "RegularizationParameter") -> Logger:
    """Public function. Overwriting the logger conf for each MMD estimator training.
    Users can rewrite this function as they want.

    Args:
        base_logger: base_logger that this function takes.
        regularization_parameter: regularization parameter values.
    """
    # logger_copy = deepcopy(base_logger)
    run_exp_name: str = f"reg1_{regularization_parameter.lambda_1}-reg2_{regularization_parameter.lambda_2}"
    if isinstance(base_logger, pytorch_lightning.loggers.MLFlowLogger):
        logger_copy = pytorch_lightning.loggers.MLFlowLogger(
            experiment_name=base_logger._experiment_name, 
            run_name=run_exp_name, 
            tracking_uri=base_logger._tracking_uri, 
            tags=base_logger.tags, 
            save_dir=base_logger.save_dir, 
            log_model=base_logger._log_model, 
            prefix=base_logger._prefix, 
            artifact_location=base_logger._artifact_location,
            run_id=base_logger._run_id)
    elif isinstance(base_logger, pytorch_lightning.loggers.TensorBoardLogger):
        logger_copy = pytorch_lightning.loggers.TensorBoardLogger(
            save_dir=base_logger.save_dir, 
            name=run_exp_name, 
            version=base_logger.version, 
            log_graph=base_logger.log_graph, 
            default_hp_metric=base_logger._default_hp_metric, 
            prefix=base_logger._prefix, 
            sub_dir=base_logger._sub_dir)
    elif isinstance(base_logger, pytorch_lightning.loggers.WandbLogger):
        raise NotImplementedError()
    elif isinstance(base_logger, pytorch_lightning.loggers.CSVLogger):
        logger_copy = pytorch_lightning.loggers.CSVLogger(
            save_dir=base_logger.save_dir,
            name=run_exp_name,
            version=base_logger.version,
            prefix=base_logger._prefix,
            flush_logs_every_n_steps=base_logger._flush_logs_every_n_steps)
    else:
        raise NotImplementedError()
    # end if

    return logger_copy



def log_postprocess_manually(mmd_estimator_opt_result: "InterpretableMmdTrainResult",
                             indices_detected: ty.List[int],
                             pl_logger: Logger,
                             aux_object: ty.Optional[ty.Dict] = None) -> None:
    """public function.
    """
    seq_training_trajectory = mmd_estimator_opt_result.trajectory_record_training
    seq_validation_trajectory = mmd_estimator_opt_result.trajectory_record_validation

    for __train_trajectory in seq_training_trajectory:
        __obj = dataclasses.asdict(__train_trajectory)
        del __obj['epoch']
        pl_logger.log_metrics(__obj, step=__train_trajectory.epoch)
    # end for

    if not isinstance(pl_logger, pytorch_lightning.loggers.CSVLogger):
        # logging parameter object
        assert mmd_estimator_opt_result.training_parameter is not None
        pl_logger.log_hyperparams(dataclasses.asdict(mmd_estimator_opt_result.training_parameter))

        # CSVLogger does not support validation logging.
        for __val_trajectory in seq_validation_trajectory:
            __obj = dataclasses.asdict(__val_trajectory)
            del __obj['epoch']
            pl_logger.log_metrics(__obj, step=__val_trajectory.epoch)
        # end for

        if aux_object is not None:
            # p-value-logging
            # I design this block mainly for post-process of algorithm one.
            pl_logger.log_metrics({
                "p-value-dev-data": aux_object['p_value_dev'],
                "n-features-selected": len(indices_detected)})
            if 'p_value_test' in aux_object and aux_object['p_value_test'] is not None:
                # test data may not exist. Hence if block
                pl_logger.log_metrics({"p-value-test-data": aux_object['p_value_test']})
            # end if
        # end if
    # end if

    # logging artifacts
    if isinstance(pl_logger, pytorch_lightning.loggers.MLFlowLogger):
        path_artifact_save = Path(mkdtemp())
        # logging artifact objects. Specific command only to MLFlowLogger.
        # logging the trained model.
        path_artifact_save_features_selected = path_artifact_save / "features_selected.json"
        __dict_artifacts = {"features_selected": indices_detected}
        with path_artifact_save_features_selected.open('w') as f:
            f.write(json.dumps(__dict_artifacts))
        # end with
        pl_logger.experiment.log_artifact(
            run_id=pl_logger.run_id, 
            local_path=path_artifact_save_features_selected.as_posix())

        # ARD weights
        path_artifact_save_ard_weights = path_artifact_save / "ard_weights.pt"
        torch.save(mmd_estimator_opt_result.ard_weights_kernel_k, path_artifact_save_ard_weights.as_posix())
        pl_logger.experiment.log_artifact(
            run_id=pl_logger.run_id,
            local_path=path_artifact_save_ard_weights.as_posix())

        # model object
        path_artifact_save_model = path_artifact_save / "trained_model.pt"
        torch.save(mmd_estimator_opt_result.mmd_estimator, path_artifact_save_model.as_posix())
        pl_logger.experiment.log_artifact(
            run_id=pl_logger.run_id,
            local_path=path_artifact_save_model.as_posix())

        # # sample selection weights (if detector is sample selection based)
        # if isinstance(mmd_estimator_opt_result):
        #     # log sample selection weights when detecotr is sample selection based.
        #     path_artifact_save_sample_selection_weights = path_artifact_save / "sample_selection_weights.pt"
        #     torch.save(mmd_estimator_opt_result.selected_samples, path_artifact_save_sample_selection_weights.as_posix())
        #     pl_logger.experiment.log_artifact(
        #         run_id=pl_logger.run_id,
        #         local_path=path_artifact_save_sample_selection_weights.as_posix())
        # end if

        shutil.rmtree(path_artifact_save)  # deleting the tmp directory.
    # end for
    

# functions above are old. I will delete them in the future.
# -----------------------------------------------------------------------------------------
# I implemented the `PostProcessLoggerHandler`. 


AcceptableObjectType = ty.Union[ 
    "SubLearnerTrainingResult",
    "TrainedResultRegressionBasedVariableDetector",
    "_DaskFunctionReturn",
    "InterpretableMmdTrainResult"
]

    
class PostProcessLoggerHandler(object):
    """A utils-class for optimizations. This class is used for logging the optimization results.
    Background: in distributed learning, hard to run logging in many reasons.
    Solution: do logging after the optimization is done.
    """
    def __init__(self,
                 loggers: ty.List[str],
                 logger2config: ty.Dict[str, ty.Dict],
                 plot_backend: str = 'plotly'):
        """
        Parameters
        ----------
            loggers: list of logger name
                "mlflow", "tensorboard", "wandb", "csv"
            logger2config: dictionary, key: logger name, value: configuration of the logger.
                a dictionary that maps logger name to its configuration. See the documentation at https://lightning.ai/docs/pytorch/stable/extensions/logging.html .
                This default configuration MUST NOT have key representing `run_name` or `name`.
                Example: `{"mlflow": { "save_dir": temp_dir.as_posix(), "tracking_uri": f"file://{temp_dir.as_posix()}" }}`
            plot_backend: str, default: "plotly"
                "plotly" or "matplotlib"
        """
        self.plot_backend = plot_backend
        self.loggers = loggers
        self.logger2config = logger2config
        for __logger in self.loggers:
            assert __logger in ("mlflow", "tensorboard", "wandb", "csv"), f"logger {__logger} is not supported."
            assert __logger in logger2config, f"In logger2config, you have to define configuration for logger {__logger}."
        # end for
        
        # validation run.
        try:
            __loggers_validation = self.initialize_logger("__test", "__test")
            for __l in __loggers_validation:
                __l.log_metrics({"test": 1})
        except Exception as e:
            raise ValueError(f"The configuration of loggers is invalid. See the Exception message below: {e}") 
        
        
    def _log_optimization_trajectory(self,
                                     logger_object: Logger, 
                                     training_result: "InterpretableMmdTrainResult",
                                     plot_backend: str = 'plotly'):
        assert plot_backend in ('plotly', 'matplotlib'), "plot_backend must be either plotly or matplotlib."
        self.plot_backend = plot_backend
        
        seq_training_trajectory = training_result.trajectory_record_training
        seq_validation_trajectory = training_result.trajectory_record_validation
        
        
        assert len(seq_training_trajectory) > 0, "training trajectory is empty."

        for __train_trajectory in seq_training_trajectory:
            __obj = dataclasses.asdict(__train_trajectory)
            del __obj['epoch']
            logger_object.log_metrics(__obj, step=__train_trajectory.epoch)
        # end for

        if not isinstance(logger_object, pytorch_lightning.loggers.CSVLogger):
            # # logging parameter object
            # assert training_result.training_parameter is not None
            # logger_object.log_hyperparams(dataclasses.asdict(training_result.training_parameter))

            # CSVLogger does not support validation logging.
            for __val_trajectory in seq_validation_trajectory:
                __obj = dataclasses.asdict(__val_trajectory)
                del __obj['epoch']
                __obj_val = {f'val_{__k}': __v for __k, __v in __obj.items()}
                logger_object.log_metrics(__obj_val, step=__val_trajectory.epoch)
            # end for
        # end if
    
    def _log_summary(self, 
                     logger_object: Logger, 
                     metric_object: ty.Dict, 
                     paramter_object: ty.Optional[ty.Dict] = None):
        """Logging the following attributes
        - weights
        - detected variables
        - computation-time
        - p-value (if available)
        """
        logger_object.log_metrics(metric_object)
    
        if paramter_object is not None and not isinstance(logger_object, pytorch_lightning.loggers.CSVLogger):
            logger_object.log_hyperparams(paramter_object)
            
    def _log_artifacts(self, logger_obj: Logger, selected_variables: ty.List[int], weights: np.ndarray):
        """
        """
        temp_dir = Path(tempfile.mkdtemp())

        # save to a local disk first
        __path_save_json = temp_dir / 'detection.json'
        with __path_save_json.open('w') as f:
            f.write(json.dumps({
                "selected_variables": selected_variables,
                "weights": weights.tolist()}
                               ))
        # end with
        if isinstance(logger_obj, pytorch_lightning.loggers.MLFlowLogger):    
            logger_obj.experiment.log_artifact(run_id=logger_obj.run_id, local_path=__path_save_json)
        else:
            msg_logger.info(f"logger {logger_obj} does not support artifact logging.")
        
    def _log_plot(self, logger_obj: Logger, weights: np.ndarray):
        """Private API. Plotting the weights.
        """
        if not isinstance(logger_obj, pytorch_lightning.loggers.MLFlowLogger):
            msg_logger.error(f'logger {logger_obj} does not support plot logging.')
            return True
        
        if self.plot_backend == 'plotly':
            __x = list(range(len(weights)))
            fig = px.bar(x=__x, y=weights)
            logger_obj.experiment.log_figure(run_id=logger_obj.run_id, figure=fig, artifact_file='weights.html')
        elif self.plot_backend == 'matplotlib':
            __n_bars = len(weights)
            fig, ax = plt.subplots(figsize=(__n_bars * 0.5, 5))
            ax.bar(x=range(len(weights)), height=weights)
            logger_obj.experiment.log_figure(run_id=logger_obj.run_id, figure=fig, artifact_file='weights.png')
        else:
            raise NotImplementedError()

    def _log(self, logger_oject: Logger, target_object: AcceptableObjectType):
        """Semi-Private API. I design this API for logging these two objects,
        1. optimization result of Interpretable-CV-detection.
        2. hyper-parameter search result of baseline regression-based variable selection.
        
        Do logging for these two information,
        1. optimization trajectory. This logging is available only for MMD-optimization.
        2. summary of the optimization. This logging is available for all optimization.
        """
        if target_object.__class__.__name__ == "SubLearnerTrainingResult":
            if target_object.training_result is None:
                __metric_obj = {
                    "job_id": target_object.get_job_id_string(),
                    "is_success": 0}
                __parameter_obj = asdict(target_object.training_parameter)
                self._log_summary(logger_oject, metric_object=__metric_obj, paramter_object=__parameter_obj)
            else:
                if target_object.variable_detected is None:
                    n_selected_variables = 0
                else:
                    n_selected_variables = len(target_object.variable_detected)
                # end if
                __metric_obj = {
                    "job_id": target_object.get_job_id_string(),
                    "is_success": 1,
                    "execution_time_wallclock": target_object.execution_time_wallclock,
                    "execution_time_cpu": target_object.execution_time_cpu,
                    "p_value_selected": target_object.p_value_selected,
                    "epoch": target_object.epoch,
                    "n_selected_variables": n_selected_variables}
                __parameter_obj = asdict(target_object.training_parameter)
                
                self._log_optimization_trajectory(logger_oject, target_object.training_result)
                self._log_summary(logger_oject, metric_object=__metric_obj, paramter_object=__parameter_obj)
                if target_object.training_result.ard_weights_kernel_k is not None:
                    __weights = target_object.training_result.ard_weights_kernel_k.numpy()
                    # assert target_object.variable_detected, "variable_detected must be not None. Logically impossible."
                    self._log_artifacts(logger_oject, target_object.variable_detected, __weights)
                    self._log_plot(logger_oject, weights=__weights)
                # end if        
        elif target_object.__class__.__name__ == "TrainedResultRegressionBasedVariableDetector":
            __metric_obj = target_object.to_dict()
            del __metric_obj['weight_vector']
            del __metric_obj['selected_variable_indices']
            __metric_obj['seq_p_value_soft'] = json.dumps(__metric_obj['seq_p_value_soft'])
            __metric_obj['seq_p_value_hard'] = json.dumps(__metric_obj['seq_p_value_hard'])
            
            __parameter_obj = target_object.regression_model.get_params()
            self._log_summary(logger_oject, __metric_obj, __parameter_obj)
            # logging weights
            if target_object.weight_vector is not None:
                __weights = target_object.weight_vector
                # assert target_object.selected_variable_indices, "variable_detected must be not None. Logically impossible."
                self._log_artifacts(logger_oject, target_object.selected_variable_indices, __weights)
                self._log_plot(logger_oject, weights=__weights)
            # end if
        elif target_object.__class__.__name__ == "_DaskFunctionReturn":
            # output from Optuna-based L1/L2 search.
            __metric_obj = {
                "optuna_trial_id": target_object.trial.number,
                "is_success": 1,
                "optuna_objective_value": target_object.optuna_objective_value,
                "p_value_dev": target_object.p_value_dev,
                "test_power_dev": target_object.test_power_dev,
                "execution_time_wallclock": target_object.execution_time_wallclock,
                "execution_time_cpu": target_object.execution_time_cpu,
                "epochs": target_object.epochs,
                "ratio_variable": target_object.ratio_variable,
                "n_selected_variables": len(target_object.selected_variables)
            }
            assert target_object.mmd_train_result is not None, "mmd_train_result must be not None. Logically impossible."
            assert target_object.mmd_train_result.training_parameter is not None, "training_parameter must be not None. Logically impossible."
            __parameter_obj = asdict(target_object.mmd_train_result.training_parameter)
            self._log_optimization_trajectory(logger_oject, target_object.mmd_train_result)
            self._log_summary(logger_oject, metric_object=__metric_obj, paramter_object=__parameter_obj)
            if target_object.mmd_train_result.ard_weights_kernel_k is not None:
                __weights = target_object.mmd_train_result.ard_weights_kernel_k.numpy()
                assert target_object.selected_variables is not None, "variable_detected must be not None. Logically impossible."
                self._log_artifacts(logger_oject, target_object.selected_variables, __weights)
                self._log_plot(logger_oject, weights=__weights)
            # end if
        elif target_object.__class__.__name__ == "_AlgorithmOneRangeFunctionReturn":
            # output from Algorithm One output.
            if target_object.p_value_test is None:
                __p_value_test = -1.0
            else:
                __p_value_test = target_object.p_value_test
            # end if
            
            __metric_obj = {
                "is_success": 1.0 if target_object.is_success else 0.0,
                "test_power_dev": target_object.test_power_dev,
                "p_value_dev": target_object.p_value_dev,
                "p_value_test": __p_value_test,
                "execution_time_wallclock": target_object.execution_time_wallclock,
                "execution_time_cpu": target_object.execution_time_cpu,
                "epochs": target_object.epochs,           
            }
            __parameter_obj = asdict(target_object.trained_result.training_parameter)
            self._log_optimization_trajectory(logger_oject, target_object.trained_result)
            self._log_summary(logger_oject, metric_object=__metric_obj, paramter_object=__parameter_obj)
            if target_object.trained_result.ard_weights_kernel_k is not None:
                __weights = target_object.trained_result.ard_weights_kernel_k.numpy()
                # assert target_object.indices_detected, "variable_detected must be not None. Logically impossible."
                self._log_artifacts(logger_oject, target_object.indices_detected, __weights)
                self._log_plot(logger_oject, weights=__weights)
            # end if
        elif target_object.__class__.__name__ == "InterpretableMmdTrainResult":
            # output of an MMD Opt Variable Detector.
            __epochs = target_object.trajectory_record_training[-1].epoch
            __metric_obj = {
                "epochs": __epochs
            }
            __parameter_obj = asdict(target_object.training_parameter)
            self._log_optimization_trajectory(logger_oject, target_object)
            self._log_summary(logger_oject, metric_object=__metric_obj, paramter_object=__parameter_obj)
            if target_object.ard_weights_kernel_k is not None:
                __weights = target_object.ard_weights_kernel_k.numpy()
                self._log_artifacts(logger_oject, [], __weights)
                self._log_plot(logger_oject, weights=__weights)
            # end if
        elif target_object.__class__.__name__ == "CrossValidationAggregatedResult":
            # posting the result of CV-aggregation.
            assert hasattr(target_object, 'stability_score_matrix'), "stability_score_matrix must be in the object."
            assert hasattr(target_object, 'stable_s_hat'), "stable_s_hat must be in the object."
            assert hasattr(target_object, 'array_s_hat'), "array_s_hat must be in the object."
            assert hasattr(target_object, 'lambda_labels'), "lambda_labels must be in the object."
            assert target_object.stability_score_matrix is None or isinstance(target_object.stability_score_matrix, torch.Tensor), "stability_score_matrix must be None or torch.Tensor."
            assert target_object.stable_s_hat is None or isinstance(target_object.stable_s_hat, list), "stable_s_hat must be None or list."
            assert target_object.array_s_hat is None or isinstance(target_object.array_s_hat, torch.Tensor), "array_s_hat must be None or torch.Tensor."
            assert target_object.lambda_labels is None or isinstance(target_object.lambda_labels, list), "lambda_labels must be None or list."
            
            # visualization of bar plot.
            if target_object.array_s_hat is not None:
                self._log_plot(logger_oject, target_object.array_s_hat.numpy())
            # end if
            
            # visualization of heatmap.
            if target_object.stability_score_matrix is not None:
                _label_lambdas = target_object.lambda_labels
                _xarray_heatmap = xr.DataArray(target_object.stability_score_matrix.numpy(),
                                               dims=('lambda_label', 'dimension'),
                                               coords={'lambda_label': _label_lambdas})
                _fig = px.imshow(_xarray_heatmap, color_continuous_scale='RdBu', origin='lower')
                if isinstance(logger_oject, pytorch_lightning.loggers.MLFlowLogger):
                    logger_oject.experiment.log_figure(run_id=logger_oject.run_id, figure=_fig, artifact_file='heatmap.html')
                # end if
            # end if
            
            # information into JSON
            if isinstance(target_object.stable_s_hat, list):
                __weights = target_object.array_s_hat.numpy() if isinstance(target_object.array_s_hat, torch.Tensor) else np.array([])
                self._log_artifacts(logger_oject, target_object.stable_s_hat, __weights)
            # end if
        else:
            raise NotImplementedError("The input object is not supported.")
    
    
    def log(self, loggers: ty.List[Logger], target_object: AcceptableObjectType):
        """Public API. I design this API for logging these two objects,
        1. optimization result of Interpretable-CV-detection.
        2. hyper-parameter search result of baseline regression-based variable selection.
        
        Do logging for these two information,
        1. optimization trajectory. This logging is available only for MMD-optimization.
        2. summary of the optimization. This logging is available for all optimization.
        """
        for __logger in loggers:
            self._log(__logger, target_object)
            __logger.finalize(status='success')
        # end for
        
    def initialize_logger(self, run_name: str, group_name: ty.Optional[str] = None) -> ty.List[Logger]:
        """Public API. Generating a list of loggers, following the default configuration that you put `__init__`.
        
        Parameters
        -----------------------
        run_name: name of the run.
        group_name: name of the group. Only used for `MLFlowLogger` and `WandbLogger`.
            In `MLFlowLogger`, this is the name of the experiment.
            In `WandbLogger`, this is the name of the project.
        """
        loggers = []
        
        for __logger in self.loggers:
            logger_default_config = self.logger2config[__logger]
            if __logger == "mlflow":
                assert "tracking_uri" in logger_default_config, f'tracking_uri must be in the key.'
                __l = pytorch_lightning.loggers.MLFlowLogger(
                    experiment_name=group_name,
                    run_name=run_name,
                    **logger_default_config)
            elif __logger == "tensorboard":
                __l = pytorch_lightning.loggers.TensorBoardLogger(
                    name=run_name,
                    **logger_default_config)
            elif __logger == "wandb":
                __l = pytorch_lightning.loggers.WandbLogger(
                    name=run_name,
                    project=group_name,
                    **logger_default_config)
            elif __logger == "csv":
                __l = pytorch_lightning.loggers.CSVLogger(
                    name=run_name,
                    **logger_default_config)
            else:
                raise NotImplementedError()
            # end if
            loggers.append(__l)
        # end for
        return loggers
        