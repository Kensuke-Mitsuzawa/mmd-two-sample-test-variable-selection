import typing
import logging

import toml
import tempfile
import functools
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers

from mmd_tst_variable_detector.detection_algorithm.commons import RegularizationParameter
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import (QuadraticMmdEstimator, LinearMmdEstimator, MmdValues)
from mmd_tst_variable_detector.kernels.gaussian_kernel import (QuadraticKernelGaussianKernel, LinearMMDGaussianKernel)
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
    InterpretableMmdTrainResult, 
    InterpretableMmdTrainParameters, 
    InterpretableMmdDetector)
from mmd_tst_variable_detector.detection_algorithm import PytorchLightningDefaultArguments
from mmd_tst_variable_detector.datasets.ram_backend_static_dataset import SimpleDataset
from mmd_tst_variable_detector.detection_algorithm import ConvergenceEarlyStop, VariableEarlyStopping, ArdWeightsEarlyStopping
from mmd_tst_variable_detector import logger as debug_logger
from mmd_tst_variable_detector.utils import evaluate_variable_detection

from mmd_tst_variable_detector.exceptions import OptimizationException, SameDataException

from . import data_generator


# import data_generator

debug_logger.setLevel(logging.DEBUG)

random_seed = 1234

def test_early_stopping_nan_objective_value():
    """Test if the early stopping works when the objective value is nan."""
    
    # Test condition depends on the random sampling.
    # So, I iterate until the test condition is satisfied.
    true_test_confition = False
    while true_test_confition:
        
        random_gen = np.random.default_rng(random_seed)
    
        data_x = torch.from_numpy(random_gen.normal(loc=0, scale=1.0, size=(100, 10)))
        data_y = torch.from_numpy(random_gen.normal(loc=0, scale=1.0, size=(100, 10)))
        
        my_dataset = SimpleDataset(data_x, data_y)

        kernel = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
        mmd_estimator = QuadraticMmdEstimator(kernel)

        parameters = InterpretableMmdTrainParameters(
            batch_size=-1,
            regularization_parameter=RegularizationParameter(0.001, 0.0),
            limit_steps_early_stop_nan=100,
            is_use_log=1)

        try:
            var_trainer = InterpretableMmdDetector(
                mmd_estimator=mmd_estimator, 
                training_parameter=parameters, 
                dataset_train=my_dataset, 
                dataset_validation=my_dataset)
        except OptimizationException:
            debug_logger.debug("OptimizationException is raised. Test pass.")
            return True
        # end try
        trainer_args = PytorchLightningDefaultArguments(
            max_epochs=500,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir='/tmp'        
        )
        
        trainer = pl.Trainer(**trainer_args.as_dict())
        try:
            trainer.fit(var_trainer)
        except OptimizationException:
            __trained_obj = var_trainer.get_trained_variables()
            __seq_trajectory = __trained_obj.trajectory_record_training
            # loss values should be NAN.
            assert all([np.isnan(__traj.loss) for __traj in __seq_trajectory[:-100]])
            
            debug_logger.debug("OptimizationException is raised. Test pass.")
            true_test_confition = True
            return True
        else:
            raise AssertionError("OptimizationException is not raised.")
        # end try


def test_early_stopping_negative_mmd():
    """Test if the early stopping works when the objective value is negative.
    Condition: data from the same prob. distributions, small regularization parameter.
    """
    # Test condition depends on the random sampling.
    # So, I iterate until the test condition is satisfied.
    true_test_confition = False
    while true_test_confition:
        
        random_gen = np.random.default_rng(random_seed)
        
        data_x = torch.from_numpy(random_gen.normal(loc=0, scale=1.0, size=(100, 10)))
        data_y = torch.from_numpy(random_gen.normal(loc=0, scale=1.0, size=(100, 10)))
        
        my_dataset = SimpleDataset(data_x, data_y)

        kernel = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
        mmd_estimator = QuadraticMmdEstimator(kernel)

        parameters = InterpretableMmdTrainParameters(
            batch_size=-1,
            regularization_parameter=RegularizationParameter(0.0, 0.0),
            limit_steps_early_stop_nan=100,
            is_use_log=-1)
        var_trainer = InterpretableMmdDetector(
            mmd_estimator=mmd_estimator, 
            training_parameter=parameters, 
            dataset_train=my_dataset, 
            dataset_validation=my_dataset)
        trainer_args = PytorchLightningDefaultArguments(
            max_epochs=500,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir='/tmp'        
        )
        
        trainer = pl.Trainer(**trainer_args.as_dict())
        try:
            trainer.fit(var_trainer)
        except OptimizationException:
            __trained_obj = var_trainer.get_trained_variables()
            __seq_trajectory = __trained_obj.trajectory_record_training
            # MMD value should be below zero.
            assert all([__traj.mmd < 0.0 for __traj in __seq_trajectory[:-100]])            
            
            debug_logger.debug("OptimizationException is raised. Test pass.")
        else:
            raise AssertionError("OptimizationException is not raised.")
        # end try



def init_tensorboard(tensorboard_config: typing.Dict
                     ) -> typing.Optional[pytorch_lightning.loggers.TensorBoardLogger]:
    from pytorch_lightning.loggers import TensorBoardLogger
    is_tensorboard_logger = tensorboard_config['is_log']
    if is_tensorboard_logger:
        return TensorBoardLogger(save_dir=tensorboard_config['save_dir'])
    else:
        return None


def test_quadratic_MmdVariableTrainer(resource_path_root: Path):
    assert (resource_path_root / "test_settings.toml").exists()
    test_config = toml.loads((resource_path_root / "test_settings.toml").open().read())

    objective_functions = ['ratio', 'mmd']
    for __obj_func in objective_functions:

        loggers = []

        if test_config[test_quadratic_MmdVariableTrainer.__name__]['tensorboard_logger']['is_log']:
            __tf_logger = init_tensorboard(test_config[test_quadratic_MmdVariableTrainer.__name__]['tensorboard_logger'])
            debug_logger.debug(f"tensorboard_logger: {__tf_logger}")
            loggers.append(__tf_logger)
        # end if

        t_xy, ground_truth = data_generator.test_data_xy_linear(sample_size=300)
        my_dataset = SimpleDataset(t_xy[0], t_xy[1])

        initial_ard = torch.ones(my_dataset.get_dimension_flattened())

        kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
        kernel.compute_length_scale_dataset(my_dataset)
        kernel.set_length_scale()

        mmd_estimator = QuadraticMmdEstimator(kernel)
        early_stopping = ConvergenceEarlyStop(ignore_epochs=500)
        var_early_stopper = VariableEarlyStopping()
        weights_stopper = ArdWeightsEarlyStopping()

        lr_scheduler = functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min')

        parameters = InterpretableMmdTrainParameters(
            batch_size=test_config[test_quadratic_MmdVariableTrainer.__name__]['batch_size'],
            regularization_parameter=RegularizationParameter(0.001, 0.0),
            objective_function=__obj_func,
            lr_scheduler=lr_scheduler,  # type: ignore
            optimizer_args={'lr': 0.01})
        var_trainer = InterpretableMmdDetector(
            mmd_estimator=mmd_estimator, 
            training_parameter=parameters, 
            dataset_train=my_dataset, 
            dataset_validation=my_dataset)

        trainer = pl.Trainer(
            max_epochs=test_config[test_quadratic_MmdVariableTrainer.__name__]['max_epochs'],
            precision=32,
            callbacks=[weights_stopper],
            logger=loggers,
            accelerator='cpu',
            devices='auto',
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir='/tmp')
        
        # Compiling the MMD opt.
        var_trainer = torch.compile(var_trainer)
        
        trainer.fit(var_trainer)
        trained_var = var_trainer.get_trained_variables()

        trained_var.plot_ard_weights()
        trained_var.plot_trajectory()
        trained_var.plot_ard_weights(mode='matplotlib')
        trained_var.plot_trajectory(mode='matplotlib')


        assert isinstance(trained_var, InterpretableMmdTrainResult)

        res, __varables = evaluate_variable_detection.evaluate_trained_variables(
            trained_var.ard_weights_kernel_k,
            ground_truth)

        if __obj_func == 'ratio':
            assert res.f1 > 0.5, f'Variable selection failed. Something is wrong in the algorithm.'
        
        __temp_model_path = Path(tempfile.mktemp())
        trainer.save_checkpoint(__temp_model_path / 'pl_checkpoint.pt')
        if isinstance(loggers[0], pytorch_lightning.loggers.WandbLogger):
            loggers[0].experiment.config.update({"f1": res.f1, "precision": res.precision, "recall": res.recall})
        # end if
        torch.save(res, __temp_model_path / 'trained_model.pt')

        model_resume = InterpretableMmdDetector.load_from_checkpoint(checkpoint_path=__temp_model_path / 'pl_checkpoint.pt')
        trainer = pl.Trainer(
            max_epochs=2 * test_config[test_quadratic_MmdVariableTrainer.__name__]['max_epochs'],
            precision=32,
            callbacks=early_stopping,
            logger=loggers,
            accelerator='cpu',
            devices='auto',
            enable_progress_bar=True,
            enable_checkpointing=False,
            default_root_dir='/tmp')
        trainer.fit(model_resume)
        if isinstance(loggers[0], pytorch_lightning.loggers.WandbLogger):
            loggers[0].experiment.config.update({"f1_resume": res.f1, "precision_resume": res.precision, "recall_resume": res.recall})
        # end if
        for __logger in loggers:
            if hasattr(__logger, "experiment.finish()"):
                __logger.experiment.finish()



def test_linear_MmdVariableTrainer(resource_path_root: Path):
    assert (resource_path_root / "test_settings.toml").exists()
    test_config = toml.loads((resource_path_root / "test_settings.toml").open().read())

    objective_functions = ['ratio', 'mmd']
    for __obj_func in objective_functions:

        loggers = []
        
        t_xy, ground_truth = data_generator.test_data_xy_linear(sample_size=300)
        my_dataset = SimpleDataset(t_xy[0], t_xy[1])

        initial_ard = torch.ones(my_dataset.get_dimension_flattened())

        kernel = LinearMMDGaussianKernel(ard_weights=initial_ard)
        kernel.compute_length_scale_dataset(my_dataset)
        kernel.set_length_scale()

        mmd_estimator = LinearMmdEstimator(kernel)

        lr_scheduler = functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min')

        parameters = InterpretableMmdTrainParameters(batch_size=test_config[test_linear_MmdVariableTrainer.__name__]['batch_size'],
                                        objective_function=__obj_func,
                                        lr_scheduler=lr_scheduler,  # type: ignore
                                        optimizer_args={'lr': 0.01})
        var_trainer = InterpretableMmdDetector(mmd_estimator=mmd_estimator, training_parameter=parameters,
                                         dataset_train=my_dataset, dataset_validation=my_dataset)
        DefaultEarlyStoppingRule = [ConvergenceEarlyStop(ignore_epochs=500), VariableEarlyStopping()]
        trainer = pl.Trainer(
            max_epochs=test_config[test_linear_MmdVariableTrainer.__name__]['max_epochs'],
            precision=32,
            callbacks=DefaultEarlyStoppingRule,
            logger=loggers,
            accelerator='cpu',
            devices='auto',
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir='/tmp'            
        )
        trainer.fit(var_trainer)
        trained_var = var_trainer.get_trained_variables()
        assert isinstance(trained_var, InterpretableMmdTrainResult)

        res, __variables = evaluate_variable_detection.evaluate_trained_variables(
            trained_var.ard_weights_kernel_k,
            ground_truth)


if __name__ == '__main__':
    p = Path('./testresources')
    # test_linear_MmdVariableTrainer(p)
    test_quadratic_MmdVariableTrainer(p)
