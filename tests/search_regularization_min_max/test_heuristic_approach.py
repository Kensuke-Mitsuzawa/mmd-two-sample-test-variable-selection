from pathlib import Path
import toml
import functools

import torch
import pytorch_lightning as pl

# from mmd_tst_variable_detector.detection_algorithm.search_regularization_min_max.hueristic_approach import heuristic_search_regularization_min_max, SelectionResult
from mmd_tst_variable_detector.detection_algorithm.search_regularization_min_max.hueristic_approach_dask import run_parameter_space_search, SelectionResult
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.early_stoppings import ConvergenceEarlyStop
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import RegularizationParameter
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdTrainParameters
from mmd_tst_variable_detector.detection_algorithm import PytorchLightningDefaultArguments

from distributed import Client, LocalCluster

# from .. import data_generator
from tests import data_generator


torch.cuda.is_available = lambda : False


def run_dask_mode_thread() -> Client:
    cluster = LocalCluster(
        '127.0.0.1:8787',
        n_workers=4, 
        threads_per_worker=1, 
        memory_limit='2GB',
        processes=False)
    
    dask_client = Client(cluster)

    return dask_client


def test_search_regularization_min_max(resource_path_root: Path):
    assert (resource_path_root / "test_settings.toml").exists()
    test_config = toml.loads((resource_path_root / "test_settings.toml").open().read())

    config_obj = test_config[test_search_regularization_min_max.__name__]

    t_xy, ground_truth = data_generator.test_data_xy_linear(sample_size=200)

    my_dataset = SimpleDataset(t_xy[0], t_xy[1])
    initial_ard_weights = torch.ones(t_xy[0].shape[1])
    kernel_obj = QuadraticKernelGaussianKernel(ard_weights=initial_ard_weights)
    kernel_obj.compute_length_scale_dataset(my_dataset)
    kernel_obj.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel_obj)

    DefaultEarlyStoppingRule = ConvergenceEarlyStop()

    pl_trainer_conf = PytorchLightningDefaultArguments(
        accelerator='cpu',
        max_epochs=config_obj['max_epochs'],
        callbacks=DefaultEarlyStoppingRule,
        check_val_every_n_epoch=100,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    lr_scheduler = functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min')
    training_parameter = InterpretableMmdTrainParameters(
        batch_size=len(my_dataset),
        optimizer_args={'lr': 0.01},
        lr_scheduler=lr_scheduler
    )

    # dask_client = run_dask_mode_thread()
    # max_try = 10
    dask_client = None
    max_try = 2

    selected_reg_parameters = run_parameter_space_search(
        dataset=my_dataset,
        mmd_estimator=mmd_estimator,
        training_parameter=training_parameter,
        pytorch_trainer_config=pl_trainer_conf,
        initial_regularization_search=RegularizationParameter(0.001, 0.0),
        max_try=max_try,
        dask_client=dask_client
    )
    assert isinstance(selected_reg_parameters, SelectionResult)


if __name__ == '__main__':
    p = Path('../testresources')
    test_search_regularization_min_max(p)
