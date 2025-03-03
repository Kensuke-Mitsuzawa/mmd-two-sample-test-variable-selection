from pathlib import Path
import toml
import functools
import shutil

import torch
import pytorch_lightning as pl

from distributed import Client, LocalCluster

from mmd_tst_variable_detector.detection_algorithm.search_regularization_min_max.optuna_upper_lower_search import run_parameter_space_search, SelectionResult
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import RegularizationParameter
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdTrainParameters
from mmd_tst_variable_detector.detection_algorithm import PytorchLightningDefaultArguments

# from .. import data_generator
from tests import data_generator


torch.cuda.is_available = lambda : False


def test_run_parameter_space_search(resource_path_root: Path):
    assert (resource_path_root / "test_settings.toml").exists()
    test_config = toml.loads((resource_path_root / "test_settings.toml").open().read())

    # config_obj = test_config[ray_bayesopt_search_regularization_min_max.__name__]

    t_xy, ground_truth = data_generator.test_data_xy_linear(sample_size=200)

    my_dataset = SimpleDataset(t_xy[0], t_xy[1])
    initial_ard_weights = torch.ones(t_xy[0].shape[1])
    kernel_obj = QuadraticKernelGaussianKernel(ard_weights=initial_ard_weights)
    kernel_obj.compute_length_scale_dataset(my_dataset)
    kernel_obj.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel_obj)

    lr_scheduler = functools.partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min')
    training_parameter = InterpretableMmdTrainParameters(
        batch_size=len(my_dataset),
        optimizer_args={'lr': 0.01},
        lr_scheduler=lr_scheduler
    )

        # a set of loggers
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.loggers import MLFlowLogger
    from pytorch_lightning.loggers import TensorBoardLogger
    
    base_loggers = [
        CSVLogger(save_dir='/tmp/test_ray_opt/csv_logger'),
        MLFlowLogger(save_dir='/tmp/test_ray_opt/mlflow_logger'),
        TensorBoardLogger(save_dir='/tmp/test_ray_opt/tensorboard_logger')
    ]

    path_ray_root = Path('/tmp/test_ray_opt')
    
    # dask_cluster = LocalCluster()
    # dask_client = dask_cluster.get_client()
    dask_client = None

    selected_reg_parameters = run_parameter_space_search(
        dataset_train=my_dataset,
        pytorch_trainer_config=PytorchLightningDefaultArguments(max_epochs=10, enable_progress_bar=False),
        mmd_estimator=mmd_estimator,
        base_training_parameter=training_parameter,
        initial_regularization_search_search_lower=RegularizationParameter(0.01, 0.0),
        n_trials=2,
        dask_client=dask_client
    )
    assert isinstance(selected_reg_parameters, SelectionResult)

    shutil.rmtree(path_ray_root.as_posix())


# if __name__ == '__main__':
#     p = Path('../testresources')
#     test_ray_bayesopt_search_regularization_min_max(p)
