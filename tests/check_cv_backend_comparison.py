import shutil
from tempfile import mkdtemp
import time

import torch
import pytorch_lightning as pl

from distributed import Client, LocalCluster

import toml
from pathlib import Path

from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import InterpretableMmdTrainParameters
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector import DistributedComputingParameter
from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.cross_validation_detector import (
    CrossValidationAlgorithmParameter,
    CrossValidationTrainParameters,
    CrossValidationInterpretableVariableDetector,
    PostProcessLoggerHandler,
    PytorchLightningDefaultArguments,
    RegularizationParameter
)

import typing as t
import torch
import numpy as np
import random
import dask
import dask.config


def sample_gaussian(x, sample: int) -> np.ndarray:
    return np.random.normal(loc=1000, scale=1, size=(sample,))


def test_data_xy_linear(dim_size: int = 20,
                        sample_size: int = 1000,
                        ratio_dependent_variables: float = 0.1,
                        func_noise_function: t.Callable[[float, int], np.ndarray] = sample_gaussian
                        ) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.List[int]]:
    """
    :param dim_size:
    :param sample_size:
    :param ratio_dependent_variables:
    :return: (data samples, ground-truth)
    """
    # dimension size
    sample_x = np.random.normal(loc=10, scale=1, size=(sample_size, dim_size))
    sample_y = np.random.normal(loc=10, scale=1, size=(sample_size, dim_size))

    # the number of dimensions to be replaced
    n_replaced = int(dim_size * ratio_dependent_variables)
    dim_ground_truth = random.sample(range(0, dim_size), k=n_replaced)

    # transformation equation
    for dim_replace in dim_ground_truth:
        y_value = func_noise_function(0.0, sample_size)
        sample_y[:, dim_replace] = y_value
    # end for

    x_tensor = torch.tensor(sample_x)
    y_tensor = torch.tensor(sample_y)
    return (x_tensor, y_tensor), dim_ground_truth



def test_data_discrete_category(dim_size: int = 20,
                                sample_size: int = 1000,
                                ratio_dependent_variables: float = 0.1,
                                num_max_category: int = 5
                                ) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.List[int]]:
    """
    :param dim_size:
    :param sample_size:
    :param ratio_dependent_variables:
    :return: (data samples, ground-truth)
    """
    # dimension size
    sample_x = np.zeros(shape=(sample_size, dim_size))
    sample_y = np.zeros(shape=(sample_size, dim_size))

    # discrete value sampling
    category_labels_population = range(0, num_max_category)
    weights_distribution = np.random.dirichlet(np.ones(num_max_category), size=1)[0]
    for __d in range(0, dim_size):
        sample_x[:, __d] = np.random.choice(category_labels_population, sample_size, p=weights_distribution)
        sample_y[:, __d] = np.random.choice(category_labels_population, sample_size, p=weights_distribution)
    # end for

    # the number of dimensions to be replaced
    n_replaced = int(dim_size * ratio_dependent_variables)
    dim_ground_truth = random.sample(range(0, dim_size), k=n_replaced)

    # transformation equation
    # noise is from poisson distribution
    for dim_replace in dim_ground_truth:
        __poisson_sample = np.random.poisson(num_max_category, sample_size)
        __poisson_sample[__poisson_sample > num_max_category] = 0
        sample_y[:, dim_replace] = __poisson_sample
    # end for

    x_tensor = torch.tensor(sample_x)
    y_tensor = torch.tensor(sample_y)
    return (x_tensor, y_tensor), dim_ground_truth



def test_check_comparison_dask_modes(resource_path_root: Path):
    """We have to switch Dask running mode: thread, daemon-true and daemon-false.
    I check if all mode will generate the same output in the almost similar run time.
    """
    dask.config.set(distributed__worker__daemon=False)
    
    temp_dir = Path(mkdtemp())
    temp_dir.mkdir(parents=True, exist_ok=True)

    path_mlflow = temp_dir / "mlruns"
    path_mlflow.mkdir(parents=True, exist_ok=True)

    t_xy, __ = test_data_xy_linear(sample_size=500)
    my_dataset = SimpleDataset(t_xy[0], t_xy[1])
    # ------------------------------------------
    
    candidate_regularization_parameter = [
        RegularizationParameter(0.1, 0.0),
        RegularizationParameter(0.01, 0.0),
    ]
    n_subsampling = 2
    max_epochs = 1000
    
    def run_dask_mode_daemon_one_process():
        # Normal Training
        start_time = time.time()
        cluster = LocalCluster(
            '127.0.0.1:8787',
            n_workers=4, 
            threads_per_worker=1, 
            memory_limit='2GB')
        dask_client = Client(cluster)
        

        algorithm_param = CrossValidationAlgorithmParameter(
            candidate_regularization_parameter=candidate_regularization_parameter,
            n_subsampling=n_subsampling)
        dist_param = DistributedComputingParameter(
            dask_scheduler_address=dask_client.scheduler.address,
            n_joblib=0)
        base_train_param = InterpretableMmdTrainParameters(
            batch_size=-1,
            n_workers_train_dataloader=0,
            n_workers_validation_dataloader=0,
            dataloader_persistent_workers=False,
            optimizer_args={'lr': 0.1})

        ss_param = CrossValidationTrainParameters(
            algorithm_parameter=algorithm_param,
            base_training_parameter=base_train_param,
            distributed_parameter=dist_param,
            computation_backend='dask'
        )
        kernel = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
        mmd_estimator = QuadraticMmdEstimator(kernel)
        pl_argument = PytorchLightningDefaultArguments(
                max_epochs=max_epochs,
                accelerator='cpu')
        ss_trainer = CrossValidationInterpretableVariableDetector(
            pytorch_trainer_config=pl_argument,
            training_parameter=ss_param,
            estimator=mmd_estimator)
        result = ss_trainer.run_cv_detection(
            training_dataset=my_dataset,
            validation_dataset=my_dataset)
        
        end_time = time.time()
        exec_time = end_time - start_time
    
        dask_client.close()
        cluster.close()
        
        shutil.rmtree('/tmp/mmd-tst-variable-detector/lightning_logs')
        
        return result, exec_time
    # end def
    
    def run_dask_mode_daemon_false_multi_prcess():
        start_time = time.time()
        cluster = LocalCluster(
            '127.0.0.1:8787',
            n_workers=4, 
            threads_per_worker=1, 
            memory_limit='2GB')
        
        dask.config.set(distributed__worker__daemon=False)
        dask_client = Client(cluster)
        
        
        algorithm_param = CrossValidationAlgorithmParameter(
            candidate_regularization_parameter=candidate_regularization_parameter,
            n_subsampling=n_subsampling)
        dist_param = DistributedComputingParameter(
            dask_scheduler_address=dask_client.scheduler.address,
            n_joblib=0)
        base_train_param = InterpretableMmdTrainParameters(
            batch_size=-1,
            n_workers_train_dataloader=1,
            n_workers_validation_dataloader=1,
            dataloader_persistent_workers=True,
            optimizer_args={'lr': 0.1})

        ss_param = CrossValidationTrainParameters(
            algorithm_parameter=algorithm_param,
            base_training_parameter=base_train_param,
            distributed_parameter=dist_param,
            computation_backend='dask'
        )
        kernel = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
        mmd_estimator = QuadraticMmdEstimator(kernel)
        pl_argument = PytorchLightningDefaultArguments(
                max_epochs=max_epochs,
                accelerator='cpu')
        ss_trainer = CrossValidationInterpretableVariableDetector(
            pytorch_trainer_config=pl_argument,
            training_parameter=ss_param,
            estimator=mmd_estimator)
        result = ss_trainer.run_cv_detection(
            training_dataset=my_dataset,
            validation_dataset=my_dataset)
        
        end_time = time.time()
        exec_time = end_time - start_time
    
        dask_client.close()
        cluster.close()
        
        shutil.rmtree('/tmp/mmd-tst-variable-detector/lightning_logs')
                    
        return result, exec_time
    
    
    def run_dask_mode_thread():
        start_time = time.time()
        cluster = LocalCluster(
            '127.0.0.1:8787',
            n_workers=4, 
            threads_per_worker=1, 
            memory_limit='2GB',
            processes=False)
        
        dask_client = Client(cluster)
        
        
        algorithm_param = CrossValidationAlgorithmParameter(
            candidate_regularization_parameter=candidate_regularization_parameter,
            n_subsampling=n_subsampling)
        dist_param = DistributedComputingParameter(
            dask_scheduler_address=dask_client.scheduler.address,
            n_joblib=0)
        base_train_param = InterpretableMmdTrainParameters(
            batch_size=-1,
            n_workers_train_dataloader=1,
            n_workers_validation_dataloader=1,
            dataloader_persistent_workers=True,
            optimizer_args={'lr': 0.1})

        ss_param = CrossValidationTrainParameters(
            algorithm_parameter=algorithm_param,
            base_training_parameter=base_train_param,
            distributed_parameter=dist_param,
            computation_backend='dask'
        )
        kernel = QuadraticKernelGaussianKernel.from_dataset(my_dataset)
        mmd_estimator = QuadraticMmdEstimator(kernel)
        pl_argument = PytorchLightningDefaultArguments(
                max_epochs=max_epochs,
                accelerator='cpu')
        ss_trainer = CrossValidationInterpretableVariableDetector(
            pytorch_trainer_config=pl_argument,
            training_parameter=ss_param,
            estimator=mmd_estimator)
        result = ss_trainer.run_cv_detection(
            training_dataset=my_dataset,
            validation_dataset=my_dataset)
        
        end_time = time.time()
        exec_time = end_time - start_time
    
        dask_client.close()
        cluster.close()
        
        shutil.rmtree('/tmp/mmd-tst-variable-detector/lightning_logs')
           
        return result, exec_time        
    # end def
    # ------------------------------------------
    
    result_daemon_false_multi_prcess, exec_time_daemon_false_multi_prcess = run_dask_mode_daemon_false_multi_prcess()
    result_thread, exec_time_thread = run_dask_mode_thread()
    result_daemon_one_process, exec_time_daemon_one_process = run_dask_mode_daemon_one_process()

    msg_true = f'true answer variable -> {__}'
    msg_detection_result = f'{result_daemon_one_process.stable_s_hat}, {result_daemon_false_multi_prcess.stable_s_hat}, {result_thread.stable_s_hat}'
    msg_exec_time = f'{exec_time_daemon_one_process}, {exec_time_daemon_false_multi_prcess}, {exec_time_thread}'
    
    print(msg_true)
    print(msg_detection_result)
    print(msg_exec_time)


if __name__ == '__main__':
    test_check_comparison_dask_modes(Path.cwd())
