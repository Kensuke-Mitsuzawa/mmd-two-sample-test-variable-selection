import shutil
import tempfile
from pathlib import Path
import pytest
import torch

from mmd_tst_variable_detector.accelerator_optimizations import DeviceSlotManager
from mmd_tst_variable_detector.datasets import SimpleDataset
from mmd_tst_variable_detector.detection_algorithm.detection_algorithm_one import (
    detection_algorithm_one,
    AlgorithmOneResult,
    AlgorithmOneIndividualResult,
)
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
    InterpretableMmdTrainParameters,
    RegularizationParameter,
)
from mmd_tst_variable_detector.detection_algorithm.pytorch_lightning_trainer import (
    PytorchLightningDefaultArguments,
)
from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
from tests import data_generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for concurrent GPU test")
def test_detection_algorithm_one_concurrent_gpu():
    """Verify detection_algorithm_one executes correctly using concurrent GPU dispatcher."""
    t_xy, ground_truth = data_generator.test_data_xy_linear(
        sample_size=60,
        dim_size=6,
        ratio_dependent_variables=0.3,
        random_seed=42,
    )
    dataset_train = SimpleDataset(t_xy[0][:30], t_xy[1][:30])
    dataset_dev = SimpleDataset(t_xy[0][30:45], t_xy[1][30:45])
    dataset_test = SimpleDataset(t_xy[0][45:], t_xy[1][45:])

    initial_ard = torch.ones(t_xy[0].shape[1])
    kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
    kernel.compute_length_scale_dataset(dataset_train)
    kernel.set_length_scale()
    mmd_estimator = QuadraticMmdEstimator(kernel)

    training_params = InterpretableMmdTrainParameters(batch_size=30)
    pl_config = PytorchLightningDefaultArguments(
        accelerator="gpu",
        devices=[0],
        max_epochs=3,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    cluster, client = DeviceSlotManager.create_gpu_cluster(
        n_gpus=1,
        k_slots_per_gpu=2,
    )

    candidate_regularizations = [
        RegularizationParameter(0.01, 0.0),
        RegularizationParameter(0.05, 0.0),
    ]

    temp_dir = Path(tempfile.mkdtemp()) / "test_algo_one_concurrent_gpu"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = detection_algorithm_one(
            mmd_estimator=mmd_estimator,
            pytorch_trainer_config=pl_config,
            base_training_parameter=training_params,
            dataset_training=dataset_train,
            dataset_dev=dataset_dev,
            dataset_test=dataset_test,
            candidate_regularization_parameters=candidate_regularizations,
            path_work_dir=temp_dir,
            dask_client=client,
            n_permutation_test=5,
        )

        assert isinstance(result, AlgorithmOneResult)
        assert isinstance(result.selected_model, AlgorithmOneIndividualResult)
        assert len(result.trained_models) == 2
        for model in result.trained_models:
            assert isinstance(model, AlgorithmOneIndividualResult)
            assert model.trained_ard_weights is not None
        # end for
    finally:
        client.close()
        cluster.close()
        shutil.rmtree(temp_dir.as_posix(), ignore_errors=True)
    # end try
