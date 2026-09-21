import shutil
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch

from mmd_tst_variable_detector.interface import (
    Interface,
    InterfaceConfigArgs,
    DetectorAlgorithmConfigArgs,
    MmdOptimizationOption,
)
from mmd_tst_variable_detector.interface.module_configs import (
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DistributedConfigArgs,
    CvSelectionConfigArgs,
    AlgorithmOneConfigArgs,
    BaselineMmdConfigArgs,
)
from mmd_tst_variable_detector.assessment_helper.data_generator import sampling_from_distribution
from mmd_tst_variable_detector.detection_algorithm.pure_pytorch_trainer import PurePytorchTrainer
from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
    InterpretableMmdDetector,
    LegacyInterpretableMmdDetector,
)
from mmd_tst_variable_detector.detection_algorithm.pytorch_lightning_trainer import (
    create_mmd_trainer,
    get_mmd_detector_class,
    PytorchLightningDefaultArguments,
)
from mmd_tst_variable_detector.detection_algorithm.search_regularization_min_max import RegularizationSearchParameters


def test_mmd_optimization_option_defaults_and_resolutions():
    """Verify default values and accelerator resolution."""
    opt = MmdOptimizationOption()
    assert opt.trainer_backend == "pure_pytorch"
    assert opt.matrix_computation == "auto"
    assert opt.use_fused_kernel is None
    assert opt.use_legacy_optimization is False

    # Resolution on GPU
    resolved_gpu = opt.resolve_for_accelerator("cuda")
    assert resolved_gpu.trainer_backend == "pure_pytorch"
    assert resolved_gpu.matrix_computation == "fused"
    assert resolved_gpu.use_fused_kernel is True

    # Resolution on CPU
    resolved_cpu = opt.resolve_for_accelerator("cpu")
    assert resolved_cpu.trainer_backend == "pure_pytorch"
    assert resolved_cpu.matrix_computation == "eager"
    assert resolved_cpu.use_fused_kernel is False

    # Resolution with legacy flow
    legacy_opt = MmdOptimizationOption(use_legacy_optimization=True)
    resolved_legacy = legacy_opt.resolve_for_accelerator("cuda")
    assert resolved_legacy.trainer_backend == "lightning"
    assert resolved_legacy.matrix_computation == "eager"
    assert resolved_legacy.use_fused_kernel is False
    assert resolved_legacy.use_legacy_optimization is True


def test_detector_algorithm_config_args_sync():
    """Verify DetectorAlgorithmConfigArgs synchronizes options properly."""
    # Test setting via individual fields
    config1 = DetectorAlgorithmConfigArgs(
        trainer_backend="pure_pytorch",
        matrix_computation="fused",
        use_fused_kernel=True,
    )
    assert config1.mmd_optimization_option is not None
    assert config1.mmd_optimization_option.trainer_backend == "pure_pytorch"
    assert config1.mmd_optimization_option.use_fused_kernel is True

    # Test setting via dict
    config2 = DetectorAlgorithmConfigArgs(
        mmd_optimization_option={"trainer_backend": "lightning", "use_legacy_optimization": True}
    )
    assert isinstance(config2.mmd_optimization_option, MmdOptimizationOption)
    assert config2.trainer_backend == "lightning"
    assert config2.use_legacy_optimization is True


def test_factory_functions():
    """Verify create_mmd_trainer and get_mmd_detector_class."""
    # Test detector class selection
    assert get_mmd_detector_class(use_legacy_optimization=False) is InterpretableMmdDetector
    assert get_mmd_detector_class(use_legacy_optimization=True) is LegacyInterpretableMmdDetector

    # Test pure_pytorch trainer creation
    cfg = PytorchLightningDefaultArguments(max_epochs=10, accelerator="auto")
    trainer_pure = create_mmd_trainer(cfg, trainer_backend="pure_pytorch", use_fused_kernel=True)
    assert isinstance(trainer_pure, PurePytorchTrainer)
    assert trainer_pure.use_fused_kernel is True
    assert trainer_pure.max_epochs == 10

    # Test lightning trainer creation
    import pytorch_lightning as pl
    trainer_pl = create_mmd_trainer(cfg, trainer_backend="lightning")
    assert isinstance(trainer_pl, pl.Trainer)


def _generate_test_data():
    x_np, y_np, _ = sampling_from_distribution(
        n_sample=30,
        dimension_size=5,
        mixture_rate=0.2,
        distribution_conf_p={"type": "gaussian", "mu": 0.0, "sigma": 1.0},
        distribution_conf_q={"type": "gaussian", "mu": 2.0, "sigma": 1.0},
    )
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def test_interface_algorithm_one_with_pure_pytorch_and_fused():
    """Test AlgorithmOne through Interface using PurePyTorch and fused kernel on CUDA."""
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    data_x, data_y = _generate_test_data()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        config_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                path_work_dir=temp_dir,
                train_accelerator=accelerator,
                distributed_config_detection=DistributedConfigArgs(distributed_mode="single"),
            ),
            approach_config_args=ApproachConfigArgs(
                approach_data_representation="sample_based",
                approach_variable_detector="interpretable_mmd",
                approach_interpretable_mmd="algorithm_one",
            ),
            data_config_args=DataSetConfigArgs(
                data_x_train=data_x,
                data_y_train=data_y,
                data_x_test=None,
                data_y_test=None,
                dataset_type_backend="ram",
                dataset_type_charactersitic="static",
            ),
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                trainer_backend="pure_pytorch",
                matrix_computation="auto",
                mmd_algorithm_one_args=AlgorithmOneConfigArgs(
                    max_epoch=20,
                    n_permutation_test=10,
                    parameter_search_parameter=RegularizationSearchParameters(
                        n_search_iteration=2,
                        n_regularization_parameter=2,
                    ),
                ),
            ),
        )

        interface_inst = Interface(config_args)
        interface_inst.fit()
        result = interface_inst.get_result()

        assert result.detection_result_sample_based is not None
        assert isinstance(result.detection_result_sample_based.weights, np.ndarray)
        assert len(result.detection_result_sample_based.weights) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_interface_cv_selection_with_pure_pytorch():
    """Test CrossValidation through Interface using PurePyTorch."""
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    data_x, data_y = _generate_test_data()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        config_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                path_work_dir=temp_dir,
                train_accelerator=accelerator,
                distributed_config_detection=DistributedConfigArgs(distributed_mode="single"),
            ),
            approach_config_args=ApproachConfigArgs(
                approach_data_representation="sample_based",
                approach_variable_detector="interpretable_mmd",
                approach_interpretable_mmd="cv_selection",
            ),
            data_config_args=DataSetConfigArgs(
                data_x_train=data_x,
                data_y_train=data_y,
                data_x_test=None,
                data_y_test=None,
                dataset_type_backend="ram",
                dataset_type_charactersitic="static",
            ),
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                trainer_backend="pure_pytorch",
                matrix_computation="auto",
                mmd_cv_selection_args=CvSelectionConfigArgs(
                    max_epoch=20,
                    n_subsampling=2,
                    n_permutation_test=10,
                    parameter_search_parameter=RegularizationSearchParameters(
                        n_search_iteration=1,
                        n_regularization_parameter=1,
                    ),
                ),
            ),
        )

        interface_inst = Interface(config_args)
        interface_inst.fit()
        result = interface_inst.get_result()

        assert result.detection_result_sample_based is not None
        assert isinstance(result.detection_result_sample_based.weights, np.ndarray)
        assert len(result.detection_result_sample_based.weights) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_interface_legacy_optimization_flow():
    """Test legacy optimization flow through Interface."""
    data_x, data_y = _generate_test_data()
    temp_dir = Path(tempfile.mkdtemp())

    try:
        config_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                path_work_dir=temp_dir,
                train_accelerator="cpu",
                distributed_config_detection=DistributedConfigArgs(distributed_mode="single"),
            ),
            approach_config_args=ApproachConfigArgs(
                approach_data_representation="sample_based",
                approach_variable_detector="interpretable_mmd",
                approach_interpretable_mmd="baseline_mmd",
            ),
            data_config_args=DataSetConfigArgs(
                data_x_train=data_x,
                data_y_train=data_y,
                data_x_test=None,
                data_y_test=None,
                dataset_type_backend="ram",
                dataset_type_charactersitic="static",
            ),
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                use_legacy_optimization=True,
                mmd_baseline_args=BaselineMmdConfigArgs(
                    max_epoch=10,
                    n_permutation_test=10,
                ),
            ),
        )

        assert config_args.detector_algorithm_config_args.use_legacy_optimization is True

        interface_inst = Interface(config_args)
        interface_inst.fit()
        result = interface_inst.get_result()

        assert result.detection_result_sample_based is not None
        assert isinstance(result.detection_result_sample_based.weights, np.ndarray)
        assert len(result.detection_result_sample_based.weights) == 5
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
