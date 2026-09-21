from dataclasses import dataclass
from typing import List
import typing as ty

from .module_configs.resource_config import ResourceConfigArgs
from .module_configs.approach_config import ApproachConfigArgs
from .module_configs.dataset_config import DataSetConfigArgs
from .module_configs.algorithm_configs.algorithm_config import (
    AlgorithmOneConfigArgs,
    CvSelectionConfigArgs,
    BaselineMmdConfigArgs,
    LinearVariableSelectionConfigArgs
)
from .module_configs.algorithm_configs.module_optimisation_config import (
    MmdOptimisationConfigTemplate,
    ConfigTPamiDraft,
    ConfigRapid,
)


import typing as ty
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch


from ..logger_unit import handler

from ..baselines.regression_based_variable_selection.tst_based_regression_tuner import TstBasedRegressionTunerResult
from ..detection_algorithm.cross_validation_detector.cross_validation_detector import CrossValidationTrainedParameter
from ..detection_algorithm.detection_algorithm_one import AlgorithmOneResult

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



from pydantic import BaseModel, Field, ConfigDict


class MmdOptimizationOption(BaseModel):
    """Pydantic model to configure MMD optimization options.

    Attributes
    ----------
    trainer_backend : str
        Runner engine: 'pure_pytorch' (pure PyTorch loop) or 'lightning' (PyTorch Lightning pl.Trainer).
        Default is 'pure_pytorch'.
    matrix_computation : str
        Pairwise distance and kernel matrix computation mode: 'auto', 'fused' (Triton CUDA kernel),
        or 'eager' (PyTorch native). Default is 'auto' (resolves to 'fused' on GPU, 'eager' on CPU).
    use_fused_kernel : ty.Optional[bool]
        Boolean flag to directly enable/disable the fused Triton CUDA kernel. If None, resolved from matrix_computation.
    use_legacy_optimization : bool
        If True, forces the legacy optimization flow: LegacyInterpretableMmdDetector with pl.Trainer,
        legacy early stoppers, and eager matrix computation. Default is False.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trainer_backend: str = Field(
        default="pure_pytorch",
        description="Runner engine: 'pure_pytorch' or 'lightning'."
    )
    matrix_computation: str = Field(
        default="auto",
        description="Pairwise distance and kernel matrix computation: 'auto', 'fused', or 'eager'."
    )
    use_fused_kernel: ty.Optional[bool] = Field(
        default=None,
        description="Boolean flag to directly enable/disable the fused Triton CUDA kernel."
    )
    use_legacy_optimization: bool = Field(
        default=False,
        description="Forces legacy optimization flow: LegacyInterpretableMmdDetector with pl.Trainer and eager matrix computation."
    )

    def resolve_for_accelerator(self, accelerator: str = "auto") -> "MmdOptimizationOption":
        """Resolve matrix_computation and use_fused_kernel based on target accelerator."""
        if self.use_legacy_optimization:
            return MmdOptimizationOption(
                trainer_backend="lightning",
                matrix_computation="eager",
                use_fused_kernel=False,
                use_legacy_optimization=True,
            )
        # end if
        acc = accelerator.lower() if isinstance(accelerator, str) else "auto"
        is_gpu = acc in ("gpu", "cuda") or (acc == "auto" and torch.cuda.is_available())

        if self.use_fused_kernel is not None:
            fused = bool(self.use_fused_kernel)
            matrix_comp = "fused" if fused else "eager"
        elif self.matrix_computation == "fused":
            fused = True
            matrix_comp = "fused"
        elif self.matrix_computation == "eager":
            fused = False
            matrix_comp = "eager"
        else:  # "auto"
            fused = is_gpu
            matrix_comp = "fused" if is_gpu else "eager"
        # end if

        return MmdOptimizationOption(
            trainer_backend=self.trainer_backend,
            matrix_computation=matrix_comp,
            use_fused_kernel=fused,
            use_legacy_optimization=self.use_legacy_optimization,
        )


@dataclass
class DetectorAlgorithmConfigArgs:
    mmd_cv_selection_args: ty.Optional[ty.Union[str, ty.Dict, CvSelectionConfigArgs]] = None
    mmd_algorithm_one_args: ty.Optional[ty.Union[str, ty.Dict, AlgorithmOneConfigArgs]] = None
    mmd_baseline_args: ty.Optional[ty.Union[str, ty.Dict, BaselineMmdConfigArgs]] = None
    linear_variable_selection_args: ty.Optional[ty.Union[str, ty.Dict, LinearVariableSelectionConfigArgs]] = None
    mmd_optimiser_configs: MmdOptimisationConfigTemplate = ConfigTPamiDraft()

    # MMD optimization controls
    trainer_backend: str = "pure_pytorch"
    matrix_computation: str = "auto"
    use_fused_kernel: ty.Optional[bool] = None
    use_legacy_optimization: bool = False
    mmd_optimization_option: ty.Optional[ty.Union[MmdOptimizationOption, ty.Dict]] = None

    def __post_init__(self):
        if self.mmd_baseline_args == '':
            self.mmd_baseline_args = None
        elif isinstance(self.mmd_baseline_args, dict):
            self.mmd_baseline_args = BaselineMmdConfigArgs(**self.mmd_baseline_args)
        # end if

        if self.mmd_cv_selection_args == '':
            self.mmd_cv_selection_args = None
        elif isinstance(self.mmd_cv_selection_args, dict):
            self.mmd_cv_selection_args = CvSelectionConfigArgs(**self.mmd_cv_selection_args)
        # end if
        if self.mmd_algorithm_one_args == '':
            self.mmd_algorithm_one_args = None
        elif isinstance(self.mmd_algorithm_one_args, dict):
            self.mmd_algorithm_one_args = AlgorithmOneConfigArgs(**self.mmd_algorithm_one_args)
        # end if
        if self.linear_variable_selection_args == '':
            self.linear_variable_selection_args = None
        elif isinstance(self.linear_variable_selection_args, dict):
            self.linear_variable_selection_args = LinearVariableSelectionConfigArgs(**self.linear_variable_selection_args)
        # end if

        # Handle MmdOptimizationOption
        if isinstance(self.mmd_optimization_option, dict):
            self.mmd_optimization_option = MmdOptimizationOption(**self.mmd_optimization_option)
        elif self.mmd_optimization_option is None:
            self.mmd_optimization_option = MmdOptimizationOption(
                trainer_backend=self.trainer_backend,
                matrix_computation=self.matrix_computation,
                use_fused_kernel=self.use_fused_kernel,
                use_legacy_optimization=self.use_legacy_optimization,
            )
        # end if
        # synchronize fields
        self.trainer_backend = self.mmd_optimization_option.trainer_backend
        self.matrix_computation = self.mmd_optimization_option.matrix_computation
        self.use_fused_kernel = self.mmd_optimization_option.use_fused_kernel
        self.use_legacy_optimization = self.mmd_optimization_option.use_legacy_optimization


@dataclass
class InterfaceConfigArgs:
    """The class for configuration of `Interface`.
    
    Parameters
    ----------
    resource_config_args: ResourceConfigArgs
        Configuration for resource.
    approach_config_args: ApproachConfigArgs
        Configuration for approach.
    data_config_args: DataSetConfigArgs
        Configuration for dataset.
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
        Configuration for detector algorithm.
    """
    resource_config_args: ResourceConfigArgs
    approach_config_args: ApproachConfigArgs
    data_config_args: DataSetConfigArgs
    detector_algorithm_config_args: DetectorAlgorithmConfigArgs
    
    def __post_init__(self):
        pass