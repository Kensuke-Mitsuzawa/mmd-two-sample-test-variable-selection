import math
import typing as ty
import abc
import copy
import logging

import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import Callback, EarlyStopping

from .trajectory_noise_reduction import (
    TrajectoryNoiseReduction,
    MinMaxScalerVectorized,
    DefaultSavgolParameters,
    LegacyTrajectoryNoiseReduction,
    LegacyMinMaxScalerVectorized,
)
from ...detection_algorithm.commons import TrajectoryRecord
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Legacy Implementations (Preserved for Benchmark Comparison: regex ^Legacy)
# ---------------------------------------------------------------------------

class LegacyBaseAutoStopRule(EarlyStopping, LegacyTrajectoryNoiseReduction, metaclass=abc.ABCMeta):
    """Legacy base class of auto-stopping-rules"""

    def __init__(
        self,
        monitor="both_loss",
        check_span=100,
        ignore_epochs: int = 500,
        check_on_train_epoch_end: bool = False,
        is_noise_reduction: bool = True,
        algorithm_noise_reduction: str = "savgol_filter",
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters),
    ):
        EarlyStopping.__init__(
            self,
            monitor=monitor,
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        LegacyTrajectoryNoiseReduction.__init__(
            self,
            check_span=check_span,
            ignore_epochs=ignore_epochs,
            algorithm_noise_reduction=algorithm_noise_reduction,
            args_noise_reduction=args_noise_reduction,
        )
        self.min_max_scaler = LegacyMinMaxScalerVectorized(feature_range=(0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.computed_ratio = -1.0
    # end def
# end class


class LegacyConvergenceEarlyStop(LegacyBaseAutoStopRule):
    """Legacy custom convergence-monitoring early stopping rule."""

    def __init__(
        self,
        monitor="both_loss",
        check_span: int = 100,
        ignore_epochs: int = 500,
        threshold_convergence_ratio: float = 0.001,
        threshold_oscillation_abs_range: float = -1,
        check_on_train_epoch_end: bool = True,
        is_noise_reduction: bool = False,
        algorithm_noise_reduction: str = "savgol_filter",
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters),
        case_insufficient_indicator: str = 'warning'
    ):
        assert monitor in ('both_loss', 'train_loss', 'val_loss'), f"monitor must be 'both_loss', 'train_loss', or 'val_loss'. But {monitor}."
        super().__init__(
            monitor=monitor,
            check_on_train_epoch_end=check_on_train_epoch_end,
            check_span=check_span,
            is_noise_reduction=is_noise_reduction,
            algorithm_noise_reduction=algorithm_noise_reduction,
            args_noise_reduction=args_noise_reduction,
        )
        self.threshold_convergence_ratio = threshold_convergence_ratio
        self.threshold_oscillation_abs_range = threshold_oscillation_abs_range
        self.ignore_epochs = ignore_epochs
        assert case_insufficient_indicator in ['warning', 'ignore', 'exception']
        self.case_insufficient_indicator = case_insufficient_indicator
    # end def

    @staticmethod
    def __generate_combine_stack(
        tensor_loss_training: torch.Tensor,
        tensor_loss_validation: torch.Tensor
    ) -> torch.Tensor:
        tensor_loss_both = tensor_loss_training + tensor_loss_validation
        return tensor_loss_both
    # end def

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        logs = trainer.callback_metrics
        __monitor_field = 'loss' if self.monitor == 'both_loss' else self.monitor

        if (
            self.monitor not in list(logs.keys()) and self.monitor != "both_loss"
        ):
            return
        # end if

        current_epoch = trainer.current_epoch
        span_cehck_start = current_epoch - self.check_span

        if current_epoch < self.ignore_epochs:
            return
        # end if

        tensor_loss_training: torch.Tensor = trainer.lightning_module.loss_training  # type: ignore
        tensor_loss_validation: torch.Tensor = trainer.lightning_module.loss_validation  # type: ignore

        tensor_loss_training = torch.nan_to_num(tensor_loss_training)
        tensor_loss_validation = torch.nan_to_num(tensor_loss_validation)

        if self.monitor == 'both_loss':
            tensor_loss_both = self.__generate_combine_stack(tensor_loss_training, tensor_loss_validation)
            tensor_loss = tensor_loss_both
        elif self.monitor == 'train_loss':
            tensor_loss = tensor_loss_training
        elif self.monitor == 'val_loss':
            tensor_loss = tensor_loss_validation
        else:
            raise NotImplementedError(f'No monitor metric is defined. {self.monitor}')
        # end if

        if self.is_noise_reduction:
            __ = self.run_noise_reduction(tensor_loss.detach().cpu().numpy())
            tensor_history = torch.tensor(__)
        else:
            tensor_history = tensor_loss
        # end if

        scaled_indicator = self.min_max_scaler(tensor_history) + 1
        scaled_indicator_in_range = scaled_indicator[span_cehck_start:(current_epoch + 1)]

        should_stop, computed_ratio = self.is_convergence_optimization(
            scaled_indicator_in_range,
            threshold_convergence_ratio=self.threshold_convergence_ratio,
            threshold_oscilation_abs_range=self.threshold_oscillation_abs_range,
        )
        reason = f"Stopped because of a convergence on {__monitor_field}"
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            self.computed_ratio = computed_ratio
        # end if
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)
        # end if
    # end def
# end class


# ---------------------------------------------------------------------------
# High-Performance Device-Resident (GPU & CPU) Implementations
# ---------------------------------------------------------------------------

class BaseAutoStopRule(EarlyStopping, TrajectoryNoiseReduction, metaclass=abc.ABCMeta):
    """Device-agnostic base class of auto-stopping-rules (GPU & CPU)."""

    def __init__(
        self,
        monitor="both_loss",
        check_span=100,
        ignore_epochs: int = 500,
        check_on_train_epoch_end: bool = False,
        is_noise_reduction: bool = True,
        algorithm_noise_reduction: str = "savgol_filter",
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters),
    ):
        EarlyStopping.__init__(
            self,
            monitor=monitor,
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        TrajectoryNoiseReduction.__init__(
            self,
            check_span=check_span,
            ignore_epochs=ignore_epochs,
            algorithm_noise_reduction=algorithm_noise_reduction,
            args_noise_reduction=args_noise_reduction,
        )
        self.min_max_scaler = MinMaxScalerVectorized(feature_range=(0.0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.computed_ratio = -1.0
    # end def
# end class


class ConvergenceEarlyStop(BaseAutoStopRule):
    """High-performance convergence-monitoring early stopping rule.

    Performs all operations (slicing, combination, noise reduction, and scaling)
    directly on `self.device` (GPU VRAM or CPU RAM) without synchronization stalls.
    """

    def __init__(
        self,
        monitor="both_loss",
        check_span: int = 100,
        ignore_epochs: int = 500,
        threshold_convergence_ratio: float = 0.001,
        threshold_oscillation_abs_range: float = -1,
        check_on_train_epoch_end: bool = True,
        is_noise_reduction: bool = False,
        algorithm_noise_reduction: str = "savgol_filter",
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters),
        case_insufficient_indicator: str = 'warning'
    ):
        assert monitor in ('both_loss', 'train_loss', 'val_loss'), f"monitor must be 'both_loss', 'train_loss', or 'val_loss'. But {monitor}."
        super().__init__(
            monitor=monitor,
            check_on_train_epoch_end=check_on_train_epoch_end,
            check_span=check_span,
            is_noise_reduction=is_noise_reduction,
            algorithm_noise_reduction=algorithm_noise_reduction,
            args_noise_reduction=args_noise_reduction,
        )
        self.threshold_convergence_ratio = threshold_convergence_ratio
        self.threshold_oscillation_abs_range = threshold_oscillation_abs_range
        self.ignore_epochs = ignore_epochs
        assert case_insufficient_indicator in ['warning', 'ignore', 'exception']
        self.case_insufficient_indicator = case_insufficient_indicator
    # end def

    @staticmethod
    def _generate_combine_stack(
        tensor_loss_training: torch.Tensor,
        tensor_loss_validation: torch.Tensor
    ) -> torch.Tensor:
        """Combine training and validation loss on device."""
        return tensor_loss_training + tensor_loss_validation
    # end def

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Device-resident early stopping convergence check."""
        current_epoch = trainer.current_epoch
        if current_epoch < self.ignore_epochs:
            return
        # end if

        logs = trainer.callback_metrics
        monitor_field = 'loss' if self.monitor == 'both_loss' else self.monitor
        if self.monitor not in list(logs.keys()) and self.monitor != "both_loss":
            return
        # end if

        span_check_start = max(0, current_epoch - self.check_span)
        n_points = current_epoch + 1

        lightning_mod = trainer.lightning_module
        raw_loss_train = getattr(lightning_mod, 'loss_training', None)
        raw_loss_val = getattr(lightning_mod, 'loss_validation', None)

        if raw_loss_train is None:
            return
        # end if

        tensor_loss_training = torch.nan_to_num(raw_loss_train[:n_points])
        if raw_loss_val is not None:
            tensor_loss_validation = torch.nan_to_num(raw_loss_val[:n_points])
        else:
            tensor_loss_validation = torch.zeros_like(tensor_loss_training)
        # end if

        if self.monitor == 'both_loss':
            tensor_loss = self._generate_combine_stack(tensor_loss_training, tensor_loss_validation)
        elif self.monitor == 'train_loss':
            tensor_loss = tensor_loss_training
        elif self.monitor == 'val_loss':
            tensor_loss = tensor_loss_validation
        else:
            raise NotImplementedError(f'No monitor metric is defined: {self.monitor}')
        # end if

        if self.is_noise_reduction:
            tensor_history = self.run_noise_reduction(tensor_loss)
        else:
            tensor_history = tensor_loss
        # end if

        scaled_indicator = self.min_max_scaler(tensor_history) + 1.0
        scaled_indicator_in_range = scaled_indicator[span_check_start:n_points]

        should_stop, computed_ratio = self.is_convergence_optimization(
            scaled_indicator_in_range,
            threshold_convergence_ratio=self.threshold_convergence_ratio,
            threshold_oscilation_abs_range=self.threshold_oscillation_abs_range,
        )

        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            self.computed_ratio = computed_ratio
            reason = f"Stopped because of convergence on {monitor_field}"
            if self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)
            # end if
        # end if
    # end def
# end class
