import math
import typing as t
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
)
from ...detection_algorithm.commons import TrajectoryRecord
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



class BaseAutoStopRule(EarlyStopping, TrajectoryNoiseReduction, metaclass=abc.ABCMeta):
    """A base class of auto-stopping-rules"""

    def __init__(
        self,
        monitor="both_loss",
        check_span=100,
        ignore_epochs: int = 500,
        check_on_train_epoch_end: bool = False,
        is_noise_reduction: bool = True,
        algorithm_noise_reduction: str = "savgol_filter",
        args_noise_reduction: t.Dict = copy.deepcopy(DefaultSavgolParameters),
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
        self.min_max_scaler = MinMaxScalerVectorized(feature_range=(0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.computed_ratio = -1.0


class ConvergenceEarlyStop(BaseAutoStopRule):
    """A custom Convergence-monitoring early stopping rule.

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
        args_noise_reduction: t.Dict = copy.deepcopy(DefaultSavgolParameters),
        case_insufficient_indicator: str = 'warning'
    ):
        """
        
        Parameters
        ---------------
        monitor: 'both_loss` activates "stop" when train_loss and val_loss are converged.
        check_span: Windows size to judge the convergence.
        ignore_epochs: Window size to ignore the rule.
        threshold_convergence_ratio: The threshold of the convergence ratio.
        threshold_oscillation_abs_range:
        check_on_train_epoch_end: If True, the rule is checked on train_epoch_end.
        is_noise_reduction: If True, the noise reduction is applied to the monitored values. Helpful when your batch size is small size.
        algorithm_noise_reduction: The algorithm of the noise reduction.
        args_noise_reduction: The arguments of the noise reduction.
        case_insufficient_indicator: The case when the number of indicators is insufficient.
        """
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

        # self.metric2tuple_index = {
        #     "epoch": 0,
        #     "train_mmd2": 1,
        #     "train_var": 2,
        #     "train_ratio": 3,
        #     "train_loss": 4,
        #     "val_mmd2": 1,
        #     "val_var": 2,
        #     "val_ratio": 3,
        #     "val_loss": 4,
        #     "both_loss": -1,
        # }
        # assert (
        #     monitor in self.metric2tuple_index
        # ), f"{monitor} is not defined in the early stopping rules."
        
        assert case_insufficient_indicator in ['warning', 'ignore', 'exception']
        self.case_insufficient_indicator = case_insufficient_indicator
        
    # @staticmethod
    # def __generate_combine_stack(trainer: "pl.Trainer") -> t.List[TrajectoryRecord]:
    #     # I genrate a combines value "train_loss" : "val_loss"
    #     stack_log_train: t.List[TrajectoryRecord] = trainer.lightning_module.stack_training_log
    #     dict_epoch2log_val: t.Dict[int, TrajectoryRecord] = {
    #         d_obj.epoch: d_obj
    #         for d_obj in trainer.lightning_module.stack_validation_log
    #     }
    #     __stack_los_common = [
    #         (d_obj, dict_epoch2log_val[d_obj.epoch])
    #         for d_obj in stack_log_train
    #         if d_obj.epoch in dict_epoch2log_val
    #     ]
    #     # combine two stack sequence
    #     stack_log = []
    #     for t_d_obj in __stack_los_common:
    #         train_loss = t_d_obj[0].loss
    #         # val_loss may be NAN sometimes.
    #         val_loss = 0.0 if np.isnan(t_d_obj[1].loss) else t_d_obj[1].loss

    #         both_loss = train_loss + val_loss

    #         stack_log.append(
    #             TrajectoryRecord(
    #                 epoch=t_d_obj[0].epoch,
    #                 mmd=-1.0,
    #                 var=-1.0,
    #                 ratio=-1.0,
    #                 loss=both_loss,
    #             )
    #         )

    #     return stack_log


    @staticmethod
    def __generate_combine_stack(
        tensor_loss_training: torch.Tensor,
        tensor_loss_validation: torch.Tensor) -> torch.Tensor:
        
        tensor_loss_both = tensor_loss_training + tensor_loss_validation
        
        return tensor_loss_both


    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        __monitor_field = 'loss' if self.monitor == 'both_loss' else self.monitor

        if (
            self.monitor not in list(logs.keys()) and self.monitor != "both_loss"
        ):  # short circuit if metric not present
            return

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

        # if __monitor_field == "loss":
        #     stack_log = self.__generate_combine_stack(trainer)
        # else:
        #     if "train" in __monitor_field:
        #         stack_log = trainer.lightning_module.stack_training_log
        #     else:
        #         stack_log = trainer.lightning_module.stack_validation_log
        #     # end if
        # # end if
        # array_epoch_number = np.array([d.epoch for d in stack_log])
        
        
        # if len(array_epoch_number[array_epoch_number >= span_cehck_start]) == 0:
        #     return
        # elif len(array_epoch_number[array_epoch_number >= span_cehck_start]) < 2:
        #     if self.case_insufficient_indicator == 'ignore':
        #         return
        #     elif self.case_insufficient_indicator == 'warning':
        #         logger.warning(f"Too few indicators. There are only {len(array_epoch_number)} indicators. Hint: Put more `check_span` and less `check_val_every_n_epoch`.")
        #     elif self.case_insufficient_indicator == 'exception':
        #         msg = "Too few indicators. Put more `check_span` and less `check_val_every_n_epoch`."
        #         msg += f" There are only {len(array_epoch_number)} indicators."
        #         raise Exception(msg)
        #     else:
        #         raise NotImplementedError()
        # # end if

        # monitor_history = [getattr(d, __monitor_field) for d in stack_log]
        # if all(isinstance(v, float) for v in monitor_history):
        #     monitor_history = [torch.tensor([v]) for v in monitor_history]
        # # end if
        # tensor_history = torch.stack(monitor_history)
        # if len(tensor_history.shape) == 2:
        #     tensor_history = tensor_history.reshape(tensor_history.shape[0])
        # # end if
        
        if self.is_noise_reduction:
            __ = self.run_noise_reduction(tensor_loss.detach().cpu().numpy())
            tensor_history = torch.tensor(__)
        else:
            tensor_history = tensor_loss
        # end if

        scaled_indicator = self.min_max_scaler(tensor_history) + 1
        # assert len(scaled_indicator) == len(array_epoch_number)
        # scaled_indicator_in_range = scaled_indicator[
        #     array_epoch_number >= span_cehck_start
        # ]
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
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)


# ----------------------------------------


