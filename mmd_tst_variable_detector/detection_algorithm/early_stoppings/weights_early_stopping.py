import typing as ty
import logging

import pytorch_lightning as pl
from scipy.stats import wasserstein_distance
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping

from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class LegacyArdWeightsEarlyStopping(EarlyStopping):
    """Legacy auto-stopping rule based on ARD weights Wasserstein distance."""

    def __init__(
        self,
        check_per_iteration: int = 1,
        span_convergence_decision: int = 50,
        ignore_epochs: int = 200,
        ratio_convergence_threshold: float = 0.1,
        check_on_train_epoch_end: bool = False
    ):
        assert ignore_epochs >= 200, "ignore_epochs must be greater than 200. For safety reasons."
        super().__init__(
            monitor='variance',
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        self.ard_weights_initial: ty.Optional[torch.Tensor] = None
        self.__record_ws_distance = []
        self.ratio_convergence_threshold = ratio_convergence_threshold
        self.check_per_iteration = check_per_iteration
        self.span_convergence_decision = span_convergence_decision
        self.ignore_epochs = ignore_epochs
    # end def

    def __is_converged(self, trainer: "pl.Trainer") -> ty.Tuple[bool, float]:
        current_epoch = trainer.current_epoch
        seq_ws_distance = self.__record_ws_distance[(current_epoch - self.span_convergence_decision):]
        current_ratio = abs(seq_ws_distance[-1] - seq_ws_distance[0]) / self.span_convergence_decision
        if current_ratio <= self.ratio_convergence_threshold:
            return True, current_ratio
        else:
            return False, current_ratio
        # end if
    # end def

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or self.ard_weights_initial is None:
            __ard_weights_initial = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach().cpu() ** 2
            self.ard_weights_initial = __ard_weights_initial / torch.max(__ard_weights_initial)
            return
        # end if

        __ard_w_current = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach().cpu() ** 2
        ard_w_current = __ard_w_current / torch.max(__ard_w_current)
        ws_distance = wasserstein_distance(self.ard_weights_initial.numpy(), ard_w_current.numpy())
        self.__record_ws_distance.append(ws_distance)

        if current_epoch < self.ignore_epochs:
            return
        # end if
        if current_epoch % self.check_per_iteration != 0:
            return
        # end if

        is_converged, current_ratio = self.__is_converged(trainer)
        if is_converged:
            self.stopped_epoch = True
            trainer.should_stop = True
            msg = f"Stopped because of convergence in WS-distance of ARD weights. {current_ratio} during {self.span_convergence_decision} epochs."
            logger.debug(msg)
            reason = msg
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)
            # end if
        # end if
    # end def
# end class


class ArdWeightsEarlyStopping(LegacyArdWeightsEarlyStopping):
    """Optimized auto-stopping rule based on ARD weights Wasserstein distance."""

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or self.ard_weights_initial is None:
            ard_w_raw = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach() ** 2
            ard_w_max = torch.max(ard_w_raw)
            self.ard_weights_initial = (ard_w_raw / ard_w_max).cpu()
            return
        # end if

        # Skip costly CPU sync and scipy call until near the convergence decision span
        start_tracking_epoch = max(0, self.ignore_epochs - self.span_convergence_decision)
        if current_epoch < start_tracking_epoch:
            return
        # end if

        ard_w_raw = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach() ** 2
        ard_w_max = torch.max(ard_w_raw)
        ard_w_current = (ard_w_raw / ard_w_max).cpu()
        ws_distance = wasserstein_distance(self.ard_weights_initial.numpy(), ard_w_current.numpy())
        self._LegacyArdWeightsEarlyStopping__record_ws_distance.append(ws_distance)

        if current_epoch < self.ignore_epochs:
            return
        # end if
        if current_epoch % self.check_per_iteration != 0:
            return
        # end if

        record_ws = self._LegacyArdWeightsEarlyStopping__record_ws_distance
        if len(record_ws) < self.span_convergence_decision:
            return
        # end if

        seq_ws_distance = record_ws[-self.span_convergence_decision:]
        current_ratio = abs(seq_ws_distance[-1] - seq_ws_distance[0]) / self.span_convergence_decision

        if current_ratio <= self.ratio_convergence_threshold:
            self.stopped_epoch = True
            trainer.should_stop = True
            msg = f"Stopped because of convergence in WS-distance of ARD weights. {current_ratio} during {self.span_convergence_decision} epochs."
            logger.debug(msg)
            if self.verbose:
                self._log_info(trainer, msg, self.log_rank_zero_only)
            # end if
        # end if
    # end def
# end class
