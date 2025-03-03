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


class ArdWeightsEarlyStopping(EarlyStopping):
    """auto-stopping-rules if the discrete variable is same during the certain epochs."""

    def __init__(
        self,
        check_per_iteration: int = 1,
        span_convergence_decision: int = 50,
        ignore_epochs: int = 200,
        ratio_convergence_threshold: float = 0.1,
        check_on_train_epoch_end: bool = False):
        """"
        Parameters
        ------------
        check_per_iteration:
            The number of epochs to check the convergence.
        span_convergence_decision:
        ratio_convergence_threshold:
        ignore_epochs:
            The number of epochs to ignore the convergence check.
            The early stopping is not performed during this period.
            The stopping criteria starts from this epoch.
            Hence, the minimul stopping epoch will be `ignore_epochs + (check_per_iteration * limit_same_variable)`.
        check_on_train_epoch_end:
            If True, the convergence check is performed at the end of each epoch.
        """
        assert ignore_epochs >= 200, "ignore_epochs must be greater than 200. For the safety reasons."
        
        EarlyStopping.__init__(
            self,
            monitor='variance',
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )

        # --------------------
        # attributes
        self.ard_weights_initial: ty.Optional[torch.Tensor] = None
        self.__record_ws_distance = []
        
        self.ratio_convergence_threshold = ratio_convergence_threshold
        self.check_per_iteration = check_per_iteration
        self.span_convergence_decision = span_convergence_decision
        self.ignore_epochs = ignore_epochs
        # --------------------
        
    def __is_converged(self, trainer: "pl.Trainer") -> ty.Tuple[bool, float]:
        """checking convergence of the Wasserstein distance.
        
        The convegence decision is by ratio of the check span, `span_convergence_decision`.
        The decision is by a slope of ws distance.
        """
        current_epoch = trainer.current_epoch

        seq_ws_distance = self.__record_ws_distance[(current_epoch - self.span_convergence_decision):]
        current_ratio = abs(seq_ws_distance[-1] - seq_ws_distance[0]) / self.span_convergence_decision
        
        if current_ratio <= self.ratio_convergence_threshold:
            return True, current_ratio
        else:
            return False, current_ratio
        
    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        current_epoch = trainer.current_epoch
        
        if current_epoch == 0 or self.ard_weights_initial is None:
            __ard_weights_initial = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach().cpu() ** 2
            self.ard_weights_initial = __ard_weights_initial / torch.max(__ard_weights_initial)
            return 
        # end if
        
        # computing ws distance, saving inso a list
        __ard_w_current = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach().cpu() ** 2
        ard_w_current = __ard_w_current / torch.max(__ard_w_current)
        ws_distance = wasserstein_distance(self.ard_weights_initial.numpy(), ard_w_current.numpy())
        self.__record_ws_distance.append(ws_distance)

        # comment: shortcutting blocks. Do not evaluate variables.
        if current_epoch < self.ignore_epochs:
            return
        # end if
        if current_epoch % self.check_per_iteration != 0:
            return
        # end if
        
        # --------------------
        is_converged, current_ratio = self.__is_converged(trainer)

        if is_converged:
            self.stopped_epoch = True
            trainer.should_stop = True
            msg = f"Stopped because of convergence in WS-distance of ARD weights. {current_ratio} during {self.span_convergence_decision} epochs."
            logger.debug(msg)
            reason = msg
        
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)


# ----------------------------------------


