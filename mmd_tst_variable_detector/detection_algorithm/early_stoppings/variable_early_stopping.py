import typing as t
import logging

import pytorch_lightning as pl

import numpy as np
import torch

from pytorch_lightning.callbacks import Callback, EarlyStopping
from ...utils import detect_variables
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class VariableEarlyStopping(EarlyStopping):
    """auto-stopping-rules if the discrete variable is same during the certain epochs."""

    def __init__(
        self,
        check_per_iteration: int = 1,
        limit_same_variable: int = 100,
        ignore_epochs: int = 300,
        check_on_train_epoch_end: bool = False):
        """"
        Parameters
        ------------
        check_per_iteration:
            The number of epochs to check the convergence.
        limit_same_variable:
            The number of epochs to check the convergence.
            We check the variables during `check_per_iteration` * `limit_same_variable` epochs.
            If the discrete variable are constant in this period, the training is stopped.
        ignore_epochs:
            The number of epochs to ignore the convergence check.
            The early stopping is not performed during this period.
            The stopping criteria starts from this epoch.
            Hence, the minimul stopping epoch will be `ignore_epochs + (check_per_iteration * limit_same_variable)`.
        check_on_train_epoch_end:
            If True, the convergence check is performed at the end of each epoch.
        """
        assert ignore_epochs > 200, "ignore_epochs must be greater than 200. For the safety reasons."
        
        EarlyStopping.__init__(
            self,
            monitor='variance',
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )

        # --------------------
        # attributes
        self.__record_variable_history = []
        
        self.check_per_iteration = check_per_iteration
        self.limit_same_variable = limit_same_variable
        self.ignore_epochs = ignore_epochs
        # --------------------
        
        
    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        current_epoch = trainer.current_epoch

        # comment: shortcutting blocks. Do not evaluate variables.
        if current_epoch < self.ignore_epochs:
            return
        # end if
        if current_epoch % self.check_per_iteration != 0:
            return
        else:
            ard_weights: torch.Tensor = trainer.lightning_module.mmd_estimator.kernel_obj.ard_weights.detach().cpu()
            variables = tuple(detect_variables(ard_weights ** 2))
            self.__record_variable_history.append(variables)
        # end if
        
        if len(self.__record_variable_history) < self.limit_same_variable:
            return
        # end if
        
        # stack_log_train: t.List[TrajectoryRecord] = trainer.lightning_module.stack_training_log  # type: ignore
        # records_in_check_span = stack_log_train[len(stack_log_train) - self.limit_same_variable:]
        records_in_check_span = self.__record_variable_history[len(self.__record_variable_history) - self.limit_same_variable:]
        variables_set = set(records_in_check_span)
                
        if len(variables_set) == 1:
            if len(list(variables_set)[0]) == len(ard_weights):
                return
            # end if            
            self.stopped_epoch = True
            trainer.should_stop = True
            logger.info(f"Stopped because of a convergence on discrete variable. {variables_set}")
            reason = f"Stopped because of a convergence on discrete variable. {variables_set} during {self.limit_same_variable} epochs."
        
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)


# ----------------------------------------


