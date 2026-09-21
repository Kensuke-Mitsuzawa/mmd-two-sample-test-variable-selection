import typing as ty
import logging

import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping

from ...utils import detect_variables
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)


class LegacyVariableEarlyStopping(EarlyStopping):
    """Auto-stopping-rules if the discrete variable is same during certain epochs (Legacy implementation)."""

    def __init__(
        self,
        check_per_iteration: int = 1,
        limit_same_variable: int = 100,
        ignore_epochs: int = 300,
        check_on_train_epoch_end: bool = False
    ):
        assert ignore_epochs > 200, "ignore_epochs must be greater than 200. For safety reasons."
        super().__init__(
            monitor='variance',
            patience=99999,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        self.__record_variable_history = []
        self.check_per_iteration = check_per_iteration
        self.limit_same_variable = limit_same_variable
        self.ignore_epochs = ignore_epochs
    # end def

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        current_epoch = trainer.current_epoch
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

        records_in_check_span = self.__record_variable_history[len(self.__record_variable_history) - self.limit_same_variable:]
        variables_set = set(records_in_check_span)

        if len(variables_set) == 1:
            if len(list(variables_set)[0]) == len(ard_weights):
                return
            # end if
            self.stopped_epoch = True
            trainer.should_stop = True
            logger.info(f"Stopped because of convergence on discrete variable: {variables_set}")
            reason = f"Stopped because of convergence on discrete variable: {variables_set} during {self.limit_same_variable} epochs."
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)
            # end if
        # end if
    # end def
# end class


class VariableEarlyStopping(LegacyVariableEarlyStopping):
    """Auto-stopping-rules if the discrete variable is constant across check epochs."""
    pass
# end class
