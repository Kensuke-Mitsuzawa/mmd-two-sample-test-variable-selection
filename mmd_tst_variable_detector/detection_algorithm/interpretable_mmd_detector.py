import abc
import logging

import torch
import copy
import numpy as np
import typing as ty
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, asdict
import pytorch_lightning as pl
import pytorch_lightning.loggers


from ..mmd_estimator.mmd_estimator import (
    BaseMmdEstimator,
    MmdValues,
    BatchSizeError,
    QuadraticMmdEstimator,
    LinearMmdEstimator,
)
from .commons import (
    TrainingStatistics,
    TrajectoryRecord,
    RegularizationParameter,
    DebugContainerNan,
    InterpretableMmdTrainParameters,
    InterpretableMmdTrainResult,
)
# from ..utils import detect_variables

from ..exceptions import OptimizationException
from ..datasets.base import BaseDataset
from ..kernels.base import BaseKernel

from ..logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



class BaseInterpretableMmdDetector(abc.ABC):
    @abc.abstractmethod
    def copy_detector(self) -> "BaseInterpretableMmdDetector":
        # Copy the detector itself. Do copy-job when you copy dataset object.
        raise NotImplementedError()




class InterpretableMmdDetector(pl.LightningModule, BaseInterpretableMmdDetector):
    def __init__(
        self,
        mmd_estimator: BaseMmdEstimator,
        training_parameter: InterpretableMmdTrainParameters,
        dataset_train: BaseDataset,
        dataset_validation: BaseDataset,
        is_debug: bool = False,
        is_tune_dataset_batch: bool = False,
        is_shuffle_dataset: bool = False,
    ):
        """
        Parameters
        ----------
        mmd_estimator: MMD estimator object.
        training_parameter:
        dataset_train:
        dataset_validation:
        is_debug: Debug mode saves values and kernel matrix when NAN occurs during the optimization.
        """
        super().__init__()
        
        # -------------------------------------------
        # attribute definitions
        self.training_parameter = training_parameter
        self.mmd_estimator = copy.deepcopy(mmd_estimator)
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.is_shuffle_dataset = is_shuffle_dataset
        
        self.nan_counter = 0  # count frequency that objective value goes to NAN
        self.is_debug = is_debug

        # stacks for training information logging.
        # Logger object refers to this attribute.
        # NOTE: This stack may cause slow-down of the optimisation speed IF you keep adding object every epoch.
        # I recommend to record it per 100 epochs or wider interval.
        self.stack_training_log = []
        self.stack_validation_log = []
        
        # torch object to record loss values.
        # I use these values for early stopping.
        # I define fields here. I set an actual object at `setup` method.
        self.loss_training: torch.Tensor
        self.loss_validation: torch.Tensor
        
        # A temporal variable to record the current epoch.
        # I use it for outputing the value at the end state of the training.
        self.current_mean_metric_training: ty.Dict[str, float] = {}
        self.current_mean_metric_validation: ty.Dict[str, float] = {}
        
        # private attributes. 
        # used for checking early-stopping of `nan-obj` and `negative-mmd`.
        self.__count_continious_nan: int = 0
        self.__count_continious_negative_mmd: int = 0
        # -------------------------------------------
        assert len(self.dataset_train) > 0, "dataset_train is len() == 0. Please check your dataset."
        assert len(self.dataset_validation) > 0, "dataset_validation is len() == 0. Please check your dataset."
        
        # -------------------------------------------

        if is_tune_dataset_batch:
            logger.info(
                "Executing tuning batch size. Turn off is_tune_dataset_batch = False \
                    if this process takes too long time and when your specified batch size is surely always mmd2 > 0.")
            
            dataset_train, training_parameter, is_shuffle_dataset = tune_dataset_batch_size(
                dataset_train, training_parameter, mmd_estimator)
            logger.info("End tuning batch size.")
        # end if

        if training_parameter.is_use_log == -1:
            # comment: when your data pair (X, Y) are quite similar, MMD2 value may become negative.
            # this if block check if the initial MMD2 value is always positive.
            _ratio_positive_mmd = self.__check_initial_mmd(dataset_train)
            if _ratio_positive_mmd < 0.5:
                logger.info(
                    f"Switching `is_use_log = 0`, which means the objective function is not log."
                    f"Setting `is_use_log = 1` if you want to use log objective function."
                )
                training_parameter.is_use_log = 0
            # end if
        elif training_parameter.is_use_log == 1:
            _ratio_positive_mmd = self.__check_initial_mmd(dataset_train)
            if _ratio_positive_mmd < 0.5:
                raise OptimizationException(
                    f"Ratio of batch having MMD > 0.0 is {_ratio_positive_mmd}. "
                    f"Optimization is not able to be performed.")
            # end if
        # end if

        self.dataset_train.close()
        self.dataset_validation.close()
        self.save_hyperparameters()

    def copy_detector(self) -> "InterpretableMmdDetector":
        """Copy the detector itself. Do copy-job when you copy dataset object.
        """
        dataset_train = self.dataset_train.copy_dataset()
        dataset_validation = self.dataset_validation.copy_dataset()

        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        
        return copy.deepcopy(self)

    def setup(self, stage: str):
        # comment: these stacks are used for aggerating training information at each epoch.
        # in `epoch_end` function, these stacks are reset to `[]`.
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.steps_nan_obj = []
        
        # I set the loss values for early stopping.
        max_epochs: int = self.trainer.max_epochs  # type: ignore
        assert max_epochs > 0, "max_epochs must be greater than 0."
        assert max_epochs is not None, "max_epochs must be set."
        self.loss_training = torch.zeros(max_epochs)
        self.loss_validation = torch.zeros(max_epochs)

    def configure_optimizers(self):
        list_trainable_parameters = self.mmd_estimator.kernel_obj._get_trainable_parameters()
        params_target = list_trainable_parameters

        self.mmd_estimator.kernel_obj.bandwidth.requires_grad = False

        if self.training_parameter.optimizer_args is None:
            optimizer = torch.optim.Adam(params_target)
        else:
            optimizer = torch.optim.Adam(
                params_target, **self.training_parameter.optimizer_args
            )
        # end if

        if self.training_parameter.lr_scheduler is not None:
            logger.debug(f"Using LR scheduler on {self.training_parameter.lr_scheduler_monitor_on}")
            _scheduler: ty.Callable[[torch.optim.Optimizer], ReduceLROnPlateau] = \
                copy.deepcopy(self.training_parameter.lr_scheduler)
            scheduler = _scheduler(optimizer=optimizer)
            if self.training_parameter.lr_scheduler_monitor_on in ("train_loss", "val_loss"):
                assert scheduler.mode == "min"
            # end if

            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
                "monitor": self.training_parameter.lr_scheduler_monitor_on,
            }
        else:
            return optimizer

    def train_dataloader(self):
        if self.training_parameter.batch_size == -1:
            batch_size = len(self.dataset_train)
        else:
            batch_size = self.training_parameter.batch_size
        # end if
        return DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=self.is_shuffle_dataset,
            num_workers=self.training_parameter.n_workers_train_dataloader,
            persistent_workers=self.training_parameter.dataloader_persistent_workers)

    def val_dataloader(self):
        if self.training_parameter.batch_size == -1:
            batch_size = len(self.dataset_validation)
        else:
            batch_size = self.training_parameter.batch_size
        # end if
        return DataLoader(
            self.dataset_validation,
            batch_size=batch_size,
            shuffle=self.is_shuffle_dataset,
            num_workers=self.training_parameter.n_workers_validation_dataloader,
            persistent_workers=self.training_parameter.dataloader_persistent_workers
        )

    def __check_initial_mmd(self, dataset: BaseDataset) -> float:
        """Checking if MMD value is always positive with the initial given parameters.
        """
        # set the dataloader with the default value.
        if self.training_parameter.batch_size == -1:
            batch_size = len(dataset)
        else:
            batch_size = self.training_parameter.batch_size
        # end if

        dataset_loader = DataLoader(
            dataset,
            batch_size=batch_size
        )
        __stack = []
        for x, y in dataset_loader:
            mmd_values: MmdValues = self.mmd_estimator.forward(x, y)
            __stack.append(mmd_values.mmd.detach().item())
            # end if
        # end for
        _positive_mmd = [__mmd for __mmd in __stack if __mmd > 0.0]
        ratio_positive_mmd = len(_positive_mmd) / len(__stack)
        return ratio_positive_mmd

    def generate_regularization_term(
        self, regularization_parameter: ty.Tuple[float, float]
    ) -> torch.Tensor:
        """Generate a regularization term.
        
        Parameters
        ----------
        regularization_parameter:
        
        Returns
        -------
        Regularization term.
        """
        if regularization_parameter[0] == 0.0 and regularization_parameter[1] == 0.0:
            __reg = torch.tensor([0.0], device=self.device)
        else:
            l1_terms = torch.sum(torch.abs(self.mmd_estimator.kernel_obj.ard_weights))
            l1 = regularization_parameter[0] * l1_terms

            l2_terms = torch.sum(torch.pow(self.mmd_estimator.kernel_obj.ard_weights, 2))
            l2 = torch.div(regularization_parameter[1], 2) * l2_terms
            __reg = l1 + l2
        # end if

        return __reg

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> MmdValues:
        mmd_variables = self.mmd_estimator.forward(x, y)
        return mmd_variables

    def function_loss(
        self, mmd_variable: MmdValues, reg_term: torch.Tensor
    ) -> torch.Tensor:
        if self.training_parameter.objective_function == "ratio":
            if self.training_parameter.is_use_log == 1:
                assert mmd_variable.ratio is not None
                obj_reg = -(torch.log(mmd_variable.ratio)) + reg_term
            else:
                assert mmd_variable.ratio is not None
                obj_reg = -mmd_variable.ratio + reg_term
        elif self.training_parameter.objective_function == "mmd":
            if self.training_parameter.is_use_log == 0:
                obj_reg = -(torch.log(mmd_variable.mmd)) + reg_term
            else:
                obj_reg = -mmd_variable.mmd + reg_term
        else:
            raise ValueError("Invalid objective function is specified.")
        # end if
        return obj_reg
    
    def __is_activate_early_stopping(self) -> bool:
        """Checking count of NAN-obj and Negative-MMD.
        
        Raises
        ------
        OptimizationException
            When the count of NAN-obj or Negative-MMD is over the limit.
        
        Note
        ----
        This 'early-stopping' is not early stopping callbacks of Pytorch Lightning.
        This method checks just private values.
        """
        if self.__count_continious_nan > self.training_parameter.limit_steps_early_stop_nan:
            raise OptimizationException(
                f'Objective value becomes NAN for {self.__count_continious_nan} times.'
                'More than your configuration value: {self.training_parameter.limit_steps_early_stop_nan} times.')
        # end if
        if self.__count_continious_negative_mmd > self.training_parameter.limit_steps_early_stop_negative_mmd:
            raise OptimizationException(
                f'MMD becomes negative for {self.__count_continious_negative_mmd} times. '
                'More than your configuration value: {self.training_parameter.limit_steps_early_stop_negative_mmd} times.')
        # end if
        
        return True

    def training_step(self, 
                      batch: ty.Tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int) -> ty.Optional[ty.Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        batch: a list of object having two elements. VariableZ object for XY and X-times-Y.
        batch_idx: a data index number for the given batch.
        
        Raises
        ------
        `OptimizationException` when the objective value becomes NAN or MMD becomes negative more than given limit threshold.
        
        Returns
        -------
        a dictionary object.
        """
        reg_term = self.generate_regularization_term(
            self.training_parameter.regularization_parameter
        )
        # define the objective-value
        mmd_variable = self.forward(x=batch[0], y=batch[1])
        obj_reg = self.function_loss(mmd_variable, reg_term)

        if mmd_variable.mmd < 0.0:
            # comment: when your data pair (X, Y) are quite similar, MMD2 value may become negative.
            self.__count_continious_negative_mmd += 1
            self.__is_activate_early_stopping()
        else:
            self.__count_continious_negative_mmd = 0
        # end if

        if torch.isnan(obj_reg):
            # comment: `obj_reg` becomes NAN when a data pair (x, y) are similar and you use log() transformation.
            self.__count_continious_nan += 1
            self.__is_activate_early_stopping()
            
            self.nan_counter += 1
            if self.is_debug:
                kernel_obj: BaseKernel = copy.deepcopy(self.mmd_estimator.kernel_obj)
                kernel_obj.ard_weights.requires_grad = False
                __mmd_values = MmdValues(
                    mmd=mmd_variable.mmd.detach(),
                    variance=mmd_variable.ratio.detach(),
                    ratio=mmd_variable.variance.detach(),
                    kernel_matrix_obj=mmd_variable.kernel_matrix_obj)
                debug_obj = DebugContainerNan(
                    epoch=self.trainer.current_epoch,
                    global_step=self.trainer.global_step,
                    mmd_values=__mmd_values,
                    batch_xy=(batch[0], batch[1]))
                self.steps_nan_obj.append(debug_obj)
            # end if
            return None
        else:
            # reset the counter to 0
            self.__count_continious_nan = 0
            
            # TODO            
            # do variable selection and recording the count.
            # comment: it may take additional execution cost. Hence, I make it optional. 
            # You can change the configuration via the parameter configuration.
            # if self.training_parameter.is_log_discrete_variables:
            #     # comment: I prefer to use tuple for memory efficiency.
            #     __variables = tuple(detect_variables(self.mmd_estimator.kernel_obj.ard_weights.detach()))
            # else:
            #     __variables  = ()
            # # end if
            
            if self.training_parameter.objective_function == "ratio":
                __values = {
                    "loss": obj_reg,
                    "mmd2": mmd_variable.mmd,
                    "ratio": mmd_variable.ratio,
                    "variance": mmd_variable.variance,
                    # "discrete_variables": __variables
                }
            elif self.training_parameter.objective_function == "mmd":
                __values = {
                    "loss": obj_reg,
                    "mmd2": mmd_variable.mmd,
                    "ratio": mmd_variable.ratio,
                    "variance": mmd_variable.variance,
                    # "discrete_variables": __variables
                }
            else:
                raise NotImplementedError()
            # end if
            self.training_step_outputs.append(__values)
            return __values

    def on_train_epoch_end(self) -> None:
        # sometimes, self.training_step_outputs becomes a blank list.
        if len(self.training_step_outputs) > 0:
            loss_mean = torch.stack([d["loss"] for d in self.training_step_outputs]).mean()
            mmd_mean = torch.stack([d["mmd2"] for d in self.training_step_outputs]).mean()
            var_mean = torch.stack([d["variance"] for d in self.training_step_outputs]).mean()
            ratio_mean = torch.stack([d["ratio"] for d in self.training_step_outputs]).mean()
            
            # variable_last_step = self.training_step_outputs[-1]["discrete_variables"]

            self.log_dict({
                    "epoch": self.trainer.current_epoch,
                    "train_loss": loss_mean,
                    "train_mmd2": mmd_mean,
                    "train_variance": var_mean,
                    "train_ratio": ratio_mean,
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"]
                    }, on_step=False, on_epoch=True)
            self.current_mean_metric_training = {
                    "epoch": self.trainer.current_epoch,
                    "loss": loss_mean.item(),
                    "mmd2": mmd_mean.item(),
                    "variance": var_mean.item(),
                    "ratio": ratio_mean.item()}            

            # ------------------------------------------------
            # recording a loss value for the early stopping.
            self.loss_training[self.trainer.current_epoch] = loss_mean.detach().cpu()

            # ------------------------------------------------
            # recording the trajectory information.
            is_record_training_log = (self.training_parameter.frequency_epoch_trajectory_record > 0) and \
                (self.trainer.current_epoch % self.training_parameter.frequency_epoch_trajectory_record == 0)
            
            if is_record_training_log:
                self.stack_training_log.append(
                    TrajectoryRecord(
                        self.current_epoch,
                        mmd_mean.detach().item(),
                        var_mean.detach().item(),
                        ratio_mean.detach().item(),
                        loss_mean.detach().item()))
            # end if
        # end if

        # resetting the list object to avoid occupying too much memory.
        self.training_step_outputs = []

    def validation_step(self, batch: ty.Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> ty.Dict[str, torch.Tensor]:
        reg_term = self.generate_regularization_term(
            self.training_parameter.regularization_parameter
        )
        # define the objective-value
        mmd_variable = self.forward(x=batch[0], y=batch[1])

        obj_reg = self.function_loss(mmd_variable, reg_term)

        if self.training_parameter.objective_function == "ratio":
            __values = {
                "loss": obj_reg,
                "mmd2": mmd_variable.mmd,
                "ratio": mmd_variable.ratio,
                "variance": mmd_variable.variance,
            }
        elif self.training_parameter.objective_function == "mmd":
            __values = {
                "loss": obj_reg,
                "mmd2": mmd_variable.mmd,
                "ratio": mmd_variable.ratio,
                "variance": mmd_variable.variance,
            }
        else:
            raise NotImplementedError()
        # end if
        self.validation_step_outputs.append(__values)
        return __values

    def on_validation_epoch_end(self):
        # sometimes, self.training_step_outputs becomes a blank list.
        if len(self.validation_step_outputs) > 0:
            loss_mean = torch.stack([d["loss"] for d in self.validation_step_outputs]).mean()
            mmd_mean = torch.stack([d["mmd2"] for d in self.validation_step_outputs]).mean()
            var_mean = torch.stack([d["variance"] for d in self.validation_step_outputs]).mean()
            ratio_mean = torch.stack([d["ratio"] for d in self.validation_step_outputs]).mean()

            self.log_dict({
                    "epoch": self.trainer.current_epoch,
                    "val_loss": loss_mean,
                    "val_mmd2": mmd_mean,
                    "val_variance": var_mean,
                    "val_ratio": ratio_mean,},
                on_step=False,
                on_epoch=True)
            
            self.current_mean_metric_validation = {
                    "epoch": self.trainer.current_epoch,
                    "loss": loss_mean.item(),
                    "mmd2": mmd_mean.item(),
                    "variance": var_mean.item(),
                    "ratio": ratio_mean.item()}
            # ------------------------------------------------
            # recording a loss value for the early stopping.
            self.loss_validation[self.trainer.current_epoch] = loss_mean.detach().cpu()

            # ------------------------------------------------
            # recording the trajectory information.
            is_log_epoch = self.trainer.current_epoch % self.training_parameter.frequency_epoch_trajectory_record == 0
            if self.training_parameter.frequency_epoch_trajectory_record > 0 and (is_log_epoch or self.trainer.check_val_every_n_epoch > 1):
                self.stack_validation_log.append(
                    TrajectoryRecord(
                        self.current_epoch,
                        mmd_mean.detach().item(),
                        var_mean.detach().item(),
                        ratio_mean.detach().item(),
                        loss_mean.detach().item()))
        # end if

        self.validation_step_outputs = []

    def teardown(self, stage: ty.Optional[str] = None) -> None:
        pass

    def on_save_checkpoint(self, checkpoint: ty.Dict[str, ty.Any]) -> None:
        checkpoint["stack_training_log"] = self.stack_training_log
        checkpoint["stack_validation_log"] = self.stack_validation_log

    def on_load_checkpoint(self, checkpoint: ty.Dict[str, ty.Any]) -> None:
        self.stack_training_log = checkpoint["stack_training_log"]
        self.stack_validation_log = checkpoint["stack_validation_log"]

    def on_train_end(self) -> None:
        """A procedure when the training is finished.
        I write out metrics at the last training/validation steps.
        """
        d_metric_train = self.current_mean_metric_training
        d_metric_val = self.current_mean_metric_validation

        if d_metric_train == {} or d_metric_val == {}:
            # comment: there a possibility that the dict is empty.
            self.stack_training_log.append(
                TrajectoryRecord(
                    epoch=int(d_metric_train["epoch"]),
                    mmd=d_metric_train["mmd2"],
                    var=d_metric_train["variance"],
                    ratio=d_metric_train["ratio"],
                    loss=d_metric_train["loss"])
            )
            self.stack_validation_log.append(
                TrajectoryRecord(
                    epoch=int(d_metric_val["epoch"]),
                    mmd=d_metric_val["mmd2"],
                    var=d_metric_val["variance"],
                    ratio=d_metric_val["ratio"],
                    loss=d_metric_val["loss"])        
            )

    # ------------------------------------------------
    # A custom method. Non-standard Pytorch-Lightning method.
    
    def get_trained_variables(
        self, is_square_ard_weights: bool = True
    ) -> InterpretableMmdTrainResult:
        """Getting the trained variables.
        :param is_square_ard_weights: returning ARD weights in a squared form.
        :return: `InterpretableMmdTrainResult`
        """
        __ard_kernel_k = self.mmd_estimator.kernel_obj.ard_weights.detach().cpu()

        stats = TrainingStatistics(
            global_step=self.trainer.global_step, nan_frequency=self.nan_counter
        )

        training_configurations = {
            "kernel": self.mmd_estimator.kernel_obj.__class__,
            "mmd_estimator": self.mmd_estimator.__class__,
        }
        
        mmd_estimator_hyperparameters = self.mmd_estimator.get_hyperparameters()

        return InterpretableMmdTrainResult(
            ard_weights_kernel_k=torch.pow(__ard_kernel_k, 2) if is_square_ard_weights else __ard_kernel_k,
            mmd_estimator=self.mmd_estimator.state_dict(),
            training_stats=stats,
            trajectory_record_training=self.stack_training_log,
            trajectory_record_validation=self.stack_validation_log,
            training_parameter=self.training_parameter,
            training_configurations=training_configurations,
            mmd_estimator_hyperparameters=mmd_estimator_hyperparameters
        )



# TODO move this function to somewhere....module.

def tune_dataset_batch_size(
    dataset: BaseDataset,
    training_parameter_obj: InterpretableMmdTrainParameters,
    mmd_estimator: BaseMmdEstimator,
    batch_size_min: int = -1,
    batch_size_max: int = -1,
    batch_size_step: int = -1,
    n_max_try: int = 100,
) -> ty.Tuple[BaseDataset, InterpretableMmdTrainParameters, bool]:
    """Finding a suitable dataset and batch size.

    Background: MMD values becomes a minus value when X and Y have a few dependencies (really similar).
    This function attempts to find a condition that makes MMD > 0.

    :param dataset:
    :param mmd_estimator:
    :param training_parameter_obj:
    :param batch_size_min:
    :param batch_size_max:
    :param batch_size_step:
    :param n_max_try:
    :return: (Z-dataset, batch-size)
    """
    # set the dataloader with the default value.
    dataset_loader = DataLoader(dataset, batch_size=training_parameter_obj.batch_size)
    __stack = []
    for x, y in dataset_loader:
        mmd_values: MmdValues = mmd_estimator.forward(x, y)
        if mmd_values.mmd > 0.0:
            # if MMD2 value > 0.0, the current configuration is satisfied.
            __stack.append(mmd_values.mmd.detach().item())
        # end if
    # end for

    if len(__stack) > 0:
        return dataset, training_parameter_obj, False
    # end if

    if batch_size_min == -1:
        batch_size_min = int(len(dataset) / 10)
    # end if
    if batch_size_max == -1:
        batch_size_max = len(dataset)
    # end if
    if batch_size_step == -1:
        batch_size_step = int((batch_size_max - batch_size_min) / 10)
    # end if

    if isinstance(mmd_estimator, LinearMmdEstimator):
        batch_size_candidate = [
            v for v in range(batch_size_min, batch_size_max) if v % 4 == 0
        ]
    else:
        batch_size_candidate = list(
            range(batch_size_min, batch_size_max, batch_size_step)
        )
    # end if

    n_try_current = 0
    # try re-shuffle dataset and try different batch size.
    while n_try_current < n_max_try:
        for batch_size in batch_size_candidate:
            __stack = []
            # set a batch-size candidate
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for x, y in dataset_loader:
                try:
                    mmd_values: MmdValues = mmd_estimator.forward(x, y)
                except BatchSizeError:
                    continue
                # end try

                if mmd_values.mmd > 0.0:
                    training_parameter_obj.batch_size = batch_size
                    # if MMD2 value > 0.0, the current configuration is satisfied.
                    __stack.append(mmd_values.mmd.detach().item())
                    # return z_dataset_variable_pair, training_parameter_obj
                # end if
            # end for
            if len(__stack) > 0:
                return dataset, training_parameter_obj, False
            # end if
        # end for
        n_try_current += 1
    # end while

    n_try_current = 0
    # try re-shuffle dataset and try different batch size.
    while n_try_current < n_max_try:
        for batch_size in batch_size_candidate:
            __stack = []
            # set a batch-size candidate
            dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x, y in dataset_loader:
                try:
                    mmd_values: MmdValues = mmd_estimator.forward(x, y)
                except BatchSizeError:
                    continue
                # end try

                if mmd_values.mmd > 0.0:
                    training_parameter_obj.batch_size = batch_size
                    # if MMD2 value > 0.0, the current configuration is satisfied.
                    __stack.append(mmd_values.mmd.detach().item())
                    # return z_dataset_variable_pair, training_parameter_obj
                # end if
            # end for
            if len(__stack) > 0:
                return dataset, training_parameter_obj, True
            # end if
        # end for
        n_try_current += 1
    # end while

    raise Exception(
        "Script attempted to find a suitable dataset and batch size. However, it failed."
    )



# ---------------------------------------------------------------------------
# For older version

MmdVariableTrainer = InterpretableMmdDetector
