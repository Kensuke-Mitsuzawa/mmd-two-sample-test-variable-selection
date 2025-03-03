import dataclasses
import typing as ty
from typing import Union, Optional, List, Iterable
from datetime import datetime

from pathlib import Path
# from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from pytorch_lightning.plugins import PLUGIN_INPUT

# from mmd_tst_variable_detector.utils.early_stopping import DefaultEarlyStoppingRule


@dataclasses.dataclass
class PytorchLightningDefaultArguments:
    """Default argument for Pytorch Lightning Trainer 2.1.2"""
    accelerator: Union[str, Accelerator] ='auto'
    strategy: Union[str, Strategy] = "auto"
    devices: Union[List[int], str, int] = "auto"
    num_nodes: int = 1
    precision: ty.Optional[str] = None
    logger: Optional[Union[Logger, Iterable[Logger], bool]] = None
    callbacks: Optional[Union[List[Callback], Callback]] = None
    fast_dev_run: bool = False
    max_epochs: int = 9999
    min_epochs: ty.Optional[int] = None
    max_steps: int = -1
    min_steps: ty.Optional[int] = None
    max_time: ty.Optional[float] = None
    limit_train_batches: ty.Optional[int]= None
    limit_val_batches: ty.Optional[int] = None
    limit_test_batches: ty.Optional[int] = None
    limit_predict_batches: ty.Optional[int] = None
    overfit_batches: float = 0.0
    val_check_interval: ty.Optional[bool] = None
    check_val_every_n_epoch: int = 1
    num_sanity_val_steps: ty.Optional[int] = None
    log_every_n_steps: ty.Optional[int] = None
    enable_checkpointing: ty.Optional[bool] = None
    enable_progress_bar: ty.Optional[bool] = None
    enable_model_summary: ty.Optional[bool] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[Union[int, float]] = None
    gradient_clip_algorithm: Optional[str] = None
    deterministic: Optional[bool] = None
    benchmark: Optional[bool] = None
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Optional[str] = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: Optional[Path] = None
    
    def __post_process__(self):
        if self.default_root_dir is None:
            self.default_root_dir = Path('/tmp/mmd-tst-variable-detector') / datetime.now().isoformat()
    
    def as_dict(self):
        return dataclasses.asdict(self)
