from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import typing as ty

from .distributed_config import DistributedConfigArgs


@dataclass
class ResourceConfigArgs:
    """Configuration class for resource. Resource includes path to files and distributed computing configuration.
    
    Parameters
    ----------
    path_work_dir: ty.Union[str, Path]
        Path to working directory.
        All files are saved in this directory.
    dir_name_ml_logger: str
        Directory name for ml_logger.
        ml_logger is a tool for logging.
    dir_name_model: str
        Directory name for saving trained models.
    dir_name_data: str
        Directory name for saving data.
    dir_name_logs: str
        Directory name for saving logs.
    train_accelerator: str
        Train accelerator.
        Either of following choices,
        1. 'cpu': cpu.
        2. 'gpu': gpu. Not `cuda`. 
            According to the documentation: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html
    distributed_config_preprocessing: ty.Optional[DistributedConfigArgs]
        Configuration for dask cluster.
    distributed_config_detection: ty.Optional[DistributedConfigArgs]
        Configuration for dask cluster.
    """
    
    path_work_dir: ty.Union[str, Path, None] = Path('/tmp/mmd_tst_variable_detector/interface')
    
    dir_name_ml_logger: str = 'mlruns'
    dir_name_model: str = 'model'  # directory for saving trained models.
    dir_name_data: str = 'data'  # directory for saving data.
    dir_name_logs: str = 'logs'  # directory for saving logs.
    
    train_accelerator: str = 'cpu'
    
    # Distributed backend configurations    
    distributed_config_detection: DistributedConfigArgs = DistributedConfigArgs()

    dask_config_detection: ty.Optional[DistributedConfigArgs] = None

    
    def __post_init__(self):        
        if self.path_work_dir is None:
            self.path_work_dir = Path('/tmp/mmd_tst_variable_detector/interface') / datetime.now().isoformat()
        
        if isinstance(self.path_work_dir, str):
            self.path_work_dir = Path(self.path_work_dir)
        # end if
        if not self.path_work_dir.exists():
            # logger.debug(f'Creating working directory: {self.path_work_dir}')
            self.path_work_dir.mkdir(parents=True, exist_ok=True)
        # end if    
        
        # I have to set values to `distributed_config_***` becasue a lot of modules use these values.
        self.dask_config_detection = self.distributed_config_detection
