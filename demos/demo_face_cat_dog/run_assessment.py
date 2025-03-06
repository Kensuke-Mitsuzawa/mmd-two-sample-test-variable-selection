from pathlib import Path
import typing as ty
import toml
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np

import torch
import pytorch_lightning as pl

import logzero
from logzero import logger

import PIL.Image

from mmd_tst_variable_detector.datasets.base import BaseDataset
from mmd_tst_variable_detector import (
    # interface
    Interface,
    InterfaceConfigArgs,
    ResourceConfigArgs,
    ApproachConfigArgs,
    DataSetConfigArgs,
    DetectorAlgorithmConfigArgs,
    CvSelectionConfigArgs,
    BasicVariableSelectionResult,
    OutputObject,
    DistributedConfigArgs,
    RegularizationSearchParameters
)


ACCEPTABLE_METHODS = ('wasserstein_independence', 'interpretable_mmd')

class PicturesDatasetGenerator(object):
    """Iterative dataset"""

    def __init__(
            self,
            image_size: ty.Tuple[int, int],
            path_dir_x: Path,
            path_dir_y: Path,
            sample_size_limit: int,
            file_extension: str = 'jpg',
            selected_variables: ty.Optional[ty.List[int]] = None):

        super().__init__()
        self.path_dir_x = path_dir_x
        self.path_dir_y = path_dir_y
        self.image_size = image_size

        logger.info('listing up jpg files...')
        self.seq_files_x = list(sorted(self.path_dir_x.rglob(f'*{file_extension}')))
        self.seq_files_y = list(sorted(self.path_dir_y.rglob(f'*{file_extension}')))

        assert len(self.seq_files_x) > 0, f'No files found in {self.path_dir_x}'
        assert len(self.seq_files_y) > 0, f'No files found in {self.path_dir_y}'

        if sample_size_limit is None:
            self.sample_size_limit = min(len(self.seq_files_x), len(self.seq_files_y))
        else:
            self.sample_size_limit = sample_size_limit
        # end if

        self.selected_variables = selected_variables

    def __del__(self):
        pass
    
    def __func_overlay_detected_variables(self, path_file: Path) -> np.ndarray:
        """Executing gray-scale and re-sizing.
        """
        im = PIL.Image.open(path_file)
        im = im.convert('L')  # gray scale
        im = im.resize(self.image_size)  # re-sizing.
        image_array_original = np.array(im)

        return image_array_original

    def __getitem__(self, idx: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
        path_file_x = self.seq_files_x[idx]
        path_file_y = self.seq_files_y[idx]

        array_x = self.__func_overlay_detected_variables(path_file_x)
        array_y = self.__func_overlay_detected_variables(path_file_y)

        return torch.from_numpy(array_x), torch.from_numpy(array_y)

    def __len__(self) -> int:
        return self.sample_size_limit

    def get_feature_size(self) -> int:
        return self.image_size[0] * self.image_size[1]




# ------------------------------------------------------------------------------
# config obj

@dataclass
class BaseConfig:
    path_experiment_root: ty.Union[Path, str]
    file_name_sqlite3: str = "exp_result.sqlite3"
    name_experiment_db: str = "experiment.json"

    dir_name_data: str = "data"
    dir_models: str = "models"
    dir_logs: str = "logs"    

    def __post_init__(self):
        self.path_experiment_root = Path(self.path_experiment_root)


@dataclass
class DataSettingConfig:
    target_image_size: ty.List[int]
    
    path_dir_data_source_x: ty.Union[str, Path]
    path_dir_data_source_y: ty.Union[str, Path]

    data_mode: str = "cat_and_dogs"
    file_extension: str = 'jpg'

    file_name_x: str = "cat.pt"
    file_name_y: str = "dog.pt"
    
    def __post_init__(self):
        self.path_dir_data_source_x = Path(self.path_dir_data_source_x)
        self.path_dir_data_source_y = Path(self.path_dir_data_source_y)


@dataclass
class DataGenerationConfig:
    sample_size_train: int
    sample_size_test: int
    
    
@dataclass
class MmdBaselineConfig:
    MAX_EPOCH: int = 9999
    
@dataclass
class CvSelectionConfig:
    MAX_EPOCH: int = 9999
    candidate_regularization_parameter: str = 'auto'
    n_regularization_parameter: int = 5

    n_subsampling: int = 10
    
    n_search_iteration: int = 10
    n_max_concurrent: int = 3


@dataclass
class ComputationalResourceConfig:
    train_accelerator: str = 'cpu'

    def __post_init__(self):
        assert self.train_accelerator in ['cpu', 'cuda']

@dataclass
class RootConfig:
    base: BaseConfig
    data_setting: DataSettingConfig
    data_generation: DataGenerationConfig
    mmd_baseline: MmdBaselineConfig
    cv_selection: CvSelectionConfig
    computational_resource: ComputationalResourceConfig
    detection_approaches: ty.List[str] = ACCEPTABLE_METHODS
    dataset_type_backend: str = 'ram'
    is_pre_reload_dataset: bool = True

    def __post_init__(self):
        assert self.dataset_type_backend in ['ram', 'file', 'flexible-file']
        if self.is_pre_reload_dataset:
            logger.info('Pre-reloaded dataset is used. All data are loaded into RAM memory.')
            self.dataset_type_backend = 'ram'
        # end if


import dacite

def main(path_toml_config: Path):
    assert path_toml_config.exists(), f'Not found: {path_toml_config}'
    __config_obj = toml.loads(path_toml_config.open().read())
    config_obj = dacite.from_dict(data_class=RootConfig, data=__config_obj)

    path_root_dir: Path = config_obj.base.path_experiment_root  # type: ignore
    path_root_dir.mkdir(parents=True, exist_ok=True)
    
    path_dir_log = path_root_dir / config_obj.base.dir_logs
    path_dir_log.mkdir(parents=True, exist_ok=True)
    logzero.logfile(path_dir_log / 'log.txt', maxBytes=1e6, backupCount=3)
    
    path_dir_data = path_root_dir / config_obj.base.dir_name_data
    path_dir_data_train = path_dir_data / 'train'
    path_dir_data_test = path_dir_data / 'test'
    path_dir_data_train.mkdir(parents=True, exist_ok=True)
    path_dir_data_test.mkdir(parents=True, exist_ok=True)
    
    # do data conversion jpg -> pt
    sample_size_limit = config_obj.data_generation.sample_size_train + config_obj.data_generation.sample_size_test
    logger.info('Generating the dataset...')
    assert isinstance(config_obj.data_setting.path_dir_data_source_x, Path)
    assert isinstance(config_obj.data_setting.path_dir_data_source_y, Path)
    data_generator = PicturesDatasetGenerator(
        image_size=tuple(config_obj.data_setting.target_image_size),  # type: ignore
        path_dir_x=config_obj.data_setting.path_dir_data_source_x,
        path_dir_y=config_obj.data_setting.path_dir_data_source_y,
        file_extension=config_obj.data_setting.file_extension,
        sample_size_limit=sample_size_limit)
    
    # splitting data into train(dev included) and test
    seq_path_xy_train = []
    seq_path_xy_test = []

    seq_x_train = []
    seq_y_train = []
    seq_x_test = []
    seq_y_test = []
    
    logger.info('Generating train/dev datasets...')
    for __iter_no in tqdm(range(config_obj.data_generation.sample_size_train)):
        __pair_xy = data_generator.__getitem__(__iter_no)
        
        __path_data_dir = path_dir_data_train / str(__iter_no)
        __path_data_dir.mkdir(parents=True, exist_ok=True)
        
        __path_x = __path_data_dir / 'x.pt'
        __path_y = __path_data_dir / 'y.pt'
        
        if config_obj.is_pre_reload_dataset:
            __x = torch.load(__path_x)['array']
            __y = torch.load(__path_y)['array']
            seq_x_train.append(__x)
            seq_y_train.append(__y)
        else:
            torch.save({'array': __pair_xy[0]}, __path_x)
            torch.save({'array': __pair_xy[1]}, __path_y)
        
            seq_path_xy_train.append(__path_data_dir)
        # end if
    # end for

    logger.info('Generating test datasets...')
    test_data_index_from = config_obj.data_generation.sample_size_train
    test_data_index_until = config_obj.data_generation.sample_size_train + config_obj.data_generation.sample_size_test
    for __iter_no in tqdm(range(test_data_index_from, test_data_index_until)):
        __pair_xy = data_generator.__getitem__(__iter_no)
        
        __path_data_dir = path_dir_data_test / str(__iter_no)
        __path_data_dir.mkdir(parents=True, exist_ok=True)
        
        __path_x = __path_data_dir / 'x.pt'
        __path_y = __path_data_dir / 'y.pt'
        
        if config_obj.is_pre_reload_dataset:
            __x = torch.load(__path_x)['array']
            __y = torch.load(__path_y)['array']
            seq_x_test.append(__x)
            seq_y_test.append(__y)
        else:
            torch.save({'array': __pair_xy[0]}, __path_x)
            torch.save({'array': __pair_xy[1]}, __path_y)
            
            seq_path_xy_test.append(__path_data_dir)
        # end if
    # end for
    logger.info('Data is ready!')
    

    if config_obj.is_pre_reload_dataset:
        logger.info('Pre-reloaded dataset is used. All data are loaded into RAM memory.')
        assert len(seq_x_train) > 0, 'No data found.'
        assert len(seq_y_train) > 0, 'No data found.'
        assert len(seq_x_test) > 0, 'No data found.'
        assert len(seq_y_test) > 0, 'No data found.'
        tensor_x_train = torch.stack(seq_x_train)
        tensor_y_train = torch.stack(seq_y_train)
        tensor_x_test = torch.stack(seq_x_test)
        tensor_y_test = torch.stack(seq_y_test)
        
        data_config_args = DataSetConfigArgs(
                data_x_train=tensor_x_train,
                data_y_train=tensor_y_train,
                data_x_test=tensor_x_test,
                data_y_test=tensor_y_test,
                dataset_type_backend=config_obj.dataset_type_backend,
                dataset_type_charactersitic='static')
    else:
        assert len(seq_path_xy_train) > 0, 'No data found.'
        assert len(seq_path_xy_test) > 0, 'No data found.'
        data_config_args = DataSetConfigArgs(
                data_x_train=path_dir_data_train,
                data_y_train=path_dir_data_train,
                data_x_test=path_dir_data_test,
                data_y_test=path_dir_data_test,
                dataset_type_backend=config_obj.dataset_type_backend,
                dataset_type_charactersitic='static')
    # end if

    path_work_dir = path_root_dir / 'work_dir'
    path_work_dir.mkdir(parents=True, exist_ok=True)
    
    path_detection_output = path_dir_data / 'detection_output'
    path_detection_output.mkdir(parents=True, exist_ok=True)
    

    detection_approaches = config_obj.detection_approaches

    distributed_config = DistributedConfigArgs(
        distributed_mode='single',
        dask_scheduler_host=None)

    parameter_search_parameter = RegularizationSearchParameters(
        n_regularization_parameter=config_obj.cv_selection.n_regularization_parameter,
        n_search_iteration=config_obj.cv_selection.n_search_iteration,
        max_concurrent_job=config_obj.cv_selection.n_max_concurrent
    )

    for __detection_approach in detection_approaches:
        # run the algorithm by interface.
        interface_args = InterfaceConfigArgs(
            resource_config_args=ResourceConfigArgs(
                train_accelerator=config_obj.computational_resource.train_accelerator,
                path_work_dir=path_work_dir,
                distributed_config_detection=distributed_config),  #comment: 8 threads is best choice.
            approach_config_args=ApproachConfigArgs(
                approach_data_representation='sample_based',
                approach_variable_detector=__detection_approach,
                approach_interpretable_mmd='cv_selection'),
            data_config_args=data_config_args,
            detector_algorithm_config_args=DetectorAlgorithmConfigArgs(
                mmd_cv_selection_args=CvSelectionConfigArgs(
                    max_epoch=config_obj.cv_selection.MAX_EPOCH,
                    parameter_search_parameter=parameter_search_parameter,
                    n_subsampling=config_obj.cv_selection.n_subsampling,
                ))
        )
    
        __interface = Interface(interface_args)
        __interface.fit()
        result_obj = __interface.get_result(output_mode='verbose')
        assert isinstance(result_obj, OutputObject)
        
        detection_obj_json: str = result_obj.as_json()
        
        with open(path_detection_output / f'{__detection_approach}.json', 'w') as f:
            f.write(detection_obj_json)
        # end with
        

if __name__ == "__main__":
    import sys
    from argparse import ArgumentParser

    opt = ArgumentParser()
    opt.add_argument('--path_config', type=str, required=True)
    __args = opt.parse_args()

    logger.info("---- Begin of the script ----")
    main(Path(__args.path_config))
    logger.info("---- End of the script ----")