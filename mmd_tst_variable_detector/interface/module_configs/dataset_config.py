from dataclasses import dataclass
from pathlib import Path
import typing as ty


from .static_module import (
    PossibleDataCharacteristic, 
    PossibleDatasetBackend,
)
from ..dataset_generater import PossibleInputType



@dataclass
class DataSetConfigArgs:
    """
    Parameters
    ----------
    data_x_train: DataPossibleArgumentType
        Data for SampleSet x.
        When configuration is from toml, either of following choices,
        1. path to the file where data is stored
        2. path to the directory where a set of files are stored.
    data_y_train: DataPossibleArgumentType
        Data for SampleSet y.
        The same discription as `data_x`.
    data_x_test: DataPossibleArgumentType
        Data for SampleSet x.
        The same discription as `data_x`.
        You can leave it blank if you do not have test data.
    data_y_test: DataPossibleArgumentType
        Data for SampleSet y.
        The same discription as `data_x`.
        You can leave it blank if you do not have test data.
    dataset_type_backend: str
        Backend of dataset.
        Either of following choices,
        1. 'ram': data is stored in RAM.
        2. 'file': data is stored in file.
        3. 'flexible-file': data is stored in file, and you can specify the number of data to be loaded.
    dataset_type_charactersitic: str
        Charactersitic of dataset.
        Either of following choices,
        1. 'static': data is static.
        2. 'sensor_st': data is sensor data and static.
        3. 'trajectory_st': data is trajectory data and static.
    file_name_x: str
        File name of data for SampleSet x.
        This parameter is used only when `dataset_type_backend` is 'file'.
    file_name_y: str
        File name of data for SampleSet y.
        The same discription as `file_name_x`.
    ratio_train_test: float
        ratio of splitting dataset into train and test.
        Not used when you give `data_x_test` and `data_y_test`.
        When `ratio_train_test = -1`, no splitting.
    """
    data_x_train: PossibleInputType
    data_y_train: PossibleInputType
    
    data_x_test: ty.Optional[PossibleInputType]
    data_y_test: ty.Optional[PossibleInputType]
    
    dataset_type_backend: str
    dataset_type_charactersitic: str
    
    key_name_array: str = 'array'
    file_name_x: str = 'x.pt'
    file_name_y: str = 'y.pt'
        
    time_aggregation_per: int = 1
    ratio_train_test: float = 0.8
    
    def __post_init__(self):        
        self.dataset_type_backend = self.dataset_type_backend.lower()
        self.dataset_type_charactersitic = self.dataset_type_charactersitic.lower()
        
        assert self.dataset_type_backend in PossibleDatasetBackend
        assert self.dataset_type_charactersitic in PossibleDataCharacteristic

        if isinstance(self.data_x_train, str):
            self.data_x_train = Path(self.data_x_train)
        # end if
        if isinstance(self.data_y_train, str):
            self.data_y_train = Path(self.data_y_train)
        # end if

        if self.data_x_test == '':
            assert self.data_y_test == '', 'data_x_test is not given, but data_y_test is given.'
            self.data_x_test = None
            self.data_y_test = None
        else:
            # logger.info(f'`ratio_train_test`={self.ratio_train_test} is ignored. \
            #     I use `data_x_test` and `data_y_test` instead.')
            
            if isinstance(self.data_x_test, str):
                self.data_x_test = Path(self.data_x_test)
            # end if
            if isinstance(self.data_y_test, str):
                self.data_y_test = Path(self.data_y_test)
        # end if

