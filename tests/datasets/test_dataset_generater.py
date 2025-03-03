from pathlib import Path
import typing as ty
from tempfile import mkdtemp
import shutil

import numpy as np
import torch

from mmd_tst_variable_detector.interface.dataset_generater import DatasetGenerater
from mmd_tst_variable_detector.datasets import (
    FileBackendStaticDataset,
    SimpleDataset,
    FileBackendSensorSampleBasedDataset
)


def __gen_file_vector(path_tmp_save_root: Path, data_form: str) -> ty.Tuple[ty.List[Path], ty.List[Path]]:
    assert data_form in ('vector', 'matrix')
    
    seq_path_parent_x = []
    seq_path_parent_y = []
    
    # generating 10 files.
    for i in range(10):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent_x.append(__path_parent / 'x.pt')
        seq_path_parent_y.append(__path_parent / 'y.pt')
        
        if data_form == 'vector':
            __x = np.random.random(size=(1000,))
            __y = np.random.random(size=(1000,))
        elif data_form == 'matrix':
            __x = np.random.random(size=(1000, 100))
            __y = np.random.random(size=(1000, 100))
        else:
            raise ValueError(f'Unexpected data_form: {data_form}')
        # end if

        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
    # end for
    return seq_path_parent_x, seq_path_parent_y



def test_dataset_init_test_file_sensor_st_sample_based():
    path_tmp_work = Path(mkdtemp())
    seq_path_x, seq_path_y = __gen_file_vector(path_tmp_work, 'matrix')

    dataset_generater = DatasetGenerater(
        data_x=seq_path_x,
        data_y=seq_path_y,
        dataset_type_backend='file',
        dataset_type_charactersitic='sensor_st',
        dataset_type_algorithm='sample_based',
        time_aggregation_per=10)
    seq_dataset = dataset_generater.get_dataset()
    assert len(seq_dataset) == 1
    assert all([isinstance(__d, FileBackendSensorSampleBasedDataset) for __d  in seq_dataset])

    shutil.rmtree(path_tmp_work.as_posix())


def test_dataset_init_test_file_static_sample_based():
    # test case of file backend / static characteristic / sample_based algorithm
    path_tmp_save_root_first = Path(mkdtemp()) / 'test-dataset_first'
    seq_path_x, seq_path_y = __gen_file_vector(path_tmp_save_root_first, 'vector')
    
    dataset_generater = DatasetGenerater(
        data_x=seq_path_x,
        data_y=seq_path_y,
        dataset_type_backend='file',
        dataset_type_charactersitic='static',
        dataset_type_algorithm='sample_based')
    dataset = dataset_generater.get_dataset()
    assert len(dataset) == 1
    assert isinstance(dataset[0], FileBackendStaticDataset)
    shutil.rmtree(path_tmp_save_root_first.as_posix())


def test_dataset_init_test_ram_static_sample_based():
    # test case of ram backend / static characteristic / sample_based algorithm
    array_x = np.random.random(size=(1000, 100))
    array_y = np.random.random(size=(1000, 100))
    
    dataset_generater = DatasetGenerater(
        data_x=array_x,
        data_y=array_y,
        dataset_type_backend='ram',
        dataset_type_charactersitic='static',
        dataset_type_algorithm='sample_based')
    dataset = dataset_generater.get_dataset()
    assert len(dataset) == 1
    assert isinstance(dataset[0], SimpleDataset)
