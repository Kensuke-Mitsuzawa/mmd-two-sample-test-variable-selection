from pathlib import Path
from tempfile import mkdtemp
import shutil
import typing as ty
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from mmd_tst_variable_detector.datasets import FileBackendOneTimeLoadStaticDataset



def __gen_file_vector(path_tmp_save_root: Path) -> ty.List[Path]:
    seq_path_parent = []
    
    # generating 10 files.
    for i in range(10):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent.append(__path_parent)
        
        __x = np.random.random(size=(1000,))
        __y = np.random.random(size=(1000,))
        
        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
    # end for
    return seq_path_parent


def __gen_file_matrix(path_tmp_save_root: Path) -> ty.List[Path]:
    seq_path_parent = []
    
    # generating 10 files.
    for i in range(10):
        __path_parent = path_tmp_save_root / f'{i}'
        __path_parent.mkdir(parents=True, exist_ok=True)
        
        seq_path_parent.append(__path_parent)
        
        __x = np.random.random(size=(20, 20))
        __y = np.random.random(size=(20, 20))
        
        torch.save({'array': torch.from_numpy(__x)}, (__path_parent / 'x.pt').as_posix())
        torch.save({'array': torch.from_numpy(__y)}, (__path_parent / 'y.pt').as_posix())
    # end for
    return seq_path_parent




def test_dataset_vector():
    path_tmp_save_root_first = Path(mkdtemp()) / 'test-dataset_first'
    
    sample_set_first = __gen_file_vector(path_tmp_save_root_first)
    dataset_first = FileBackendOneTimeLoadStaticDataset(sample_set_first)
    
    dataset_first.run_files_validation()
    
    # dataset must raise assertion error.
    try:
        dataset_first.__getitem__(0)
    except AssertionError as e:
        pass
    
    # loading data to RAM
    dataset_ready = dataset_first.generate_dataset_on_ram()
    
    # test __getitem__
    t_xy = dataset_ready.__getitem__(0)
    assert isinstance(t_xy, tuple)
    assert isinstance(t_xy[0], torch.Tensor)
    assert isinstance(t_xy[1], torch.Tensor)
    # vector length 1000
    assert t_xy[0].shape[0] == 1000
    assert t_xy[1].shape[0] == 1000
    
    # test __len__
    assert dataset_ready.__len__() == 10
    
    # ----------------------------
    
    # merging test
    path_tmp_save_root_second = Path(mkdtemp()) / 'test-dataset_second'
    sample_set_second = __gen_file_vector(path_tmp_save_root_second)
    dataset_second = FileBackendOneTimeLoadStaticDataset(sample_set_second)

    dataset_obj_merged = dataset_first.merge_new_dataset(dataset_second)
    assert len(dataset_obj_merged) == 20
    
    dataset_second_ready = dataset_obj_merged.generate_dataset_on_ram()
    
    # test selected variables
    selected_variables = [0, 1, 2, 3, 4]
    dataset_obj_selected = dataset_second_ready.get_selected_variables_dataset(tuple(selected_variables))
    t_xy = dataset_obj_selected.__getitem__(0)
    assert t_xy[0].shape[0] == len(selected_variables)
    
    # test with DataLoader
    
    loader = DataLoader(dataset_second_ready, batch_size=10, pin_memory=True, num_workers=4)
    for __ in loader:
        pass
    # end for
    
    # deleting
    shutil.rmtree(path_tmp_save_root_first)
    shutil.rmtree(path_tmp_save_root_second)
    
    
def test_dataset_matrix():
    path_tmp_save_root_first = Path(mkdtemp()) / 'test-dataset_first'
    
    sample_set_first = __gen_file_matrix(path_tmp_save_root_first)
    dataset_first = FileBackendOneTimeLoadStaticDataset(sample_set_first)
    
    dataset_first.run_files_validation()
    
    dataset_ready = dataset_first.generate_dataset_on_ram()
    
    # test __getitem__
    t_xy = dataset_ready.__getitem__(0)
    assert isinstance(t_xy, tuple)
    assert isinstance(t_xy[0], torch.Tensor)
    assert isinstance(t_xy[1], torch.Tensor)
    # vector length 1000
    assert t_xy[0].shape[0] == 400
    assert t_xy[1].shape[0] == 400
    
    # test __len__
    assert dataset_first.__len__() == 10
    
    # merging test
    path_tmp_save_root_second = Path(mkdtemp()) / 'test-dataset_second'
    sample_set_second = __gen_file_matrix(path_tmp_save_root_second)
    dataset_second = FileBackendOneTimeLoadStaticDataset(sample_set_second)

    dataset_obj_merged = dataset_first.merge_new_dataset(dataset_second)
    assert len(dataset_obj_merged) == 20
    
    dataset_ready = dataset_obj_merged.generate_dataset_on_ram()
    
    # test selected variables
    selected_variables = [0, 1, 2, 3, 4]
    dataset_obj_selected = dataset_ready.get_selected_variables_dataset(tuple(selected_variables))
    t_xy = dataset_obj_selected.__getitem__(0)
    assert t_xy[0].shape[0] == len(selected_variables)
    
    # test with DataLoader
    
    loader = DataLoader(dataset_ready, batch_size=10, pin_memory=True, num_workers=4)
    for __ in loader:
        pass
    # end for
    
    # deleting
    shutil.rmtree(path_tmp_save_root_first)
    shutil.rmtree(path_tmp_save_root_second)
