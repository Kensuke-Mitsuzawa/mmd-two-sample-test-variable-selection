import itertools
import numpy as np
import typing as ty

from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.module_data_sampling import DataSampling
from tests import data_generator


from mmd_tst_variable_detector import RamBackendStaticDataset


def is_all_arrays_equal(arrays: ty.List[np.ndarray]) -> bool:
    if not arrays:
        raise ValueError("The list of arrays is empty.")
    
    seq_true = []
    
    first_array = arrays[0]
    for array in arrays[1:]:
        __is_same = np.array_equal(first_array, array), "Not all arrays are equal."
        seq_true.append(__is_same[0])
    # end for
    
    return all(seq_true)
        

def test_cross_validation():
    # Test re-loading and resume..
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=1234)
    dataset_train = RamBackendStaticDataset(t_xy_train[0], t_xy_train[1])
    
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=4321) 
    dataset_dev = RamBackendStaticDataset(t_xy_dev[0], t_xy_dev[1])

    data_sampling = DataSampling(1234)

    seq_generated_task_datasets = data_sampling.get_datasets(
        seq_parameter_type_ids=[1, 2],
        n_sampling=3,
        training_dataset=dataset_train,
        validation_dataset=dataset_dev,
        sampling_strategy='cross-validation',
        ratio_training_data=0.5)
    
    func_sort = lambda obj: obj.task_key.data_splitting_id
    g_generated_task_datasets = itertools.groupby(
        sorted(seq_generated_task_datasets, key=func_sort),
        key=func_sort)
    for splitting_id, seq_task_datasets in g_generated_task_datasets:
        seq_task_datasets = list(seq_task_datasets)
        assert len(seq_task_datasets) == 2
        
        for task_dataset in seq_task_datasets:
            assert task_dataset.task_key.data_splitting_id == splitting_id
        # end for
        
        stack_datasets_train_x = []
        stack_datasets_train_y = []
        stack_datasets_dev_x = []
        stack_datasets_dev_y = []
        for __task_container in seq_task_datasets:
            stack_datasets_train_x.append(__task_container.dataset_train._x.numpy())
            stack_datasets_train_y.append(__task_container.dataset_train._y.numpy())
            stack_datasets_dev_x.append(__task_container.dataset_dev._x.numpy())
            stack_datasets_dev_y.append(__task_container.dataset_dev._y.numpy())
        # end for
        assert is_all_arrays_equal(stack_datasets_train_x)
        assert is_all_arrays_equal(stack_datasets_train_y)
        assert is_all_arrays_equal(stack_datasets_dev_x)
        assert is_all_arrays_equal(stack_datasets_dev_y)
    # end for
# end test


def test_random_splitting():
    # Test re-loading and resume..
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=1234)
    dataset_train = RamBackendStaticDataset(t_xy_train[0], t_xy_train[1])
    
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=4321) 
    dataset_dev = RamBackendStaticDataset(t_xy_dev[0], t_xy_dev[1])

    data_sampling = DataSampling(1234)

    seq_generated_task_datasets = data_sampling.get_datasets(
        seq_parameter_type_ids=[1, 2],
        n_sampling=3,
        training_dataset=dataset_train,
        validation_dataset=dataset_dev,
        sampling_strategy='random-splitting',
        ratio_training_data=0.5)
    
    func_sort = lambda obj: obj.task_key.data_splitting_id
    g_generated_task_datasets = itertools.groupby(
        sorted(seq_generated_task_datasets, key=func_sort),
        key=func_sort)
    for splitting_id, seq_task_datasets in g_generated_task_datasets:
        seq_task_datasets = list(seq_task_datasets)
        assert len(seq_task_datasets) == 2
        
        for task_dataset in seq_task_datasets:
            assert task_dataset.task_key.data_splitting_id == splitting_id
        # end for
        
        stack_datasets_train_x = []
        stack_datasets_train_y = []
        stack_datasets_dev_x = []
        stack_datasets_dev_y = []
        for __task_container in seq_task_datasets:
            stack_datasets_train_x.append(__task_container.dataset_train._x.numpy())
            stack_datasets_train_y.append(__task_container.dataset_train._y.numpy())
            stack_datasets_dev_x.append(__task_container.dataset_dev._x.numpy())
            stack_datasets_dev_y.append(__task_container.dataset_dev._y.numpy())
        # end for
        assert is_all_arrays_equal(stack_datasets_train_x) == False
        assert is_all_arrays_equal(stack_datasets_train_y) == False
        assert is_all_arrays_equal(stack_datasets_dev_x) == False
        assert is_all_arrays_equal(stack_datasets_dev_y) == False
    # end for
# end test


def test_k_fold_cross_validation():
    # Test re-loading and resume..
    t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=1234)
    dataset_train = RamBackendStaticDataset(t_xy_train[0], t_xy_train[1])
    
    t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=4321) 
    dataset_dev = RamBackendStaticDataset(t_xy_dev[0], t_xy_dev[1])

    data_sampling = DataSampling(1234)

    seq_generated_task_datasets = data_sampling.get_datasets(
        seq_parameter_type_ids=[1, 2],
        n_sampling=3,
        training_dataset=dataset_train,
        validation_dataset=dataset_dev,
        sampling_strategy='random-splitting',
        ratio_training_data=0.5)
    
    func_sort = lambda obj: obj.task_key.data_splitting_id
    g_generated_task_datasets = itertools.groupby(
        sorted(seq_generated_task_datasets, key=func_sort),
        key=func_sort)
    for splitting_id, seq_task_datasets in g_generated_task_datasets:
        seq_task_datasets = list(seq_task_datasets)
        assert len(seq_task_datasets) == 2
        
        for task_dataset in seq_task_datasets:
            assert task_dataset.task_key.data_splitting_id == splitting_id
        # end for
        
        stack_datasets_train_x = []
        stack_datasets_train_y = []
        stack_datasets_dev_x = []
        stack_datasets_dev_y = []
        for __task_container in seq_task_datasets:
            stack_datasets_train_x.append(__task_container.dataset_train._x.numpy())
            stack_datasets_train_y.append(__task_container.dataset_train._y.numpy())
            stack_datasets_dev_x.append(__task_container.dataset_dev._x.numpy())
            stack_datasets_dev_y.append(__task_container.dataset_dev._y.numpy())
        # end for
        assert is_all_arrays_equal(stack_datasets_train_x) == False
        assert is_all_arrays_equal(stack_datasets_train_y) == False
        assert is_all_arrays_equal(stack_datasets_dev_x) == False
        assert is_all_arrays_equal(stack_datasets_dev_y) == False
    # end for
# end test


def test_deterministic_splitting(n_iterations: int = 5):    
    seq_stack = []
    for __n_iter in range(n_iterations):
        # Test re-loading and resume..
        t_xy_train, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=1234)
        dataset_train = RamBackendStaticDataset(t_xy_train[0], t_xy_train[1])
        
        t_xy_dev, __ = data_generator.test_data_xy_linear(sample_size=100, random_seed=4321)
        dataset_dev = RamBackendStaticDataset(t_xy_dev[0], t_xy_dev[1])

        data_sampling = DataSampling(1234)

        seq_generated_task_datasets = data_sampling.get_datasets(
            seq_parameter_type_ids=[1],
            n_sampling=3,
            training_dataset=dataset_train,
            validation_dataset=dataset_dev,
            sampling_strategy='cross-validation',
            ratio_training_data=0.5)
        
        assert len(seq_generated_task_datasets) == 3
        seq_stack += seq_generated_task_datasets
    # end for
    
    func_sort = lambda obj: obj.task_key.data_splitting_id
    g_generated_task_datasets = itertools.groupby(
        sorted(seq_stack, key=func_sort),
        key=func_sort)    
    
    stack_datasets_train_x = []
    stack_datasets_train_y = []
    stack_datasets_dev_x = []
    stack_datasets_dev_y = []
        
    for data_splitting_id, seq_task_datasets in g_generated_task_datasets:
        seq_task_datasets = list(seq_task_datasets)
        
        for __container in seq_task_datasets:
            assert __container.task_key.data_splitting_id == data_splitting_id
                    
            if data_splitting_id == 0:
                # save the first CV dataset.
                stack_datasets_train_x.append(__container.dataset_train._x.numpy())
                stack_datasets_train_y.append(__container.dataset_train._y.numpy())
                stack_datasets_dev_x.append(__container.dataset_dev._x.numpy())
                stack_datasets_dev_y.append(__container.dataset_dev._y.numpy())
            # end if
        # end for
    # end for

    assert is_all_arrays_equal(stack_datasets_train_x)
    assert is_all_arrays_equal(stack_datasets_train_y)
    assert is_all_arrays_equal(stack_datasets_dev_x)
    assert is_all_arrays_equal(stack_datasets_dev_y)
# end test