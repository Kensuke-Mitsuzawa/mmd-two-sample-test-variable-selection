import typing as ty
import itertools
import random
import copy
import logging
import numpy as np

from pathlib import Path

from sklearn.model_selection import KFold

from ...datasets import BaseDataset
from ...logger_unit import handler


logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



class JobKeyGeneratedDataset(ty.NamedTuple):
    parameter_type_id: int
    data_splitting_id: int


class _TaskTuple(ty.NamedTuple):
    task_key: JobKeyGeneratedDataset
    dataset_train: BaseDataset
    dataset_dev: BaseDataset
    dataset_test: ty.Optional[BaseDataset]



class DataSampling(object):
    def __init__(self,
                 seed_root_random: int):
        self.seed_root_random = seed_root_random
        
    def __random_splitting(self,
                           dataset_whole: BaseDataset,
                           seq_parameter_type_ids: ty.List[ty.Any],
                           n_cv: int,
                           ratio_training_data: float) -> ty.List[_TaskTuple]:
        combination_list = list(itertools.product(seq_parameter_type_ids, range(n_cv)))
        # generate sequence of random seeds.
        # the N(random-seed) is equal to the number of task-to-be-executed.
        local_random_gen = random.Random(self.seed_root_random)
        random_seed_ids = [local_random_gen.randint(0, 999) for _ in range(0, len(combination_list))]
        assert len(random_seed_ids) == len(combination_list)
        
        # prepare sample-ids for training and validation.
        __n_sample_train = int(len(dataset_whole) * ratio_training_data)


        seq_task_parameters: ty.List[_TaskTuple] = []
        
        for __i_sub_id, __sub_id_tuple in enumerate(combination_list):
            assert isinstance(__sub_id_tuple, tuple), 's_id_tuple must be a tuple.'
            
            _parameter_type_id = __sub_id_tuple[0]
            _cv_id = __sub_id_tuple[1]

            __agg_key = JobKeyGeneratedDataset(
                parameter_type_id=_parameter_type_id,
                data_splitting_id=_cv_id)
                
            # split the dataset into training and validation.
            __local_sampling_random_gen = random.Random(random_seed_ids[__i_sub_id])
            _sample_ids_train = __local_sampling_random_gen.sample(range(len(dataset_whole)), k=__n_sample_train)
            _sample_ids_val = list(set(range(len(dataset_whole))) - set(_sample_ids_train))
            assert len(_sample_ids_train) + len(_sample_ids_val) == len(dataset_whole), 'The sum of sample_ids_train and sample_ids_val must be equal to the length of the whole dataset.'
            assert len(set(_sample_ids_train).intersection(set(_sample_ids_val))) == 0, 'sample_ids_train and sample_ids_val must be disjoint.'            

            __, new_dataset_train = dataset_whole.get_subsample_dataset(sample_ids=_sample_ids_train)
            __, new_dataset_val = dataset_whole.get_subsample_dataset(sample_ids=_sample_ids_val)
            
            __task_args = _TaskTuple(
                task_key=__agg_key,
                dataset_train=new_dataset_train,
                dataset_dev=new_dataset_val,
                dataset_test=None)
            seq_task_parameters.append(__task_args)
        # end for
        return seq_task_parameters    

    def __cross_validation(self,
                           dataset_whole: BaseDataset,
                           seq_parameter_type_ids: ty.List[ty.Any],
                           n_cv: int,
                           ratio_training_data: float) -> ty.List[_TaskTuple]:        
        # generate sequence of random seeds.
        # the N(random-seed) is equal to the number of subsampling.
        local_random_gen = random.Random(self.seed_root_random)
        random_seed_ids = [local_random_gen.randint(0, 999) for _ in range(0, n_cv)]
        assert len(random_seed_ids) == n_cv
        
        # prepare sample-ids for training and validation.
        __n_sample_train = int(len(dataset_whole) * ratio_training_data)
        dict_cv2sample_ids = {}
        for __i_cv in range(n_cv):
            __local_sampling_random_gen = random.Random(random_seed_ids[__i_cv])
            _sample_ids_train = __local_sampling_random_gen.sample(range(len(dataset_whole)), k=__n_sample_train)
            _sample_ids_val = list(set(range(len(dataset_whole))) - set(_sample_ids_train))
            dict_cv2sample_ids[__i_cv] = (_sample_ids_train, _sample_ids_val)
            assert len(_sample_ids_train) + len(_sample_ids_val) == len(dataset_whole), 'The sum of sample_ids_train and sample_ids_val must be equal to the length of the whole dataset.'
            assert len(set(_sample_ids_train).intersection(set(_sample_ids_val))) == 0, 'sample_ids_train and sample_ids_val must be disjoint.'            
        # end for

        seq_task_parameters: ty.List[_TaskTuple] = []

        for __parameter_type_id in seq_parameter_type_ids:
            for __i_cv in range(0, n_cv):
                __agg_key = JobKeyGeneratedDataset(
                    parameter_type_id=__parameter_type_id,
                    data_splitting_id=__i_cv)
                
                # split the dataset into training and validation.
                sample_ids_train, sample_ids_val = dict_cv2sample_ids[__i_cv]
                __, new_dataset_train = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_train)
                __, new_dataset_val = dataset_whole.get_subsample_dataset(sample_ids=sample_ids_val)
                
                __task_args = _TaskTuple(
                    task_key=__agg_key,
                    dataset_train=new_dataset_train,
                    dataset_dev=new_dataset_val,
                    dataset_test=None)
                seq_task_parameters.append(__task_args)
            # end for
        # end for
        return seq_task_parameters
    
    def __k_fold_cross_validation(self,
                                  seq_parameter_type_ids: ty.List[ty.Any],
                                  dataset_whole: BaseDataset,
                                  n_cv: int) -> ty.List[_TaskTuple]:
        _k_fold = n_cv
        _k_fold_splitter = KFold(_k_fold, shuffle=False)  # shuffle off for the better re-producibility.
        
        seq_sample_ids = list(range(len(dataset_whole)))
        
        seq_task_parameters = []
        
        for __i_cv, (train_index, test_index) in enumerate(_k_fold_splitter.split(np.array(seq_sample_ids))):
            assert train_index is not None and test_index is not None
            __, _train_dataset = dataset_whole.get_subsample_dataset(sample_ids=train_index.tolist())
            __, _dev_dataset = dataset_whole.get_subsample_dataset(sample_ids=test_index.tolist())

            _new_dataset_train = _train_dataset.copy_dataset()
            _new_dataset_val = _dev_dataset.copy_dataset()
            
            for __parameter_type_id in seq_parameter_type_ids:

                __agg_key = JobKeyGeneratedDataset(
                    parameter_type_id=__parameter_type_id,
                    data_splitting_id=__i_cv)
            
                seq_task_parameters.append(_TaskTuple(
                    task_key=__agg_key,
                    dataset_train=_new_dataset_train,
                    dataset_dev=_new_dataset_val,
                    dataset_test=None))
            # end for
        # end for
        return seq_task_parameters
        
    def get_datasets(self,
                     seq_parameter_type_ids: ty.List[ty.Any],
                     n_sampling: int,
                     training_dataset: BaseDataset,
                     validation_dataset: ty.Optional[BaseDataset],
                     sampling_strategy: str,
                     ratio_training_data: float
                     ) -> ty.List[_TaskTuple]:
        """Generate arguments for distributed computing.

        Parameters
        --------------
        sub_id_tuple: ty.List[ty.Tuple[RegularizationParameter, int]]
        """
        assert sampling_strategy in ('cross-validation', 'random-splitting', 'k-fold-cross-validation'), f'sampling_strategy={sampling_strategy} is not supported.'
        
        seq_task_arguments = []
        
        if sampling_strategy in ('subsampling', 'bootstrap'):
            assert training_dataset is not None, 'training_dataset must not be None.'
            assert validation_dataset is not None, 'validation_dataset must not be None'
        else:
            # in cross-validation mode.
            assert training_dataset is not None, 'training_dataset must not be None.'
        # end if

        if sampling_strategy in ('cross-validation', 'random-splitting', 'k-fold-cross-validation'):
            if validation_dataset is None:
                logger.debug('I use the training_dataset at the whole_dataset.')
                dataset_whole = training_dataset
            else:
                dataset_whole = training_dataset.merge_new_dataset(validation_dataset)
        else:
            dataset_whole = None
        # end if

        if sampling_strategy == 'cross-validation':
            assert dataset_whole is not None, 'dataset_whole must not be None.'
            seq_task_arguments = self.__cross_validation(
                dataset_whole=dataset_whole,
                seq_parameter_type_ids=seq_parameter_type_ids,
                n_cv=n_sampling,
                ratio_training_data=ratio_training_data)
        elif sampling_strategy == 'random-splitting':
            assert dataset_whole is not None, 'dataset_whole must not be None.'
            seq_task_arguments = self.__random_splitting(
                dataset_whole=dataset_whole,
                seq_parameter_type_ids=seq_parameter_type_ids,
                n_cv=n_sampling,                
                ratio_training_data=ratio_training_data)
        elif sampling_strategy == 'k-fold-cross-validation':
            assert dataset_whole is not None, 'dataset_whole must not be None.'            
            seq_task_arguments = self.__k_fold_cross_validation(
                seq_parameter_type_ids=seq_parameter_type_ids,
                dataset_whole=dataset_whole,
                n_cv=n_sampling)
        else:
            raise NotImplementedError()
        
        return seq_task_arguments
