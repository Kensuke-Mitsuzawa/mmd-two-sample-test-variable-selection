import typing as ty
import random
import math

import numpy
from tqdm import tqdm
from datetime import datetime

import torch
import numpy as np
from ot import sliced_wasserstein_distance as swd

from torch.utils.data import DataLoader

from distributed import Client
# comment: must be from `base`, otherwise circular import error occurs.
from ..datasets.base import BaseDataset


TypeFunctionInOut = ty.Callable[[np.ndarray, np.ndarray], float]


class FunctionArgumentFull(ty.NamedTuple):
    func_distance: TypeFunctionInOut 
    z: np.ndarray
    n_x: int
    n_y: int


class FunctionArgumentBatch(ty.NamedTuple):
    func_distance: TypeFunctionInOut 
    dataset: BaseDataset
    batch_size: int
    featre_weights: ty.Optional[np.ndarray] = None



def _base_func_distance_sliced_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    _d = swd(x, y)
    return _d  # type: ignore


class PermutationTest(object):
    def __init__(self,
                 func_distance: TypeFunctionInOut = _base_func_distance_sliced_wasserstein,
                 n_permutation_test: int = 500,
                 batch_size: int = -1,
                 is_shuffle: bool = False,
                 is_normalize: bool = False,
                 enable_progress_bar: bool = False,
                 dask_client: ty.Optional[Client] = None):
        self.func_distance = func_distance
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.n_permutation_test = n_permutation_test
        self.enable_progress_bar = enable_progress_bar
        
        self.dask_client = dask_client
    
    # -----------------------------------------------------
    # static methods called by Dask
    @staticmethod
    def __get_distance_z(func_args: FunctionArgumentFull) -> float:
        
        func_distance: TypeFunctionInOut = func_args.func_distance 
        z: np.ndarray = func_args.z
        # z value must be always float.32
        z = z.astype(np.float32)
        
        n_x: int = func_args.n_x
        n_y: int = func_args.n_y
        
        z_a_ind = random.Random(datetime.now().microsecond).sample(range(0, len(z)), n_x)
        z_b_ind = list(set(range(0, len(z))) - set(z_a_ind))
        z_a = z[z_a_ind, :]
        z_b = z[z_b_ind, :]
        
        __distance = func_distance(z_a, z_b)
        if isinstance(__distance, torch.Tensor):
            __distance = __distance.cpu().detach().numpy()
        # end if
        return __distance
    
    @staticmethod
    def __sample_null_batch_dask_func(func_args: FunctionArgumentBatch) -> np.ndarray:
        func_distance: TypeFunctionInOut = func_args.func_distance
        dataset: BaseDataset = func_args.dataset
        batch_size: int = func_args.batch_size
        featre_weights: ty.Optional[np.ndarray] = func_args.featre_weights
        
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch_distance = []
        for batch in data_loader:
            x_batch = batch[0]
            y_batch = batch[1]
            z = np.concatenate([x_batch, y_batch], 0)  # SOLUTION
            if featre_weights is not None:
                if len(z.shape) == 3:
                    # comment: element-wise-product((N, |S|, C), (|S|))
                    z = np.einsum('ijk,j->ijk', z, featre_weights)
                else:
                    z = np.multiply(featre_weights, z)
                # end if
            # end if
            # z value must be always float.32
            z = z.astype(np.float32)                        
            _n_half_z = int(len(z) / 2)
            
            z_a_ind = random.Random(datetime.now().microsecond).sample(range(0, len(z)), k=_n_half_z)
            z_b_ind = list(set(range(0, len(z))) - set(z_a_ind))
            z_a = z[z_a_ind, :]
            z_b = z[z_b_ind, :]
            __distance = func_distance(z_a, z_b)
            batch_distance.append(__distance)
        # end for
        
        return np.stack(batch_distance)
    
    # -----------------------------------------------------
    
    def __sample_null_batch(self, 
                            dataset: BaseDataset, 
                            num_permutations: int = 500,
                            featre_weights: ty.Optional[np.ndarray] = None,
                            ):
        stats = []
        range_ = range(num_permutations)
        
        seq_func_requests = []
        for i in range_:
            seq_func_requests.append(FunctionArgumentBatch(
                func_distance=self.func_distance,
                dataset=dataset,
                batch_size=self.batch_size,
                featre_weights=featre_weights))
        # end for
        
        if self.dask_client is None:
            seq_dist_values = [self.__sample_null_batch_dask_func(__args) for __args in seq_func_requests]
        else:
            task_queue = self.dask_client.map(self.__sample_null_batch_dask_func, seq_func_requests)
            seq_dist_values = self.dask_client.gather(task_queue)
        # end if

        # array (num_permutations, |num_permutations|)
        assert isinstance(seq_dist_values, list)
        tensor_stack_distance = np.stack(seq_dist_values)
        distance_mean = np.mean(tensor_stack_distance, axis=1)
        # end for
        return distance_mean

    @staticmethod
    def __stack_tensor(seq_sample: ty.List[np.ndarray]) -> np.ndarray:
        sample_shape = seq_sample[0].shape
        
        if len(sample_shape) == 1:        
            if all(isinstance(_x, numpy.ndarray) for _x in seq_sample):
                tensor_obj = np.vstack(seq_sample)
            else:
                tensor_obj = np.vstack(seq_sample)
        else:
            tensor_obj = np.stack(seq_sample, axis=0)
        # end if
        return tensor_obj

    def __sample_null(self,
                      dataset: BaseDataset,
                      num_permutations: int = 500,
                      featre_weights: ty.Optional[np.ndarray] = None
                      ) -> np.ndarray:
        """Compute statistics with sub-sampled data. The process is normally called "Permutation".

        Args:
            num_permutations: #time to run permutation test.
            is_progress: boolen, if True; then show a progress bar, False not.

        Returns: array object which contains values by permutations.
        """
        # Getting dataset thorugh dataset's method. Feature-extraction is necessary here.
        with torch.no_grad():        
            _seq_pair_xy = [dataset.__getitem__(_idx) for _idx in range(0, len(dataset))]
        # end with
        
        x_all = [t_xy[0] for t_xy in _seq_pair_xy]
        y_all = [t_xy[1] for t_xy in _seq_pair_xy]
        
        x_all = self.__stack_tensor(x_all)        
        y_all = self.__stack_tensor(y_all)
                
        z = np.concatenate([x_all, y_all], 0)  # SOLUTION
        
        if featre_weights is not None:
            if len(z.shape) == 3:
                # comment: element-wise-product((N, |S|, C), (|S|))
                z = np.einsum('ijk,j->ijk', z, featre_weights)
            else:
                z = np.multiply(featre_weights, z)
            # end if
        # end if
        
        n_x = x_all.shape[0]
        n_y = y_all.shape[0]

        range_ = range(num_permutations)
        
        seq_func_args = [FunctionArgumentFull(self.func_distance, z, n_x, n_y) for _ in range_]
    
        if self.dask_client is None:
            seq_dist_values = [self.__get_distance_z(__args) for __args in seq_func_args]
        else:
            task_queue = self.dask_client.map(self.__get_distance_z, seq_func_args)
            seq_dist_values = self.dask_client.gather(task_queue)
        # end if
        
        assert isinstance(seq_dist_values, list)

        array_null_distribution = np.stack(seq_dist_values)
        # end if
        return array_null_distribution
    
    def __compute_statistic(self, dataset: BaseDataset) -> float:
        with torch.no_grad():
            _seq_pair_xy = [dataset.__getitem__(_idx) for _idx in range(0, len(dataset))]
        # end with
        x_all = [t_xy[0] for t_xy in _seq_pair_xy]
        y_all = [t_xy[1] for t_xy in _seq_pair_xy]
        
        x_all = self.__stack_tensor(x_all)
        y_all = self.__stack_tensor(y_all)
        
        x_all = x_all.astype(np.float32)
        y_all = y_all.astype(np.float32)
                        
        statistics = self.func_distance(x_all, y_all)
        # end if

        return statistics
        
    def __compute_statistic_batch(self, dataset: BaseDataset) -> float:
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        __i = 0
        statistics_batch = []
        with torch.no_grad():
            for batch in data_loader:
                __x = batch[0].numpy().astype(np.float32)
                __y = batch[1].numpy().astype(np.float32)
                __statistics = self.func_distance(__x, __y)
                # end if
                statistics_batch.append(__statistics)
                __i += 1
            # end for
        # end with   
        __np_stack = np.stack(statistics_batch)
        statistics = np.mean(__np_stack)
        return statistics

    def compute_threshold(self, stats_permutation_test: np.ndarray, alpha: float = 0.05) -> float:
        """Compute threshold against the given alpha value.

        Args:
            alpha:

        Returns: A value where the threshold exists.
        """
        values_sorted = sorted(stats_permutation_test)
        index_alpha: int = math.floor(len(values_sorted) * (1 - alpha))
        return values_sorted[index_alpha]

    def normalize_statistic(self, statistic: np.ndarray, length_x: int, length_y: int) -> np.ndarray:
        n_x = length_x
        n_y = length_y
        return n_x * n_y * statistic / (n_x + n_y)

    def compute_p_value(self, stats_permutation_test: np.ndarray, statistic: float) -> float:
        """Compute p-value based on the permutation tests.

        Args:
            statistic: A statistic value with the whole dataset.

        Returns: p-value.
        """
        # check if all values are the same
        is_all_same: np.bool_ = np.all(stats_permutation_test == stats_permutation_test[0])
        is_all_same_statistics = np.all(stats_permutation_test == statistic)
        # check if greater value than the given statistic
        # is_greater: np.bool_ = np.any(stats_permutation_test[stats_permutation_test > statistic])
        if is_all_same and is_all_same_statistics:
            return 1.0
        # elif is_greater == False:
        #     return 0.0
        else:
            values_sorted = np.sort(stats_permutation_test)
            i = self.find_position_to_insert(values_sorted, statistic)
            return 1.0 - i / len(stats_permutation_test)

    @staticmethod
    def find_position_to_insert(values_sorted: np.ndarray, statistic: float) -> int:
        i = 0
        for i, value_i in enumerate(values_sorted):
            if value_i > statistic:
                return i
            # end if
        # end if
        return i

    def run_test(self, 
                 dataset: BaseDataset, 
                 threshold_alpha: float = 0.05,
                 featre_weights: ty.Optional[ty.Union[np.ndarray, torch.Tensor]] = None
                 ) -> ty.Tuple[float, np.ndarray]:
        if isinstance(featre_weights, torch.Tensor):
            featre_weights = featre_weights.numpy()
        # end if
        assert isinstance(featre_weights, np.ndarray) or featre_weights is None
        
        if self.batch_size == -1:
            stats_permutation_test = self.__sample_null(dataset, 
                                                        num_permutations=self.n_permutation_test, 
                                                        featre_weights=featre_weights)
            data_statistics = self.__compute_statistic(dataset)
        else:
            stats_permutation_test = self.__sample_null_batch(dataset, 
                                                              num_permutations=self.n_permutation_test, 
                                                              featre_weights=featre_weights)
            data_statistics = self.__compute_statistic_batch(dataset)
        # end if
        
        threshold = self.compute_threshold(stats_permutation_test, alpha=threshold_alpha)
        p_value = self.compute_p_value(stats_permutation_test, data_statistics)

        return p_value, stats_permutation_test
