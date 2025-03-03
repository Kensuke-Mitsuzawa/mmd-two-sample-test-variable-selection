import typing as ty
import logging
from functools import partial

from scipy.stats import wasserstein_distance
from ot import sliced_wasserstein_distance

import numpy as np
import torch

from distributed import Client
from joblib import Parallel, delayed

from ..datasets.base import BaseDataset
from ..datasets.file_onetime_load_backend_static_dataset import FileBackendOneTimeLoadStaticDataset
from ..logger_unit import handler

logger = logging.getLogger(f'{__package__}.{__name__}')
logger.addHandler(handler)



def __get_wasserstein_distance(input_dataset: BaseDataset, index_dimension: int) -> ty.Tuple[int, float]:
    if isinstance(input_dataset, FileBackendOneTimeLoadStaticDataset):
        input_dataset = input_dataset.generate_dataset_on_ram()
    # end if
    
    vector_x, vector_y = input_dataset.get_samples_at_dimension_d(index_dimension)
    return index_dimension, wasserstein_distance(vector_x, vector_y)


def __get_sliced_wasserstein_distance(input_dataset: BaseDataset, index_dimension: int) -> ty.Tuple[int, float]:
    # if isinstance(input_dataset, FileBackendOneTimeLoadStaticDataset):
    if input_dataset.is_dataset_on_ram():
        input_dataset = input_dataset.generate_dataset_on_ram()
    # end if
    
    tensor_x, tensor_y = input_dataset.get_samples_at_dimension_d(index_dimension)
    
    if isinstance(tensor_x, torch.Tensor):
        array_x = tensor_x.numpy()
    else:
        array_x = tensor_x
    # end if
    
    if isinstance(tensor_y, torch.Tensor):
        array_y = tensor_y.numpy()
    else:
        array_y = tensor_y
    # end if
    
    n_sample_x = len(array_x)
    n_sample_y = len(array_y)
    
    c_size = array_x.shape[-1]
    assert c_size == array_y.shape[-1], f'Not equal dimension. {c_size} != {array_y.shape[-1]}'
    
    input_x = np.reshape(array_x, (n_sample_x, c_size))
    input_y = np.reshape(array_y, (n_sample_y, c_size))
    
    d_value = sliced_wasserstein_distance(input_x, input_y, n_projections=100)
    return index_dimension, d_value


def main(input_dataset: BaseDataset, distributed_backend: str, dask_client: ty.Optional[Client]) -> np.ndarray:
    """Public function to compute the Wasserstein weights initialization.
    
    Parameters
    -----------
    input_dataset: BaseDataset
        The input dataset.
    distributed_backend: str
    dask_client: Client
        
    Returns
    --------
    np.ndarray
        The Wasserstein weights initialization.
        The array shape is (|D|, ) when data characteristic is static,
        (|S|, |T|) for the sensor-st,
        (|A|,) for the trajectory-st.
    """
    assert distributed_backend in ('dask', 'single', 'joblib'), f'Not supported backend. {distributed_backend}'
    
    target_function = __get_wasserstein_distance
    x_dimension_x = input_dataset.get_dimension_flattened()
    # end if
    
    # Compute the Wasserstein weights initialization.
    results: list[tuple[int, float]]
    if distributed_backend == 'single':
        if dask_client is not None:
            logger.debug('Dask client object is given. However, I ignore it since `distributed_backend` is single.')
        # end if
        # Single mode
        logger.debug('Single mode')
        results = [target_function(input_dataset, __index_d) for __index_d in range(x_dimension_x)]
    elif distributed_backend == 'joblib':
        # Joblib mode
        logger.debug('Joblib mode')
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(target_function)(
                input_dataset, __index_d) for __index_d in range(x_dimension_x)
            )  # type: ignore
    else:
        # Dask mode
        logger.debug('Dask mode')
        __func_target = partial(target_function, input_dataset)
        
        futures = dask_client.map(__func_target, range(x_dimension_x))
        results = dask_client.gather(futures)  # type: ignore
    # end if
    
    __sorted_list = sorted(results, key=lambda x: x[0])
    array_weights = np.array([__t[1] for __t in __sorted_list])
    
    if np.count_nonzero(array_weights) == 0:
        logger.info(f'All weights are zero. {array_weights}')
        return array_weights
    else:    
        array_weights_normal = array_weights / array_weights.max()
        assert len(array_weights_normal) == x_dimension_x, f'Not equal dimension. {len(array_weights_normal)} != {x_dimension_x}'
        
        return array_weights_normal
