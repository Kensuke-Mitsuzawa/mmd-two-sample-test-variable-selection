import numpy as np

from ..datasets.base import BaseDataset

from .wasserstein_weights_initialization import main


"""An interface module of weights initialization.
"""



def weights_initialization(dataset_input: BaseDataset, 
                           approach_name: str = 'wasserstein', 
                           distributed_backend: str = 'dask',
                           dask_client=None) -> np.ndarray:
    """Public function.
    
    Parameters
    -----------
    dataset_input: BaseDataset
        The input dataset.
    approach_name: str
    dask_client: Client
    
    Returns
    --------
    np.ndarray
        The weights initialization. The value range is [0, 1.0]
    """
    if approach_name == 'wasserstein':
        return main(dataset_input, distributed_backend, dask_client)
    else:
        raise NotImplementedError(f'No approach named {approach_name}')
    # end if
    