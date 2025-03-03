# import datetime
# import typing as ty
# import copy

# import torch
# import numpy as np

# from pathlib import Path

# from .base import (
#     BaseDataset, 
#     BaseStaticDataset,
#     BaseRamBackendDataset,
#     BaseSampleBasedDataset,
#     Return_split_train_and_test 
# )


# class RamFlexibleBackendStaticDataset(BaseRamBackendDataset, BaseStaticDataset, BaseSampleBasedDataset):
#     def __init__(self,
#                  path_x: Path,
#                  path_y: Path,
#                  x: ty.Optional[torch.Tensor],
#                  y: ty.Optional[torch.Tensor],
#                  variable_label: ty.Optional[str] = None):
#         """A dataset container holding a pair of Z=(X, Y). This class corresponds to a variable Z_{XY}.
#         The two variables X, Y should be supposed to pairs. Call `shuffle_data` method to generate a variable Z_{X \times Y}.

#         :param x:
#         :param y:
#         """
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x)
#         if isinstance(y, np.ndarray):
#             y = torch.from_numpy(y)
#         # end if

#         self.data_format_x = self.__check_data_form(x)
#         self.data_format_y = self.__check_data_form(y)
#         self.variable_label = variable_label

#         assert len(x) == len(y), f'Not equal sample numbers. N(X)={len(x)} N(Y)={len(y)}'
#         self.n_sample = len(x)

#         self._dimension_x = self.__get_dimensions(x, self.data_format_x)
#         self._dimension_y = self.__get_dimensions(y, self.data_format_y)

#         self._x = x
#         self._y = y

#     def close(self):
#         pass

#     def __getitem__(self, idx: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
#         if self.data_format_x == 'vector':
#             __x = self._x[idx]
#         elif self.data_format_x == 'matrix':
#             __x = self.flatten_matrix_to_vector(self._x[idx])
#         else:
#             raise NotImplementedError()

#         if self.data_format_y == 'vector':
#             __y = self._y[idx]
#         elif self.data_format_y == 'matrix':
#             __y = self.flatten_matrix_to_vector(self._y[idx])
#         else:
#             raise NotImplementedError()

#         return __x, __y

#     def __len__(self) -> int:
#         return self.n_sample

#     def get_all_samples(self) -> ty.Tuple[torch.Tensor, torch.Tensor]:
#         __seq_t_xy = [self.__getitem__(__i) for __i in range(self.n_sample)]
#         __seq_x = torch.stack([t[0] for t in __seq_t_xy])
#         __seq_y = torch.stack([t[1] for t in __seq_t_xy])

#         return __seq_x, __seq_y

#     def get_samples_at_dimension_d(self, index_d: int) -> ty.Tuple[torch.Tensor, torch.Tensor]:
#         if self.data_format_x == 'vector':
#             __x = self._x[:, index_d]
#             __y = self._y[:, index_d]
#         elif self.data_format_x == 'matrix':
#             __x_orig = torch.flatten(self._x, start_dim=1)
#             __y_orig = torch.flatten(self._y, start_dim=1)
#             __x = __x_orig[:, index_d]
#             __y = __y_orig[:, index_d]
#         else:
#             raise NotImplementedError()
#         # end if
        
#         assert __x.shape[0] == __y.shape[0], f'Not equal sample numbers. N(X)={__x.shape[0]} N(Y)={__y.shape[0]}'
#         return __x, __y

#     @staticmethod
#     def __check_data_form(x: torch.Tensor) -> str:
#         """
#         Args:
#             x:
#             y:
#         Returns: either of {vector, matrix}
#         """
#         if len(x.shape) == 2:
#             return 'vector'
#         elif len(x.shape) == 3:
#             return 'matrix'
#         else:
#             raise Exception(f'Undefined case. X is {x.shape}')

#     @staticmethod
#     def __get_dimensions(data: torch.Tensor, data_format: str) -> ty.Tuple[int, ...]:
#         if data_format == 'vector':
#             return (data.shape[1],)
#         elif data_format == 'matrix':
#             return data.shape[1:]
#         else:
#             raise NotImplementedError()
        
#     def copy_dataset(self) -> "SimpleDataset":
#         return copy.deepcopy(self)

#     @staticmethod
#     def flatten_matrix_to_vector(data: torch.Tensor) -> torch.Tensor:
#         """Flattening a matrix into vector if the incoming data is a matrix form."""
#         return torch.flatten(data, start_dim=0)

#     def get_dimension_flattened(self) -> ty.Tuple[int, int]:
#         if self.data_format_x == 'vector':
#             __x = self._dimension_x[0]
#         elif self.data_format_x == 'matrix':
#             __x = np.prod(self._dimension_x).item()
#         else:
#             raise NotImplementedError()

#         if self.data_format_y == 'vector':
#             __y = self._dimension_y[0]
#         elif self.data_format_y == 'matrix':
#             __y = np.prod(self._dimension_y).item()
#         else:
#             raise NotImplementedError()

#         return __x, __y
    
#     def merge_new_dataset(self, dataset_obj: "BaseDataset") -> "BaseDataset":
#         assert type(self) == type(dataset_obj), f'Inconsistent dataset type. {type(self)} vs {type(dataset_obj)}'
        
#         sample_whole_x = torch.cat([self._x, dataset_obj._x], dim=0)
#         sample_whole_y = torch.cat([self._y, dataset_obj._y], dim=0)
        
#         new_sample_size = len(self) + len(dataset_obj)
        
#         return SimpleDataset(sample_whole_x, sample_whole_y)


#     def get_random_data_id(self, n_samples: int) -> ty.List[int]:
#         perm = torch.randperm(len(self._x))
#         idx = perm[:n_samples]
#         return idx.tolist()

#     def get_subsample_dataset(self, n_samples: int = None, sample_ids: ty.List[int] = None) -> ty.Tuple[ty.List[int], "SimpleDataset"]:
#         assert n_samples is not None or sample_ids is not None, 'Either of n_samples or sample_ids should be specified.'
#         if sample_ids is None:
#             sub_data_index = self.get_random_data_id(n_samples)
#         else:
#             sub_data_index = sample_ids
#         # end if
            
#         sub_datset = SimpleDataset(copy.deepcopy(self._x[sub_data_index, :]), copy.deepcopy(self._y[sub_data_index, :]))
#         return sub_data_index, sub_datset

#     def get_bootstrap_dataset(self, seed: int = None) -> ty.Tuple[ty.Tuple[ty.List[int], ty.List[int]], "SimpleDataset"]:
#         if seed is None:
#             seed = datetime.datetime.now().second
#         # end if
#         rng = np.random.RandomState(seed)
#         boot_sample_x_id = rng.choice(range(0, len(self._x)), size=len(self._x), replace=True)
#         boot_sample_y_id = rng.choice(range(0, len(self._y)), size=len(self._x), replace=True)
#         sub_data_index = (boot_sample_x_id, boot_sample_y_id)

#         sub_datset = SimpleDataset(copy.deepcopy(self._x[boot_sample_x_id, :]), copy.deepcopy(self._y[boot_sample_y_id, :]))
#         return sub_data_index, sub_datset

#     def get_selected_variables_dataset(self, selected_variables: ty.List[int]) -> "SimpleDataset":
#         seq_tuple_tensor_xy = [self.__getitem__(__i) for __i in range(self.__len__())]
#         dim_x, dim_y = self.get_dimension_flattened()

#         x = torch.zeros(len(seq_tuple_tensor_xy), dim_x)
#         y = torch.zeros(len(seq_tuple_tensor_xy), dim_y)

#         for __i_sample, __t_xy in enumerate(seq_tuple_tensor_xy):
#             __x_vector: torch.Tensor = __t_xy[0]
#             __y_vector: torch.Tensor = __t_xy[1]

#             x[__i_sample] = __x_vector
#             y[__i_sample] = __y_vector
#         # end for

#         return SimpleDataset(x[:, selected_variables], y[:, selected_variables])

#     def split_train_and_test(self, train_ratio: float = 0.8) -> Return_split_train_and_test:
#         n_sample_train = int(self.__len__() * train_ratio)
#         seq_sample_id = range(0, self.__len__())
        
#         seq_train_id, train_dataset = self.get_subsample_dataset(n_samples=n_sample_train)
        
#         set_test_sample_id = set(seq_sample_id) - set(seq_train_id)
#         seq_test_id, test_dataset = self.get_subsample_dataset(sample_ids=list(set_test_sample_id))        
        
#         return Return_split_train_and_test(
#             train_dataset=train_dataset,
#             test_dataset=test_dataset,
#             train_test_sample_ids=(seq_train_id, seq_test_id)
#         )


# # ======================================================

