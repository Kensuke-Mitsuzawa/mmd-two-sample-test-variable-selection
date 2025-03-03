# import typing as ty
# from pathlib import Path
# from tempfile import mkdtemp
# import shutil

# import numpy as np
# import torch
# import torch.utils.data.dataloader

# from mmd_tst_variable_detector.datasets import FileBackendSensorTimeSlicingDataset


# def generate_sensor_array(path_file_root: Path) -> ty.List[Path]:
#     """Generating 2-D array of (|S|, |T|). I split all T elements into separated files. A separated file has 1D array of (|S|). 
#     """
#     seq_file_parent_dir = []
#     path_file_root.mkdir(parents=True, exist_ok=True)
    
#     size_s = 10
#     size_t = 100
    
#     __x = np.random.normal(size=(size_s, size_t))
#     __y = np.random.normal(size=(size_s, size_t))    

#     for i in range(size_t):
#         path_dir_timestamp = path_file_root / f"{i}"
#         path_dir_timestamp.mkdir(parents=True, exist_ok=True)
        
#         torch.save({'array': torch.from_numpy(__x[:, i])}, path_dir_timestamp / "x.pt")
#         torch.save({'array': torch.from_numpy(__y[:, i])}, path_dir_timestamp / "y.pt")        
#         seq_file_parent_dir.append(path_dir_timestamp)
#     # end for

#     return seq_file_parent_dir
    


# def test_FileBackendTrajectoryTimeSlicingDataset():
#     path_root_data_dir_first = Path(mkdtemp())
#     path_root_data_dir_first.mkdir(parents=True, exist_ok=True)
    
#     seq_timestlice_files_first = generate_sensor_array(path_root_data_dir_first)
    
#     time_slicing_size = 10
#     time_start_current = 0
    
#     while (time_start_current + time_slicing_size) < 100:
#         __dataset_part = FileBackendSensorTimeSlicingDataset(
#             path_pair_parent_dir=seq_timestlice_files_first,
#             time_slice_from=time_start_current,
#             time_slice_to=(time_start_current + time_slicing_size))
#         assert len(__dataset_part) == time_slicing_size + 1, "The length of dataset is not correct."
#         __dataset_part.run_files_validation()
        
#         time_start_current = time_start_current + time_slicing_size
#     # end while
    
#     # merging test
#     dataset_first = FileBackendSensorTimeSlicingDataset(
#             path_pair_parent_dir=seq_timestlice_files_first,
#             time_slice_from=0,
#             time_slice_to=99)
#     dataset_first.run_files_validation()
       
#     path_root_data_dir_second = Path(mkdtemp())
#     path_root_data_dir_second.mkdir(parents=True, exist_ok=True)
#     seq_timestlice_files_second = generate_sensor_array(path_root_data_dir_second)

#     dataset_second = FileBackendSensorTimeSlicingDataset(
#             path_pair_parent_dir=seq_timestlice_files_second,
#             time_slice_from=0,
#             time_slice_to=99)
#     dataset_second.run_files_validation()
    
#     dataset_merged = dataset_first.merge_new_dataset(dataset_second)
#     assert len(dataset_merged) == 200
    
#     # test selected variables
#     dataset_selected = dataset_first.get_selected_variables_dataset(tuple([1, 2, 3]))
#     dim_x = dataset_selected.get_dimension_flattened()
#     assert  dim_x == 3
    
#     # test with DataLoader
#     loader = torch.utils.data.dataloader.DataLoader(dataset_first, num_workers=4, pin_memory=True)
#     for pair in loader:
#         pass
    
#     # end. deleting files.
#     shutil.rmtree(path_root_data_dir_first)
#     shutil.rmtree(path_root_data_dir_second)
