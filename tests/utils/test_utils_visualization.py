# import logging

# import torch
# import pytorch_lightning as pl
# import shutil

# import tempfile
# import toml
# from pathlib import Path

# from mmd_tst_variable_detector.datasets import SimpleDataset
# from mmd_tst_variable_detector.kernels.gaussian_kernel import QuadraticKernelGaussianKernel
# from mmd_tst_variable_detector.mmd_estimator.mmd_estimator import QuadraticMmdEstimator
# from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector import (
#     # StabilitySelectionTrainedParameter,
#     StabilitySelectionAlgorithmParameter,
#     # StabilitySelectionParameters,
#     CrossValidationTrainParameters,
#     CrossValidationAlgorithmParameter,
#     DistributedComputingParameter,
#     CrossValidationInterpretableVariableDetector
# )
# from mmd_tst_variable_detector.detection_algorithm.interpretable_mmd_detector import (
#     InterpretableMmdTrainParameters
# )
# from mmd_tst_variable_detector.detection_algorithm.cross_validation_detector.checkpoint_saver import CheckPointSaverStabilitySelection
# from mmd_tst_variable_detector.utils import utils_visualization

# from .. import data_generator

# logger = logging.getLogger(__name__)


# torch.cuda.is_available = lambda : False


# def test_visualize_stability_score_modifications(resource_path_root: Path):
#     config_obj = toml.loads((resource_path_root / "test_settings.toml").open().read())[test_visualize_stability_score_modifications.__name__]

#     logger.warning("I skip this test becasue this function has the dependency to WanDB. Waiting for the modification.")
#     return True

#     algorithm_param = StabilitySelectionAlgorithmParameter(
#         candidate_regularization_parameter=config_obj['candidate_regularization_parameter'],
#         n_subsampling=config_obj['n_subsampling'])
#     dist_param = DistributedComputingParameter(
#         dask_scheduler_address=None,
#         n_joblib=config_obj['n_joblib'])
#     base_train_param = InterpretableMmdTrainParameters(
#         batch_size=config_obj['batch_size'],
#     )


#     ss_param = CrossValidationTrainParameters(
#         algorithm_parameter=algorithm_param,
#         base_training_parameter=base_train_param,
#         distributed_parameter=dist_param,
#         computation_backend='joblib'
#     )

#     t_xy, __ = data_generator.test_data_xy_linear(sample_size=500)
#     my_dataset = SimpleDataset(t_xy[0], t_xy[1])

#     initial_ard = torch.ones(my_dataset.get_dimension_flattened()[0])

#     kernel = QuadraticKernelGaussianKernel(ard_weights=initial_ard)
#     kernel.compute_length_scale_dataset(my_dataset)
#     kernel.set_length_scale()

#     mmd_estimator = QuadraticMmdEstimator(kernel)

#     trainer_lightning = pl.Trainer(max_epochs=config_obj['max_epochs'], accelerator='cpu')

#     resume_checkpoint_saver = CheckPointSaverStabilitySelection(output_dir=config_obj["working_dir_checkpoint_network"])

#     ss_trainer = CrossValidationInterpretableVariableDetector(
#         trainer_lightning=trainer_lightning,
#         training_parameter=ss_param,
#         estimator=mmd_estimator,
#         resume_checkpoint_saver=resume_checkpoint_saver
#     )
#     ss_result = ss_trainer.run_cv_detection(training_dataset=my_dataset, validation_dataset=my_dataset)

#     fig = utils_visualization.visualize_stability_score_modifications(
#         ss_result,
#         resume_checkpoint_saver
#     )
#     __path = Path(tempfile.mktemp())
#     __path.mkdir()
#     path_png_out = __path / f'{test_visualize_stability_score_modifications.__name__}.png'
#     fig.savefig(path_png_out)

#     shutil.rmtree(__path)