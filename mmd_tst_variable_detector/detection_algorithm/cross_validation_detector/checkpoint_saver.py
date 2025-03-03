import typing
import pickle
import torch
import json

from pathlib import Path

from dataclasses import asdict

import pytorch_lightning.callbacks

from .commons import SubLearnerTrainingResult

BASENAME_CHECKPOINT = 'checkpoint'


class CheckPointSaverStabilitySelection(pytorch_lightning.callbacks.Checkpoint):
    def __init__(
            self,
            output_dir: Path,
            key_file_name: str = 'task_id.json',
            training_parameter_file_name: str = 'training_parameters.pickle',
            trained_result_obj_name: str = 'trained_detection_result.pt'):
        """

        :param output_dir: Network-Disk with dask backend, Local storage with Joblib backend.
        :param trained_parameter_file_name:
        :param trained_result_obj_name:
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        # end if
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        # end if

        self.output_dir = output_dir

        self.key_file_name = key_file_name
        self.training_parameter_file_name = training_parameter_file_name
        self.trained_result_obj_name = trained_result_obj_name

    def save_checkpoint(
            self,
            sub_leaner_training_result: SubLearnerTrainingResult):
        """Saving objects at the specified point.
        """
        # mkdir reg_param / sub_learner_id
        path_sub_dir = (self.output_dir / str(sub_leaner_training_result.job_id))
        path_sub_dir.mkdir(exist_ok=True, parents=True)

        with (path_sub_dir / self.key_file_name).open('w') as f:
            f.write(json.dumps({"task_id": str(sub_leaner_training_result.job_id)}))
        # end if

        # saving arguments object
        with (path_sub_dir / self.training_parameter_file_name).open('wb') as f:
            f.write(pickle.dumps(sub_leaner_training_result.training_parameter))
        # end with

        # saving sub-learner-object
        path_result_ = (path_sub_dir / self.trained_result_obj_name)
        torch.save(sub_leaner_training_result.__dict__, path_result_)
        # end with

    def load_checkpoint(self) -> typing.List[SubLearnerTrainingResult]:
        """

        :return: [(job-id-tuple, TrainingParameters, TrainingResult)]
        """
        seq_files = self.output_dir.rglob(f'*{self.trained_result_obj_name}')
        seq_variable_objects = []
        for path in seq_files:
            __result = torch.load(path.as_posix())
            seq_variable_objects.append(SubLearnerTrainingResult(**__result))
        # end for
        return seq_variable_objects
