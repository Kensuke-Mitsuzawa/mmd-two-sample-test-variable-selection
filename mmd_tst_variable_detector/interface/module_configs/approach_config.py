from dataclasses import dataclass
from typing import List

from .static_module import PossibleDataRepresentation, PossibleVariableDetector, PossibleInterpretableMMD


@dataclass
class ApproachConfigArgs:
    """Configuration class for approach
    
    Parameters
    ----------
    approach_data_representation: str
        Data representation.
        Either of following choices,
        1. 'sample_based': data is sample based.
    approach_variable_detector: str
        Variable detector.
        Either of following choices,
        1. 'wasserstein_independence': Independent variable detection by Wasserstein, and permutation test by sliced-wasserstein.
        2. 'linear_variable_selection': Linear Variable Selection.
        3. 'interpretable_mmd': Interpretable MMD.
    approach_interpretable_mmd: str
        Interpretable MMD.
        Either of following choices,
        1. 'cv_selection': Cross Validation Selection.
        2. 'algorithm_one': Algorithm One in the paper.
        3. 'baseline_mmd': Baseline MMD.
    """
    approach_variable_detector: str
    approach_interpretable_mmd: str
    approach_data_representation: str = 'sample_based'    
    
    def __post_init__(self):
        self.approach_data_representation = self.approach_data_representation.lower()
        self.approach_variable_detector = self.approach_variable_detector.lower()
        self.approach_interpretable_mmd = self.approach_interpretable_mmd.lower()
        assert self.approach_data_representation in PossibleDataRepresentation, f'{self.approach_data_representation} is not supported. Possible choise -> {PossibleDataRepresentation}'
        assert self.approach_variable_detector in PossibleVariableDetector, f'{self.approach_variable_detector} is not supported. Possible choise -> {PossibleVariableDetector}'
        assert self.approach_interpretable_mmd in PossibleInterpretableMMD, f'{self.approach_interpretable_mmd} is not supported. Possible choise -> {PossibleInterpretableMMD}'
