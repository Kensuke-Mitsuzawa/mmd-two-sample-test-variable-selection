"""Legacy compatibility shim for detection_algorithm_one.

Redirects all imports to mmd_tst_variable_detector.detection_algorithm.reg_selection.
"""
from .reg_selection import (
    RegSelectionIndividualResult,
    RegSelectionResult,
    RegSelectionVariableDetector,
    detection_reg_selection,
    AlgorithmOneIndividualResult,
    AlgorithmOneResult,
    AlgorithmOneVariableDetector,
    detection_algorithm_one,
    PossibleTypeRegularizationParameter,
    _RegSelectionRangeFunctionRequestPayload,
    _RegSelectionRangeFunctionReturn,
    _AlgorithmOneRangeFunctionRequestPayload,
    _AlgorithmOneRangeFunctionReturn,
    _run_reg_selection_min_max_param_range,
    _run_reg_selection_search_objective_based,
    _run_algorithm_one_min_max_param_range,
    _run_algorithm_one_search_objective_based,
    _select_model,
    function_logging_manually,
)

__all__ = [
    # Legacy names
    "AlgorithmOneIndividualResult",
    "AlgorithmOneResult",
    "AlgorithmOneVariableDetector",
    "detection_algorithm_one",
    "PossibleTypeRegularizationParameter",
    "_AlgorithmOneRangeFunctionRequestPayload",
    "_AlgorithmOneRangeFunctionReturn",
    "_run_algorithm_one_min_max_param_range",
    "_run_algorithm_one_search_objective_based",
    # New names
    "RegSelectionIndividualResult",
    "RegSelectionResult",
    "RegSelectionVariableDetector",
    "detection_reg_selection",
    "_RegSelectionRangeFunctionRequestPayload",
    "_RegSelectionRangeFunctionReturn",
    "_run_reg_selection_min_max_param_range",
    "_run_reg_selection_search_objective_based",
    "_select_model",
    "function_logging_manually",
]
