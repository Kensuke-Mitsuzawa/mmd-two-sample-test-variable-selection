from ..data_generator import test_data_xy_linear

from mmd_tst_variable_detector.baselines import l2_distance_detector


def test_l2_distance_detector():
    t_data_xy, ground_truth = test_data_xy_linear()
    detected_indices, __ = l2_distance_detector.get_l2_distance_based_variables(t_data_xy[0], t_data_xy[1], threshold=0.01)
    assert len(detected_indices) > 0
