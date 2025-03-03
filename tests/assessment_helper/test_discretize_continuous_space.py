import numpy as np

from mmd_tst_variable_detector.assessment_helper.discretize_continuous_space import main
from mmd_tst_variable_detector.assessment_helper import data_generator_brownian_motion


def test_discretize_continuous_space():
    n_timestamp = 100
    
    x0_x = np.random.normal(0, 1, size=(100, 2))
    x0_y = np.random.normal(0, 1, size=(100, 2))    
    
    sim_x = data_generator_brownian_motion.brownian_flexibile(
        x0=x0_x, 
        size_t=n_timestamp, 
        seed=42,
        func_transition_prob_distribution=data_generator_brownian_motion.default_func_transition_prob,
        func_transform_step=data_generator_brownian_motion.default_func_transform_step)
    sim_y = data_generator_brownian_motion.brownian_flexibile(
        x0=x0_y, 
        size_t=n_timestamp, 
        seed=42,
        func_transition_prob_distribution=data_generator_brownian_motion.default_func_transition_prob,
        func_transform_step=data_generator_brownian_motion.default_func_transform_step)

    n_division_per_space = 10
    n_sensors = n_division_per_space * n_division_per_space
    container_x, container_y = main(simulation_x=sim_x, simulation_y=sim_y, n_division=n_division_per_space)
    
    assert len(container_x.sensor_objects) == n_sensors
    assert container_x.array_agent_position.shape[0] == n_sensors
    assert container_x.array_agent_position.shape[1] == n_timestamp
    assert container_x.array_sensor_count.shape[0] == n_sensors
    assert container_x.array_sensor_count.shape[1] == n_timestamp
