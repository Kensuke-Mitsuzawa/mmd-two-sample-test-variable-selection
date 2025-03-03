from dataclasses import dataclass
import typing as ty
import itertools
import logging
import collections

import numpy as np

logger = logging.getLogger(f'{__package__}.{__name__}')


@dataclass
class SensorObject:
    sensor_id: int
    x_range_from: np.ndarray
    y_range_from: np.ndarray

    x_range_to: np.ndarray
    y_range_to: np.ndarray
    
    bin_id_one: int
    bin_id_two: int
    

@dataclass
class DiscretizeObject:
    sensor_objects: ty.List[SensorObject]
    array_sensor_count: np.ndarray
    array_agent_position: np.ndarray



def define_sensor_spaces(simulation_x: np.ndarray, 
                         simulation_y: np.ndarray, 
                         n_division: int) -> ty.List[SensorObject]:
    """Defining sensors and its coverage spaces.
    
    Args
    -----
    simulation_x: (|A|, |T|, 2)
    simulation_y: (|A|, |T|, 2)
    n_division: int
    
    Returns
    -------
    list_sensor_objects: list of SensorObject
    """
    # getting max and min of C=2 space.
    space_coordinate_one_x = simulation_x[:, :, 0].flatten()
    space_coordinate_two_x = simulation_x[:, :, 1].flatten()

    space_coordinate_one_y = simulation_y[:, :, 0].flatten()
    space_coordinate_two_y = simulation_y[:, :, 1].flatten()    
    
    max_position_one = np.max([space_coordinate_one_x, space_coordinate_one_y])
    min_position_one = np.min([space_coordinate_one_x, space_coordinate_one_y])
    
    max_position_two = np.max([space_coordinate_two_x, space_coordinate_two_y])
    min_position_two = np.min([space_coordinate_two_x, space_coordinate_two_y])
    
    space_segment_one = (max_position_one - min_position_one) / n_division
    space_segment_two = (max_position_two - min_position_two) / n_division

    seq_sensor_objects = []
    sensor_i = 0
    
    for __i_one in range(0, n_division):
        for __i_two in range(0, n_division):
            _x_range_from = min_position_one + __i_one * space_segment_one
            _y_range_from = min_position_two + __i_two * space_segment_two

            _x_range_to = min_position_one + (__i_one + 1) * space_segment_one
            _y_range_to = min_position_two + (__i_two + 1) * space_segment_two

            _sensor_obj = SensorObject(
                sensor_id=sensor_i,
                x_range_from=_x_range_from,
                y_range_from=_y_range_from,
                x_range_to=_x_range_to,
                y_range_to=_y_range_to,
                bin_id_one=__i_one,
                bin_id_two=__i_two)
            seq_sensor_objects.append(_sensor_obj)
            sensor_i += 1
            
            assert _x_range_from < max_position_one and _x_range_to <= (max_position_one + space_segment_one)
            assert _y_range_from < max_position_two and _y_range_to <= (max_position_two + space_segment_two)
        # end for
    # end for
    return seq_sensor_objects

    
def extract_bins(sensor_spaces: ty.List[SensorObject]) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Extracting bins from sensor spaces.
    
    Args
    -----
    sensor_spaces: list of SensorObject
    
    Returns
    -------
    bins_x: np.ndarray
    bins_y: np.ndarray
    """
    min_x = np.min([sensor_space.x_range_from for sensor_space in sensor_spaces])
    min_y = np.min([sensor_space.y_range_from for sensor_space in sensor_spaces])
    
    bins_x = np.array(list(set([min_x] + [sensor_space.x_range_to for sensor_space in sensor_spaces])))
    bins_y = np.array(list(set([min_y] + [sensor_space.y_range_to for sensor_space in sensor_spaces])))
    
    return np.sort(bins_x), np.sort(bins_y)

    
def count_agents(simulation_output: np.ndarray, 
                 sensor_spaces: ty.List[SensorObject]
                 ) -> np.ndarray:
    """Counting agents in each sensor space.
    
    Args
    -----
    simulation_output: (|A|, |T|, 2)
    sensor_spaces: list of SensorObject
    
    Parameters
    ----------
    array_sensor_count: (|S|, |T|)
    """
    bins_x, bins_y = extract_bins(sensor_spaces)
    
    n_sensors: int = len(sensor_spaces)
    n_timesteps: int = simulation_output.shape[1]
    n_agent = simulation_output.shape[0]
    array_sensor_count = np.zeros([n_sensors, n_timesteps])

    for _time_step in range(n_timesteps):
        _status_at_t = simulation_output[:, _time_step, :]
        _position_x = _status_at_t[:, 0]  # (|A|,)
        _position_y = _status_at_t[:, 1]  # (|A|,)

        _histogram, _xedges, _yedges = np.histogram2d(_position_x, _position_y, bins=(bins_x, bins_y))
        _vector_count_sensor = _histogram.flatten()  # (|S|,)
        
        array_sensor_count[:, _time_step] = _vector_count_sensor
        if np.sum(_vector_count_sensor) != n_agent:
            logger.warning(f"At time-step={_time_step}. The sum of agents in each sensor space must be equal to the total number of agents.")
    # end for

    return array_sensor_count


def find_agent_bins(simulation_output: np.ndarray, 
                    sensor_spaces: ty.List[SensorObject]) -> np.ndarray:
    """Find the bins of agents in each sensor space.
    
    Args
    -----
    simulation_output: (|A|, |T|, 2)
    sensor_spaces: list of SensorObject
    
    """	
    bins_x, bins_y = extract_bins(sensor_spaces)
    
    n_agents: int = simulation_output.shape[0]
    n_timestep: int = simulation_output.shape[1]

    array_agent_positions = np.zeros((n_agents, n_timestep, 2))

    for _n_time_step in range(n_timestep):
        _array_at_time = simulation_output[:, _n_time_step, :]

        _hitx = np.digitize(_array_at_time[:, 0], bins_x, right=True)
        _hity = np.digitize(_array_at_time[:, 1], bins_y, right=True)

        _hitbins = np.array(list(zip(_hitx, _hity)))
        array_agent_positions[:, _n_time_step, :] = _hitbins
    # end for

    return array_agent_positions


def get_sensor_id_position(array_position_sensor: np.ndarray,
                           sensor_spaces: ty.List[SensorObject]) -> np.ndarray:
    """
    Return
    ------
    array_sensor_id: (|A|, |T|). An element value is a sensor id.
    """
    bins_x, bins_y = extract_bins(sensor_spaces)
    
    bin_id2sensor_id = {
        (sensor_obj.bin_id_one, sensor_obj.bin_id_two): sensor_obj.sensor_id 
        for sensor_obj in sensor_spaces}
    
    def __get_sensor_id(array_bin_id: np.ndarray) -> int:
        """Get sensor id from bin id."""
        # index-id must be -1.
        # if index-id is 0, then I do +1
        # if index-id is N(bins), then I do -1.
        __bin_id_one = int(array_bin_id[0])
        __bin_id_two = int(array_bin_id[1])
        if __bin_id_one == 0:
            __bin_id_one += 1
        elif __bin_id_one == len(bins_x):
            __bin_id_one -= 1
        # en if
        
        if __bin_id_two == 0:
            __bin_id_two += 1
        elif __bin_id_two == len(bins_y):
            __bin_id_two -= 1
        # end if
        
        __bin_id_one -= 1        
        __bin_id_two -= 1
        sensor_id = bin_id2sensor_id[(__bin_id_one, __bin_id_two)]
        return sensor_id
    
    array_sensor_ids = np.apply_along_axis(__get_sensor_id, 2, array_position_sensor)
    return array_sensor_ids


def count_sensor_frequency(aray_position_sensor_id: np.ndarray, n_sensors: int) -> np.ndarray:
    """Counting the frequency of sensor id in each time-step.
    
    Args
    -----
    aray_position_sensor_id_x: (|A|, |T|)
    
    Returns
    -------
    array_sensor_count: (|S|, |T|)
    """
    n_timesteps = aray_position_sensor_id.shape[1]
    
    array_sensor_count = np.zeros((n_sensors, n_timesteps))
    
    for _time_step in range(n_timesteps):
        _sensor_id_at_t = aray_position_sensor_id[:, _time_step]
        _counter_sensor_id = collections.Counter(_sensor_id_at_t.tolist())
        for __sensor_id, __count_freq in _counter_sensor_id.items():
            array_sensor_count[__sensor_id, _time_step] = __count_freq
        # end for
    # end for
    
    return array_sensor_count


def validate_sensor_id_position(array_sensor_id: np.ndarray,
                                array_sensor_count: np.ndarray):
    for __index_time_t in range(0, len(array_sensor_id)):
        _counter_sensor_id = collections.Counter(array_sensor_id[:, __index_time_t].tolist())
        for __sensor_id, __count_freq in _counter_sensor_id.items():
            __count_hist_value = array_sensor_count[__sensor_id, __index_time_t]
            if __count_freq != __count_hist_value:
                raise Exception(f"sensor_id={__sensor_id}, time={__index_time_t}. The true-count is {__count_hist_value}, count-by-position={__count_freq}")
        # end for
    # end for


def main(
    simulation_x: np.ndarray,
    simulation_y: np.ndarray,
    n_division: int,
    seq_sensors: ty.Optional[ty.List[SensorObject]] = None,
    is_validation_double_check: bool = False) -> ty.Tuple[DiscretizeObject, DiscretizeObject]:
    """
    Args
    -----
    simulation_x: (|A|, |T|, 2)
    simulation_y: (|A|, |T|, 2)
    
    
    Returns
    -------
    seq_simulation_objects, SensorObject : the length of sensors.
    array_sensor_x: (|S|, |T|)
    array_sensor_y: (|S|, |T|)
    array_agent_position_x: (|A|, |T|, 2)
    array_agent_position_y: (|A|, |T|, 2)
    """
    if seq_sensors is None:
        seq_sensors = define_sensor_spaces(
            simulation_x=simulation_x,
            simulation_y=simulation_y,
            n_division=n_division)
    
    # counting agents in each sensor space.
    # the return array is (|S|, |T|)
    # sensor_count_x = count_agents(
    #     simulation_output=simulation_x,
    #     sensor_spaces=seq_sensors)
    # sensor_count_y = count_agents(
    #     simulation_output=simulation_y,
    #     sensor_spaces=seq_sensors)
    
    # I get bins-index of agents in each sensor space.
    # So, the array space is (|A|, |T|, 2)
    array_position_sensor_x = find_agent_bins(simulation_x, seq_sensors)
    array_position_sensor_y = find_agent_bins(simulation_y, seq_sensors)
    
    # I convert bins-index to sensor-id.
    # So, the array space is (|A|, |T|)
    aray_position_sensor_id_x = get_sensor_id_position(array_position_sensor_x, seq_sensors)
    aray_position_sensor_id_y = get_sensor_id_position(array_position_sensor_y, seq_sensors)    
    
    sensor_count_x = count_sensor_frequency(aray_position_sensor_id_x, n_sensors=len(seq_sensors))
    sensor_count_y = count_sensor_frequency(aray_position_sensor_id_y, n_sensors=len(seq_sensors))
    
    if is_validation_double_check:
        logger.debug(f'Running validation on array x.')
        validate_sensor_id_position(aray_position_sensor_id_x, sensor_count_x)
        logger.debug(f'Done validation on x.')
        
        logger.debug(f'Running validation on array y.')
        validate_sensor_id_position(aray_position_sensor_id_y, sensor_count_y)
        logger.debug(f'Done validation on y.')
    # end if
        
    container_x = DiscretizeObject(
        sensor_objects=seq_sensors,
        array_sensor_count=sensor_count_x,
        array_agent_position=array_position_sensor_x)
    containr_y = DiscretizeObject(
        sensor_objects=seq_sensors,
        array_sensor_count=sensor_count_y,
        array_agent_position=array_position_sensor_y)
    
    return container_x, containr_y
