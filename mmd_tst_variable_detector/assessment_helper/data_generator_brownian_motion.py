"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).

This script comes from https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html

This script can make Brownian motion up 2D case.
"""
import typing as ty

from math import sqrt
from scipy.stats import norm
import numpy as np


import logging

logger = logging.getLogger(__name__)


TypeFuncDistributionTransition = ty.Callable[[int, int, ty.Optional[int]], np.ndarray]
TypeFuncTransformation = ty.Callable[[np.ndarray], np.ndarray]


def default_func_transition_prob(n_agent: int, n_coordinate: int, seed: ty.Optional[int] = None) -> np.ndarray:
    """Returning (|A|, |C|) sampled from N(0, 5.0)"""
    gen = np.random.default_rng(seed)
    return gen.normal(loc=0.0, scale=5.0, size=(n_agent, n_coordinate))


def default_func_transform_step(array_t: np.ndarray) -> np.ndarray:
    return array_t


def brownian_flexibile(x0: np.ndarray, 
                       size_t: int,
                       snapshot_frequency: float = 1.0,
                       func_transition_prob_distribution: TypeFuncDistributionTransition = default_func_transition_prob,
                       func_transform_step: TypeFuncTransformation = default_func_transform_step,
                       seed: ty.Optional[int] = None) -> np.ndarray:
    """Generate an instance of Brownian motion (i.e. the Wiener process)
    
    Parameters
    ----------
    x0 : float or numpy array (or something that can be converted to a numpy array using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    size_t : int
        The number of steps to take.
    snapshot_frequency: float
        must be between [0.0, 1.0]. When 1.0, saving position at every step.
        When < 1.0, saving position at every step * snapshot_frequency.
    func_transition_prob_distribution : callable
        A probability function which add an term between t and t+1.
    func_transform_step : callable
        A function which transforms the value of t+1.
    seed : int or None
        Random seed.
        
    Returns
    -------
    A numpy array of floats with shape (|A|, |Snap|, |C|).
    |Snap| = |T| * (1 / snapshot_frequency)
    """
    assert 0.0 <= snapshot_frequency <= 1.0, f'snapshot_frequency must be between [0.0, 1.0]'

    if seed is not None:
        logger.debug(f'Fixing random seed to {seed}')
        np.random.seed(seed)
    # end if
    
    x0 = np.asarray(x0)
    

    # size of (n-agent, time-axis, n-coordinate)
    steps_per_time_step = int(1 / snapshot_frequency)
    n_time_saving = int(size_t * steps_per_time_step)
    n_agent = len(x0)
    n_coordinate = x0.shape[1]
    array_size = (len(x0), n_time_saving, x0.shape[1])
    trajectory_agent = np.zeros(array_size)
    trajectory_agent[:, 0, :] = x0
    
    __t_step = 1
    __i_time_snapshot = 1
    while __i_time_snapshot < n_time_saving:
        # loop per a time-step.
        __prob_term = func_transition_prob_distribution(n_agent, n_coordinate, seed)  # size of (n-agent, n-coordinate)
        # getting the probability term.
        __value_t = trajectory_agent[:, (__t_step - 1), :]  # size of (n-agent, n-coordinate)
        # positions at the t+1 step.
        __position_time_steo_plus_1 = func_transform_step(__value_t + __prob_term)
        
        __t_1_per_snapshot_step = __position_time_steo_plus_1 / steps_per_time_step
        # loop per a snapshot steps.
        for __i in range(1, steps_per_time_step + 1):
            # saving the position at every step * snapshot_frequency.
            trajectory_agent[:, __i_time_snapshot, :] = __t_1_per_snapshot_step * __i
            __i_time_snapshot += 1
            # end condition
            if __i_time_snapshot == n_time_saving:
                break
            # end if
        # end for
        # trajectory_agent[:, __t_step, :] = __t_1
        # +1 for the time-step.
        __t_step += 1
        
    # end while

    return trajectory_agent



def brownian(x0: np.ndarray, n: int, dt: float, delta: float, out=None, seed=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """
    if seed is not None:
        logger.debug(f'Fixing random seed to {seed}')
        np.random.seed(seed)
    # end if

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out