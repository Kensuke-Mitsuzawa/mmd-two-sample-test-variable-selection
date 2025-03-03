import typing as t
import copy
import numpy as np
import pandas
import torch

from scipy.signal import savgol_filter



DefaultRollingMeanParameters = {
    'window': 500
}

DefaultSavgolParameters = {
    'window_length': 501,
    'polyorder': 1
}


class MinMaxScalerVectorized(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor):
        """Fit features

        Parameters
        ----------
        stacked_features : tuple, list
            List of stacked features.

        Returns
        -------
        tensor
            A tensor with scaled features using requested preprocessor.
        """
        # Feature range
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor


class TrajectoryNoiseReduction(object):
    """A base class of auto-stopping-rules"""
    def __init__(self,
                 check_span: int = 100,
                 ignore_epochs: int = 500,
                 is_noise_reduction: bool = True,
                 algorithm_noise_reduction: str = 'savgol_filter',
                 args_noise_reduction: t.Dict = copy.deepcopy(DefaultSavgolParameters)):

        self.check_span = check_span
        self.ignore_epochs = ignore_epochs
        self.min_max_scaler = MinMaxScalerVectorized(feature_range=(0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.algorithm_noise_reduction = algorithm_noise_reduction
        self.args_noise_reduction = args_noise_reduction
        self.computed_ratio = -1.0

    def run_noise_reduction(self, input_sequence: np.ndarray, window_length: t.Optional[int] = None) -> np.ndarray:
        args_noise_reduction = self.args_noise_reduction
        if self.algorithm_noise_reduction == 'rolling_mean':
            output_sequence = pandas.Series(input_sequence).rolling(**args_noise_reduction).mean().to_numpy()
            output_sequence = np.nan_to_num(output_sequence)
        elif self.algorithm_noise_reduction == 'savgol_filter':
            # auto-adjustment of windwos-length
            if window_length is not None:
                window_length_ = window_length + 1 if window_length % 2 == 0 else window_length
                args_noise_reduction['window_length'] = window_length_
            # end if

            if len(input_sequence) < args_noise_reduction['window_length'] or args_noise_reduction['window_length'] < 0:
                # auto-adjustment of windwos-length
                args_noise_reduction['window_length'] = int(len(input_sequence) / 2)
                args_noise_reduction['window_length'] = (args_noise_reduction['window_length'] + 1) \
                    if args_noise_reduction['window_length'] % 2 == 0 else args_noise_reduction['window_length']
            # end if
            
            assert args_noise_reduction['window_length'] > 1.0, \
                f"window_length must be greater than 1.0, but {args_noise_reduction['window_length']}. \
                Debug info: len(input_sequence) = {len(input_sequence)}, window_length = {args_noise_reduction['window_length']}, polyorder = {args_noise_reduction['polyorder']}"

            output_sequence = savgol_filter(input_sequence, **args_noise_reduction)
            output_sequence = np.nan_to_num(output_sequence)
        else:
            raise Exception()
        return output_sequence

    def is_oscillation_span(
        self,
        scaled_indicator: torch.Tensor,
        acceptance_range: float = 0.01) -> bool:
        """True if the trajectory has the big oscillation, False not.
        Args:
            acceptance_range: If (max_span - min_span) < acceptance_range, the function does not recognize the big oscilation.
        """
        if (torch.max(scaled_indicator[1:-1]) - torch.min(scaled_indicator[1:-1])) < acceptance_range:
            return False
        # end if

        # scaled_indicator = minmax_scale(seq_indicator) + 1  # plus one to avoid a 0 value.
        span_start = len(scaled_indicator) - self.check_span
        log_first = scaled_indicator[span_start]
        log_last = scaled_indicator[-1]

        min_span = torch.min(scaled_indicator[1:-1])
        max_span = torch.max(scaled_indicator[1:-1])

        first_last_ratio = log_first / log_last
        min_max_ratio = max_span / min_span

        __is_big_oscillation = abs(1 - first_last_ratio) < abs(1 - min_max_ratio)
        is_big_oscillation = bool(__is_big_oscillation)
        return is_big_oscillation

    def is_mal_optimization(
            self,
            seq_indicator: torch.Tensor,
            threshold_mal_opt_ratio: float = 0.02,
            is_up_expected: bool = False) -> t.Tuple[bool, float]:
        """True if the training trajectory is malformed else False.
        If MMD^2 decreases or Ratio decreases, the detection is MAL.
        Args:
            threshold_mal_opt_ratio: Mal-trajectory if r < (1 - threshold_mal_opt_ratio) and (1 + threshold_mal_opt_ratio) < r, where r = (min / max).
              0.02 is recommended for the 100 epochs.
        """
        is_oscilation = self.is_oscillation_span(seq_indicator)
        if is_oscilation:
            return False, -1.0
        # end if
        span_start = len(seq_indicator) - self.check_span
        __diff_ratio = seq_indicator[-1] / seq_indicator[span_start]
        if is_up_expected:
            if seq_indicator[0] < seq_indicator[-1]:
                return False, -1.0
            else:
                if __diff_ratio < (1 - threshold_mal_opt_ratio):
                    return True, __diff_ratio.item()
                else:
                    return False, __diff_ratio.item()
                # end if
            # end if
        else:
            if seq_indicator[0] > seq_indicator[-1]:
                return False, -1.0
            else:
                if __diff_ratio > (1 + threshold_mal_opt_ratio):
                    return True, __diff_ratio.item()
                else:
                    return False, __diff_ratio.item()
                # end if
            # end if

        # end if

    def is_convergence_optimization(
            self,
            scaled_seq_indicator_in_range: torch.Tensor,
            threshold_convergence_ratio: float = 0.001,
            threshold_oscilation_abs_range: float = 0.005) -> t.Tuple[bool, float]:
        """True if the trajectory is in converging, False otherwise.
        Args:
            threshold_convergence_ratio
        Return:
            is_stop, computed_ratio
        """
        if threshold_oscilation_abs_range != -1:
            is_oscillation = self.is_oscillation_span(scaled_seq_indicator_in_range, acceptance_range=threshold_oscilation_abs_range)
            if is_oscillation:
                return False, -1.0
        # end if

        # span_start = len(scaled_seq_indicator) - self.check_span
        # __seq_indicator = scaled_seq_indicator[span_start:]

        ratio_first_end = scaled_seq_indicator_in_range[-1] / scaled_seq_indicator_in_range[0]
        __is_convergence = (1 - threshold_convergence_ratio) < ratio_first_end < (1 + threshold_convergence_ratio)
        is_convergence = bool(__is_convergence)
        # if is_convergence:
        #     print()
        return is_convergence, ratio_first_end.item()