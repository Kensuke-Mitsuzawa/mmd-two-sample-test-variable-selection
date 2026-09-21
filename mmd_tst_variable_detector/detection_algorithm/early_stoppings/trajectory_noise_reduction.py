import typing as ty
import copy
import numpy as np
import pandas
import torch
import torch.nn.functional as F

from scipy.signal import savgol_filter, savgol_coeffs


DefaultRollingMeanParameters = {
    'window': 500
}

DefaultSavgolParameters = {
    'window_length': 501,
    'polyorder': 1
}


# ---------------------------------------------------------------------------
# Legacy Implementations (Preserved for Benchmark Comparison: regex ^Legacy)
# ---------------------------------------------------------------------------

class LegacyMinMaxScalerVectorized(object):
    """MinMax Scaler (Legacy implementation with in-place mutation)"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        a, b = self.feature_range

        dist = tensor.max(dim=0, keepdim=True)[0] - tensor.min(dim=0, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=0, keepdim=True)[0])
        tensor.mul_(b - a).add_(a)

        return tensor
    # end def
# end class


class LegacyTrajectoryNoiseReduction(object):
    """Legacy base class for trajectory noise reduction"""
    def __init__(
        self,
        check_span: int = 100,
        ignore_epochs: int = 500,
        is_noise_reduction: bool = True,
        algorithm_noise_reduction: str = 'savgol_filter',
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters)
    ):
        self.check_span = check_span
        self.ignore_epochs = ignore_epochs
        self.min_max_scaler = LegacyMinMaxScalerVectorized(feature_range=(0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.algorithm_noise_reduction = algorithm_noise_reduction
        self.args_noise_reduction = args_noise_reduction
        self.computed_ratio = -1.0
    # end def

    def run_noise_reduction(self, input_sequence: np.ndarray, window_length: ty.Optional[int] = None) -> np.ndarray:
        args_noise_reduction = self.args_noise_reduction
        if self.algorithm_noise_reduction == 'rolling_mean':
            output_sequence = pandas.Series(input_sequence).rolling(**args_noise_reduction).mean().to_numpy()
            output_sequence = np.nan_to_num(output_sequence)
        elif self.algorithm_noise_reduction == 'savgol_filter':
            if window_length is not None:
                window_length_ = window_length + 1 if window_length % 2 == 0 else window_length
                args_noise_reduction['window_length'] = window_length_
            # end if

            if len(input_sequence) < args_noise_reduction['window_length'] or args_noise_reduction['window_length'] < 0:
                args_noise_reduction['window_length'] = int(len(input_sequence) / 2)
                args_noise_reduction['window_length'] = (args_noise_reduction['window_length'] + 1) \
                    if args_noise_reduction['window_length'] % 2 == 0 else args_noise_reduction['window_length']
            # end if

            assert args_noise_reduction['window_length'] > 1.0, \
                f"window_length must be greater than 1.0, but {args_noise_reduction['window_length']}."

            output_sequence = savgol_filter(input_sequence, **args_noise_reduction)
            output_sequence = np.nan_to_num(output_sequence)
        else:
            raise Exception()
        # end if
        return output_sequence
    # end def

    def is_oscillation_span(
        self,
        scaled_indicator: torch.Tensor,
        acceptance_range: float = 0.01
    ) -> bool:
        if (torch.max(scaled_indicator[1:-1]) - torch.min(scaled_indicator[1:-1])) < acceptance_range:
            return False
        # end if

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
    # end def

    def is_mal_optimization(
        self,
        seq_indicator: torch.Tensor,
        threshold_mal_opt_ratio: float = 0.02,
        is_up_expected: bool = False
    ) -> ty.Tuple[bool, float]:
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
    # end def

    def is_convergence_optimization(
        self,
        scaled_seq_indicator_in_range: torch.Tensor,
        threshold_convergence_ratio: float = 0.001,
        threshold_oscilation_abs_range: float = 0.005
    ) -> ty.Tuple[bool, float]:
        if threshold_oscilation_abs_range != -1:
            is_oscillation = self.is_oscillation_span(scaled_seq_indicator_in_range, acceptance_range=threshold_oscilation_abs_range)
            if is_oscillation:
                return False, -1.0
            # end if
        # end if

        ratio_first_end = scaled_seq_indicator_in_range[-1] / scaled_seq_indicator_in_range[0]
        __is_convergence = (1 - threshold_convergence_ratio) < ratio_first_end < (1 + threshold_convergence_ratio)
        is_convergence = bool(__is_convergence)
        return is_convergence, ratio_first_end.item()
    # end def
# end class


# ---------------------------------------------------------------------------
# High-Performance Device-Resident (GPU & CPU) Implementations
# ---------------------------------------------------------------------------

class MinMaxScalerVectorized(object):
    """MinMax Scaler (Non-destructive, works on any torch device: CPU or GPU)

    Transforms each channel to the range [a, b].
    """

    def __init__(self, **kwargs):
        self.feature_range = (0.0, 1.0)
        self.__dict__.update(kwargs)
    # end def

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fit features non-destructively on device.

        Parameters
        ----------
        tensor : torch.Tensor
            A 1D or 2D tensor.

        Returns
        -------
        torch.Tensor
            A new scaled tensor.
        """
        a, b = self.feature_range
        tensor_min = tensor.min(dim=0, keepdim=True)[0]
        tensor_max = tensor.max(dim=0, keepdim=True)[0]
        dist = tensor_max - tensor_min
        dist = torch.where(dist == 0.0, torch.ones_like(dist), dist)
        scale = 1.0 / dist
        scaled = (tensor - tensor_min) * scale * (b - a) + a
        return scaled
    # end def
# end class


class TrajectoryNoiseReduction(object):
    """Base class of auto-stopping-rules with native GPU & CPU acceleration."""

    def __init__(
        self,
        check_span: int = 100,
        ignore_epochs: int = 500,
        is_noise_reduction: bool = True,
        algorithm_noise_reduction: str = 'savgol_filter',
        args_noise_reduction: ty.Dict = copy.deepcopy(DefaultSavgolParameters)
    ):
        self.check_span = check_span
        self.ignore_epochs = ignore_epochs
        self.min_max_scaler = MinMaxScalerVectorized(feature_range=(0.0, 1.0))
        self.is_noise_reduction = is_noise_reduction
        self.algorithm_noise_reduction = algorithm_noise_reduction
        self.args_noise_reduction = copy.deepcopy(args_noise_reduction)
        self.computed_ratio = -1.0
        self._cached_kernel: ty.Optional[torch.Tensor] = None
        self._cached_window_key: ty.Optional[ty.Tuple[int, int, str, torch.device]] = None
    # end def

    def run_noise_reduction(
        self,
        input_sequence: ty.Union[np.ndarray, torch.Tensor],
        window_length: ty.Optional[int] = None
    ) -> ty.Union[np.ndarray, torch.Tensor]:
        """Apply noise reduction on trajectory sequence.

        Operates directly on GPU/CPU torch.Tensor via Conv1D without device transfers,
        or on np.ndarray via standard scipy/pandas routines.
        """
        if isinstance(input_sequence, torch.Tensor):
            return self._run_noise_reduction_torch(input_sequence, window_length)
        else:
            return self._run_noise_reduction_numpy(input_sequence, window_length)
        # end if
    # end def

    def is_oscillation_span(
        self,
        scaled_indicator: torch.Tensor,
        acceptance_range: float = 0.01
    ) -> bool:
        """True if the trajectory has big oscillation, False otherwise."""
        if len(scaled_indicator) < 3:
            return False
        # end if
        min_inner = torch.min(scaled_indicator[1:-1])
        max_inner = torch.max(scaled_indicator[1:-1])
        if (max_inner - min_inner) < acceptance_range:
            return False
        # end if

        span_start = max(0, len(scaled_indicator) - self.check_span)
        log_first = scaled_indicator[span_start]
        log_last = scaled_indicator[-1]

        first_last_ratio = log_first / log_last
        min_max_ratio = max_inner / min_inner

        is_big_oscillation = bool(torch.abs(1.0 - first_last_ratio) < torch.abs(1.0 - min_max_ratio))
        return is_big_oscillation
    # end def

    def is_mal_optimization(
        self,
        seq_indicator: torch.Tensor,
        threshold_mal_opt_ratio: float = 0.02,
        is_up_expected: bool = False
    ) -> ty.Tuple[bool, float]:
        """True if the training trajectory is malformed, else False."""
        is_oscillation = self.is_oscillation_span(seq_indicator)
        if is_oscillation:
            return False, -1.0
        # end if
        span_start = max(0, len(seq_indicator) - self.check_span)
        diff_ratio = seq_indicator[-1] / seq_indicator[span_start]
        diff_ratio_val = diff_ratio.item()

        if is_up_expected:
            if seq_indicator[0] < seq_indicator[-1]:
                return False, -1.0
            else:
                is_mal = diff_ratio_val < (1.0 - threshold_mal_opt_ratio)
                return is_mal, diff_ratio_val
            # end if
        else:
            if seq_indicator[0] > seq_indicator[-1]:
                return False, -1.0
            else:
                is_mal = diff_ratio_val > (1.0 + threshold_mal_opt_ratio)
                return is_mal, diff_ratio_val
            # end if
        # end if
    # end def

    def is_convergence_optimization(
        self,
        scaled_seq_indicator_in_range: torch.Tensor,
        threshold_convergence_ratio: float = 0.001,
        threshold_oscilation_abs_range: float = 0.005
    ) -> ty.Tuple[bool, float]:
        """True if the trajectory is converging, False otherwise."""
        if threshold_oscilation_abs_range != -1:
            is_oscillation = self.is_oscillation_span(
                scaled_seq_indicator_in_range,
                acceptance_range=threshold_oscilation_abs_range
            )
            if is_oscillation:
                return False, -1.0
            # end if
        # end if

        ratio_first_end = scaled_seq_indicator_in_range[-1] / scaled_seq_indicator_in_range[0]
        is_convergence = bool((1.0 - threshold_convergence_ratio) < ratio_first_end < (1.0 + threshold_convergence_ratio))
        return is_convergence, ratio_first_end.item()
    # end def

    def _run_noise_reduction_torch(
        self,
        tensor_sequence: torch.Tensor,
        window_length: ty.Optional[int] = None
    ) -> torch.Tensor:
        """Apply noise reduction on PyTorch tensor directly on its device (CPU or GPU)."""
        seq_len = len(tensor_sequence)
        if seq_len < 3:
            return tensor_sequence
        # end if

        args_reduction = dict(self.args_noise_reduction)

        if self.algorithm_noise_reduction == 'rolling_mean':
            window = args_reduction.get('window', 500)
            if seq_len < window or window <= 0:
                window = max(3, seq_len // 2)
            # end if
            window = min(window, seq_len)
            kernel = (torch.ones(window, dtype=tensor_sequence.dtype, device=tensor_sequence.device) / window).view(1, 1, -1)
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded = F.pad(tensor_sequence.view(1, 1, -1), (pad_left, pad_right), mode='replicate')
            output_tensor = F.conv1d(padded, kernel).view(-1)
            return torch.nan_to_num(output_tensor)
        elif self.algorithm_noise_reduction == 'savgol_filter':
            w_len = args_reduction.get('window_length', 501)
            polyorder = args_reduction.get('polyorder', 1)
            if window_length is not None:
                w_len = window_length + 1 if window_length % 2 == 0 else window_length
            # end if
            if seq_len < w_len or w_len <= 0:
                w_len = max(3, seq_len // 2)
                w_len = (w_len + 1) if w_len % 2 == 0 else w_len
            # end if
            if w_len > seq_len:
                w_len = seq_len if seq_len % 2 != 0 else max(3, seq_len - 1)
            # end if
            if w_len <= polyorder:
                return tensor_sequence
            # end if

            cache_key = (w_len, polyorder, tensor_sequence.dtype, tensor_sequence.device)
            if self._cached_window_key == cache_key and self._cached_kernel is not None:
                kernel = self._cached_kernel
            else:
                coeffs = savgol_coeffs(w_len, polyorder)
                kernel = torch.tensor(
                    coeffs[::-1].copy(),
                    dtype=tensor_sequence.dtype,
                    device=tensor_sequence.device
                ).view(1, 1, -1)
                self._cached_kernel = kernel
                self._cached_window_key = cache_key
            # end if

            pad = w_len // 2
            padded = F.pad(tensor_sequence.view(1, 1, -1), (pad, pad), mode='replicate')
            output_tensor = F.conv1d(padded, kernel).view(-1)
            return torch.nan_to_num(output_tensor)
        else:
            raise NotImplementedError(f"Unsupported noise reduction algorithm: {self.algorithm_noise_reduction}")
        # end if
    # end def

    def _run_noise_reduction_numpy(
        self,
        input_sequence: np.ndarray,
        window_length: ty.Optional[int] = None
    ) -> np.ndarray:
        """Apply noise reduction on numpy array for fallback compatibility."""
        args_noise_reduction = dict(self.args_noise_reduction)
        if self.algorithm_noise_reduction == 'rolling_mean':
            output_sequence = pandas.Series(input_sequence).rolling(**args_noise_reduction).mean().to_numpy()
            output_sequence = np.nan_to_num(output_sequence)
        elif self.algorithm_noise_reduction == 'savgol_filter':
            if window_length is not None:
                window_length_ = window_length + 1 if window_length % 2 == 0 else window_length
                args_noise_reduction['window_length'] = window_length_
            # end if
            if len(input_sequence) < args_noise_reduction['window_length'] or args_noise_reduction['window_length'] < 0:
                args_noise_reduction['window_length'] = int(len(input_sequence) / 2)
                args_noise_reduction['window_length'] = (args_noise_reduction['window_length'] + 1) \
                    if args_noise_reduction['window_length'] % 2 == 0 else args_noise_reduction['window_length']
            # end if
            w_len = args_noise_reduction['window_length']
            if w_len > len(input_sequence):
                w_len = len(input_sequence) if len(input_sequence) % 2 != 0 else max(3, len(input_sequence) - 1)
                args_noise_reduction['window_length'] = w_len
            # end if
            if w_len <= args_noise_reduction.get('polyorder', 1):
                return input_sequence
            # end if
            output_sequence = savgol_filter(input_sequence, **args_noise_reduction)
            output_sequence = np.nan_to_num(output_sequence)
        else:
            raise NotImplementedError(f"Unsupported noise reduction algorithm: {self.algorithm_noise_reduction}")
        # end if
        return output_sequence
    # end def
# end class