import typing as ty
import abc
from sympy import im

import torch

from ..datasets import BaseDataset
from ..kernels.base import (BaseKernel, KernelMatrixObject)
from ..kernels.gaussian_kernel import QuadraticKernelGaussianKernel, LinearMMDGaussianKernel
from ..kernels import QuadraticKernelMatrixContainer
from .commons import (
    ArgumentParameters, 
    MmdValues, 
)

import importlib.metadata


class BatchSizeError(Exception):
    pass




class BaseMmdEstimator(torch.nn.Module):
    def __init__(self,
                 kernel_obj: BaseKernel):
        super().__init__()
        self.kernel_obj = kernel_obj
        
    @classmethod
    @abc.abstractmethod
    def from_dataset(cls, dataset: BaseDataset) -> "BaseMmdEstimator":
        """Public method to create an estimator from a dataset."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_hyperparameters(self) -> ArgumentParameters:
        raise NotImplementedError()
        

class QuadraticMmdEstimator(BaseMmdEstimator):
    def __init__(self,
                 kernel_obj: BaseKernel,
                 unit_diagonal: bool = False,
                 biased: bool = False,
                 is_compute_variance: bool = True,
                 variance_term: str = 'liu_2020',
                 constant_optimization: float = 1e-8):
        super().__init__(kernel_obj)
        assert kernel_obj.kernel_computation_type == 'quadratic', \
            'This class expects Quadratic Kernels. Your kernel is NOT quadratic kernel(s).'

        self.unit_diagonal = unit_diagonal
        self.biased = biased

        self.kernel_obj = kernel_obj

        self.is_compute_variance = is_compute_variance

        # following Learning Deep Kernels for Non-Parametric Two-Sample Tests, 2020
        self.constant_optimization = constant_optimization

        self.variance_term = variance_term
        if variance_term == 'liu_2020':
            self.func_mmd_var = self._mmd_variance_liu2020
        elif variance_term == 'sutherland_2017':
            self.func_mmd_var = self._mmd_variance_sutherland2017
        else:
            raise NotImplementedError('Not Defined.')
        
    @classmethod
    def from_dataset(cls, 
                    dataset: BaseDataset, 
                    kernel_class: ty.Type[BaseKernel],
                    **kwargs) -> "QuadraticMmdEstimator":
        """Public method to create an estimator from a dataset."""        
        kernel_obj = kernel_class.from_dataset(dataset, **kwargs)
        
        return cls(kernel_obj=kernel_obj, 
                   unit_diagonal=False, 
                   biased=False, 
                   is_compute_variance=True, 
                   variance_term='liu_2020', 
                   constant_optimization=1e-8)
        
        
    def get_hyperparameters(self) -> ArgumentParameters:
        # distance class
        class_name_distance = self.kernel_obj.distance_module.__class__.__name__
        hyperparameters_distance = self.kernel_obj.distance_module.get_hyperparameters()
        
        # kernel class
        class_name_kernel = self.kernel_obj.__class__.__name__
        hyperparameters_kernel = self.kernel_obj.get_hyperparameters()
        
        # estimator class
        class_name_mmd_estimator = self.__class__.__name__
        hyperparameters_mmd_estimator = {
            'unit_diagonal': self.unit_diagonal,
            'biased': self.biased,
            'is_compute_variance': self.is_compute_variance,
            'variance_term': self.variance_term,
            'constant_optimization': self.constant_optimization
        }
        
        try:
            package_version = importlib.metadata.version(__package__)
        except importlib.metadata.PackageNotFoundError:
            package_version = 'unknown'
        # end try

        return ArgumentParameters(
            distance_class_name=class_name_distance,
            kernel_class_name=class_name_kernel,
            mmd_estimator_class_name=class_name_mmd_estimator,
            distance_object_arguments=hyperparameters_distance,
            kernel_object_arguments=hyperparameters_kernel,
            mmd_object_arguments=hyperparameters_mmd_estimator,
            package_version=package_version
        )

    def _mmd_variance_liu2020(self, kernel_matrix_obj: KernelMatrixObject, use_1sample_U: bool=True) -> MmdValues:
        """Computing MMD and Variance of MMD. The original code is from Liu's codebase.
        :param kernel_matrix_obj:
        :param use_1sample_U:
        :return:
        """
        # TODO: where is the constant value??
        k_obj: QuadraticKernelMatrixContainer = kernel_matrix_obj.kernel_matrix_container

        Kxxy = torch.cat((k_obj.k_xx, k_obj.k_xy), 1)
        Kyxy = torch.cat((k_obj.k_xy.transpose(0, 1), k_obj.k_yy), 1)
        Kxyxy = torch.cat((Kxxy, Kyxy), 0)
        nx = k_obj.k_xx.shape[0]
        ny = k_obj.k_yy.shape[0]

        if self.biased == False:
            xx = torch.div((torch.sum(k_obj.k_xx) - torch.sum(torch.diag(k_obj.k_xx))), (nx * (nx - 1)))
            yy = torch.div((torch.sum(k_obj.k_yy) - torch.sum(torch.diag(k_obj.k_yy))), (ny * (ny - 1)))
            # one-sample U-statistic.
            if use_1sample_U:
                xy = torch.div((torch.sum(k_obj.k_xy) - torch.sum(torch.diag(k_obj.k_xy))), (nx * (ny - 1)))
            else:
                xy = torch.div(torch.sum(k_obj.k_xy), (nx * ny))
            mmd2 = xx - 2 * xy + yy
        else:
            xx = torch.div((torch.sum(k_obj.k_xx)), (nx * nx))
            yy = torch.div((torch.sum(k_obj.k_yy)), (ny * ny))
            # one-sample U-statistic.
            if use_1sample_U:
                xy = torch.div((torch.sum(k_obj.k_xy)), (nx * ny))
            else:
                xy = torch.div(torch.sum(k_obj.k_xy), (nx * ny))
            mmd2 = xx - 2 * xy + yy
        # end if

        if self.is_compute_variance:
            hh = k_obj.k_xx + k_obj.k_yy - k_obj.k_xy - k_obj.k_xy.transpose(0, 1)
            V1 = torch.dot(hh.sum(1) / ny, hh.sum(1) / ny) / ny
            V2 = (hh).sum() / (nx) / nx
            varEst = 4 * (V1 - V2 ** 2)
        else:
            varEst = torch.tensor([0.0])
        # end if

        ratio = mmd2 / torch.sqrt(varEst + self.constant_optimization)

        return MmdValues(mmd2, varEst, ratio)

    def _mmd_variance_sutherland2017(self, kernel_matrix_object: KernelMatrixObject) -> MmdValues:
        """A custom function inspired from Liu, 2020. A constant value C is a new term.
        MMD^2: as described in Sutherland, 2017
        Variance: as described in Sutherland, 2017
        Ratio: MMD^2 / (Var + C)
        :param kernel_matrix_object:
        :return:
        """
        k_obj: QuadraticKernelMatrixContainer = kernel_matrix_object.kernel_matrix_container
        m = k_obj.k_xx.shape[0]  # Assumes X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if self.unit_diagonal:
            diag_x = diag_y = 1
            sum_diag_x = sum_diag_y = m
            sum_diag2_x = sum_diag2_y = m
        else:
            diag_x = torch.diagonal(k_obj.k_xx)
            diag_y = torch.diagonal(k_obj.k_yy)

            sum_diag_x = diag_x.sum()
            sum_diag_y = diag_y.sum()

            sum_diag2_x = diag_x.dot(diag_x)
            sum_diag2_y = diag_y.dot(diag_y)
        # end if
        # Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        kt_xx_sums = torch.sum(k_obj.k_xx, dim=1) - diag_x
        # Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        kt_yy_sums = torch.sum(k_obj.k_yy, dim=1) - diag_y
        # K_XY_sums_0 = K_XY.sum(axis=0)
        k_xy_sums_0 = torch.sum(k_obj.k_xy, dim=0)
        # K_XY_sums_1 = K_XY.sum(axis=1)
        k_xy_sums_1 = torch.sum(k_obj.k_xy, dim=1)

        kt_xx_sum = kt_xx_sums.sum()
        kt_yy_sum = kt_yy_sums.sum()
        k_xy_sum = k_xy_sums_0.sum()

        kt_xx_2_sum = (k_obj.k_xx ** 2).sum() - sum_diag2_x
        kt_yy_2_sum = (k_obj.k_yy ** 2).sum() - sum_diag2_y
        k_xy_2_sum = (k_obj.k_xy ** 2).sum()

        if self.biased:
            mmd2 = ((kt_xx_sum + sum_diag_x) / (m * m) + (kt_yy_sum + sum_diag_y) / (m * m) - 2 * k_xy_sum / (m * m))
        else:
            mmd2 = (kt_xx_sum / (m * (m - 1)) + kt_yy_sum / (m * (m - 1)) - 2 * k_xy_sum / (m * m))
        # end if

        if self.is_compute_variance:
            var_est = (
                    2 / (m ** 2 * (m - 1) ** 2) * (
                    2 * kt_xx_sums.dot(kt_xx_sums) - kt_xx_2_sum
                    + 2 * kt_yy_sums.dot(kt_yy_sums) - kt_yy_2_sum)
                    - (4 * m - 6) / (m ** 3 * (m - 1) ** 3) * (kt_xx_sum ** 2 + kt_yy_sum ** 2)
                    + 4 * (m - 2) / (m ** 3 * (m - 1) ** 2) * (
                            k_xy_sums_1.dot(k_xy_sums_1) + k_xy_sums_0.dot(k_xy_sums_0))
                    - 4 * (m - 3) / (m ** 3 * (m - 1) ** 2) * k_xy_2_sum
                    - (8 * m - 12) / (m ** 5 * (m - 1)) * k_xy_sum ** 2
                    + 8 / (m ** 3 * (m - 1)) * (
                            1 / m * (kt_xx_sum + kt_yy_sum) * k_xy_sum
                            - kt_xx_sums.dot(k_xy_sums_1)
                            - kt_yy_sums.dot(k_xy_sums_0))
            )
        else:
            var_est = torch.tensor([0.0])
        # end if

        ratio = mmd2 / torch.sqrt(var_est + self.constant_optimization)

        return MmdValues(mmd2, var_est, ratio)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> MmdValues:
        """Computes MMD value.
        Returns:
            MmdValues: named tuple object.
        """
        v_kernel_matrix_object = self.kernel_obj.compute_kernel_matrix(x, y)
        
        is_same_sample_size = v_kernel_matrix_object.x_size == v_kernel_matrix_object.y_size
        if is_same_sample_size == False and self.variance_term == 'liu_2020':
            # Note: (Liu et al., 2020)'s MMD formula supposes that same sample size. Thus, we should use Sutherland's MMD when n != m.
            # logger.warning('The sample sizes are different. The variance term is not reliable.')
            self.func_mmd_var = self._mmd_variance_sutherland2017
        # end if

        computed_values = self.func_mmd_var(v_kernel_matrix_object)
        return computed_values


class LinearMmdEstimator(BaseMmdEstimator):
    def __init__(self,
                 kernel_obj: BaseKernel,
                 unit_diagonal: bool = False,
                 biased: bool = False,
                 is_compute_variance: bool = True,
                 variance_term: str = 'sutherland_2017',
                 constant_optimization: float = 1e-8):
        super().__init__(kernel_obj)
        assert kernel_obj.kernel_computation_type == 'linear', \
            'This class expects linear Kernels. Your kernel is NOT linear kernel(s).'

        self.kernel_obj = kernel_obj
        self.unit_diagonal = unit_diagonal
        self.biased = biased

        self.is_compute_variance = is_compute_variance

        # following Learning Deep Kernels for Non-Parametric Two-Sample Tests, 2020
        self.constant_optimization = torch.tensor([constant_optimization])

        if variance_term == 'sutherland_2017':
            self.func_mmd_var = self._mmd_variance_sutherland2017
        else:
            raise NotImplementedError('Not Defined.')
        
    @classmethod
    def from_dataset(cls, 
                     dataset: BaseDataset, 
                     kernel_class: ty.Type[LinearMMDGaussianKernel],
                     **kwargs) -> "LinearMmdEstimator":
        """Public method to create an estimator from a dataset."""
        
        kernel_obj: LinearMMDGaussianKernel = kernel_class.from_dataset(dataset, **kwargs)
        assert isinstance(kernel_obj, LinearMMDGaussianKernel)
        
        return cls(kernel_obj=kernel_obj,
                   unit_diagonal=False,
                   biased=False,
                   is_compute_variance=True,
                   variance_term='sutherland_2017',
                   constant_optimization=1e-8)

    def get_hyperparameters(self) -> ArgumentParameters:
        # distance class
        class_name_distance = self.kernel_obj.distance_module.__class__.__name__
        hyperparameters_distance = self.kernel_obj.distance_module.get_hyperparameters()
                
        class_name_kernel = self.kernel_obj.__class__.__name__
        hyperparameters_kernel = self.kernel_obj.get_hyperparameters()
        
        class_name_mmd_estimator = self.__class__.__name__
        hyperparameters_mmd_estimator = {
            'unit_diagonal': self.unit_diagonal,
            'biased': self.biased,
            'is_compute_variance': self.is_compute_variance,
            'constant_optimization': self.constant_optimization.item(),
        }
        
        try:
            package_version = importlib.metadata.version(__package__)
        except importlib.metadata.PackageNotFoundError:
            package_version = 'unknown'
        # end try

        return ArgumentParameters(
            distance_class_name=class_name_distance,
            kernel_class_name=class_name_kernel,
            mmd_estimator_class_name=class_name_mmd_estimator,
            distance_object_arguments=hyperparameters_distance,
            kernel_object_arguments=hyperparameters_kernel,
            mmd_object_arguments=hyperparameters_mmd_estimator,
            package_version=package_version
        )

    def _mmd_variance_sutherland2017(self, kernel_matrix_object: KernelMatrixObject) -> MmdValues:
        """A custom function inspired from Liu, 2020. A constant value C is a new term.
        MMD^2: as described in Sutherland, 2017
        Variance: as described in Sutherland, 2017
        Ratio: MMD^2 / (Var + C)
        :param kernel_matrix_object:
        :return:
        """
        assert kernel_matrix_object.x_size == kernel_matrix_object.y_size

        k_obj = kernel_matrix_object.kernel_matrix_container
        # mmd2 = k_obj.k_h.mean() + self.constant_optimization
        m_power2 = kernel_matrix_object.x_size ** 2
        m = kernel_matrix_object.x_size
        mmd2 = (k_obj.k_h.sum() / (m_power2 - m)) + self.constant_optimization

        m = (kernel_matrix_object.x_size // 2) * 2
        approx_var = 1 / 2 * ((k_obj.k_h[:m:2] - k_obj.k_h[1:m:2]) ** 2).mean()

        # ratio = mmd2 / torch.sqrt(approx_var + self.constant_optimization)
        ratio = mmd2 / torch.sqrt(approx_var)

        return MmdValues(mmd2, approx_var, ratio)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> MmdValues:
        """Computes MMD value.
        Returns:
            MmdValues: named tuple object.
        """
        try:
            v_kernel_matrix_object = self.kernel_obj.compute_kernel_matrix(x, y)
            computed_values = self.func_mmd_var(v_kernel_matrix_object)
        except RuntimeError:
            raise BatchSizeError(f'Batch size = {len(x)} does not fit to the linear estimator. Try with more samples.')
        # end if
        return computed_values
