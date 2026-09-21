import typing as ty
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
# end try


if HAS_TRITON:
    @triton.jit
    def _fused_gaussian_kernel_fwd_kernel(
        X_ptr, Y_ptr, W_ptr, BW_ptr, Out_ptr,
        M, N, D,
        stride_xm, stride_xd,
        stride_yn, stride_yd,
        stride_om, stride_on,
        stride_bw,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        r"""Triton JIT Forward Kernel for Fused ARD Gaussian Kernel Matrix.

        Mathematical Definition:
        -------------------------
        Given matrices $X \in \mathbb{R}^{M \times D}$, $Y \in \mathbb{R}^{N \times D}$,
        ARD weight vector $w \in \mathbb{R}^D$, and bandwidth vector $\sigma^2 \in \mathbb{R}^D$,
        computes the Gaussian Gram matrix $K \in \mathbb{R}^{M \times N}$ where:

        .. math::
            K_{ij} = \exp\left( - \sum_{d=0}^{D-1} \frac{w_d^2 \cdot (x_{id} - y_{jd})^2}{\sigma_d^2} \right)

        Thread-Block Parallelization Strategy:
        --------------------------------------
        - The $M \times N$ output matrix is partitioned into 2D tiles of size
          $\text{BLOCK\_M} \times \text{BLOCK\_N}$ (e.g., $32 \times 32$).
        - Each thread block $(pid_m, pid_n)$ loads the corresponding subset of rows
          $i \in [pid_m \cdot B_M, (pid_m + 1) \cdot B_M)$ from $X$ and
          $j \in [pid_n \cdot B_N, (pid_n + 1) \cdot B_N)$ from $Y$.
        - The inner loop over feature dimension $d \in \{0, \dots, D-1\}$ accumulates
          the weighted squared distances entirely in on-chip GPU SRAM registers.
        - Finally, the exponential $\exp(-\cdot)$ is computed and written to global VRAM once.
        """
        # Block IDs along the 2D grid
        # pid_m \in \{0, \dots, \lceil M / \text{BLOCK\_M} \rceil - 1\}
        pid_m = tl.program_id(0)
        # pid_n \in \{0, \dots, \lceil N / \text{BLOCK\_N} \rceil - 1\}
        pid_n = tl.program_id(1)

        # Compute tile offset indices for rows i and columns j:
        # i \in [pid_m \cdot B_M, (pid_m + 1) \cdot B_M)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # j \in [pid_n \cdot B_N, (pid_n + 1) \cdot B_N)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Boundary masks to prevent out-of-bounds memory access when M or N is not a multiple of block size:
        # \text{mask}_m(i) = \mathbb{I}(i < M), \quad \text{mask}_n(j) = \mathbb{I}(j < N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Initialize accumulator in fast GPU registers:
        # \text{acc}_{ij}^{(0)} = 0.0, \quad \forall (i, j) \in \text{BLOCK\_M} \times \text{BLOCK\_N}
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over each feature dimension d \in \{0, \dots, D-1\}
        for d in range(D):
            # Load scalar ARD weight: w_d = W[d]
            w_val = tl.load(W_ptr + d)

            # Load bandwidth value: \sigma_d^2 = \text{BW}[d \cdot \text{stride\_bw}]
            bw_val = tl.load(BW_ptr + d * stride_bw)

            # Compute scale factor: \alpha_d = \frac{w_d^2}{\sigma_d^2}
            w_scale = (w_val * w_val) / bw_val

            # Load row coordinates for dimension d:
            # x_{i, d} = X[i \cdot \text{stride\_xm} + d \cdot \text{stride\_xd}]
            x_vals = tl.load(X_ptr + offs_m * stride_xm + d * stride_xd, mask=mask_m, other=0.0)

            # Load column coordinates for dimension d:
            # y_{j, d} = Y[j \cdot \text{stride\_yn} + d \cdot \text{stride\_yd}]
            y_vals = tl.load(Y_ptr + offs_n * stride_yn + d * stride_yd, mask=mask_n, other=0.0)

            # Compute pairwise 1D difference vector via broadcasting:
            # \Delta_{ij, d} = x_{i, d} - y_{j, d}
            diff = x_vals[:, None] - y_vals[None, :]

            # Accumulate weighted squared difference into register tile:
            # \text{acc}_{ij}^{(d+1)} = \text{acc}_{ij}^{(d)} + \alpha_d \cdot (x_{i, d} - y_{j, d})^2
            acc += (diff * diff) * w_scale
        # end for

        # Compute element-wise exponential:
        # K_{ij} = \exp(-\text{acc}_{ij}^{(D)}) = \exp\left( - \sum_{d=0}^{D-1} \frac{w_d^2 \cdot (x_{id} - y_{jd})^2}{\sigma_d^2} \right)
        k_val = tl.exp(-acc)

        # Write output tile to global VRAM:
        # \text{Out}[i \cdot \text{stride\_om} + j \cdot \text{stride\_on}] = K_{ij}
        tl.store(
            Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            k_val,
            mask=mask_m[:, None] & mask_n[None, :]
        )
    # end def


    @triton.jit
    def _fused_gaussian_kernel_bwd_w_kernel(
        GradOut_ptr, K_ptr, X_ptr, Y_ptr, W_ptr, BW_ptr, GradW_partial_ptr,
        M, N, D,
        stride_gm, stride_gn,
        stride_km, stride_kn,
        stride_xm, stride_xd,
        stride_yn, stride_yd,
        stride_bw,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        r"""Triton JIT Backward Kernel for Fused ARD Gaussian Kernel Gradients.

        Mathematical Derivation via Chain Rule:
        ---------------------------------------
        Let $\mathcal{L}$ be the scalar objective function, and let $G \in \mathbb{R}^{M \times N}$
        be the upstream gradient tensor:

        .. math::
            G_{ij} = \frac{\partial \mathcal{L}}{\partial K_{ij}}

        The gradient of the Gram matrix element $K_{ij}$ with respect to ARD weight $w_d$ is:

        .. math::
            \frac{\partial K_{ij}}{\partial w_d}
            = \frac{\partial}{\partial w_d} \exp\left( - \sum_{k=0}^{D-1} \frac{w_k^2 \cdot (x_{ik} - y_{jk})^2}{\sigma_k^2} \right)
            = K_{ij} \cdot \left( - \frac{2 w_d}{\sigma_d^2} \cdot (x_{id} - y_{jd})^2 \right)

        By the multivariable chain rule, the total gradient for weight $w_d$ is the sum over all pairs $(i, j)$:

        .. math::
            \frac{\partial \mathcal{L}}{\partial w_d}
            = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} \frac{\partial \mathcal{L}}{\partial K_{ij}} \cdot \frac{\partial K_{ij}}{\partial w_d}
            = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} G_{ij} \cdot K_{ij} \cdot \left( - \frac{2 w_d}{\sigma_d^2} \right) \cdot (x_{id} - y_{jd})^2

        Thread-Block Parallelization Strategy:
        --------------------------------------
        - Uses a 3D grid: $(pid_m, pid_n, d)$ where dimension $d \in \{0, \dots, D-1\}$.
        - Each thread block computes the partial sum of gradients for tile $(pid_m, pid_n)$ on feature $d$.
        - The partial sum is saved to `GradW_partial_ptr`, and reduced across blocks in PyTorch.
        """
        # Block IDs along 3D grid
        pid_m = tl.program_id(0)  # Row tile index
        pid_n = tl.program_id(1)  # Column tile index
        d = tl.program_id(2)      # Feature dimension index d \in \{0, \dots, D-1\}

        # Tile indices for rows i and columns j
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # 2D boundary masks: \text{mask}_{ij} = \mathbb{I}(i < M \land j < N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_2d = mask_m[:, None] & mask_n[None, :]

        # Load scalar parameter values for dimension d:
        # w_d = W[d], \quad \sigma_d^2 = \text{BW}[d \cdot \text{stride\_bw}]
        w_val = tl.load(W_ptr + d)
        bw_val = tl.load(BW_ptr + d * stride_bw)

        # Common derivative factor: \gamma_d = -\frac{2 w_d}{\sigma_d^2}
        factor = (-2.0 * w_val) / bw_val

        # Load upstream loss gradient: G_{ij} = \text{GradOut}[i, j]
        g = tl.load(GradOut_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn, mask=mask_2d, other=0.0)

        # Load forward kernel value: K_{ij} = K[i, j]
        k = tl.load(K_ptr + offs_m[:, None] * stride_km + offs_n[None, :] * stride_kn, mask=mask_2d, other=0.0)

        # Load coordinates for dimension d: x_{i, d} and y_{j, d}
        x_vals = tl.load(X_ptr + offs_m * stride_xm + d * stride_xd, mask=mask_m, other=0.0)
        y_vals = tl.load(Y_ptr + offs_n * stride_yn + d * stride_yd, mask=mask_n, other=0.0)

        # Compute pairwise coordinate difference: \Delta_{ij, d} = x_{i, d} - y_{j, d}
        diff = x_vals[:, None] - y_vals[None, :]

        # Element-wise gradient contribution:
        # \delta_{ij, d} = G_{ij} \cdot K_{ij} \cdot \left( - \frac{2 w_d}{\sigma_d^2} \right) \cdot (x_{i, d} - y_{j, d})^2
        grad_elem = g * k * (diff * diff) * factor

        # Intra-block reduction across tile:
        # S_{\text{block}}(pid_m, pid_n, d) = \sum_{i \in \text{tile}_m} \sum_{j \in \text{tile}_n} \delta_{ij, d}
        block_sum = tl.sum(grad_elem)

        # Store partial block sum in global memory buffer:
        # \text{idx} = (pid_m \cdot \text{num\_n\_blocks} + pid_n) \cdot D + d
        num_n_blocks = tl.num_programs(1)
        idx = (pid_m * num_n_blocks + pid_n) * D + d
        tl.store(GradW_partial_ptr + idx, block_sum)
    # end def
# end if


class FusedGaussianKernelFunction(torch.autograd.Function):
    r"""PyTorch Autograd Function for Fused Gaussian ARD Kernel using OpenAI Triton.

    Mathematical Representation:
    -----------------------------
    Forward:
    .. math::
        K_{ij} = \exp\left( - \sum_{d=0}^{D-1} \frac{w_d^2 \cdot (x_{id} - y_{jd})^2}{\sigma_d^2} \right)

    Backward w.r.t $w_d$:
    .. math::
        \frac{\partial \mathcal{L}}{\partial w_d}
        = \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} G_{ij} \cdot K_{ij} \cdot \left( - \frac{2 w_d}{\sigma_d^2} \right) \cdot (x_{id} - y_{jd})^2
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        ard_weights: torch.Tensor,
        bandwidth: torch.Tensor
    ) -> torch.Tensor:
        r"""Compute forward Gram matrix $K(x, y)$ on GPU via Triton.

        Args:
            ctx: PyTorch autograd context.
            x: Input sample tensor $X \in \mathbb{R}^{M \times D}$.
            y: Input sample tensor $Y \in \mathbb{R}^{N \times D}$.
            ard_weights: ARD weight parameter vector $w \in \mathbb{R}^D$.
            bandwidth: Bandwidth parameter tensor $\sigma^2 \in \mathbb{R}^D$ or scalar.

        Returns:
            Gram matrix $K \in \mathbb{R}^{M \times N}$.
        """
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32).contiguous()
        y = y.to(dtype=torch.float32).contiguous()
        ard_weights = ard_weights.to(dtype=torch.float32).contiguous()
        bandwidth = bandwidth.to(dtype=torch.float32).contiguous()

        if not HAS_TRITON or x.device.type != "cuda":
            # Eager PyTorch Fallback for CPU execution:
            # xw = X \odot (w / \sqrt{\sigma^2}), \quad yw = Y \odot (w / \sqrt{\sigma^2})
            gamma = torch.reciprocal(torch.sqrt(bandwidth))
            xw = x * (ard_weights * gamma)
            yw = y * (ard_weights * gamma)
            dist = torch.cdist(xw, yw) ** 2
            out = torch.exp(-dist).to(dtype=orig_dtype)
            ctx.save_for_backward(x, y, ard_weights, bandwidth, out)
            ctx.use_triton = False
            ctx.orig_dtype = orig_dtype
            return out
        # end if

        M, D = x.shape
        N, _ = y.shape
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        stride_bw = 1 if bandwidth.numel() > 1 else 0
        BLOCK = 32
        grid = ((M + BLOCK - 1) // BLOCK, (N + BLOCK - 1) // BLOCK)

        _fused_gaussian_kernel_fwd_kernel[grid](
            x, y, ard_weights, bandwidth, out,
            M, N, D,
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            out.stride(0), out.stride(1),
            stride_bw,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK
        )

        ctx.save_for_backward(x, y, ard_weights, bandwidth, out)
        ctx.use_triton = True
        ctx.orig_dtype = orig_dtype
        return out.to(dtype=orig_dtype)
    # end def

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> ty.Tuple[None, None, torch.Tensor, None]:
        r"""Compute backward gradients $\frac{\partial \mathcal{L}}{\partial w}$ w.r.t ARD weights.

        Args:
            ctx: PyTorch autograd context containing saved tensors $(X, Y, w, \sigma^2, K)$.
            grad_output: Upstream gradient $G = \frac{\partial \mathcal{L}}{\partial K} \in \mathbb{R}^{M \times N}$.

        Returns:
            Tuple of gradients: (None for X, None for Y, grad_w for ard_weights, None for bandwidth).
        """
        x, y, ard_weights, bandwidth, out = ctx.saved_tensors
        grad_output = grad_output.to(dtype=torch.float32).contiguous()
        orig_dtype = ctx.orig_dtype

        if not ctx.use_triton:
            # Analytical CPU fallback:
            # \frac{\partial \mathcal{L}}{\partial w_d} = \sum_{i, j} (K_{ij} \cdot G_{ij}) \cdot \left(-\frac{2 w_d}{\sigma_d^2}\right) \cdot (x_{id} - y_{jd})^2
            factor = (-2.0 * ard_weights) / bandwidth  # (D,)
            diff_sq = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2  # (M, N, D)
            k_g = (out * grad_output).unsqueeze(-1)  # (M, N, 1)
            grad_w = torch.sum(k_g * diff_sq * factor, dim=(0, 1)).to(dtype=orig_dtype)
            return None, None, grad_w, None
        # end if

        M, D = x.shape
        N, _ = y.shape
        BLOCK = 32
        num_m = (M + BLOCK - 1) // BLOCK
        num_n = (N + BLOCK - 1) // BLOCK
        grid = (num_m, num_n, D)
        partial_grad = torch.empty((num_m * num_n, D), device=x.device, dtype=torch.float32)
        stride_bw = 1 if bandwidth.numel() > 1 else 0

        _fused_gaussian_kernel_bwd_w_kernel[grid](
            grad_output, out, x, y, ard_weights, bandwidth, partial_grad,
            M, N, D,
            grad_output.stride(0), grad_output.stride(1),
            out.stride(0), out.stride(1),
            x.stride(0), x.stride(1),
            y.stride(0), y.stride(1),
            stride_bw,
            BLOCK_M=BLOCK, BLOCK_N=BLOCK
        )
        # Sum partial block sums across all grid tiles:
        # \frac{\partial \mathcal{L}}{\partial w_d} = \sum_{\text{blocks}} \text{partial\_grad}[\text{block}, d]
        grad_w = partial_grad.sum(dim=0).to(dtype=orig_dtype)
        return None, None, grad_w, None
    # end def
# end class


def compute_fused_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    ard_weights: torch.Tensor,
    bandwidth: torch.Tensor
) -> torch.Tensor:
    r"""Public helper function to compute the Gaussian kernel matrix using the fused Triton kernel.

    Mathematical Formula:
    ---------------------
    .. math::
        K_{ij} = \exp\left( - \sum_{d=0}^{D-1} \frac{w_d^2 \cdot (x_{id} - y_{jd})^2}{\sigma_d^2} \right)

    Args:
        x: Input sample tensor $X \in \mathbb{R}^{M \times D}$.
        y: Input sample tensor $Y \in \mathbb{R}^{N \times D}$.
        ard_weights: ARD weight parameter vector $w \in \mathbb{R}^D$.
        bandwidth: Bandwidth tensor $\sigma^2 \in \mathbb{R}^D$ or scalar.

    Returns:
        Gram matrix $K \in \mathbb{R}^{M \times N}$.
    """
    return FusedGaussianKernelFunction.apply(x, y, ard_weights, bandwidth)
# end def
