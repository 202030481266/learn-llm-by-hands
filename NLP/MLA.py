# deepseek v3 tech report https://arxiv.org/pdf/2412.19437
# reference blog article: https://zhuanlan.zhihu.com/p/23062701108
# reference code implementation: 
# https://github.com/ThomasVonWu/Interview-Note/blob/main/%E5%A4%A7%E6%A8%A1%E5%9E%8B/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F/DeepSeek_MLA.py#L367
# https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl
from triton import Config

from typing import Literal, Optional, Tuple


class MyRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _norm(self, x: torch.FloatTensor):
        # (b, s, c ) / (b, s, 1 ) => (b, s, c )
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.FloatTensor):
        return self.weight * self._norm(x)


class RMSNorm(nn.Module):
    """Official Implementation
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [16, 32, 64]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() > 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    # dtype = torch.bfloat16 # 后面接上 RMSnorm 类型会跳转float32
    dtype = torch.float32

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype or Linear.dtype)
        )
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(
                torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = False, dtype=None
    ):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """

    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn（Yet another RoPE extensioN）
    # 是通过一种高效的计算方法来扩展模型的上下文窗口，比以前的方法减少10倍
    # 的令牌和2.5倍的训练步骤。它引入了一个ramp函数，并将该函数合并到方法依赖函数中.
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 0.707


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim  # 64
    seqlen = args.max_seq_len  # 4096*4
    beta_fast = args.beta_fast  # 32
    beta_slow = args.beta_slow  # 1
    base = args.rope_theta  # 10000.0
    factor = args.rope_factor  # 40.0

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (max_seq_len,  dim // 2)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied. (bs, seq_len, num_heads, dim)
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings. (seq_len, dim//2)

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    # (2, 100, 16, 64) => (2, 100, 16, 32, 2) => (2, 100, 16, 32)
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


# Ref: https://github.com/deepseek-ai/DeepSeek-V3/blob/b5d872ead062c94b852d75ce41ae0b10fcfa1c86/inference/model.py#L393
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"


#############################################################
####### MLA, this is the part we need to focus on ###########
#############################################################
class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim # 隐藏层维度
        self.n_heads = args.n_heads # 注意力头数
        self.n_local_heads = args.n_heads // world_size # 本地注意力头数
        self.q_lora_rank = args.q_lora_rank # q矩阵低秩矩阵的秩
        self.kv_lora_rank = args.kv_lora_rank # kv矩阵低秩矩阵的秩
        self.qk_nope_head_dim = args.qk_nope_head_dim # 非位置编码的qk矩阵维度
        self.qk_rope_head_dim = args.qk_rope_head_dim # 位置编码的qk矩阵维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim # qk矩阵头维度
        self.v_head_dim = args.v_head_dim # v矩阵头维度
        
        # Q 矩阵相关
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.qk_head_dim) # MHA的设定 (7168, 128+64) = (7168, 192)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank) # q 的降维矩阵 (7168, 1536)，假设这里是1536, W_{DQ}
            self.q_norm = RMSNorm(self.q_lora_rank) 
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.qk_head_dim * self.n_heads) # 假设头128个, (1536, 192 * 128), W_{UQR}
        
        # kv cache 相关
        self.wkv_a = linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim) # kv cache的降维矩阵 (7168, 512 + 64), W_{DKVR}
        self.kv_norm = RMSNorm(self.kv_lora_rank) # 针对 kv cache 矩阵的RMSNorm
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)) # (512, 128 * (128 + 128)) # W_{UKV}
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim) # (dim, dim) 
        self.softmax_scale = self.qk_head_dim ** -0.5 
        
        if args.max_seq_len > args.original_seq_len: # 需要设置模型的外推
            # 通常来说矩阵向量都是 (batch_size, seq_len, heads, head_dim)
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # kv cache , C_{KV}
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False) 
            # K矩阵的位置编码， k_{R}
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False) 
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size=2, seq_len=1, dim=7168).
            start_pos (int): Starting position in the sequence for caching. (seq_len, d//2).
                                         KV cache 起始填充位置以通过：推理过程中记录上一个预
                                         测的token生成KV填充至KV cache lists的结束的位置end_pos计算得到。
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen

        # q矩阵相关
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            # (2, 1, 7168) -> (2, 1, 1536) -> (2, 1, 192 * 128)
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        # (2, 1, 192 * 128) -> (2, 1, 128, 192)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # (2, 1, 128, 192) -> (2, 1, 128, 128) (2, 1, 128, 64)
        q_nope, q_pe = torch.split(q, [self.qk_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis) # 添加ROPE
        
        # kv cache 相关
        # (2, 1, 7168) -> (2, 1, 512 + 64)
        kv = self.wkv_a(x) 
        # (2, 1, 512 + 64) -> (2, 1, 512), (2, 1, 64)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # (2, 1, 1, 64)
        k_pe = apply_rotary_emb(k_pe, freqs_cis) # 添加ROPE
        
        if attn_impl == "naive":
            # 将q_nope和q_pe拼接起来, (2, 1, 128, 192)
            q = torch.cat([q_nope, q_pe], dim=-1)
            # (2, 1, 512) -> (2, 1, 128 * (128 + 128))
            kv = self.wkv_b(self.kv_norm(kv))
            # (2, 1, 128 * (128 + 128)) -> (2, 1, 128, 128 + 128)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # (2, 1, 128, 128 + 128) -> (2, 1, 128, 128), (2, 1, 128, 128)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # 这里因为共享了k_pe，所以需要扩展到 (2, 1, 128, 64)，满足128个head
            # (2, 1, 128, 128 + 64)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 更新kv cache
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算attention scores
            # 注意乘上softmax_scale，这个主要是为了增强上下文的性能，详细参考YaRN论文
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = (self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size))
            # (128 * (128 + 128), 512) -> (128, 128, 512)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # 计算 q_c * W_{UKV}^T，这里对d求和
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 更新kv cache
            # (2, 1, 512)
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(self.kv)
            # (2, 1, 1, 64) -> (2, 1, 64)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 计算 q_c * W_{UKV}^T + q_r * k_r^T , 分别对 c 和 r 求和
            scores = (
                (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) + 
                torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
            )
        if mask is not None:
            # 通常来说mask的形状是(B,1,T) 或者 (B,T)
            # 使用unsqueeze扩展维度，可以广播到每一个head中
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        if attn_impl == "naive":
            # 加权求和
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 原理和计算attention那里很像，先对t求和，然后再对c求和
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2)) # (bsz, seq_len, dim)
        return x