import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from torch.nn.parameter import Parameter


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048


class ColumnParallelLinear(nn.Module):
    """A simplified linear layer without column parallelism.

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        init_method: method to initialize weights. Note that bias is always set to zero.
        keep_master_weight_for_test: Keep master weight for testing (optional).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Weight shape is (out_features, in_features) for standard linear layer
        # Y = AX + b
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.master_weight = None
        if keep_master_weight_for_test:
            self.master_weight = self.weight.clone()
        with torch.no_grad():
            init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = F.linear(input_, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    """A simplified linear layer without row parallelism.

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias.
        init_method: method to initialize weights. Note that bias is always set to zero.
        keep_master_weight_for_test: Keep master weight for testing (optional).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super(RowParallelLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.master_weight = None
        if keep_master_weight_for_test:
            self.master_weight = self.weight.clone()
        with torch.no_grad():
            init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = F.linear(input_, self.weight, self.bias)
        return output


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), base: float = 1e6):
    """
    旋转位置编码的预处理，计算theta矩阵，两两配对
    Rotary Positional Encoding (RoPE) precomputation
    """
    # theta_i = 1 / (base ** (2i / d))，标准旋转位置编码计算
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    # 外积计算, [end, dim//2]
    freqs = torch.outer(t, freqs).float()
    # 复数表示: cos(theta) + i*sin(theta)
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape freqs_cis for broadcasting with input tensor x.

    Args:
        freqs_cis: [seq_len, dim] - 预计算的频率张量
        x: [batch_size, seq_len, num_heads, head_dim] - 输入张量
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        xq: [batch_size, seq_len, num_heads, head_dim] - 查询张量
        xk: [batch_size, seq_len, num_heads, head_dim] - 键张量
        freqs_cis: [seq_len, dim] - 预计算的频率张量

    Returns:
        xq_out, xk_out: 应用旋转位置编码后的查询和键张量
    """
    # 转换为复数形式: [b, s, n_h, d_h//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 广播形状: [1, seq_len, 1, head_dim // 2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 应用旋转并转回实数
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


#############################################################
########### MHA (Multi-Head Attention) #####################
#############################################################
class MHA(nn.Module):
    """
    经典的Multi-Head Attention实现

    与GQA的区别：
    - MHA中每个attention head都有独立的K和V投影
    - 不需要像GQA那样重复KV（repeat_kv）
    - 计算量更大，但表达能力更强

    核心公式：
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        # MHA中每个head的维度
        self.head_dim = args.dim // self.n_heads

        # QKV投影: [b, s, d] -> [b, s, n_h * d_h]
        self.wq = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False, init_method=lambda x: x
        )
        self.wk = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False, init_method=lambda x: x
        )
        self.wv = ColumnParallelLinear(
            args.dim, args.n_heads * self.head_dim, bias=False, init_method=lambda x: x
        )

        # 输出投影: [b, s, n_h * d_h] -> [b, s, d]
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim, args.dim, bias=False, init_method=lambda x: x
        )

        # KV Cache用于推理加速
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_k = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim, device=device
        )
        self.cache_v = torch.zeros(
            args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播

        Args:
            x: [batch_size, seq_len, dim] - 输入张量
            start_pos: int - 当前序列的起始位置（用于KV cache）
            freqs_cis: [seq_len, head_dim] - 预计算的旋转位置编码
            mask: [batch_size, n_heads, seq_len, cache_len+seq_len] - 注意力掩码

        Returns:
            output: [batch_size, seq_len, dim] - 注意力输出
        """
        bsz, seqlen, _ = x.shape

        # QKV投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑为多头形式: [b, s, n_h * d_h] -> [b, s, n_h, d_h]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # 应用旋转位置编码（RoPE）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 更新KV Cache
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 获取完整的缓存的keys和values
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 转置用于矩阵乘法: [b, n_h, s, d_h]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数: QK^T / sqrt(d_h)
        # [b, n_h, s, d_h] @ [b, n_h, d_h, cache_len+s] -> [b, n_h, s, cache_len+s]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用因果掩码（如果提供）
        if mask is not None:
            scores = scores + mask

        # Softmax归一化
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(xq)

        # 加权求和: [b, n_h, s, cache_len+s] @ [b, n_h, cache_len+s, d_h] -> [b, n_h, s, d_h]
        output = torch.matmul(scores, values)

        # 合并多头并输出投影
        # [b, n_h, s, d_h] -> [b, s, n_h, d_h] -> [b, s, n_h * d_h]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # [b, s, n_h * d_h] -> [b, s, d]
        return self.wo(output)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = ModelArgs(dim=512, n_heads=8, max_batch_size=2, max_seq_len=128)

    mha = MHA(args).to(device)

    # 预计算位置编码
    freqs_cis = precompute_pos_cis(args.dim // args.n_heads, end=args.max_seq_len).to(device)

    # 模拟输入
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, args.dim).to(device)

    # 前向传播
    output = mha(x, start_pos=0, freqs_cis=freqs_cis[:seq_len])

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("MHA module test passed!")
