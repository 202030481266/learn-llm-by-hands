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
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
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
        stride: For the strided linear layers (kept for compatibility, though typically 1).
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

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        # Weight shape is (out_features, in_features) for standard linear layer
        # Y = AX + b
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            # Always initialize bias to zero
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight
        self.master_weight = None
        if keep_master_weight_for_test:
            self.master_weight = self.weight.clone()
        with torch.no_grad():
            init_method(self.weight)  # Apply the initialization method directly

    def get_master_weight(self) -> torch.Tensor:
        """Returns the weight tensor (or master weight if kept for testing)."""
        return self.master_weight if self.master_weight is not None else self.weight

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Directly perform linear transformation
        # y = x * A^T + b
        output = F.linear(input_, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    """A simplified linear layer without row parallelism.

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias.
        init_method: method to initialize weights. Note that bias is always set to zero.
        stride: For the strided linear layers (kept for compatibility, though typically 1).
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

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight
        self.master_weight = None
        if keep_master_weight_for_test:
            self.master_weight = self.weight.clone()
        with torch.no_grad():
            init_method(self.weight)

    def get_master_weight(self) -> torch.Tensor:
        """Returns the weight tensor (or master weight if kept for testing)."""
        return self.master_weight if self.master_weight is not None else self.weight

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Directly perform linear transformation
        # y = x * A^T + b
        output = F.linear(input_, self.weight, self.bias)
        return output


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), base: float = 1e6):
    """
    旋转位置编码的预处理，计算theta矩阵，两两配对
    """
    # theta_i = 1 / (base ** (2i / d))，标准旋转位置编码计算
    # [dim//2,]
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [end,]
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # 外积计算, [end,1] outer [dim//2,] -> [end, dim//2]
    # 这里得到end行，第m行代表的是位置m的编码[m*theta_1, m*theta_2, ...m*theta_i,...m*theta_{dim//2}]
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # abs*(cos(theta)+i*sin(theta))
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 输入：
    # freqs_cis: [seq_len, dim] - 预计算的频率张量
    # x: [batch_size, seq_len, num_heads, head_dim] - 输入张量（比如 xq 或 xk）
    
    ndim = x.ndim  # x 的维度数，通常是 3
    assert 0 <= 1 < ndim  # 确保 x 至少是 2 维
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # 检查 freqs_cis 的形状匹配 x 的 seq_len 和最后一维
    # seq_len和最后一个维度是一样的
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # [1, seq_len, 1, dim]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 输入：
    # xq: [batch_size, seq_len, num_q_heads, head_dim] - 查询张量
    # xk: [batch_size, seq_len, num_k_heads, head_dim] - 键张量
    # freqs_cis: [seq_len, dim] - 预计算的频率张量（通常 dim = head_dim）
    
    # 将 xq 和 xk 重塑并转换为复数形式
    # view_as_complex输入的矩阵的最后一个维度必须是2，分别表示实部和虚部, cosi + jsini
    # xq_: [batch_size, seq_len, num_q_heads, head_dim // 2] - 复数形式的查询张量
    # xk_: [batch_size, seq_len, num_k_heads, head_dim // 2] - 复数形式的键张量
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 重塑 freqs_cis 以匹配 xq_ 的形状并进行广播
    # [1, seq_len, 1, head_dim // 2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # 应用旋转编码并展平
    # [batch_size, seq_len, num_q_heads, head_dim // 2] * [1, seq_len, 1, head_dim // 2] -> 
    #       [batch_size, seq_len, num_q_heads, head_dim // 2]
    # xq_out: [batch_size, seq_len, num_q_heads, head_dim] - 旋转后的查询张量
    # xk_out: [batch_size, seq_len, num_k_heads, head_dim] - 旋转后的键张量
    # 这里设计到了负数的计算点
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeak_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # GQA的实现，将key和value重复n_rep次，仅仅创建视图
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )

#############################################################
####### GQA, this is the part we need to focus on ###########
#############################################################
class GQA(nn.Module):
    """
    No parallelization, just for practice.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads
        
        # [b, s, d] -> [b, s, n_qh * d_h]
        self.wq = ColumnParallelLinear(args.dim, args.n_heads * self.head_dim, bias=False, init_method=lambda x: x)
        # [b, s, d] -> [b, s, n_kvh * d_h]
        self.wk = ColumnParallelLinear(args.dim, args.n_kv_heads * self.head_dim, bias=False, init_method=lambda x: x)
        # [b, s, d] -> [b, s, n_kvh * d_h]
        self.wv = ColumnParallelLinear(args.dim, args.n_kv_heads * self.head_dim, bias=False, init_method=lambda x: x)
        # [b, s, n_qh * d_h] -> [b, s, d]
        self.wo = RowParallelLinear(args.n_heads * self.head_dim, args.dim, bias=False, init_method=lambda x: x)
        
        # kv cache
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim).cuda()
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, args.n_kv_heads, self.head_dim).cuda()

    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape # [b, s, d]
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) 
        # [b, s, n_qh * d_h] -> [b, s, n_qh, d_h]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        # [b, s, n_kvh * d_h] -> [b, s, n_kvh, d_h]
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        # [b, s, n_kvh * d_h] -> [b, s, n_kvh, d_h]
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        # 添加旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 转换到相同的设备和数据类型上
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos:start_pos+seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos+seqlen] = xv
        
        keys = self.cache_k[:bsz, :start_pos+seqlen]
        values = self.cache_v[:bsz, :start_pos+seqlen]
        keys = repeak_kv(keys, self.n_rep) # [bsz, cache_len + seq_len, n_qh, d_h]
        values = repeak_kv(values, self.n_rep) # [bsz, cache_len + seq_len, n_qh, d_h]
        
        # attention分数计算
        xq = xq.transpose(1, 2) # [bsz, n_qh, s, d_h]
        keys = keys.transpose(1, 2) # [bsz, n_qh, cache_len + seq_len, d_h]
        # [bsz, n_qh, s, d_h] * [bsz, n_qh, d_h, cache_len + seq_len] -> [bsz, n_qh, s, cache_len + seq_len]
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        # [bsz, n_qh, s, cache_len + seq_len]
        scores = F.softmax(scores, dim=-1)
        # [bsz, n_qh, s, cache_len + seq_len] * [bsz, n_qh, cache_len + seq_len, d_h] -> [bsz, n_qh, s, d_h]
        output = torch.matmul(scores, values)
        # [bsz, n_qh, s, d_h] -> [bsz, s, n_qh, d_h] -> [bsz, s, n_qh * d_h]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # [bsz, s, n_qh * d_h] * [n_qh * d_h, d] -> [bsz, s, d]
        return self.wo(output)
        
        
