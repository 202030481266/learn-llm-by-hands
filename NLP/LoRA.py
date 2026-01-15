# LoRA (Low-Rank Adaptation) Implementation
# Reference: https://arxiv.org/abs/2106.09685 (LoRA: Low-Rank Adaptation of Large Language Models)
# Used in: Almost all modern LLM fine-tuning (LLaMA-Adapter, Alpaca-LoRA, QLoRA, etc.)

"""
LoRA核心思想：
==============
在大模型微调时，不直接更新原始权重W，而是学习一个低秩分解：
    W' = W + ΔW = W + BA

其中：
    - W: 原始预训练权重 [out_features, in_features]，frozen
    - B: 低秩矩阵 [out_features, r]，可训练
    - A: 低秩矩阵 [r, in_features]，可训练
    - r: 秩（rank），通常 r << min(out_features, in_features)

优势：
    1. 参数高效：只需要训练 r*(in+out) 个参数，而不是 in*out
    2. 无推理延迟：训练后可将BA合并到W中
    3. 模块化：可以切换不同的LoRA适配器
    4. 保持基础模型能力：原始权重不变

典型应用场景：
    - 在Attention层的Q、K、V、O投影上添加LoRA
    - 在FFN层的上下投影上添加LoRA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA配置参数"""
    r: int = 8                           # LoRA秩（低秩分解的维度）
    lora_alpha: int = 16                 # 缩放因子
    lora_dropout: float = 0.0            # Dropout概率
    target_modules: List[str] = None     # 要应用LoRA的模块名称
    merge_weights: bool = False          # 是否合并权重到原始矩阵
    
    def __post_init__(self):
        if self.target_modules is None:
            # 默认应用到Attention的Q、V投影
            self.target_modules = ["wq", "wv"]


class LoRALinear(nn.Module):
    """
    带LoRA的线性层
    
    实现了 Y = X(W + αBA/r) + bias 的计算
    
    其中：
        - W: 原始权重，frozen
        - A: 低秩矩阵A [r, in_features]，用kaiming初始化
        - B: 低秩矩阵B [out_features, r]，用零初始化
        - α: 缩放因子（lora_alpha）
        - r: 低秩维度
    
    Args:
        in_features: 输入维度
        out_features: 输出维度
        r: LoRA秩
        lora_alpha: 缩放因子
        lora_dropout: Dropout概率
        merge_weights: 是否在推理时合并权重
    
    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)
    
    Example:
        >>> lora_linear = LoRALinear(768, 768, r=8, lora_alpha=16)
        >>> x = torch.randn(2, 128, 768)
        >>> output = lora_linear(x)
        >>> print(output.shape)  # torch.Size([2, 128, 768])
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        
        # 原始线性层权重（frozen）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 初始化原始权重
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # LoRA参数
        if r > 0:
            # A矩阵: [r, in_features]
            self.lora_A = nn.Parameter(torch.empty(r, in_features))
            # B矩阵: [out_features, r]
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            
            # 缩放因子: α/r
            self.scaling = lora_alpha / r
            
            # Dropout
            self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
            
            # 初始化A矩阵（kaiming初始化）
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # B矩阵初始化为0，确保初始时ΔW=BA=0
        
        # 标记权重是否已合并
        self.merged = False
        
        # 冻结原始权重
        self.weight.requires_grad = False
    
    def merge(self):
        """
        将LoRA权重合并到原始权重中
        
        合并后: W' = W + α*BA/r
        推理时无额外计算开销
        """
        if self.r > 0 and not self.merged:
            # W = W + scaling * (B @ A)
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A)
            self.merged = True
    
    def unmerge(self):
        """
        将LoRA权重从原始权重中分离
        
        用于需要继续训练或切换LoRA适配器的场景
        """
        if self.r > 0 and self.merged:
            # W = W - scaling * (B @ A)
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        如果权重已合并: Y = XW' + bias
        否则: Y = XW + X(BA)*scaling + bias
        """
        if self.r > 0 and not self.merged:
            # 原始线性变换
            result = F.linear(x, self.weight, self.bias)
            # 添加LoRA增量: X @ A^T @ B^T * scaling
            lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_output * self.scaling
            return result
        else:
            # 权重已合并，或r=0
            return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}, lora_alpha={self.lora_alpha}'


class LoRAEmbedding(nn.Module):
    """
    带LoRA的Embedding层
    
    用于微调词嵌入表，适用于需要适应新词汇或领域的场景
    
    Args:
        num_embeddings: 词表大小
        embedding_dim: 嵌入维度
        r: LoRA秩
        lora_alpha: 缩放因子
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        lora_alpha: int = 16,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.padding_idx = padding_idx
        
        # 原始Embedding
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)
        
        # LoRA参数
        if r > 0:
            # A矩阵: [r, num_embeddings]
            self.lora_A = nn.Parameter(torch.zeros(r, num_embeddings))
            # B矩阵: [embedding_dim, r]
            self.lora_B = nn.Parameter(torch.zeros(embedding_dim, r))
            
            self.scaling = lora_alpha / r
            
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)
        
        self.merged = False
        self.weight.requires_grad = False
    
    def merge(self):
        if self.r > 0 and not self.merged:
            # W = W + scaling * (B @ A).T
            self.weight.data += self.scaling * (self.lora_B @ self.lora_A).T
            self.merged = True
    
    def unmerge(self):
        if self.r > 0 and self.merged:
            self.weight.data -= self.scaling * (self.lora_B @ self.lora_A).T
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # 原始嵌入
            result = F.embedding(
                x, self.weight, self.padding_idx
            )
            # LoRA增量: 通过one-hot索引获取对应的LoRA权重
            # after_A: [batch, seq_len, r]
            after_A = F.embedding(x, self.lora_A.T, self.padding_idx)
            # result += (after_A @ B.T) * scaling
            result = result + (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.embedding(x, self.weight, self.padding_idx)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    将模型中除LoRA参数外的所有参数设为不可训练
    
    Args:
        model: 要处理的模型
        bias: 偏置处理方式
            - 'none': 所有偏置frozen
            - 'all': 所有偏置可训练
            - 'lora_only': 只有LoRA层的偏置可训练
    
    Example:
        >>> model = SomeTransformer()
        >>> apply_lora_to_model(model, config)
        >>> mark_only_lora_as_trainable(model, bias='lora_only')
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    
    if bias == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for module in model.modules():
            if isinstance(module, LoRALinear) and module.bias is not None:
                module.bias.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取模型中所有LoRA参数的state_dict
    
    用于保存LoRA适配器，可以单独保存和加载
    
    Args:
        model: 包含LoRA层的模型
    
    Returns:
        只包含LoRA参数的字典
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict


def set_lora_state_dict(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> None:
    """
    加载LoRA参数到模型中
    
    Args:
        model: 包含LoRA层的模型
        lora_state_dict: LoRA参数字典
    """
    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)


def apply_lora_to_linear(
    linear: nn.Linear,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> LoRALinear:
    """
    将普通的nn.Linear转换为LoRALinear
    
    保持原始权重不变，添加LoRA参数
    
    Args:
        linear: 原始线性层
        r: LoRA秩
        lora_alpha: 缩放因子
        lora_dropout: Dropout概率
    
    Returns:
        带LoRA的线性层
    
    Example:
        >>> linear = nn.Linear(768, 768)
        >>> lora_linear = apply_lora_to_linear(linear, r=8)
    """
    lora_linear = LoRALinear(
        linear.in_features,
        linear.out_features,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=linear.bias is not None,
    )
    
    # 复制原始权重
    lora_linear.weight.data.copy_(linear.weight.data)
    if linear.bias is not None:
        lora_linear.bias.data.copy_(linear.bias.data)
    
    return lora_linear


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型的参数量
    
    Args:
        model: 模型
    
    Returns:
        (总参数量, 可训练参数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class LoRAAttention(nn.Module):
    """
    带LoRA的Multi-Head Attention层示例
    
    展示如何将LoRA应用到Attention的Q、K、V、O投影上
    
    Args:
        dim: 模型维度
        n_heads: 注意力头数
        r: LoRA秩
        lora_alpha: 缩放因子
        lora_dropout: Dropout概率
        lora_targets: 要应用LoRA的目标，可选 ['q', 'k', 'v', 'o']
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_targets: List[str] = None,
    ):
        super().__init__()
        
        if lora_targets is None:
            lora_targets = ['q', 'v']  # 默认只对Q和V添加LoRA
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # 根据target决定是否使用LoRA
        # Q投影
        if 'q' in lora_targets:
            self.wq = LoRALinear(dim, dim, r=r, lora_alpha=lora_alpha, 
                                  lora_dropout=lora_dropout, bias=False)
        else:
            self.wq = nn.Linear(dim, dim, bias=False)
        
        # K投影
        if 'k' in lora_targets:
            self.wk = LoRALinear(dim, dim, r=r, lora_alpha=lora_alpha,
                                  lora_dropout=lora_dropout, bias=False)
        else:
            self.wk = nn.Linear(dim, dim, bias=False)
        
        # V投影
        if 'v' in lora_targets:
            self.wv = LoRALinear(dim, dim, r=r, lora_alpha=lora_alpha,
                                  lora_dropout=lora_dropout, bias=False)
        else:
            self.wv = nn.Linear(dim, dim, bias=False)
        
        # O投影
        if 'o' in lora_targets:
            self.wo = LoRALinear(dim, dim, r=r, lora_alpha=lora_alpha,
                                  lora_dropout=lora_dropout, bias=False)
        else:
            self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, 1, seq_len, seq_len] 或 None
        
        Returns:
            output: [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.shape
        
        # QKV投影
        q = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)
    
    def merge_lora(self):
        """合并所有LoRA权重"""
        for module in [self.wq, self.wk, self.wv, self.wo]:
            if isinstance(module, LoRALinear):
                module.merge()
    
    def unmerge_lora(self):
        """分离所有LoRA权重"""
        for module in [self.wq, self.wk, self.wv, self.wo]:
            if isinstance(module, LoRALinear):
                module.unmerge()


# QLoRA相关（量化LoRA）
class QuantizedLoRALinear(nn.Module):
    """
    量化版本的LoRA线性层（QLoRA的核心组件）
    
    QLoRA = 4-bit量化基础模型 + LoRA适配器
    
    注意：这是一个简化版本，实际QLoRA需要配合bitsandbytes库
    这里展示的是概念性实现
    
    关键思想：
        - 基础权重W用低精度（4-bit/8-bit）存储
        - LoRA参数A、B用全精度（fp16/bf16）存储
        - 推理时：Y = Dequantize(W_quant)X + (BA)X * scaling
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        bits: int = 8,  # 量化位数
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.bits = bits
        
        # 模拟量化权重（实际应使用专门的量化库）
        # 这里只是概念展示
        self.register_buffer('weight_quantized', 
                           torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        
        # LoRA参数（全精度）
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.scaling = lora_alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def quantize_weight(self, weight: torch.Tensor):
        """量化权重（简化版）"""
        max_val = weight.abs().max(dim=1, keepdim=True)[0]
        scale = max_val / (2 ** (self.bits - 1) - 1)
        weight_quantized = (weight / scale).round().clamp(
            -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        ).to(torch.int8)
        return weight_quantized, scale.squeeze()
    
    def dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        return self.weight_quantized.float() * self.weight_scale.unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化基础权重
        weight = self.dequantize_weight()
        # 基础计算
        result = F.linear(x, weight)
        # LoRA增量
        result = result + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


if __name__ == "__main__":
    print("=" * 70)
    print("LoRA (Low-Rank Adaptation) - 完整实现演示")
    print("=" * 70)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # =====================================================
    # 测试1: 基础LoRALinear
    # =====================================================
    print("\n" + "-" * 50)
    print("测试1: 基础LoRALinear层")
    print("-" * 50)
    
    in_features, out_features = 768, 768
    r = 8
    lora_alpha = 16
    
    lora_linear = LoRALinear(
        in_features, out_features,
        r=r, lora_alpha=lora_alpha
    ).to(device)
    
    x = torch.randn(2, 128, in_features, device=device)
    output = lora_linear(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"原始权重参数量: {in_features * out_features:,}")
    print(f"LoRA参数量: {r * in_features + r * out_features:,}")
    print(f"参数节省比例: {1 - (r * (in_features + out_features)) / (in_features * out_features):.2%}")
    
    # 测试权重合并
    print("\n测试权重合并与分离:")
    output_before = lora_linear(x)
    lora_linear.merge()
    output_after_merge = lora_linear(x)
    lora_linear.unmerge()
    output_after_unmerge = lora_linear(x)
    
    print(f"合并前后输出差异: {(output_before - output_after_merge).abs().max().item():.2e}")
    print(f"分离后输出差异: {(output_before - output_after_unmerge).abs().max().item():.2e}")
    
    # =====================================================
    # 测试2: 将普通Linear转换为LoRALinear
    # =====================================================
    print("\n" + "-" * 50)
    print("测试2: 将nn.Linear转换为LoRALinear")
    print("-" * 50)
    
    original_linear = nn.Linear(512, 512, bias=False).to(device)
    lora_converted = apply_lora_to_linear(original_linear, r=4, lora_alpha=8).to(device)
    
    x2 = torch.randn(1, 64, 512, device=device)
    
    # 初始时LoRA输出应该和原始相同（因为B初始化为0）
    original_output = original_linear(x2)
    lora_output = lora_converted(x2)
    
    print(f"转换后初始输出差异: {(original_output - lora_output).abs().max().item():.2e}")
    print("(差异应接近0，因为LoRA的B矩阵初始化为0)")
    
    # =====================================================
    # 测试3: LoRAAttention
    # =====================================================
    print("\n" + "-" * 50)
    print("测试3: LoRAAttention层")
    print("-" * 50)
    
    dim, n_heads = 512, 8
    lora_attn = LoRAAttention(
        dim=dim,
        n_heads=n_heads,
        r=8,
        lora_alpha=16,
        lora_targets=['q', 'v']  # 只对Q和V添加LoRA
    ).to(device)
    
    x3 = torch.randn(2, 64, dim, device=device)
    
    # 创建因果掩码
    seq_len = x3.shape[1]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    output3 = lora_attn(x3, mask=mask)
    print(f"输入形状: {x3.shape}")
    print(f"输出形状: {output3.shape}")
    
    # 统计参数
    total, trainable = count_parameters(lora_attn)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    print(f"可训练参数占比: {trainable/total:.2%}")
    
    # =====================================================
    # 测试4: 参数冻结与保存
    # =====================================================
    print("\n" + "-" * 50)
    print("测试4: 参数冻结与LoRA权重保存")
    print("-" * 50)
    
    # 冻结非LoRA参数
    mark_only_lora_as_trainable(lora_attn, bias='none')
    
    total_after, trainable_after = count_parameters(lora_attn)
    print(f"冻结后可训练参数量: {trainable_after:,}")
    print(f"冻结后可训练参数占比: {trainable_after/total_after:.2%}")
    
    # 保存LoRA权重
    lora_state = get_lora_state_dict(lora_attn)
    print(f"LoRA state_dict包含 {len(lora_state)} 个参数:")
    for name in lora_state.keys():
        print(f"  - {name}: {lora_state[name].shape}")
    
    # =====================================================
    # 测试5: LoRAEmbedding
    # =====================================================
    print("\n" + "-" * 50)
    print("测试5: LoRAEmbedding层")
    print("-" * 50)
    
    vocab_size, embed_dim = 32000, 512
    lora_embed = LoRAEmbedding(
        vocab_size, embed_dim,
        r=8, lora_alpha=16
    ).to(device)
    
    token_ids = torch.randint(0, vocab_size, (2, 128), device=device)
    embed_output = lora_embed(token_ids)
    
    print(f"输入形状: {token_ids.shape}")
    print(f"输出形状: {embed_output.shape}")
    print(f"原始Embedding参数量: {vocab_size * embed_dim:,}")
    print(f"LoRA参数量: {8 * vocab_size + 8 * embed_dim:,}")
    
    # =====================================================
    # 参数效率总结
    # =====================================================
    print("\n" + "=" * 70)
    print("LoRA参数效率分析")
    print("=" * 70)
    
    # 假设一个7B参数的LLM
    print("\n假设场景: 7B参数LLM微调")
    print("-" * 50)
    
    # 典型LLaMA-7B配置
    llm_dim = 4096
    llm_heads = 32
    llm_layers = 32
    
    # 每层的Q、V投影参数
    qv_params_per_layer = 2 * (llm_dim * llm_dim)  # Q和V各一个
    total_qv_params = qv_params_per_layer * llm_layers
    
    # LoRA参数（r=8）
    lora_r = 8
    lora_params_per_layer = 2 * (lora_r * llm_dim + lora_r * llm_dim)  # A和B
    total_lora_params = lora_params_per_layer * llm_layers
    
    print(f"原始Q、V参数量: {total_qv_params:,} ({total_qv_params/1e9:.2f}B)")
    print(f"LoRA参数量 (r={lora_r}): {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    print(f"参数减少比例: {1 - total_lora_params/total_qv_params:.4%}")
    print(f"\n如果使用r=16: {2 * total_lora_params:,} ({2 * total_lora_params/1e6:.2f}M)")
    print(f"如果使用r=32: {4 * total_lora_params:,} ({4 * total_lora_params/1e6:.2f}M)")
    
    print("\n" + "=" * 70)
    print("LoRA模块测试完成！")
    print("=" * 70)

