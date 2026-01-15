# KV Cache Implementation for LLM Inference
# KV Cache是大语言模型推理加速的核心技术
# 通过缓存已计算的Key和Value，避免重复计算

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class CacheConfig:
    """KV Cache配置"""
    max_batch_size: int = 32          # 最大批量大小
    max_seq_len: int = 4096           # 最大序列长度
    n_heads: int = 32                 # 注意力头数
    head_dim: int = 128               # 每个头的维度
    n_kv_heads: Optional[int] = None  # KV头数（GQA时使用）
    dtype: torch.dtype = torch.float32  # 默认float32以兼容CPU


#############################################################
########### 静态KV Cache (Static KV Cache) ##################
#############################################################
class StaticKVCache(nn.Module):
    """
    静态KV Cache - 预分配固定大小的缓存
    
    特点：
    - 预先分配固定大小的内存，避免动态分配开销
    - 适用于序列长度可预测的场景
    - 内存使用固定，无碎片化问题
    
    缓存形状: [batch_size, max_seq_len, n_heads, head_dim]
    
    工作原理：
    训练时:  Q, K, V 都是完整序列 [B, S, H, D]
    推理时:  
        Step 1: 处理prompt，存储K,V到cache [B, prompt_len, H, D]
        Step 2: 生成token 1，只计算新token的Q,K,V，
                新K,V追加到cache，Q与完整cache做attention
        Step N: 重复Step 2，cache逐步增长
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__()
        self.config = config
        n_kv_heads = config.n_kv_heads or config.n_heads
        
        # 预分配缓存 [batch, max_seq, n_kv_heads, head_dim]
        self.register_buffer(
            "cache_k",
            torch.zeros(
                config.max_batch_size,
                config.max_seq_len,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                config.max_batch_size,
                config.max_seq_len,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        
        # 当前序列位置
        self.seq_pos = 0
        
    def update(
        self,
        key: torch.Tensor,          # [batch, seq_len, n_kv_heads, head_dim]
        value: torch.Tensor,        # [batch, seq_len, n_kv_heads, head_dim]
        start_pos: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存并返回完整的K,V序列
        
        Args:
            key: 新的Key张量
            value: 新的Value张量
            start_pos: 起始位置（如果为None，则使用内部计数器）
            
        Returns:
            cached_k: 完整的Key缓存 [batch, cache_len, n_kv_heads, head_dim]
            cached_v: 完整的Value缓存 [batch, cache_len, n_kv_heads, head_dim]
        """
        batch_size, seq_len = key.shape[:2]
        
        if start_pos is None:
            start_pos = self.seq_pos
        
        # 确保缓存与输入数据类型和设备一致
        if self.cache_k.dtype != key.dtype or self.cache_k.device != key.device:
            self.cache_k = self.cache_k.to(dtype=key.dtype, device=key.device)
            self.cache_v = self.cache_v.to(dtype=value.dtype, device=value.device)
            
        # 将新的K,V写入缓存
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = key
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = value
        
        # 更新位置计数器
        self.seq_pos = start_pos + seq_len
        
        # 返回截止到当前位置的缓存
        return (
            self.cache_k[:batch_size, :self.seq_pos],
            self.cache_v[:batch_size, :self.seq_pos]
        )
    
    def reset(self):
        """重置缓存"""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.seq_pos = 0
        
    def get_seq_length(self) -> int:
        """获取当前缓存的序列长度"""
        return self.seq_pos


#############################################################
########### 动态KV Cache (Dynamic KV Cache) #################
#############################################################
class DynamicKVCache(nn.Module):
    """
    动态KV Cache - 按需增长的缓存
    
    特点：
    - 根据实际序列长度动态分配内存
    - 内存效率更高（不预分配最大长度）
    - 适用于序列长度变化大的场景
    - 有动态分配开销
    
    实现方式：使用Python List存储，按需concatenate
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__()
        self.config = config
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        
        # 使用List动态存储
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        
    def update(
        self,
        key: torch.Tensor,          # [batch, seq_len, n_kv_heads, head_dim]
        value: torch.Tensor,        # [batch, seq_len, n_kv_heads, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存并返回完整的K,V序列
        
        Args:
            key: 新的Key张量
            value: 新的Value张量
            
        Returns:
            cached_k: 完整的Key缓存
            cached_v: 完整的Value缓存
        """
        if self.cache_k is None:
            # 第一次调用，直接使用输入
            self.cache_k = key
            self.cache_v = value
        else:
            # 拼接新的K,V到缓存
            self.cache_k = torch.cat([self.cache_k, key], dim=1)
            self.cache_v = torch.cat([self.cache_v, value], dim=1)
            
        return self.cache_k, self.cache_v
    
    def reset(self):
        """重置缓存"""
        self.cache_k = None
        self.cache_v = None
        
    def get_seq_length(self) -> int:
        """获取当前缓存的序列长度"""
        if self.cache_k is None:
            return 0
        return self.cache_k.shape[1]


#############################################################
########### 滑动窗口KV Cache (Sliding Window Cache) #########
#############################################################
class SlidingWindowCache(nn.Module):
    """
    滑动窗口KV Cache - 只保留最近的N个token
    
    特点：
    - 固定大小的滑动窗口，内存使用恒定
    - 适用于长序列生成（如Mistral）
    - 窗口外的信息会被丢弃
    - 结合局部注意力使用效果更好
    
    实现：使用环形缓冲区（Ring Buffer）
    
    注意：滑动窗口注意力在某些情况下可能会丢失长距离依赖，
    但在实践中，结合多层堆叠可以有效扩展感受野。
    """
    
    def __init__(self, config: CacheConfig, window_size: int = 4096):
        super().__init__()
        self.config = config
        self.window_size = window_size
        n_kv_heads = config.n_kv_heads or config.n_heads
        
        # 预分配窗口大小的缓存
        self.register_buffer(
            "cache_k",
            torch.zeros(
                config.max_batch_size,
                window_size,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        self.register_buffer(
            "cache_v",
            torch.zeros(
                config.max_batch_size,
                window_size,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        
        # 当前写入位置（环形缓冲区指针）
        self.write_pos = 0
        # 缓存中的有效token数量
        self.filled_length = 0
        
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新滑动窗口缓存
        
        Args:
            key: 新的Key张量 [batch, seq_len, n_kv_heads, head_dim]
            value: 新的Value张量
            
        Returns:
            cached_k: 窗口内的Key缓存
            cached_v: 窗口内的Value缓存
        """
        batch_size, seq_len = key.shape[:2]
        
        # 确保缓存与输入数据类型和设备一致
        if self.cache_k.dtype != key.dtype or self.cache_k.device != key.device:
            self.cache_k = self.cache_k.to(dtype=key.dtype, device=key.device)
            self.cache_v = self.cache_v.to(dtype=value.dtype, device=value.device)
        
        # 处理输入序列
        for i in range(seq_len):
            # 写入环形缓冲区
            pos = self.write_pos % self.window_size
            self.cache_k[:batch_size, pos] = key[:, i]
            self.cache_v[:batch_size, pos] = value[:, i]
            self.write_pos += 1
            self.filled_length = min(self.filled_length + 1, self.window_size)
            
        # 返回有效部分的缓存
        if self.filled_length < self.window_size:
            # 缓存未满，直接返回填充部分
            return (
                self.cache_k[:batch_size, :self.filled_length],
                self.cache_v[:batch_size, :self.filled_length]
            )
        else:
            # 缓存已满，需要重新排列成正确顺序
            # 环形缓冲区的起始位置
            start = self.write_pos % self.window_size
            indices = torch.cat([
                torch.arange(start, self.window_size),
                torch.arange(0, start)
            ])
            return (
                self.cache_k[:batch_size].index_select(1, indices.to(self.cache_k.device)),
                self.cache_v[:batch_size].index_select(1, indices.to(self.cache_v.device))
            )
    
    def reset(self):
        """重置缓存"""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.write_pos = 0
        self.filled_length = 0
        
    def get_seq_length(self) -> int:
        """获取当前缓存的有效长度"""
        return self.filled_length


#############################################################
########### 分页KV Cache (Paged KV Cache) ###################
#############################################################
class PagedKVCache(nn.Module):
    """
    分页KV Cache - 类似vLLM的PagedAttention简化版
    
    特点：
    - 将缓存分成固定大小的页（Block）
    - 支持非连续内存分配
    - 更好的内存利用率，减少碎片化
    - 支持多请求共享KV Cache（如beam search）
    
    核心思想：
    - 传统Cache: 每个序列预分配max_seq_len的连续内存
    - Paged Cache: 按需分配固定大小的页，页可以不连续
    
    内存结构：
    - Block Pool: 预分配的内存池，包含多个Block
    - Block Table: 记录每个序列使用哪些Block
    
    优势：
    1. 减少内存浪费（不需要按max_seq_len预分配）
    2. 支持共享（beam search中多个候选可共享prefix的Block）
    3. 更好的批处理（不同长度的序列可以高效batching）
    """
    
    def __init__(
        self,
        config: CacheConfig,
        block_size: int = 16,       # 每个block包含的token数
        num_blocks: int = 1024      # 总block数量
    ):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_blocks = num_blocks
        n_kv_heads = config.n_kv_heads or config.n_heads
        
        # Block Pool: [num_blocks, block_size, n_kv_heads, head_dim]
        self.register_buffer(
            "key_cache",
            torch.zeros(
                num_blocks,
                block_size,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(
                num_blocks,
                block_size,
                n_kv_heads,
                config.head_dim,
                dtype=config.dtype
            )
        )
        
        # 空闲block列表
        self.free_blocks: List[int] = list(range(num_blocks))
        
        # Block Table: 每个序列的block映射 {seq_id: [block_ids]}
        self.block_tables: dict = {}
        
        # 每个序列的当前长度
        self.seq_lengths: dict = {}
        
    def allocate_blocks(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        为序列分配所需的blocks
        
        Args:
            seq_id: 序列ID
            num_tokens: 需要存储的token数量
            
        Returns:
            分配的block ID列表
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
            self.seq_lengths[seq_id] = 0
            
        # 计算需要新分配的block数量
        current_blocks = len(self.block_tables[seq_id])
        new_blocks_needed = max(0, num_blocks_needed - current_blocks)
        
        if new_blocks_needed > len(self.free_blocks):
            raise RuntimeError(f"Not enough free blocks. Need {new_blocks_needed}, have {len(self.free_blocks)}")
            
        # 分配新blocks
        for _ in range(new_blocks_needed):
            block_id = self.free_blocks.pop()
            self.block_tables[seq_id].append(block_id)
            
        return self.block_tables[seq_id]
    
    def update(
        self,
        seq_id: int,
        key: torch.Tensor,          # [seq_len, n_kv_heads, head_dim]
        value: torch.Tensor,        # [seq_len, n_kv_heads, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新分页缓存
        
        Args:
            seq_id: 序列ID
            key: 新的Key张量（单个序列）
            value: 新的Value张量
            
        Returns:
            cached_k: 完整的Key缓存
            cached_v: 完整的Value缓存
        """
        seq_len = key.shape[0]
        start_pos = self.seq_lengths.get(seq_id, 0)
        total_len = start_pos + seq_len
        
        # 分配blocks
        block_ids = self.allocate_blocks(seq_id, total_len)
        
        # 写入数据
        for i in range(seq_len):
            pos = start_pos + i
            block_idx = pos // self.block_size
            offset = pos % self.block_size
            block_id = block_ids[block_idx]
            
            self.key_cache[block_id, offset] = key[i]
            self.value_cache[block_id, offset] = value[i]
            
        self.seq_lengths[seq_id] = total_len
        
        # 收集完整的缓存
        cached_k = self._gather_cache(seq_id, self.key_cache)
        cached_v = self._gather_cache(seq_id, self.value_cache)
        
        return cached_k, cached_v
    
    def _gather_cache(self, seq_id: int, cache: torch.Tensor) -> torch.Tensor:
        """从分页缓存中收集连续的张量"""
        block_ids = self.block_tables[seq_id]
        seq_len = self.seq_lengths[seq_id]
        
        # 收集所有blocks的数据
        all_data = []
        for block_id in block_ids:
            all_data.append(cache[block_id])
            
        # 拼接并截断到实际长度
        gathered = torch.cat(all_data, dim=0)[:seq_len]
        return gathered
    
    def free_sequence(self, seq_id: int):
        """释放序列占用的blocks"""
        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id]:
                self.free_blocks.append(block_id)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]
            
    def reset(self):
        """重置所有缓存"""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.free_blocks = list(range(self.num_blocks))
        self.block_tables.clear()
        self.seq_lengths.clear()


#############################################################
########### 带KV Cache的注意力计算 ###########################
#############################################################
class AttentionWithKVCache(nn.Module):
    """
    带KV Cache的多头注意力实现
    
    演示如何在注意力计算中使用KV Cache
    
    计算流程：
    1. 计算Q, K, V投影
    2. 对K, V应用旋转位置编码（可选）
    3. 更新KV Cache
    4. 计算注意力分数 Q @ K^T / sqrt(d)
    5. 应用mask
    6. Softmax + V加权求和
    7. 输出投影
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        max_batch_size: int = 32,
        cache_type: str = "static"  # "static", "dynamic", "sliding"
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // self.n_kv_heads  # GQA重复因子
        
        # QKV投影
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
        # 创建KV Cache
        config = CacheConfig(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            head_dim=self.head_dim,
            n_kv_heads=self.n_kv_heads
        )
        
        if cache_type == "static":
            self.cache = StaticKVCache(config)
        elif cache_type == "dynamic":
            self.cache = DynamicKVCache(config)
        elif cache_type == "sliding":
            self.cache = SlidingWindowCache(config, window_size=4096)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        self.cache_type = cache_type
        
    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        为GQA重复KV头
        
        Args:
            x: [batch, seq_len, n_kv_heads, head_dim]
            
        Returns:
            [batch, seq_len, n_heads, head_dim]
        """
        if self.n_rep == 1:
            return x
        batch, seq_len, n_kv_heads, head_dim = x.shape
        x = x[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)
        
    def forward(
        self,
        x: torch.Tensor,                    # [batch, seq_len, dim]
        start_pos: int = 0,                 # 用于静态缓存
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, dim]
            start_pos: 序列起始位置（静态缓存使用）
            mask: 注意力掩码
            
        Returns:
            output: [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV投影
        q = self.wq(x)  # [batch, seq_len, n_heads * head_dim]
        k = self.wk(x)  # [batch, seq_len, n_kv_heads * head_dim]
        v = self.wv(x)  # [batch, seq_len, n_kv_heads * head_dim]
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 更新KV Cache
        if isinstance(self.cache, StaticKVCache):
            k, v = self.cache.update(k, v, start_pos)
        else:
            k, v = self.cache.update(k, v)
            
        # 为GQA重复KV
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)
        
        # 转置用于注意力计算: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用mask
        if mask is not None:
            scores = scores + mask
            
        # Softmax
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 加权求和
        output = torch.matmul(attn, v)
        
        # 合并头并输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        
        return output
    
    def reset_cache(self):
        """重置KV Cache"""
        self.cache.reset()


#############################################################
########### 工具函数 ########################################
#############################################################
def create_causal_mask(
    seq_len: int,
    cache_len: int = 0,
    device: torch.device = None
) -> torch.Tensor:
    """
    创建因果注意力掩码
    
    Args:
        seq_len: 当前序列长度
        cache_len: 缓存长度
        device: 设备
        
    Returns:
        mask: [1, 1, seq_len, cache_len + seq_len]
    """
    total_len = cache_len + seq_len
    mask = torch.full(
        (seq_len, total_len),
        float("-inf"),
        device=device
    )
    # 因果掩码：只能看到当前位置及之前的token
    mask = torch.triu(mask, diagonal=cache_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16
) -> str:
    """
    估算KV Cache的内存占用
    
    公式: 2 * batch_size * seq_len * n_layers * n_kv_heads * head_dim * bytes_per_element
    
    2 是因为要存储 K 和 V 两个缓存
    
    Args:
        batch_size: 批量大小
        seq_len: 序列长度
        n_layers: 层数
        n_kv_heads: KV头数
        head_dim: 头维度
        dtype: 数据类型
        
    Returns:
        内存占用的字符串描述
    """
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }.get(dtype, 2)
    
    # K和V各需要的内存
    kv_cache_bytes = 2 * batch_size * seq_len * n_layers * n_kv_heads * head_dim * bytes_per_element
    
    # 转换为合适的单位
    if kv_cache_bytes >= 1024 ** 3:
        return f"{kv_cache_bytes / (1024 ** 3):.2f} GB"
    elif kv_cache_bytes >= 1024 ** 2:
        return f"{kv_cache_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{kv_cache_bytes / 1024:.2f} KB"


#############################################################
########### 测试代码 ########################################
#############################################################
if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache Implementation - Test and Demonstration")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 测试配置
    batch_size = 2
    n_heads = 8
    n_kv_heads = 2  # GQA配置
    head_dim = 64
    dim = n_heads * head_dim
    
    config = CacheConfig(
        max_batch_size=4,
        max_seq_len=1024,
        n_heads=n_heads,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads
    )
    
    # ===================== 测试静态KV Cache =====================
    print("\n" + "-" * 50)
    print("Testing Static KV Cache")
    print("-" * 50)
    
    static_cache = StaticKVCache(config).to(device)
    
    # 模拟prefill阶段（处理prompt）
    prompt_len = 32
    k1 = torch.randn(batch_size, prompt_len, n_kv_heads, head_dim, device=device)
    v1 = torch.randn(batch_size, prompt_len, n_kv_heads, head_dim, device=device)
    
    cached_k, cached_v = static_cache.update(k1, v1, start_pos=0)
    print(f"After prefill: cache shape = {cached_k.shape}")
    
    # 模拟decode阶段（逐token生成）
    for step in range(3):
        k_new = torch.randn(batch_size, 1, n_kv_heads, head_dim, device=device)
        v_new = torch.randn(batch_size, 1, n_kv_heads, head_dim, device=device)
        cached_k, cached_v = static_cache.update(k_new, v_new)
        print(f"After decode step {step + 1}: cache length = {static_cache.get_seq_length()}")
    
    # ===================== 测试动态KV Cache =====================
    print("\n" + "-" * 50)
    print("Testing Dynamic KV Cache")
    print("-" * 50)
    
    dynamic_cache = DynamicKVCache(config)
    
    # Prefill
    cached_k, cached_v = dynamic_cache.update(k1.cpu(), v1.cpu())
    print(f"After prefill: cache shape = {cached_k.shape}")
    
    # Decode
    for step in range(3):
        k_new = torch.randn(batch_size, 1, n_kv_heads, head_dim)
        v_new = torch.randn(batch_size, 1, n_kv_heads, head_dim)
        cached_k, cached_v = dynamic_cache.update(k_new, v_new)
        print(f"After decode step {step + 1}: cache length = {dynamic_cache.get_seq_length()}")
    
    # ===================== 测试滑动窗口Cache =====================
    print("\n" + "-" * 50)
    print("Testing Sliding Window KV Cache (window=64)")
    print("-" * 50)
    
    sliding_cache = SlidingWindowCache(config, window_size=64).to(device)
    
    # 测试超过窗口大小的情况
    for i in range(100):
        k = torch.randn(batch_size, 1, n_kv_heads, head_dim, device=device)
        v = torch.randn(batch_size, 1, n_kv_heads, head_dim, device=device)
        cached_k, cached_v = sliding_cache.update(k, v)
        if i % 20 == 0:
            print(f"Step {i}: cache length = {sliding_cache.get_seq_length()}, shape = {cached_k.shape}")
    
    # ===================== 测试完整的注意力模块 =====================
    print("\n" + "-" * 50)
    print("Testing Attention with KV Cache")
    print("-" * 50)
    
    for cache_type in ["static", "dynamic", "sliding"]:
        print(f"\n>>> Cache type: {cache_type}")
        
        attn = AttentionWithKVCache(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=128,
            max_batch_size=4,
            cache_type=cache_type
        ).to(device)
        
        # Prefill
        x = torch.randn(batch_size, 32, dim, device=device)
        mask = create_causal_mask(32, 0, device)
        output = attn(x, start_pos=0, mask=mask)
        print(f"  Prefill output shape: {output.shape}")
        
        # Decode
        for step in range(3):
            x_new = torch.randn(batch_size, 1, dim, device=device)
            cache_len = attn.cache.get_seq_length()
            mask = create_causal_mask(1, cache_len, device)
            output = attn(x_new, start_pos=cache_len, mask=mask)
        print(f"  After 3 decode steps, cache length: {attn.cache.get_seq_length()}")
        
        attn.reset_cache()
    
    # ===================== 内存估算 =====================
    print("\n" + "-" * 50)
    print("KV Cache Memory Estimation")
    print("-" * 50)
    
    # 模拟LLaMA 7B的参数
    print("\nLLaMA 7B-like model (32 layers, 32 heads, head_dim=128):")
    for seq_len in [512, 2048, 4096, 8192]:
        mem = estimate_kv_cache_memory(
            batch_size=1,
            seq_len=seq_len,
            n_layers=32,
            n_kv_heads=32,
            head_dim=128,
            dtype=torch.float16
        )
        print(f"  seq_len={seq_len}: {mem}")
    
    # 模拟LLaMA 70B的参数（GQA，n_kv_heads=8）
    print("\nLLaMA 70B-like model with GQA (80 layers, 8 KV heads, head_dim=128):")
    for seq_len in [512, 2048, 4096, 8192]:
        mem = estimate_kv_cache_memory(
            batch_size=1,
            seq_len=seq_len,
            n_layers=80,
            n_kv_heads=8,
            head_dim=128,
            dtype=torch.float16
        )
        print(f"  seq_len={seq_len}: {mem}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

