# KV Cache - LLM推理加速的核心技术

## 1. 什么是KV Cache？

KV Cache（Key-Value Cache）是大语言模型（LLM）推理过程中最重要的优化技术之一。它通过缓存已计算的 Key 和 Value 张量，避免在自回归生成时重复计算历史token的 K 和 V。

### 1.1 为什么需要KV Cache？

在自回归语言模型中，生成每个新token时都需要对所有历史token进行注意力计算：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**不使用KV Cache的情况**：

```
Step 1: 输入 [t1]           → 计算 Q1, K1, V1
Step 2: 输入 [t1, t2]       → 重新计算 Q1, K1, V1, Q2, K2, V2  ❌ 重复计算
Step 3: 输入 [t1, t2, t3]   → 重新计算所有                      ❌ 重复计算
```

**使用KV Cache的情况**：

```
Step 1: 输入 [t1]           → 计算并缓存 K1, V1
Step 2: 输入 [t2]           → 计算 K2, V2，从缓存获取 K1, V1   ✅ 无重复计算
Step 3: 输入 [t3]           → 计算 K3, V3，从缓存获取 K1:2, V1:2 ✅ 无重复计算
```

### 1.2 计算复杂度对比

| 方法 | 生成 N 个 token 的复杂度 |
|------|------------------------|
| 无 KV Cache | O(N³) |
| 有 KV Cache | O(N²) |

节省的计算量随序列长度增加而显著增加！

## 2. KV Cache的工作原理

### 2.1 推理的两个阶段

LLM推理分为两个阶段：

**Prefill（预填充）阶段**：
- 处理完整的输入prompt
- 一次性计算所有token的 Q, K, V
- 将 K, V 存入缓存

**Decode（解码）阶段**：
- 逐个生成新token
- 只计算新token的 Q, K, V
- 新的 K, V 追加到缓存
- Q 与完整的缓存 K, V 做注意力计算

```
┌─────────────────────────────────────────────────────────────┐
│                         Prefill Phase                        │
│  Input: "What is the capital of France?"                    │
│  Output: K_cache, V_cache for all input tokens              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         Decode Phase                         │
│  Step 1: Generate "The" → append K, V to cache              │
│  Step 2: Generate "capital" → append K, V to cache          │
│  Step 3: Generate "is" → append K, V to cache               │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 缓存的形状

```python
# 单层的KV Cache形状
cache_k: [batch_size, seq_len, n_kv_heads, head_dim]
cache_v: [batch_size, seq_len, n_kv_heads, head_dim]

# 完整模型的KV Cache（所有层）
total_cache: [2, n_layers, batch_size, seq_len, n_kv_heads, head_dim]
#             ↑
#          K和V两个缓存
```

## 3. KV Cache的实现策略

### 3.1 静态KV Cache（Static KV Cache）

**特点**：
- 预分配固定大小的内存（max_batch_size × max_seq_len）
- 内存使用固定，无分配开销
- 简单高效，但可能浪费内存

```python
class StaticKVCache:
    def __init__(self, max_batch, max_seq, n_heads, head_dim):
        # 预分配最大容量
        self.cache_k = torch.zeros(max_batch, max_seq, n_heads, head_dim)
        self.cache_v = torch.zeros(max_batch, max_seq, n_heads, head_dim)
        self.seq_pos = 0
    
    def update(self, k, v, start_pos):
        # 写入缓存的指定位置
        seq_len = k.shape[1]
        self.cache_k[:, start_pos:start_pos+seq_len] = k
        self.cache_v[:, start_pos:start_pos+seq_len] = v
        return self.cache_k[:, :start_pos+seq_len], self.cache_v[:, :start_pos+seq_len]
```

**适用场景**：
- 序列长度可预测
- 批处理大小固定
- 追求最低延迟

### 3.2 动态KV Cache（Dynamic KV Cache）

**特点**：
- 按需分配内存
- 内存效率更高
- 有动态分配开销

```python
class DynamicKVCache:
    def __init__(self):
        self.cache_k = None
        self.cache_v = None
    
    def update(self, k, v):
        if self.cache_k is None:
            self.cache_k = k
            self.cache_v = v
        else:
            # 动态拼接
            self.cache_k = torch.cat([self.cache_k, k], dim=1)
            self.cache_v = torch.cat([self.cache_v, v], dim=1)
        return self.cache_k, self.cache_v
```

**适用场景**：
- 序列长度变化大
- 内存受限
- 不追求极致性能

### 3.3 滑动窗口KV Cache（Sliding Window Cache）

**特点**：
- 只保留最近 W 个token的 K, V
- 内存使用恒定 O(W)
- 适用于超长序列
- Mistral等模型使用此技术

```python
class SlidingWindowCache:
    def __init__(self, window_size):
        self.window_size = window_size
        self.cache_k = torch.zeros(batch, window_size, n_heads, head_dim)
        self.cache_v = torch.zeros(batch, window_size, n_heads, head_dim)
        self.write_pos = 0  # 环形缓冲区指针
```

**原理**：
```
窗口大小 W = 4:

Step 5: [t1, t2, t3, t4, t5] → 缓存 [t2, t3, t4, t5]  # t1被丢弃
Step 6: [t1, t2, t3, t4, t5, t6] → 缓存 [t3, t4, t5, t6]  # t2被丢弃
```

**适用场景**：
- 超长序列生成
- 内存受限
- 任务不需要全局上下文

### 3.4 分页KV Cache（Paged KV Cache / vLLM）

**特点**：
- 将缓存分成固定大小的页（Block）
- 按需分配页，减少内存碎片
- 支持多请求共享缓存（如beam search）
- vLLM的核心技术

```python
class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=1024):
        # 预分配Block Pool
        self.key_cache = torch.zeros(num_blocks, block_size, n_heads, head_dim)
        self.value_cache = torch.zeros(num_blocks, block_size, n_heads, head_dim)
        
        # Block Table: 记录每个序列使用哪些block
        self.block_tables = {}  # {seq_id: [block_ids]}
```

**内存布局对比**：
```
传统缓存（连续内存）:
┌────────────────────────────────────┐
│ Seq 1: [████████____________]      │  # 浪费空间
│ Seq 2: [████████████________]      │  # 浪费空间
│ Seq 3: [████________________]      │  # 浪费空间
└────────────────────────────────────┘

分页缓存（非连续内存）:
┌────────────────────────────────────┐
│ Block Pool: [B0][B1][B2][B3][B4]...│
│ Seq 1 → [B0, B2]                   │  # 按需分配
│ Seq 2 → [B1, B3, B5]               │  # 无浪费
│ Seq 3 → [B4]                       │
└────────────────────────────────────┘
```

**适用场景**：
- 高吞吐量服务
- 动态batch调度
- 内存利用率要求高

## 4. KV Cache的内存占用

### 4.1 计算公式

```
KV Cache内存 = 2 × batch_size × seq_len × n_layers × n_kv_heads × head_dim × bytes
               ↑
            K和V两个缓存
```

### 4.2 实际模型的内存占用（FP16）

| 模型 | 层数 | KV头数 | head_dim | 序列长度 | KV Cache内存 |
|------|------|--------|----------|----------|-------------|
| LLaMA 7B | 32 | 32 | 128 | 4K | 4 GB |
| LLaMA 13B | 40 | 40 | 128 | 4K | 6.25 GB |
| LLaMA 70B (GQA) | 80 | 8 | 128 | 4K | 2.5 GB |
| Mistral 7B | 32 | 8 | 128 | 32K | 8 GB |

### 4.3 GQA对内存的影响

Grouped Query Attention (GQA) 通过减少 KV 头数来降低缓存大小：

```
MHA (n_kv_heads = n_heads):     KV Cache = 100%
GQA (n_kv_heads = n_heads/4):   KV Cache = 25%   ← 节省75%
MQA (n_kv_heads = 1):           KV Cache = 3.125% ← 节省96.875%
```

## 5. 优化技术

### 5.1 量化KV Cache

将 K, V 从 FP16 量化到 INT8 或 INT4：

```python
# FP16 → INT8: 内存减半
cache_k_int8 = torch.quantize_per_tensor(cache_k, scale, zero_point, torch.qint8)

# FP16 → INT4: 内存减少75%
cache_k_int4 = quantize_to_int4(cache_k)  # 自定义实现
```

### 5.2 KV Cache压缩

- **H2O (Heavy-Hitter Oracle)**: 只保留"重要"的token
- **StreamingLLM**: 保留开头token + 滑动窗口
- **Scissorhands**: 基于注意力分数动态剪枝

### 5.3 Speculative Decoding

- 使用小模型生成多个候选token
- 大模型验证并一次性处理多个token
- 减少decode步骤数

## 6. 代码示例

### 6.1 基本使用

```python
from KVCache import StaticKVCache, CacheConfig, AttentionWithKVCache

# 创建配置
config = CacheConfig(
    max_batch_size=4,
    max_seq_len=2048,
    n_heads=32,
    head_dim=128,
    n_kv_heads=8  # GQA
)

# 创建注意力模块
attn = AttentionWithKVCache(
    dim=4096,
    n_heads=32,
    n_kv_heads=8,
    cache_type="static"
)

# Prefill
prompt = torch.randn(1, 128, 4096)
output = attn(prompt, start_pos=0)

# Decode
for _ in range(100):
    new_token = torch.randn(1, 1, 4096)
    cache_len = attn.cache.get_seq_length()
    output = attn(new_token, start_pos=cache_len)
```

### 6.2 内存估算

```python
from KVCache import estimate_kv_cache_memory

# 估算LLaMA 7B的KV Cache内存
mem = estimate_kv_cache_memory(
    batch_size=1,
    seq_len=4096,
    n_layers=32,
    n_kv_heads=32,
    head_dim=128,
    dtype=torch.float16
)
print(f"KV Cache memory: {mem}")  # 约4 GB
```

## 7. 总结

| 缓存类型 | 内存效率 | 延迟 | 实现复杂度 | 适用场景 |
|---------|---------|------|-----------|---------|
| 静态 | 低 | 最低 | 简单 | 固定长度，低延迟 |
| 动态 | 中 | 中 | 简单 | 通用场景 |
| 滑动窗口 | 高 | 低 | 中等 | 超长序列 |
| 分页 | 最高 | 中 | 复杂 | 高吞吐服务 |

**最佳实践**：
1. 对于单请求低延迟场景，使用静态KV Cache
2. 对于高吞吐服务场景，使用分页KV Cache（vLLM）
3. 对于超长序列，考虑滑动窗口或压缩技术
4. 尽可能使用GQA减少KV头数
5. 考虑量化KV Cache进一步减少内存

## 参考资料

1. [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
2. [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245)
3. [Mistral: Sliding Window Attention](https://arxiv.org/abs/2310.06825)
4. [StreamingLLM](https://arxiv.org/abs/2309.17453)
5. [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)

