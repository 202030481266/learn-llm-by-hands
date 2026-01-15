# LoRA (Low-Rank Adaptation) 低秩适应

## 📚 论文信息

- **论文**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **作者**: Edward J. Hu, Yelong Shen, Phillip Wallis, et al. (Microsoft)
- **发布时间**: 2021年6月

## 🎯 核心思想

LoRA的核心洞察是：**预训练模型的权重更新具有低"内在秩"（low intrinsic rank）**。

传统微调需要更新完整的权重矩阵 $W \in \mathbb{R}^{d \times k}$，参数量巨大。LoRA提出用低秩分解来近似权重更新：

$$W' = W + \Delta W = W + BA$$

其中：
- $W$: 原始预训练权重，**冻结不更新**
- $B \in \mathbb{R}^{d \times r}$: 低秩矩阵
- $A \in \mathbb{R}^{r \times k}$: 低秩矩阵  
- $r \ll \min(d, k)$: 秩（通常 $r = 4, 8, 16, 32$）

## 🔍 数学原理

### 前向传播

对于输入 $x$，原始线性层输出为：
$$h = Wx$$

添加LoRA后：
$$h = Wx + \Delta Wx = Wx + BAx = (W + BA)x$$

为了控制更新幅度，引入缩放因子 $\alpha$：
$$h = Wx + \frac{\alpha}{r}BAx$$

### 参数初始化

- **A矩阵**: 使用Kaiming/He初始化（高斯分布）
- **B矩阵**: **初始化为零**

这样确保训练开始时 $\Delta W = BA = 0$，不改变原始模型的行为。

### 参数量对比

| 方法 | 参数量 | 示例 (d=4096, k=4096) |
|------|--------|----------------------|
| 全量微调 | $d \times k$ | 16.8M |
| LoRA (r=8) | $r \times (d + k)$ | 65.5K |
| LoRA (r=16) | $r \times (d + k)$ | 131K |
| LoRA (r=32) | $r \times (d + k)$ | 262K |

**节省比例**: 使用 $r=8$ 时，参数量减少 **99.6%**！

## 🏗️ 模块结构

```
                    ┌─────────────────┐
                    │   输入 x        │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 │                 ▼
    ┌──────────────┐         │         ┌──────────────┐
    │  冻结的 W    │         │         │   A (r×k)    │ ← 可训练
    │  (d×k)       │         │         └──────┬───────┘
    └──────┬───────┘         │                │
           │                 │                ▼
           │                 │         ┌──────────────┐
           │                 │         │   B (d×r)    │ ← 可训练
           │                 │         └──────┬───────┘
           │                 │                │
           │                 │                │ × α/r
           │                 │                │
           └────────┬────────┴────────┬───────┘
                    │      相加       │
                    ▼                 ▼
              ┌─────────────────────────┐
              │      输出 h = Wx + BAx  │
              └─────────────────────────┘
```

## 📝 代码实现

### 核心LoRA线性层

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        # 原始权重（冻结）
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False
        
        # LoRA参数
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # 初始化为0
        
        self.scaling = lora_alpha / r
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A)
    
    def forward(self, x):
        # 原始输出 + LoRA增量
        result = F.linear(x, self.weight)
        result += (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result
```

### 权重合并（推理优化）

训练完成后，可以将LoRA权重合并到原始权重中，**推理时无额外开销**：

```python
def merge(self):
    """W' = W + scaling * BA"""
    self.weight.data += self.scaling * (self.lora_B @ self.lora_A)

def unmerge(self):
    """W = W' - scaling * BA"""
    self.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
```

## 🎮 应用场景

### 在Attention中应用LoRA

通常只对 **Query (Q)** 和 **Value (V)** 投影添加LoRA：

```python
# 典型配置
lora_targets = ['q', 'v']  # 只对Q和V添加LoRA

# Q投影
self.wq = LoRALinear(dim, dim, r=8, lora_alpha=16)
# V投影  
self.wv = LoRALinear(dim, dim, r=8, lora_alpha=16)
# K和O保持原始
self.wk = nn.Linear(dim, dim)
self.wo = nn.Linear(dim, dim)
```

**为什么选择Q和V？**
- 原论文实验表明，只训练Q和V效果最好
- K投影对性能影响较小
- 减少了一半的LoRA参数

## 🔧 超参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **r** | 8-64 | 秩越大，表达能力越强，但参数越多 |
| **lora_alpha** | 16-32 | 通常设为 $2r$ 或固定值 |
| **lora_dropout** | 0.05-0.1 | 防止过拟合 |
| **target_modules** | ["q", "v"] | 建议至少包含Q和V |

### 不同任务的推荐配置

```python
# 简单任务（文本分类）
LoRAConfig(r=4, lora_alpha=8)

# 中等任务（指令微调）
LoRAConfig(r=8, lora_alpha=16)

# 复杂任务（多任务学习）
LoRAConfig(r=16, lora_alpha=32)

# 最大表达能力
LoRAConfig(r=64, lora_alpha=128)
```

## 🚀 LoRA变体

### 1. QLoRA (Quantized LoRA)

结合4-bit量化，进一步减少显存：

```python
# 基础权重: 4-bit量化存储
# LoRA参数: fp16/bf16全精度
Y = Dequantize(W_quant)·X + BA·X × scaling
```

**显存节省**: 在7B模型上可将微调显存从>70GB降至~6GB

### 2. LoRA+

改进的优化策略：
- A矩阵使用较大学习率
- B矩阵使用较小学习率

### 3. DoRA (Weight-Decomposed LoRA)

分解权重的magnitude和direction：
$$W' = m \frac{W + BA}{\|W + BA\|}$$

### 4. AdaLoRA

自适应调整不同层的秩 $r$。

## 📊 实验效果

在LLaMA-7B上的典型表现：

| 方法 | 可训练参数 | 显存 | 效果 |
|------|-----------|------|------|
| 全量微调 | 7B | ~120GB | 100% |
| LoRA (r=8) | 4.2M | ~18GB | 97% |
| LoRA (r=16) | 8.4M | ~20GB | 98% |
| QLoRA (r=8) | 4.2M | ~6GB | 95% |

## 💡 最佳实践

### 1. 训练技巧

```python
# 只训练LoRA参数
for name, param in model.named_parameters():
    if 'lora_' not in name:
        param.requires_grad = False

# 使用较小的学习率
optimizer = AdamW(model.parameters(), lr=1e-4)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### 2. 多LoRA切换

```python
# 保存LoRA权重
lora_state = get_lora_state_dict(model)
torch.save(lora_state, "lora_adapter.pt")

# 加载不同的LoRA适配器
lora_chinese = torch.load("lora_chinese.pt")
lora_code = torch.load("lora_code.pt")

# 运行时切换
set_lora_state_dict(model, lora_chinese)  # 中文对话
set_lora_state_dict(model, lora_code)     # 代码生成
```

### 3. 推理优化

```python
# 训练完成后合并权重
model.merge_lora()

# 此时推理与原始模型完全相同
# 无额外计算开销！
output = model(input)
```

## 🔗 相关工作

- **Adapter**: 在层之间插入小型网络
- **Prefix-Tuning**: 在输入前添加可学习的前缀
- **Prompt-Tuning**: 只训练软提示向量
- **BitFit**: 只训练偏置项

LoRA相比这些方法的优势：
- ✅ 无推理延迟（可合并）
- ✅ 参数效率高
- ✅ 易于切换和组合
- ✅ 与量化技术兼容

## 📖 参考资料

1. [LoRA原论文](https://arxiv.org/abs/2106.09685)
2. [QLoRA论文](https://arxiv.org/abs/2305.14314)
3. [PEFT库 (Hugging Face)](https://github.com/huggingface/peft)
4. [LLaMA-Adapter](https://arxiv.org/abs/2303.16199)

