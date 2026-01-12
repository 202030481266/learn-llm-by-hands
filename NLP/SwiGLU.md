# SwiGLU 激活函数

## 概述

SwiGLU (Swish-Gated Linear Unit) 是一种基于门控线性单元 (GLU) 的激活函数，使用 SiLU (Swish) 作为门控激活函数。它是现代大语言模型中最主流的 FFN 激活函数，被 LLaMA、Mistral、DeepSeek、ChatGLM 等主流模型采用。

## 论文来源

- **GLU Variants Improve Transformer** (Shazeer, 2020)
  - 论文链接: https://arxiv.org/abs/2002.05202
  - Google 提出的 GLU 变体，在 Transformer 中显著提升了性能

## 数学定义

### 1. 基础 GLU (Gated Linear Unit)

$$\text{GLU}(x) = (xW) \otimes \sigma(xV)$$

其中：
- $x$ 是输入
- $W, V$ 是权重矩阵
- $\sigma$ 是 sigmoid 函数
- $\otimes$ 表示逐元素相乘

### 2. SwiGLU 定义

SwiGLU 使用 SiLU (Swish) 替代 sigmoid：

$$\text{SwiGLU}(x) = \text{SiLU}(xW_g) \otimes (xW_v) W_o$$

展开写：

$$\text{SwiGLU}(x) = \left( xW_g \cdot \text{sigmoid}(xW_g) \right) \otimes (xW_v) W_o$$

在 FFN 中的完整形式：

$$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{SiLU}(xW_1) \otimes xW_3 \right) W_2$$

### 3. SiLU (Swish) 激活函数

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

## 与传统 ReLU FFN 的对比

### Vanilla FFN (原始 Transformer)

$$\text{FFN}_{\text{ReLU}}(x) = \text{ReLU}(xW_1) W_2$$

**参数量**: $d \times 4d + 4d \times d = 8d^2$

### SwiGLU FFN

$$\text{FFN}_{\text{SwiGLU}}(x) = \left( \text{SiLU}(xW_1) \otimes xW_3 \right) W_2$$

**参数量**: $d \times \frac{8}{3}d \times 3 \approx 8d^2$ (与 ReLU 版本相近)

**注意**: 实际实现中，SwiGLU 的隐藏层维度通常设为 $\frac{8}{3}d$ 而非 $4d$，使参数量与原始 FFN 相当。

### 对比表

| 特性 | ReLU FFN | SwiGLU FFN |
|------|----------|------------|
| 参数量 | $8d^2$ | $\approx 8d^2$ (3个矩阵，每个 $\frac{8}{3}d$) |
| 激活函数 | ReLU | SiLU (Swish) |
| 门控机制 | 无 | 有 (双路径) |
| 性能 | 基准 | +1-2% perplexity |
| 平滑性 | 不可导 | 处处可导 |

## 为什么 SwiGLU 更好？

### 1. 门控机制

SwiGLU 引入了门控机制，允许网络动态调节信息流：

$$\text{output} = \text{gate} \otimes \text{value}$$

门控值决定保留多少 value 信息，类似 LSTM 的门控思想。

### 2. 平滑性

SiLU 是处处可导的平滑函数，没有 ReLU 的"死神经元"问题：

$$\text{SiLU}'(x) = \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x))$$

### 3. 负值区域

SiLU 在负值区域有非零输出，保留了更多信息：

```
SiLU(-2) ≈ -0.24  (有输出)
ReLU(-2) = 0      (无输出)
```

### 4. 自门控特性

SiLU 本身具有自门控特性：对于大输入，$\sigma(x) \to 1$，$\text{SiLU}(x) \to x$

## 实现细节

### 隐藏层维度计算

LLaMA 使用的标准公式：

```python
hidden_dim = int((4 * dim) * 2 / 3)  # = 8/3 * dim
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
```

其中 `multiple_of=256` 用于对齐，提高 GPU 计算效率。

### 代码结构

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.w1 = nn.Linear(dim, hidden_dim)  # Gate
        self.w2 = nn.Linear(hidden_dim, dim)  # Output
        self.w3 = nn.Linear(dim, hidden_dim)  # Value

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

## 其他 GLU 变体

### GeGLU (Gated GELU)

$$\text{GeGLU}(x) = \text{GELU}(xW_1) \otimes (xW_3) W_2$$

- 使用 GELU 替代 SiLU
- 用于 PaLM、T5 等模型
- 性能略逊于 SwiGLU

### ReGLU (Gated ReLU)

$$\text{ReGLU}(x) = \text{ReLU}(xW_1) \otimes (xW_3) W_2$$

- 使用 ReLU 替代 SiLU
- 原始论文中效果不如 SwiGLU

### PAIGLU (Gated Gated-Linear)

更复杂的门控变体，实际应用较少。

## 使用 SwiGLU 的模型

| 模型 | 年份 | 备注 |
|------|------|------|
| LLaMA/LLaMA 2/3 | 2023 | 隐藏层维度 = 8/3 * dim |
| Mistral 7B | 2023 | 与 LLaMA 相同 |
| DeepSeek/V2/V3 | 2024 | MoE 层也使用 SwiGLU |
| ChatGLM | 2023 | 使用 FFN + SwiGLU |
| Qwen | 2023 | 阿里模型 |
| Baichuan 2 | 2023 | 百川智能 |
| Phi-2 | 2023 | Microsoft 小模型 |

## 性能对比

根据原论文和后续研究，SwiGLU 相比 ReLU FFN：

- **Perplexity**: 降低约 5-10%
- **训练稳定性**: 更好
- **收敛速度**: 略快
- **参数效率**: 相近参数量下性能更好

## 优缺点总结

### 优点

1. **性能优越**: 在几乎所有基准测试上优于 ReLU FFN
2. **平滑可导**: 没有不可导点，优化更稳定
3. **保留负值信息**: 负值区域有输出
4. **门控机制**: 动态调节信息流
5. **参数效率**: 与 ReLU FFN 参数量相近

### 缺点

1. **计算量略高**: 需要额外的矩阵乘法和逐元素操作
2. **实现复杂**: 比单投影 ReLU FFN 复杂
3. **内存占用**: 3 个投影矩阵 vs 2 个

## 参考文献

1. Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202
2. Touvron, H. et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288
3. DeepSeek-AI (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437

## 扩展阅读

- **SiLU vs Swish**: 同一个函数，Elon Musk 的 OpenAI 团队命名为 Swish
- **GELU**: 另一种平滑激活函数，用于 BERT、GPT-2/3
- **GEGLU**: Google 的 T5 和 PaLM 使用
