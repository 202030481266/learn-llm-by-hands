## GQA

[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245)

## 旋转位置编码的巧妙计算

Llama3 中使用这种复数旋转是一种非常巧妙的旋转位置编码（RoPE，Rotary Position Embedding）实现方式。

这是该算法的详细步骤：

1. 首先，将输入的两个连续维度 `[x1, x2]` 转换为复数 `x1 + j * x2`（使用 `view_as_complex` 函数）
   
2. 然后，创建一组旋转因子 `torch.polar(torch.ones_like(freqs), freqs)`，它会创建形如 `cos(mθ) + j*sin(mθ)` 的复数，其中 `freqs` 包含了基于位置和维度的角度值

3. 当一个复数乘以另一个复数时，会发生旋转和缩放：
   - 复数 `x1 + j*x2` 乘以 `cos(mθ) + j*sin(mθ)` 得到：
   - `(x1 + j*x2) * (cos(mθ) + j*sin(mθ))`
   - `= x1*cos(mθ) - x2*sin(mθ) + j*(x1*sin(mθ) + x2*cos(mθ))`

这个结果正是对 `[x1, x2]` 应用了一个旋转矩阵的效果：
```
[cos(mθ)  -sin(mθ)]   [x1]
[sin(mθ)   cos(mθ)] * [x2]
```

这种实现方式有几个优点：
1. 计算高效 - 使用复数乘法可以避免显式构造旋转矩阵
2. 数值稳定 - 避免了使用三角函数时可能出现的数值不稳定问题
3. 代码简洁 - 使用 PyTorch 的复数操作可以用很少的代码实现旋转

这种旋转编码让模型能够更好地理解序列中的相对位置信息，而不需要像传统位置编码那样直接将绝对位置添加到嵌入中。

## 使用`repeat_kv`函数构建多份视图

具体的实现采用`expand`和`reshape`函数，创建视图。

```python
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
```


