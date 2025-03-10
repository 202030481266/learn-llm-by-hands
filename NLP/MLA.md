## 部分代码核心详解

由于官方deepseek的代码没有按照原始论文的公式敲的，主要是用了拼接矩阵的技巧优化了计算效率，不太好理解，下面是源码对应的公式。

### kv cache 相关

公式：

$$
\begin{align}
\left[c_{t}^{KV}, k_{t}^{NR}\right] &= h_{t}W^{DKVR} \\
k_{t}^{R} &= \text{ROPE}\left(k_{t}^{NR}\right) \\
\left[k_{t}^{C}, v_{t}^{C}\right] &= c_{t}^{KV}W^{UKV}
\end{align}
$$

### q矩阵相关

需要仔细看论文，论文中的 $q_{t}^{R}$ 是从压缩后的 $q_{t}^{C}$ 中计算得到的，这是为了提高计算效率，这涉及到了公式的变动。

$$
\begin{align}
c_{t}^{Q} &= h_{t}W^{DQ} \\
\left[q_{t}^{C}, q_{t}^{NR}\right] &= c_{t}^{Q}W^{UQR} \\
q_{t}^{R} &= \text{ROPE}\left(q_{t}^{NR}\right)
\end{align}
$$

### 计算attention_scores

代码中采用了einsum来计算scores，非常简洁但是不是那么容易理解，下面从公式的角度说明。

一般来说，我们如果要计算两个矩阵的乘法，在数学上可以表示为求和：

$$
C_{i, j} = \sum_{k=1}^{n} A_{i, k} \times B_{k, j}
$$

这可以看成是对维度`k`进行了压缩求和，使用einsum可以表示为：

```python
C = torch.einsum("ik,kj->ij", A, B)
```

同样的在高维矩阵中，我们有计算attention_scores的`qk`矩阵，就可以计算当前`q`中第`i`个token对于`k`中第`j`个token的attention_score。

$$
\text{scores}_{b, i, h, j} = \sum_{x=1}^{d} q_{b, i, h, x} \times k_{b, j, h, x}
$$

使用einsum可以表示为：

```python
scores = torch.einsum("bshd,bthd->bsht", q, k)
```

### 官方attention scores的计算实现

官方给定的attention scores修改了原有的公式，不是按照经典的attention计算方法，主要是利用解耦的`q`和`k`分别计算带ROPE和不带ROPE的分数，然后加和得到`q`和`k`的attention scores。

首先我们有：

$$
q_{t,i} = \left[q_{t,i}^{C}, q_{t,i}^{R}\right] \\
k_{t,i} = \left[k_{t,i}^{C}, k_{t,i}^{R}\right]
$$

参考注意力的矩阵乘法，可得：

$$
\begin{align}
\sum_{j=1}^{t}q_{t}k_{j}^T &= \sum_{j=1}^{t}\left[q_{t}^{C}, q_{t}^{R}\right]\left[k_{j}^{C}, k_{j}^{R}\right] \\
&= \sum_{j=1}^{t}\left[q_{t}^{C}(k_{j}^{C})^T + q_{t}^{R}(k_{j}^{R})^T\right]
\end{align}
$$

拓展上面的公式，可以得到直接使用kv cache中缓存的变量来计算，增加计算效率：

$$
\begin{align}
\sum_{j=1}^{t}q_{t}k_{j}^T &= \sum_{j=1}^{t}\left[q_{t}^{C}(k_{j}^{C})^T + q_{t}^{R}(k_{j}^{R})^T\right] \\
&= \sum_{j=1}^{t}\text{slice}[h_{t}W^{DQ}W^{UQR}, :d_{h}n_{h}](c_{j}^{KV}\text{slice}[W^{UKV}, :d_{h}n_{h}])^T + \text{ROPE}(q_{t}^{NR})\text{ROPE}((k_{j}^{NR}))^T \\
&= \sum_{j=1}^{t}\text{slice}[h_{t}W^{DQ}W^{UQR}, :d_{h}n_{h}]\text{slice}[W^{UKV}, :d_{h}n_{h}]^{T}(c_{j}^{KV})^{T} + \text{ROPE}(\text{slice}[h_{t}W^{DQ}W^{UQR}, d_{h}n_{h}:])(k_{j}^{R})^T
\end{align}
$$

可以观察到 $c_{j}^{KV}$ 和 $k_{j}^{R}$ 是已经缓存好的，所以可以利用缓存来直接计算attention scores。（注意这里的计算复杂度没有发生改变）









