# Attention 机制

!!! note "核心一句话"
    Attention 让模型在处理每个 token 时，能够动态地"关注"序列中其他位置的信息，权重由内容相似度决定。

## 1. 问题背景

RNN 处理序列时存在两个缺陷：

1. **长距离依赖衰减**：梯度随序列长度指数衰减，远处的信息难以传递
2. **无法并行**：必须按时间步顺序计算

Attention 机制通过**直接连接任意两个位置**解决了这两个问题。

## 2. Scaled Dot-Product Attention

### 2.1 三个角色：Q / K / V

给定输入序列 $X \in \mathbb{R}^{n \times d}$，通过三个线性变换得到：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

直觉类比：

- **Query（查询）**：当前 token 在"问"什么
- **Key（键）**：每个 token 在"说"自己是什么
- **Value（值）**：每个 token 实际携带的信息

### 2.2 计算公式

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**为什么除以 $\sqrt{d_k}$？**

当 $d_k$ 很大时，点积结果的方差也变大，softmax 会进入梯度极小的饱和区。除以 $\sqrt{d_k}$ 将方差归一化。

!!! example "数值示例"
    设 $d_k = 64$，两个随机向量的点积期望值约为 0，标准差约为 $\sqrt{64} = 8$。
    不做缩放时 softmax 输入范围约 ±8，导致梯度消失。

### 2.3 代码实现

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
    q: torch.Tensor,   # (batch, heads, seq_len, d_k)
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        output: (batch, heads, seq_len, d_v)
        weights: (batch, heads, seq_len, seq_len)  注意力权重，可用于可视化
    """
    d_k = q.size(-1)
    
    # (batch, heads, seq, seq)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # mask=True 的位置填充 -inf，softmax 后趋近于 0
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    
    return output, weights
```

## 3. Multi-Head Attention

单个 Attention 头只能关注一种"关系"。多头机制让模型从不同角度理解序列。

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

### 3.1 完整 MHA 实现

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 合并为单个矩阵乘法，更高效
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o   = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,          # (batch, seq, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        
        # 一次性计算 Q/K/V
        qkv = self.W_qkv(x)                          # (batch, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                # 各 (batch, seq, d_model)
        
        # 拆分多头：(batch, heads, seq, d_k)
        def split_heads(t):
            return t.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        
        # Attention 计算
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多头：(batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        
        return self.W_o(out)
```

### 3.2 可视化注意力权重

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(weights, tokens, head=0):
    """可视化第 head 个注意力头的权重矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        weights[0, head].detach().cpu().numpy(),
        xticklabels=tokens, yticklabels=tokens,
        cmap='Blues', ax=ax
    )
    ax.set_title(f'Attention Head {head}')
    plt.tight_layout()
    plt.show()
```

## 4. Causal Mask（因果掩码）

解码器中，token 只能关注自己及之前的 token（不能看未来）：

```python
def causal_mask(seq_len: int, device='cpu') -> torch.Tensor:
    """返回下三角矩阵（True=可见，False=遮蔽）"""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

# 使用
mask = causal_mask(seq_len=10)
# tensor([[True, False, ..., False],
#         [True, True,  ..., False],
#         ...
#         [True, True,  ..., True ]])
```

## 5. 常见问题

??? question "Self-Attention 的时间复杂度是多少？"
    $O(n^2 d)$，其中 $n$ 是序列长度，$d$ 是模型维度。  
    这是长上下文的瓶颈所在，也是 Flash Attention / 稀疏注意力要解决的核心问题。

??? question "为什么 Q 和 K 用独立的投影矩阵，而不直接用 X 做点积？"
    独立的 $W^Q$、$W^K$ 让模型学习"问什么"和"有什么"两个不同的表示空间，
    而不是强制用同一个表示既当查询又当键。实践中效果更好。

??? question "Cross-Attention 和 Self-Attention 的区别？"
    Self-Attention：Q / K / V 来自同一序列（如解码器内部）  
    Cross-Attention：Q 来自一个序列，K / V 来自另一个序列（如解码器关注编码器输出）

## 6. 本节小结

- [x] Attention 通过 Q/K/V 三角色实现动态加权
- [x] $\sqrt{d_k}$ 缩放防止 softmax 饱和
- [x] 多头机制捕捉不同维度的语义关系
- [x] Causal Mask 实现自回归生成

**下一节**：[位置编码 →](positional-encoding.md)

---

*本页最后更新：{{ git_revision_date_localized }}*
