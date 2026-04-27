# 第 5 章：Attention 机制

## 1. 本章要解决的问题

第 4 章里，我们已经沿着：

`n-gram -> 神经语言模型`

这条线看到了语言模型是怎样一步步变强的。

但当我们继续往前走时，很快会撞上一个新的核心问题：

**模型到底怎样“有选择地看上下文”？**

例如一句话：

`The animal didn't cross the street because it was too tired.`

这里的 `it` 更可能指代 `animal`，而不是 `street`。

人类读到这里，几乎会自然地把注意力放到真正相关的词上；
但如果一个模型只是机械地把固定长度上下文压进一个向量里，那么：

- 远距离依赖很容易丢失
- 不同位置的信息会被混在一起
- 模型很难动态决定“当前最该看谁”

这就是 Attention 要解决的事。

这一章的任务可以概括成一句话：

**理解模型为什么需要一种“按相关性读取上下文”的机制，以及这种机制是怎样被写成可训练计算图的。**

从全书结构上看，这一章有三个作用：

- 它承接第 4 章“语言模型为什么需要更强上下文建模”
- 它为第 6 章 Transformer 架构提供最关键的核心模块
- 它为第 7 章 Mini-GPT 的代码实现打下直接基础

如果第 4 章回答的是“模型在做什么预测任务”，那么第 5 章回答的就是：

**模型在预测下一个 token 时，究竟怎样利用前文。**

## 2. 你学完后应该会什么

- 能用自己的话解释 Attention 为什么会出现
- 能理解 query、key、value 分别在做什么
- 能写出 scaled dot-product attention 的核心公式
- 能解释为什么要除以 \( \sqrt{d_k} \)
- 能区分 self-attention、cross-attention 和 causal self-attention
- 能理解 multi-head attention 为什么有用
- 能看懂 GPT 里 attention 模块的最小实现

## 3. 核心直觉

先不急着上公式，我们先把它想成一个“按需查资料”的过程。

### 3.1 没有 Attention 时会发生什么

假设我们要预测一句话里的下一个 token。

如果模型只能把整个前文压缩成一个固定向量，那么它会遇到一个问题：

**所有历史信息都要被提前揉成一团。**

这会带来两个直接困难：

- 重要信息和不重要信息没有显式区分
- 当前预测真正需要哪部分上下文，模型很难动态选择

这也是早期 RNN / seq2seq 模型里常见的瓶颈之一：
句子越长，信息越容易在传递过程中衰减。

### 3.2 Attention 的一句话直觉

Attention 可以粗暴但有效地理解成：

**当前 token 在处理自己时，会去整个上下文里“找对自己最有帮助的信息”，再把这些信息按权重加权汇总。**

也就是说，它不是把上下文一股脑塞进来，而是会问：

- 我现在是谁？
- 我需要什么信息？
- 历史上哪些 token 和我最相关？
- 这些 token 应该各占多大权重？

### 3.3 一个最小例子

看一句英文：

`The cat chased the mouse because it was hungry`

当模型处理 `it` 之后的内容时，`it` 到底和谁更相关？

一个好的模型不会平均看待前面的每个词，而会更关注：

- `cat`
- `mouse`
- `because`

其中究竟更偏向谁，要由当前任务和训练数据共同决定。

这就是 Attention 最核心的价值：

**让“关注哪里”成为模型可以学习的事情。**

## 4. 数学定义

现在把上面的直觉写成公式。

### 4.1 Q、K、V 是什么

对于输入序列中的每个 token 表示向量 `x`，我们通常会通过三组线性变换得到：

$$
q = xW_Q,\quad k = xW_K,\quad v = xW_V
$$

把整段序列写成矩阵形式：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

这里可以先用直觉理解：

- `Query`：我现在想找什么信息
- `Key`：我身上有什么信息可供匹配
- `Value`：如果你觉得我相关，你最终拿走的内容是什么

最容易混淆的是：

**真正被加权求和的是 `V`，不是 `K`。**

`Q` 和 `K` 主要负责算“相关性分数”，`V` 才是最终被读取的内容。

### 4.2 注意力分数怎么来

对于当前位置的 query \( q_i \) 和所有位置的 key \( k_j \)，先计算相似度：

$$
s_{ij} = q_i k_j^T
$$

把所有位置一起写成矩阵：

$$
S = QK^T
$$

它表示：

**序列中每个位置，对其他所有位置的关注分数。**

### 4.3 为什么要除以 \( \sqrt{d_k} \)

如果 key 的维度是 \( d_k \)，那么标准写法是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

为什么要缩放？

因为当维度变大时，点积的数值通常也会变大，softmax 会更容易进入非常尖锐的区域，导致：

- 某几个位置的概率几乎变成 1
- 其他位置几乎变成 0
- 梯度不稳定，训练更难

所以除以 \( \sqrt{d_k} \) 的作用可以简单理解成：

**把分数的尺度控制在更稳定的范围里。**

### 4.4 softmax 在做什么

对每一行分数做 softmax 后，我们得到注意力权重：

$$
A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

这里的 \( A_{ij} \) 可以理解为：

**当第 `i` 个位置更新自己时，它应该从第 `j` 个位置拿多少信息。**

最后输出是：

$$
O = AV
$$

这表示每个位置都从整段上下文中，取回一个加权汇总后的新表示。

## 5. 从直觉到实现

这一节不再堆公式，而是按计算流程走一遍。

### 5.1 第一步：输入序列先变成向量

设输入序列长度为 \( T \)，每个 token 的隐藏维度是 \( d_{model} \)。

那么输入张量形状通常是：

$$
X \in \mathbb{R}^{T \times d_{model}}
$$

经过三组线性层后得到：

- \( Q \)：\( T \times d_k \)
- \( K \)：\( T \times d_k \)
- \( V \)：\( T \times d_v \)

### 5.2 第二步：算每个位置对所有位置的相关性

做一次矩阵乘法：

$$
QK^T
$$

结果形状是：

$$
T \times T
$$

这个矩阵非常关键，因为它直接刻画了：

**每个 token 会看向哪些 token。**

### 5.3 第三步：做 mask 和 softmax

如果是普通 self-attention，一个位置可以看整段序列。

但如果是 GPT 这样的自回归语言模型，就必须满足：

**当前位置不能偷看未来 token。**

所以在 softmax 之前，要对未来位置加上 mask。

直观上可以理解为：

- 允许看的位置保留原分数
- 不允许看的位置改成一个极小值，例如 `-inf`

这样 softmax 之后，那些位置的权重就会接近 0。

### 5.4 第四步：按权重汇总 value

注意力权重矩阵和 `V` 相乘后，每个位置都会得到一个新的表示：

$$
O = AV
$$

这个新表示不是来自某一个单独 token，而是来自：

**整段上下文中对当前任务最有帮助的信息加权组合。**

这正是 Attention 和“固定窗口拼接特征”之间最本质的区别。

## 6. Self-Attention、Cross-Attention 与 Causal Mask

### 6.1 Self-Attention

Self-Attention 指的是：

**Q、K、V 都来自同一个序列。**

也就是说，一个序列内部的每个位置，都可以和同一序列的其他位置交互。

在 Transformer 里，这是一种最基础也最重要的信息混合方式。

### 6.2 Cross-Attention

Cross-Attention 指的是：

- `Q` 来自一个序列
- `K` 和 `V` 来自另一个序列

例如机器翻译里，decoder 在生成目标语言时，可以把当前生成状态作为 query，再去读 encoder 输出的表示。

所以 cross-attention 的核心不是“更高级”，而是：

**让一个序列主动去读取另一个序列。**

### 6.3 Causal Mask 为什么是 GPT 的关键约束

GPT 做的是 next-token prediction。

如果在训练时，第 `t` 个位置可以直接看到第 `t+1` 个甚至更后面的 token，那么这个任务就被“作弊”了。

因此 decoder-only GPT 必须使用 causal mask，也叫：

- look-ahead mask
- upper-triangular mask

它保证：

**第 `t` 个位置只能看见 `1...t` 的内容，而不能看见未来。**

这也是为什么 BERT 和 GPT 在 attention 约束上会不同：

- BERT 做双向理解，通常不需要 causal mask
- GPT 做自回归生成，必须使用 causal mask

## 7. Multi-Head Attention 为什么有用

如果只有一个 attention head，会发生什么？

模型仍然可以学到相关性，但它每次只能在一种投影空间里做“匹配”和“读取”。

Multi-Head Attention 的做法是：

- 把隐藏维度切成多个头
- 每个头各自学习一套 \( W_Q, W_K, W_V \)
- 每个头独立做 attention
- 最后把多个头拼接起来，再做一次线性变换

它的直觉价值在于：

**不同头可以学习不同类型的关系。**

例如某些头更关注：

- 近距离词法依赖
- 长距离指代关系
- 句法结构
- 分隔符或位置边界

当然，现实里“每个头都有明确语义”并不总是成立，但从建模能力上说，多头确实给了模型更丰富的表达空间。

## 8. 最小代码实现

下面给一个最小可读版本的 scaled dot-product attention。

```python
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: [batch, heads, seq_len, head_dim]
    mask:    [1, 1, seq_len, seq_len] or broadcastable tensor
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = attn @ v
    return out, attn
```

如果我们要构造 GPT 风格的 causal mask，可以这样写：

```python
def build_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
```

这个 mask 的作用是保留下三角区域，也就是：

当前位置只能看自己和自己之前的位置。

如果你把这两段代码和第 7 章的 Mini-GPT 实现放在一起看，会发现 GPT 的 attention 核心其实并不神秘：

**线性投影 -> 打分 -> mask -> softmax -> 加权求和**

难点不在公式本身，而在于你是否真正理解每一步在限制什么、读取什么。

## 9. 常见误区

### 误区 1：Attention 就是在“找到最重要的一个词”

不准确。

Attention 通常不是硬选择某一个词，而是对多个位置做加权汇总。
虽然有时权重会很尖锐，但本质上它是软选择，不是离散检索。

### 误区 2：Q、K、V 是三份不同输入

不一定。

在 self-attention 中，它们通常都来自同一个输入序列，只是经过了不同的线性变换；
只有在 cross-attention 时，它们才可能来自不同来源。

### 误区 3：有了 Attention，模型就天然懂长程依赖

也不能这么说。

Attention 确实给了模型直接访问远距离 token 的路径，但模型最终能否学会有效利用这些路径，还取决于：

- 数据
- 训练目标
- 模型规模
- 优化稳定性

### 误区 4：attention 权重越可解释，模型就越好

这是一个常见但危险的过度推断。

attention map 有时能提供一定直觉，但“权重高”不一定等于“因果上最重要”。
把 attention 完全当成解释工具，通常会过度简化模型内部行为。

## 10. 面试问题

### Q1：为什么 Attention 比固定窗口或单向压缩上下文更强？

因为它允许每个位置在计算自己表示时，动态访问整段上下文，而不是把所有历史信息提前压缩成一个固定向量。这样模型更容易捕捉长距离依赖，也能更灵活地区分哪些上下文更重要。

### Q2：为什么 scaled dot-product attention 里要除以 \( \sqrt{d_k} \)？

因为点积的方差会随着维度增大而变大，直接送进 softmax 会让分布过于尖锐，导致梯度不稳定。除以 \( \sqrt{d_k} \) 可以让数值尺度更平稳。

### Q3：self-attention 和 cross-attention 的区别是什么？

self-attention 中，Q、K、V 来自同一个序列；cross-attention 中，Q 来自当前序列，而 K、V 来自另一个序列。前者强调序列内部交互，后者强调一个序列去读取另一个序列。

### Q4：GPT 为什么一定要用 causal mask？

因为 GPT 的训练目标是根据前文预测下一个 token。如果当前位置能看到未来 token，训练目标就被破坏了，模型会通过“偷看答案”得到虚假的低 loss。

### Q5：multi-head attention 的价值是什么？

它让模型可以在多个不同子空间里并行学习关系模式，而不是只用一套相似度规则处理所有依赖。这样表达能力通常更强。

## 11. 本章小结

这一章真正要建立的，不只是一个公式，而是一条清晰主线：

`为什么需要动态读取上下文 -> Q/K/V 如何把这个过程参数化 -> self-attention 如何在序列内部做信息交互 -> causal mask 如何约束生成任务 -> multi-head 如何提升表达能力`

如果你读到这里，已经能回答下面这个问题：

**语言模型在预测下一个 token 时，并不是“平均看前文”，而是在用 attention 有选择地读取前文。**

这就是 Transformer 能成立的关键前提。

下一章我们会在这个基础上继续往上走：

**把 attention、位置编码、FFN、残差连接、LayerNorm 拼成一个完整的 Transformer block。**
