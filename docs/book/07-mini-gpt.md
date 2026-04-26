# 第 7 章：从零实现一个 Mini-GPT

## 1. 本章要解决的问题

第 6 章里，我们已经知道了 Transformer 的核心零件：

- token embedding
- positional embedding
- self-attention
- FFN
- residual connection
- LayerNorm

但到那里为止，我们理解的仍然主要是：

**架构长什么样。**

而这一章要继续往前走一步，回答一个更“工程化”的问题：

**如果只保留 GPT 最核心的部分，我们能不能亲手把一个最小可运行的语言模型实现出来？**

这件事非常重要，因为很多人在学习大模型时，容易卡在一个中间状态：

- 概念都见过
- 架构图也能看懂
- 甚至知道 GPT 是 decoder-only

但一旦真的要自己实现时，就会马上遇到一连串具体问题：

- 输入的 token ids 是怎么变成隐藏向量的？
- causal mask 到底加在 attention 的哪一步？
- logits 是怎么和词表大小对应起来的？
- 训练时为什么要做 shift？
- 生成时为什么不是一次吐出整段话，而是一个 token 一个 token 地接着采样？

所以本章的任务，不再是继续解释“Transformer 是什么”，而是把前面学过的结构真正落到代码和项目里。

从全书结构上看，这一章有三个作用：

- 它承接第 6 章，把 Transformer 架构落实成一个具体可运行的 GPT
- 它为第 8 章的数据处理和第 9 章的预训练流程提供模型主体
- 它也会成为本书第一个真正可以拿来讲项目、写 README、写进简历的最小作品

如果第 6 章回答的是：

**GPT 为什么这样设计。**

那么第 7 章回答的就是：

**GPT 最小版本到底应该怎样被写出来、训起来、跑起来。**

## 2. 你学完后应该会什么

- 能说清一个 decoder-only GPT 的最小组成
- 能画出 Mini-GPT 从输入到输出的完整数据流
- 能理解 forward、loss 计算和 generate 的区别
- 能写出一个最小可运行的 GPT block
- 能解释 greedy、temperature、top-k、top-p sampling 的差异
- 能把这个项目组织成一个适合求职展示的 from-scratch 项目

## 3. 为什么这一章要亲手实现，而不是直接调用框架

今天训练和使用大模型，当然很少有人真的从零手写整个工业级模型。

现实里我们更常见的是：

- 用 Hugging Face 加载模型
- 用现成 tokenizer 和数据管线
- 用 Trainer、accelerate、deepspeed 等工具链完成训练

这些都没有问题，而且后面的章节也会进入这条路线。

但如果一开始就只停留在工具调用层，会有一个明显问题：

**你知道怎么“用模型”，却不一定真的知道“模型里面发生了什么”。**

Mini-GPT 的价值就在这里。

它不是为了和真正的 GPT-2、LLaMA、Qwen 比规模，而是为了让你第一次把下面这些东西串成一个完整闭环：

- 输入怎样进入模型
- attention 怎样在代码里实现
- 输出 logits 怎样对应词表
- loss 怎样从 next-token prediction 得到
- 生成怎样一步步进行

可以把这一章理解成：

**在进入完整预训练、微调和应用工程之前，先亲手搭出一个最小可解释的发动机。**

## 4. Mini-GPT 到底是一个什么项目

这一章建议你把 Mini-GPT 理解成一个“小而完整”的项目，而不只是一个零散代码片段。

它至少要具备下面四种能力：

### 4.1 能定义模型

也就是你能用 PyTorch 搭出一个最小的 decoder-only Transformer：

- embedding
- position embedding
- 多层 block
- 最终 LayerNorm
- language modeling head

### 4.2 能训练

也就是你可以把一段文本数据切成训练样本，计算 next-token prediction loss，并让 loss 逐渐下降。

### 4.3 能生成

也就是给模型一个 prompt，它能基于已经看到的上下文一步步往后生成新 token。

### 4.4 能展示

也就是这不只是“你本地跑过一次”的代码，而是一个你可以对别人讲清楚的项目：

- 目标是什么
- 模型结构是什么
- 训练目标是什么
- 结果如何
- 还有哪些问题和优化空间

这最后一点尤其关键。

因为从求职视角看，真正有价值的不是“我写过一段 GPT 代码”，而是：

**我能从原理、实现、实验和项目表达四个层面，把这个作品讲完整。**

## 5. 一个最小 Mini-GPT 的整体结构

先不要急着看代码，我们先把最小数据流固定下来。

假设输入是一段 token ids：

$$
x = [x_1, x_2, \dots, x_T]
$$

Mini-GPT 的前向过程大致可以写成：

$$
x
\rightarrow \text{token embedding}
\rightarrow \text{position embedding}
\rightarrow N \text{ 个 decoder blocks}
\rightarrow \text{final LayerNorm}
\rightarrow \text{lm head}
\rightarrow \text{logits}
$$

其中最核心的部分是：

### 5.1 输入层

- token ids 先查 embedding table，得到 token vectors
- 再加上 position embedding，形成初始隐藏表示

### 5.2 中间主体

重复堆叠若干个 decoder block。

每个 block 里通常包含：

- masked self-attention
- MLP / FFN
- residual connection
- LayerNorm

### 5.3 输出层

最终每个位置都会输出一个长度为 `vocab_size` 的向量，表示：

**在当前位置之后，下一个 token 各自有多大概率。**

这个向量还不是概率，而是 logits。

后面会再经过 softmax，才会变成真正的概率分布。

## 6. 为什么 GPT 是 decoder-only

第 6 章已经讲过，GPT 之所以采用 decoder-only，本质上是因为它的训练目标是：

**给定前文，预测下一个 token。**

这意味着在位置 `t` 上，模型只能使用：

- 自己当前位置
- 左边已经出现过的内容

而不能偷看右边未来的 token。

所以 GPT 最关键的限制不是“有没有 decoder”这几个字本身，而是：

**有没有 causal mask。**

它保证了注意力矩阵里，一个位置只能看见自己和过去，而看不见未来。

例如输入：

`I love deep learning`

当模型处理 `deep` 这个位置时，它可以看：

- `I`
- `love`
- `deep`

但不能看未来的 `learning`。

否则训练时就会发生“作弊”：

模型明明应该靠前文预测下一个词，却提前把答案看到了。

## 7. 一个 decoder block 里到底有什么

写 Mini-GPT 时，最值得真正吃透的是 block，而不是把整份代码机械抄下来。

因为整个模型本质上就是：

**同一个高质量 block 重复堆叠很多次。**

一个最小 decoder block 一般可以写成：

1. 先做 LayerNorm
2. 进入 masked self-attention
3. 做 residual add
4. 再做 LayerNorm
5. 进入 MLP
6. 再做 residual add

如果用 Pre-LN 结构，形式可以写成：

$$
h' = h + \text{Attention}(\text{LN}(h))
$$

$$
h'' = h' + \text{MLP}(\text{LN}(h'))
$$

这个结构里，几个模块各自承担的角色可以记成：

- attention：决定“看谁”
- MLP：决定“怎么加工看到的内容”
- residual：保留原始信息通路
- LayerNorm：让深层训练更稳定

## 8. 从输入到输出，forward 一次发生了什么

这一节非常关键。

很多人能背出模块名，但一到真正写代码就会混乱，是因为没有把 forward 的数据流想清楚。

假设：

- batch size = `B`
- sequence length = `T`
- hidden size = `C`
- vocabulary size = `V`

那么一次前向大致会经历下面这些步骤。

### 8.1 输入 token ids

模型输入通常是：

$$
X \in \mathbb{R}^{B \times T}
$$

严格说这里存的是整数 id，不是连续值向量。

例如：

```text
[[12, 981, 44, 8],
 [91, 17, 203, 5]]
```

每一行是一条样本，每个数字是一个 token id。

### 8.2 查 token embedding

通过 embedding table，把每个 id 映射成长度为 `C` 的向量。

得到：

$$
\text{tok\_emb} \in \mathbb{R}^{B \times T \times C}
$$

### 8.3 加 position embedding

同时根据位置 `0, 1, 2, ..., T-1` 取出位置向量：

$$
\text{pos\_emb} \in \mathbb{R}^{T \times C}
$$

再与 token embedding 相加：

$$
h^{(0)} = \text{tok\_emb} + \text{pos\_emb}
$$

这样每个位置就同时知道：

- 自己是谁
- 自己在第几个位置

### 8.4 经过多个 decoder blocks

初始表示进入多个 block：

$$
h^{(l+1)} = \text{Block}^{(l)}(h^{(l)})
$$

在每个 block 里，attention 负责上下文交互，MLP 负责非线性特征变换。

### 8.5 输出 logits

最后通过一个线性层投影到词表维度：

$$
\text{logits} \in \mathbb{R}^{B \times T \times V}
$$

这里的意思是：

- 对 batch 中每个样本
- 对序列中的每个位置
- 模型都会给出一个关于整个词表的打分向量

### 8.6 训练时计算 loss

如果训练目标是 next-token prediction，那么当前位置的输出要去预测“下一个位置”的 token。

所以训练时通常会做：

- `logits[:, :-1, :]`
- `targets[:, 1:]`

也就是：

- 用前 `T-1` 个位置的输出
- 对齐后 `T-1` 个位置的真实答案

这一点经常被叫做：

**shift logits / shift labels**

它是 causal language modeling 里最基础、也最容易写错的细节之一。

## 9. causal self-attention 是本章最关键的实现点

如果说整个 Mini-GPT 只能真正挑一个“必须看懂”的模块，那大概率就是 causal self-attention。

它的核心公式仍然是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

但 GPT 比普通 self-attention 多了一步：

**在 softmax 前加 mask。**

这个 mask 的作用是把未来位置打成负无穷，使它在 softmax 后概率接近 0。

例如长度为 `4` 的序列，它对应的可见性大致像这样：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这表示：

- 第 1 个 token 只能看自己
- 第 2 个 token 能看前两个
- 第 3 个 token 能看前三个
- 第 4 个 token 能看前四个

这就是“自回归”的本质约束。

## 10. 最小代码实现

下面这份代码不是工业级实现，但足够帮助你建立 Mini-GPT 的完整骨架。

### 10.1 配置类

```python
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
```

### 10.2 Causal Self-Attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)
```

### 10.3 MLP 和 Block

```python
class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### 10.4 Mini-GPT 主体

```python
class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
```

### 10.5 生成函数

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

这份代码最重要的不是一行不差地背下来，而是你要知道：

- 配置类控制模型规模
- block 是模型主体
- forward 负责训练时的 logits 和 loss
- generate 负责推理时逐步续写

## 11. 训练一个最小版本时要做什么

有了模型结构之后，下一步就是让它真的学会一点东西。

最小训练流程通常包括下面几个部分。

### 11.1 准备训练文本

最开始完全可以用一个很小的数据集，比如：

- 一篇短文章
- 一个小说片段
- 课程笔记文本
- Shakespeare 小样本

注意，这一章的重点不是追求大规模数据，而是先建立：

**从文本到 token，再到 batch，再到 loss 的闭环。**

### 11.2 构造输入和目标

如果有一个 token 序列：

```text
[10, 25, 7, 88, 19]
```

那么可以构造：

- 输入：`[10, 25, 7, 88]`
- 目标：`[25, 7, 88, 19]`

这就对应“看到前文，预测下一个 token”。

### 11.3 前向传播

把输入喂进模型，得到 logits。

### 11.4 计算损失

对 logits 和目标做交叉熵。

### 11.5 反向传播与参数更新

```python
optimizer.zero_grad()
logits, loss = model(x, y)
loss.backward()
optimizer.step()
```

### 11.6 定期采样

训练语言模型时，不能只盯着 loss。

因为有时候 loss 在降，但生成结果仍然：

- 重复
- 发散
- 只会输出高频 token

所以建议定期给一个 prompt，看看模型实际会生成什么。

## 12. 训练目标为什么是 next-token prediction

很多初学者第一次写 Mini-GPT 时，会下意识觉得：

“模型不是在理解整句话吗，为什么目标只是预测下一个 token？”

这是因为自回归语言模型的一个核心思想是：

**只要模型能反复做好 next-token prediction，它就能一步步生成整段文本。**

例如：

给定开头：

`The cat`

模型先预测下一个词可能是：

- `is`
- `sat`
- `runs`

如果它采样出了 `sat`，那么上下文就变成：

`The cat sat`

接着再预测下一个 token。

于是“整段生成”其实就是“单步预测”不断重复。

这个目标看起来局部，但效果非常强，因为语言本来就是序列展开的。

## 13. 推理时为什么要一个 token 一个 token 地生成

这一点也很容易让人困惑。

训练时我们一次输入整段序列，看起来很并行；
但推理时却要一步一步生成，这是不是很低效？

答案是：

**是的，自回归生成天生就是串行的。**

因为第 `t+1` 个 token 的分布依赖前面已经确定的内容，而这些内容在生成前并不存在。

所以推理过程通常是：

1. 输入 prompt
2. 得到最后一个位置的 logits
3. 采样出下一个 token
4. 把这个 token 拼回输入
5. 再继续下一轮

这也是为什么大模型推理优化里，KV cache 会非常重要。

这一章先不展开 KV cache 的工程细节，但你要先理解：

**逐 token 生成不是实现写得笨，而是任务形式本身决定的。**

## 14. 采样策略决定“它会怎么说话”

即使是同一个模型，同一个 prompt，不同采样策略也会让输出风格明显不同。

### 14.1 Greedy decoding

每一步都选概率最大的 token。

优点：

- 简单
- 稳定

缺点：

- 很容易重复
- 容易显得僵硬

### 14.2 Temperature sampling

在 softmax 前用 temperature 调整 logits：

$$
p_i = \text{softmax}(z_i / T)
$$

- `T < 1`：分布更尖锐，更保守
- `T > 1`：分布更平缓，更随机

### 14.3 Top-k sampling

每一步只保留概率最高的 `k` 个 token，再从里面采样。

它的作用是：

**防止模型从非常离谱的长尾 token 里乱抽。**

### 14.4 Top-p sampling

也叫 nucleus sampling。

它不是固定保留前 `k` 个 token，而是保留累计概率达到阈值 `p` 的最小集合。

这比 top-k 更自适应，因为不同上下文下，模型分布的陡峭程度并不一样。

实践里经常会看到下面这种组合：

- temperature
- top-k 或 top-p

原因很简单：

我们既想保留一定随机性，又不希望它胡说得太离谱。

## 15. 你应该观察哪些实验现象

Mini-GPT 不是“代码跑通就结束”的章节。

真正有价值的是，你要开始学会像做实验一样记录现象。

建议至少观察下面几类结果。

### 15.1 loss 是否稳定下降

这是最基础的信号。

如果 loss 完全不降，通常说明：

- 学习率有问题
- 数据构造有问题
- mask 写错了
- logits 和 labels 没有正确对齐

### 15.2 生成结果是否从乱码逐渐变得有模式

训练初期，模型往往只会输出：

- 杂乱 token
- 高频字符
- 重复片段

随着训练推进，它通常会先学到：

- 局部拼写模式
- 高频短语模式
- 句子节奏

这其实很符合语言模型从浅层统计到更高层结构逐渐建立能力的过程。

### 15.3 不同采样参数下输出差异有多大

例如你可以固定同一个 prompt，对比：

- greedy
- temperature = 0.8
- top-k = 20
- top-p = 0.9

观察它们在下面几个维度的差异：

- 流畅度
- 重复率
- 创造性
- 离谱程度

### 15.4 模型有没有明显 overfit

如果数据特别小，模型可能会：

- 训练 loss 很低
- 生成结果高度复读训练文本

这并不奇怪，反而是一个很好的教学观察：

**语言模型并不是神秘地“理解”了世界，它首先是在拟合训练分布。**

## 16. 初学者最容易犯的错误

这一节非常建议保留，因为 Mini-GPT 的很多 bug 都是“能跑，但逻辑错了”。

### 16.1 忘记加 causal mask

这会导致训练时偷看未来，loss 看起来很好，但模型学到的是作弊解法。

### 16.2 logits 和 labels 没有 shift

如果当前位置直接预测当前位置，而不是预测下一个位置，训练目标就错了。

### 16.3 维度变换写错

尤其是在 multi-head attention 里，`view`、`transpose`、`contiguous` 很容易出错。

### 16.4 生成时没有只取最后一个位置的 logits

生成下一个 token 时，只应该使用当前序列最后一个位置的输出分布。

### 16.5 训练和推理状态没有切换

比如 dropout 在生成时如果没有关掉，会让输出不稳定。

### 16.6 把“能生成句子”误认为“已经学懂语言”

Mini-GPT 能生成一些看起来像样的文本，不代表它已经拥有很强语义能力。

它可能只是学会了局部统计模式和常见续写结构。

这个区分非常重要，它会影响你后面对预训练、对齐和评估的理解。

## 17. 这一章和后面几章怎么衔接

Mini-GPT 做完以后，后面几章的逻辑会顺很多。

### 17.1 第 8 章会解决“数据怎样更规范地进来”

这一章里我们默认输入已经是 token ids。

但现实里你还要解决：

- tokenizer 怎么训练或选择
- padding 和 truncation 怎么做
- batch 怎么组织
- attention mask 怎么配合数据处理

这些会进入第 8 章。

### 17.2 第 9 章会解决“怎样把训练流程做完整”

这一章先用最小训练循环把主线跑通。

后面会进一步展开：

- optimizer 细节
- warmup
- lr schedule
- mixed precision
- checkpointing
- 更完整的 pretraining pipeline

### 17.3 后续微调与应用章会建立在这个模型视角上

无论后面讲：

- SFT
- LoRA / QLoRA
- Alignment
- RAG
- Agent

如果你脑子里一直保留这个 Mini-GPT 的最小骨架，很多新概念都会更容易落地。

## 18. 怎么把这一章写成一个求职项目

这一章之所以特别重要，不只是因为它教你实现模型，还因为它很适合沉淀成第一个像样的作品。

你可以把这个项目定位成：

`Mini-GPT from Scratch`

它想证明的能力包括：

- 理解 decoder-only Transformer 结构
- 能用 PyTorch 独立实现最小 GPT
- 能完成基础训练与采样
- 能记录实验并分析现象

### 18.1 适合写在 GitHub README 里的说法

> A minimal decoder-only GPT implemented from scratch with PyTorch, including causal self-attention, autoregressive training, and text generation.

### 18.2 适合写在简历里的说法

> 从零实现 Mini-GPT，自主完成 decoder-only Transformer、causal self-attention、next-token prediction 训练与采样生成流程，并基于小规模语料完成实验验证。

### 18.3 面试时可以怎么口头介绍

可以用下面这种结构：

1. 我先自己实现了一个最小 GPT，而不是直接调框架
2. 重点实现了 masked self-attention、Pre-LN block 和生成函数
3. 训练目标是标准的 causal language modeling
4. 我观察了 loss 下降、不同采样策略的输出差异，以及小数据上的 overfitting 现象
5. 这个项目帮助我把 Transformer 原理真正落到了代码层

这类表达会比“我学过 GPT”有说服力得多。

## 19. 本章小结

这一章最重要的，不是把工业级 GPT 复现一遍，而是第一次把下面这条主线真正打通：

**结构 -> 代码 -> 训练 -> 生成 -> 项目表达**

到这里为止，你应该已经建立起一个很重要的模型直觉：

- GPT 本质上是 decoder-only Transformer
- 它靠 causal mask 保证自回归约束
- 它的训练目标是 next-token prediction
- 它的推理方式是逐 token 生成
- 它的最小实现并不神秘，但每个细节都值得亲手走一遍

如果说前面几章是在帮你理解语言模型“为什么这样设计”，那么从这一章开始，你已经真正进入：

**自己动手把一个 LLM 最小系统搭出来。**

## 20. 面试问题

### Q1：为什么 GPT 只需要 decoder-only，而不需要 encoder？

因为 GPT 的目标是自回归 next-token prediction。当前位置只能依赖左侧上下文，所以 decoder-only 加 causal mask 就已经和训练目标天然对齐，不一定需要 encoder 的双向表示能力。

### Q2：Mini-GPT 训练时为什么要做 shift？

因为模型在位置 `t` 的输出，目标是预测位置 `t+1` 的真实 token，而不是当前位置本身。所以 logits 和 labels 需要错开一位对齐。

### Q3：为什么生成时要一个 token 一个 token 地采样？

因为自回归生成里，下一个 token 的概率依赖前面已经生成出的内容，而这些内容在生成前并不存在，所以推理天然是串行展开的。

### Q4：causal mask 的作用是什么？

它阻止当前位置在 attention 中看到未来 token，避免训练时作弊，并保证模型学习到的行为与推理时的自回归生成过程一致。

### Q5：为什么同一个模型在不同 sampling 策略下输出差异很大？

因为 logits 只是一个分布，不同 decoding 策略会以不同方式从分布中取样。greedy 更保守，temperature 会改变分布平滑度，top-k / top-p 会限制候选范围，所以生成风格会明显不同。

## 21. 延伸阅读

- Andrej Karpathy, `Let's build GPT`
- Stanford CS336, `Language Modeling from Scratch`
- Hugging Face LLM Course 中关于 causal language modeling 的章节
- 《Build a Large Language Model (From Scratch)》中关于 GPT 实现与训练的部分
