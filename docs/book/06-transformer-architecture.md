# 第 6 章：Transformer 架构

## 1. 本章要解决的问题

第 5 章里，我们已经理解了 Attention 的核心直觉：

**模型会根据当前 token 的需求，去上下文里有选择地读取信息。**

这一步非常关键，但它还不是一个完整的大模型架构。

因为当我们真的想把语言模型做出来时，很快会遇到几个更具体的问题：

- 只有 Attention，模型怎么知道 token 的先后顺序？
- 只有上下文交互，没有额外非线性变换，表达能力够吗？
- 如果把网络堆得很深，训练为什么还能稳定？
- 为什么后来会分化出 BERT、GPT、T5 这些不同结构？

所以这一章要解决的，不再是：

**Attention 是什么。**

而是：

**Attention 是怎样被组织进一个完整的 Transformer，并进一步演化成现代 LLM 主流架构的。**

从全书结构上看，这一章有三个作用：

- 它承接第 5 章，把 Attention 从“单个机制”升级成“完整系统”
- 它为第 7 章的 Mini-GPT 提供结构层面的蓝图
- 它也为后面预训练、微调、RAG、Agent 等章节建立统一模型视角

如果第 5 章回答的是：

**模型怎样动态关注上下文。**

那么第 6 章回答的就是：

**这些能力怎样被拼装成一个真正可训练、可扩展、可用于不同任务的神经网络架构。**

## 2. 你学完后应该会什么

- 能用自己的话解释 Transformer 为什么不只是 Attention
- 能说明 token embedding、position、attention、FFN、残差连接、LayerNorm 分别做什么
- 能画出一个最小 Transformer block 的数据流
- 能区分 encoder-only、decoder-only、encoder-decoder 三类架构
- 能解释 GPT 为什么采用 decoder-only
- 能把这一章的结构理解直接映射到第 7 章的 Mini-GPT 代码里

## 3. 为什么只有 Attention 还不够

很多人第一次学 Transformer 时，容易把它理解成一句话：

`Transformer = Attention`

这不算完全错，但不完整。

更准确地说，Transformer 是：

**以 Attention 为核心，再配上一组让深层网络可以表达、训练和扩展的辅助模块。**

如果只有 Attention，而没有其他配套设计，会出现几个明显问题。

### 3.1 没有位置信息，模型不知道顺序

Attention 本身并不天然理解顺序。

假设输入两个序列：

- `dog bites man`
- `man bites dog`

如果我们只把 token 向量丢进 self-attention，而不给任何位置相关信息，那么模型看到的更像是一个“词袋集合”，而不是一个有先后顺序的序列。

但语言的意义高度依赖顺序。

所以 Transformer 必须额外引入：

- positional encoding
- 或 positional embedding

来告诉模型：

**这个 token 不只是“是什么”，还要知道“它在第几个位置”。**

### 3.2 只有信息交互还不够，还需要更强表达能力

Attention 擅长做的是：

**让每个位置从其他位置读取相关信息。**

但如果整个网络只反复做“加权读取”，而没有额外的非线性变换，那么模型的表达能力会受限。

这就像一个团队里所有人都在互相交换信息，但没有人对信息做进一步加工、抽象和重组。

所以 Transformer 在 attention 之后，通常还会接一个逐位置的前馈网络（Feed-Forward Network, FFN）。

它的作用可以先粗略理解成：

- attention 负责“信息路由”
- FFN 负责“特征变换”

两者组合起来，模型才能既会看上下文，也会处理上下文。

### 3.3 网络变深之后，训练会更困难

现代大模型不是一两层，而是很多层 block 堆起来的。

一旦网络变深，就会遇到经典问题：

- 梯度传播困难
- 训练不稳定
- 前面学到的表示容易被后面破坏

所以 Transformer 借鉴了残差连接（Residual Connection）的思想。

也就是每个子层都不是只输出新结果，而是把输入保留下来，再和新结果相加。

直觉上可以理解成：

**模型不是每一层都推翻前一层，而是在原有表示上做增量修正。**

这会让深层网络更容易训练。

### 3.4 激活分布不稳定，需要规范化

深层神经网络训练时，一个常见问题是不同层输入分布会不断变化，导致训练不稳定。

Transformer 里广泛使用 LayerNorm，作用可以先简单理解成：

**把每个位置上的表示拉回一个更稳定的数值范围。**

这样做的好处是：

- 梯度更稳定
- 训练更容易收敛
- 深层堆叠时更可靠

所以到这里我们就能看出：

Transformer 不是把 Attention 单独拿出来直接堆很多层，而是围绕它补齐了一整套工程化和训练稳定性设计。

## 4. Transformer 的核心组成模块

现在我们把完整 Transformer 先拆成几个最重要的零件。

### 4.1 Token Embedding

模型拿到的原始输入并不是单词本身，而是 token id。

例如一句话：

`I love NLP`

经过 tokenizer 之后，可能变成：

`[314, 892, 10577]`

但这些整数本身没有语义。

所以第一步通常要做 embedding lookup，把每个 token id 映射成一个稠密向量：

$$
e_i = E[x_i]
$$

其中：

- \( x_i \) 是第 \( i \) 个 token id
- `E` 是 embedding matrix
- \( e_i \) 是该 token 的向量表示

可以把它理解成：

**token embedding 负责回答“这个 token 是谁”。**

### 4.2 Positional Encoding / Positional Embedding

有了 token embedding 之后，模型还缺一件事：

**这个 token 在序列中的位置。**

最早的 Transformer 论文里使用的是 sinusoidal positional encoding；
而 GPT、BERT 等很多现代模型更常用可学习的 positional embedding。

无论具体实现是什么，本质上都是要把“位置信息”注入表示中。

最常见的做法是直接相加：

$$
h_i^{(0)} = e_i + p_i
$$

其中 \( p_i \) 表示第 \( i \) 个位置的向量。

这一层负责回答：

**这个 token 不只是“谁”，还要知道“它在哪”。**

### 4.3 Multi-Head Self-Attention

这部分是 Transformer 的核心。

第 5 章里我们已经见过单头 attention 的基本形式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

而在 Transformer 里，通常不会只用一个 attention head，而是使用 multi-head attention。

它的直觉是：

**让模型从多个子空间、多个关系视角同时观察上下文。**

例如有的 head 可能更关注：

- 主谓关系
- 代词指代
- 局部搭配
- 长距离依赖

严格说，head 不一定会自动学成这么清晰的人类语法功能，但这个直觉足够帮助我们理解它为什么比单头更强。

multi-head 的常见写法是：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

其中每个 `head` 都有自己的一组 \( W_Q, W_K, W_V \)。

### 4.4 Feed-Forward Network

attention 之后，Transformer 还会接一个 position-wise FFN。

所谓 position-wise，意思是：

**每个位置都经过同一套两层 MLP，但不同位置之间在这一步不直接交互。**

常见形式是：

$$
\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

这里的激活函数 \( \sigma \) 可能是 ReLU、GELU 等。

它的主要作用不是建模序列关系，而是：

- 提升非线性表达能力
- 对 attention 汇总后的特征做进一步加工

所以可以把它理解成：

- attention 决定“看谁”
- FFN 决定“怎么处理看到的内容”

### 4.5 Residual Connection

Transformer 的每个核心子层外面，通常都会包一层残差连接。

例如：

$$
x' = x + \text{Sublayer}(x)
$$

这意味着子层不是完全替换旧表示，而是在旧表示基础上叠加一个修正项。

它的价值在深层网络里非常大：

- 缓解梯度消失
- 保留原始信息通路
- 让训练更稳定

很多时候你会发现，真正让深层模型能“堆起来”的，不只是核心模块有多聪明，还有这些看起来很朴素的训练技巧。

### 4.6 LayerNorm

Transformer 中另一个关键零件是 LayerNorm。

它通常放在子层附近，用于稳定训练。

你可以先把它理解成：

**对每个样本、每个位置的隐藏表示做标准化，让不同层之间的数值分布更可控。**

从直觉上说，它像是在提醒网络：

“不要让某一层的激活值突然爆得太大，也不要让尺度飘得太厉害。”

现代很多实现里，还会区分：

- Post-LN：先子层，再残差，再 LayerNorm
- Pre-LN：先 LayerNorm，再进子层，再做残差

工程上 Pre-LN 在深层训练里通常更稳定，所以很多现代 LLM 实现更偏向这种写法。

## 5. 一个 Transformer Block 是怎么串起来的

理解完零件之后，接下来最重要的问题是：

**这些零件到底按什么顺序工作？**

先看最经典的 Transformer block 思路。

### 5.1 输入表示

设输入序列长度为 \( T \)，隐藏维度为 \( d_{model} \)。

输入 token ids 先经过 embedding，得到：

$$
X \in \mathbb{R}^{T \times d_{model}}
$$

再加上位置向量：

$$
H^{(0)} = X_{token} + X_{pos}
$$

这时每个位置已经同时携带了：

- token 语义信息
- 位置信息

### 5.2 进入 self-attention 子层

然后每个位置通过 self-attention 与整段上下文交互。

如果是普通 Transformer encoder，一个位置可以看所有 token；
如果是 GPT 这样的生成模型，则要加 causal mask，禁止当前位置偷看未来。

这一层的作用是：

**让每个 token 根据当前任务，从整段上下文中读取自己需要的信息。**

### 5.3 残差连接和 LayerNorm

attention 子层之后，通常会接：

- residual
- LayerNorm

写成简化形式可以是：

$$
H' = \text{LayerNorm}(H + \text{Attention}(H))
$$

或者在 Pre-LN 结构里写成：

$$
H' = H + \text{Attention}(\text{LayerNorm}(H))
$$

两种写法都很常见，核心思想不变：

**attention 负责信息交互，残差和规范化负责让训练稳定。**

### 5.4 进入 FFN 子层

接下来每个位置再单独进入 FFN：

$$
H'' = \text{LayerNorm}(H' + \text{FFN}(H'))
$$

或对应的 Pre-LN 版本。

这一步不直接建模位置之间的关系，而是对当前表示做更强的非线性变换。

### 5.5 多层堆叠

一个 block 还不够。

Transformer 的能力很大程度上来自：

**把这样的 block 反复堆叠很多层。**

随着层数加深，模型可以逐步形成：

- 更复杂的上下文表示
- 更抽象的语义特征
- 更强的生成或理解能力

所以 Transformer 的一个核心思想其实很朴素：

**同一种高质量 block，稳定地重复很多次。**

## 6. 三类主流 Transformer 架构

理解 block 之后，下一步就能看懂为什么后来会出现不同“家族”。

虽然大家底层都来自 Transformer，但根据任务目标不同，主流架构大致分成三类。

## 6.1 Encoder-only

代表模型：BERT、RoBERTa

这类模型主要由 encoder stack 构成。

它的特点是：

- 每个位置通常可以看到左右两边上下文
- 更适合做表示学习和理解任务
- 常见于分类、抽取、匹配、判别任务

为什么它擅长理解？

因为对于一句完整输入，encoder-only 模型在编码时可以双向整合上下文。

例如句子里一个词的表示，不仅能参考左边，也能参考右边。

这对下面这些任务很有帮助：

- 句子分类
- 命名实体识别
- 文本匹配
- 检索编码

但它天然不是最直接的生成架构，因为生成任务要求模型在第 `t` 步只能依赖前 `t-1` 步。

## 6.2 Decoder-only

代表模型：GPT 系列、LLaMA、Qwen 等大多数现代自回归 LLM

这类模型只保留 decoder stack，但在语言建模场景里，通常只用到：

- masked self-attention
- FFN
- 残差连接
- LayerNorm

而不一定使用原始 seq2seq Transformer decoder 里的 cross-attention。

它的核心特点是：

- 使用 causal mask
- 每个位置只能看见自己和左边上下文
- 天然适配 next-token prediction

例如训练时：

输入：

`I love deep`

目标：

让模型预测下一个 token 更可能是 `learning`

这个目标和 decoder-only 结构是完全对齐的。

所以它特别适合：

- 文本续写
- 对话生成
- 代码生成
- 通用自回归语言建模

## 6.3 Encoder-decoder

代表模型：T5、BART

这类模型同时保留 encoder 和 decoder 两部分。

它的典型工作方式是：

- encoder 先把输入序列编码成上下文表示
- decoder 再基于这些表示一步步生成输出

这里 decoder 除了有 masked self-attention，还会多一个：

- cross-attention

也就是 decoder 在生成时，不仅看自己前面已经生成的 token，还会去读取 encoder 输出。

这种结构非常适合输入和输出都比较明确的 seq2seq 任务，例如：

- 机器翻译
- 摘要
- 改写
- 问答

因为它天然支持：

**先理解输入，再条件生成输出。**

## 6.4 三类架构的直观对比

可以先把它们粗略记成下面这样：

| 架构类型 | 代表模型 | 上下文可见性 | 更适合什么 |
| --- | --- | --- | --- |
| encoder-only | BERT | 双向 | 理解、表示、判别 |
| decoder-only | GPT | 仅左侧可见 | 自回归生成 |
| encoder-decoder | T5 | encoder 双向，decoder 左侧可见 | 条件生成、seq2seq |

这张表不是为了背诵，而是帮助你建立一个特别重要的意识：

**架构不是凭空选的，而是和训练目标、任务形式强相关。**

## 7. GPT 为什么采用 decoder-only

这一节是本章最重要的落点之一。

很多人会问：

既然 encoder-decoder 看起来更完整，为什么 GPT 不走那条路，而是选择 decoder-only？

原因可以从四个角度理解。

### 7.1 它和 next-token prediction 完全对齐

GPT 的核心训练目标是：

**给定前文，预测下一个 token。**

这要求模型在第 \( t \) 个位置只能看：

- \( x_1 \)
- \( x_2 \)
- ...
- \( x_{t-1} \)

而 decoder-only + causal mask 恰好就是为这种约束设计的。

所以它在结构上和训练目标天然一致。

### 7.2 它和生成过程完全一致

推理时，GPT 的工作方式也是：

1. 读入当前上下文
2. 预测下一个 token 概率分布
3. 采样或选出一个 token
4. 把它拼回上下文
5. 继续下一步

这本质上就是 decoder-only 的自回归生成过程。

所以它的训练和推理在形式上非常统一。

### 7.3 架构更简洁，更容易做大规模预训练

相较于 encoder-decoder，decoder-only 在通用语言建模场景下通常更直接：

- 输入输出统一成一条 token 序列
- 训练目标简单清晰
- 模型设计更容易标准化

这对于大规模预训练很重要。

因为一旦训练范式足够统一，数据组织、并行训练、推理部署都会更顺。

### 7.4 它特别适合“通用生成器”这个产品形态

ChatGPT、代码补全、写作助手、Agent 背后的大模型，本质上都强依赖一种能力：

**给定已有上下文，持续生成后续 token。**

而 decoder-only 恰好天然擅长这件事。

所以从研究目标到产品形态，GPT 选择 decoder-only 都很自然。

这并不意味着 encoder-only 或 encoder-decoder 不重要，而是说明：

**对于通用自回归生成，decoder-only 是最顺手的架构。**

## 8. 从结构图到代码骨架

到这里，我们已经可以把 Transformer block 映射成代码里的模块关系。

先看一个极简伪代码版本：

```python
class TransformerBlock(nn.Module):
    def __init__(self):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

如果继续向上封装成一个最小 GPT，大致会是：

```python
class GPT(nn.Module):
    def __init__(self):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, idx):
        tok = self.token_emb(idx)
        pos = self.pos_emb(position_ids)
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

这里最重要的不是记代码，而是建立对应关系：

- `token_emb`：token embedding
- `pos_emb`：位置表示
- `blocks`：重复堆叠的 Transformer blocks
- `lm_head`：把隐藏状态映射回词表 logits

这也正是第 7 章 Mini-GPT 要真正实现的内容。

所以你可以把这一章当成：

**第 7 章代码实现前的结构蓝图。**

## 9. 常见误区

### 误区 1：Transformer 就是 Attention

不对。

更准确地说，Attention 是 Transformer 的核心模块，但完整 Transformer 还包括：

- 位置表示
- FFN
- 残差连接
- LayerNorm
- 多层堆叠

真正可训练、可扩展的大模型，是这些模块共同作用的结果。

### 误区 2：有了 self-attention 就天然知道顺序

不对。

self-attention 本身只是在 token 间计算相关性，并不自带“第几个位置”的概念。

所以如果没有位置编码或位置嵌入，模型无法可靠地区分不同顺序。

### 误区 3：FFN 只是一个可有可无的小 MLP

不对。

FFN 不是装饰件，它承担了重要的非线性变换功能。

如果只有 attention 而没有 FFN，模型的信息交互有了，但表示加工能力不够强。

### 误区 4：BERT 和 GPT 的区别只是一个双向一个单向

这句话只说对了一半。

两者的差别不只是可见性不同，还包括：

- 训练目标不同
- 更适合的任务类型不同
- 输出使用方式不同

所以它们不是同一个模型换个 mask 那么简单，而是面向不同任务偏好的两条路线。

### 误区 5：decoder-only 比 encoder-decoder 更“低级”

不对。

架构没有绝对高级和低级，只有任务匹配不匹配。

对于通用自回归生成，decoder-only 往往更直接、更高效、更容易规模化；
对于翻译、摘要等条件生成任务，encoder-decoder 往往更自然。

## 10. 面试问题

### Q1：一个标准 Transformer block 通常由哪些部分组成？

通常包括：

- multi-head attention
- FFN
- residual connection
- LayerNorm

在进入 block 之前，还会有 token embedding 和 position embedding。

### Q2：为什么 Transformer 需要位置编码？

因为 self-attention 本身不天然感知顺序。

如果不显式注入位置信息，模型就很难区分：

- `A B C`
- `C B A`

这样的序列差异。

### Q3：FFN 在 Transformer 里起什么作用？

attention 负责从上下文中聚合信息，FFN 负责对每个位置的表示做进一步非线性变换，提升表达能力。

### Q4：encoder-only、decoder-only、encoder-decoder 的区别是什么？

- encoder-only 更偏理解和表示学习
- decoder-only 更偏自回归生成
- encoder-decoder 更偏条件生成和 seq2seq

### Q5：GPT 为什么采用 decoder-only？

因为 GPT 的训练目标是 next-token prediction，而 decoder-only 配合 causal mask 与这一目标天然对齐，训练和推理形式也保持一致。

## 11. 本章小结

这一章最重要的结论可以浓缩成三句话：

第一，Transformer 不只是 Attention，而是：

**Attention + position + FFN + residual + LayerNorm + 多层堆叠。**

第二，不同 Transformer 架构并不是随便分叉出来的，而是和任务形式强相关：

- BERT 偏理解
- GPT 偏生成
- T5 偏条件生成

第三，GPT 之所以成为现代通用大模型的主流路线之一，一个关键原因就在于：

**decoder-only 与 next-token prediction 和自回归生成天然匹配。**

理解了这一层，下一章再去从零实现一个 Mini-GPT，就不会只是“照着代码搭积木”，而是会知道每个模块为什么存在、为什么这样连接、为什么最终会长成 GPT 这种样子。
