# 第 8 章：Tokenizer 与训练数据入口

## 1. 本章要解决的问题

第 7 章里，我们已经从零实现了一个最小的 Mini-GPT。

但当我们真的准备训练它时，马上会遇到一个很现实的问题：

**自然语言文本到底是怎么变成模型能吃进去的张量的？**

模型并不认识“句子”“单词”或者“汉字的含义”。

它真正接收的输入是：

- 一串离散 token id
- 对应的 attention mask
- 在训练时还要有 labels

所以这一章要回答的，不只是“Tokenizer 是什么”，而是更完整的问题：

**从原始文本到可训练样本，中间到底发生了什么？**

这件事在全书结构里有三个作用：

- 它承接第 7 章，把 Mini-GPT 从“模型定义”推进到“可以吃数据训练”
- 它为第 9 章的预训练流程铺路
- 它也会让你第一次真正理解，数据处理并不是训练前的边角料，而是语言模型系统的一部分

如果第 7 章的主角是：

**模型结构本身。**

那么第 8 章的主角就是：

**文本如何被变成模型训练所需的输入格式。**

## 2. 你学完后应该会什么

- 能解释 token、vocabulary、special token 分别是什么
- 能说清为什么现代 LLM 大多使用 subword tokenizer
- 能区分 BPE、WordPiece 和 SentencePiece 的核心思路
- 能把原始文本变成 `input_ids`、`labels` 和 `attention_mask`
- 能理解 padding、truncation、packing 在训练中的作用
- 能把这一章的数据流程和第 7 章的 Mini-GPT 接起来

## 3. 为什么模型不能直接读取文本

人看到一句话：

`I love deep learning`

会自然理解这是 4 个英文单词组成的句子。

但神经网络不会这样想。

对模型来说，输入最终必须是数值张量。也就是说，在进入 embedding 层之前，文本必须先被映射成某种离散符号序列：

$$
\text{text} \rightarrow \text{tokens} \rightarrow \text{token ids}
$$

这里的 tokenizer，本质上就是一个“翻译器”：

- 输入是字符串
- 输出是 token 序列
- 再进一步映射成词表中的整数 id

例如，一句文本可能会被切成：

```text
["I", " love", " deep", " learning"]
```

然后再映射成：

```text
[40, 1842, 2387, 7123]
```

后面的 embedding 层才会把这些 id 查表变成向量。

所以 tokenizer 的作用并不是“附属小工具”，而是：

**语言进入模型世界的入口。**

如果没有 tokenizer，后面的 embedding、attention、loss 计算都无从谈起。

## 4. Tokenizer 的核心概念

### 4.1 token 是什么

token 可以理解成：

**模型处理文本时使用的最小离散单元。**

这个单元不一定等于：

- 一个单词
- 一个汉字
- 一个字符

在现代 LLM 里，token 更常见的是 subword，也就是“子词片段”。

例如：

```text
unbelievable
-> ["un", "believ", "able"]
```

或者：

```text
ChatGPT is amazing
-> ["Chat", "G", "PT", " is", " amazing"]
```

### 4.2 vocabulary 是什么

vocabulary，也就是词表，是 tokenizer 允许使用的全部 token 集合。

每个 token 都会有一个唯一 id：

$$
\text{token} \leftrightarrow \text{id}
$$

例如：

```text
"hello" -> 15496
" world" -> 995
```

模型真正训练的并不是字符串，而是这些 id 所对应的 embedding 参数。

### 4.3 special tokens 是什么

除了普通文本 token，模型里通常还会有一些特殊 token，例如：

- `BOS`：begin of sequence，序列开始
- `EOS`：end of sequence，序列结束
- `PAD`：padding token，用来把 batch 补齐到相同长度
- `UNK`：unknown token，表示未知 token

不同模型是否使用这些 token、具体 id 是多少，可能并不相同。

例如 GPT 系列常常不依赖 `PAD` 做预训练，而 BERT 类模型通常更明确地使用 padding 和 segment 相关标记。

这一点提醒我们：

**tokenizer 从来不是脱离模型单独存在的。**

它必须和模型训练目标、数据格式、推理方式一起看。

## 5. 为什么现代 LLM 更偏向 subword

如果我们随便设计文本切分方式，看起来至少有三种选择：

- 按词切
- 按字符切
- 按子词切

为什么今天大多数 LLM 都更偏向 subword tokenizer？

因为它在词级和字符级之间做了一个很好的工程折中。

### 5.1 word-level 的问题

如果按“完整单词”建词表，会遇到两个明显问题。

第一，词表会非常大。

自然语言里有：

- 词形变化
- 专有名词
- 拼写错误
- 新词
- 领域术语

如果都作为独立单词处理，词表膨胀会非常快。

第二，会有严重的 OOV 问题，也就是 out-of-vocabulary。

只要测试时遇到没见过的新词，模型就很难处理。

### 5.2 char-level 的问题

如果按字符切，OOV 问题确实会大幅缓解，因为几乎所有文本都能表示成字符序列。

但代价是序列太长。

例如一个英文单词按字符拆开后，token 数会明显上升，这会直接增加：

- 序列长度
- attention 计算成本
- 训练时间

而且字符级 token 往往缺少足够语义，模型需要在更长上下文中自己重新组合出词和短语结构。

### 5.3 subword 的折中

subword 的核心想法是：

**高频模式作为整体保留，低频词拆成更小片段。**

这样就同时获得了几个优点：

- 词表不会像 word-level 那样无限膨胀
- 新词也能通过已有子词组合表示
- 序列长度通常又不会像 char-level 那样夸张

所以从工程角度看，subword 是一个非常自然的选择。

## 6. 三类主流 tokenizer 方法

这一节不追求论文级细节，重点是建立“方法地图”。

### 6.1 BPE：Byte Pair Encoding

BPE 是大模型里最常见、也最值得优先掌握的一类思路。

它最早的直觉非常简单：

1. 先从较小粒度开始，例如字符或字节
2. 统计语料里最常一起出现的相邻 token 对
3. 把这个高频 token 对合并成一个新 token
4. 重复这个过程 many times，逐步长出词表

例如一开始可能有：

```text
"l", "o", "w"
"l", "o", "w", "e", "r"
```

如果 `("l", "o")` 很常一起出现，就先合并成 `"lo"`。

之后如果 `("lo", "w")` 也很高频，就继续合并成 `"low"`。

这样，词表会逐步从小单位长出高频片段。

BPE 的优点是：

- 直观
- 容易实现
- 对高频模式压缩效果好

它也解释了为什么很多 tokenizer 的输出看起来像“半个词、一个词根、再加一个后缀”。

### 6.2 WordPiece

WordPiece 和 BPE 很像，也是在逐步构造 subword 词表。

但它通常更强调：

**哪种合并能在统计意义上更提升当前词表对语料的表示能力。**

可以粗略理解成：

- BPE 更像“频次驱动合并”
- WordPiece 更像“概率收益驱动合并”

在工程使用上，两者给人的体验有时很接近。

### 6.3 SentencePiece

SentencePiece 更像是一套完整框架，而不只是一个单独算法名。

它有两个很重要的特点：

第一，它不强依赖空格分词。

这对中文、日文、多语言场景很有价值，因为这些语言的“词边界”本来就不像英文那样天然清晰。

第二，它可以直接把原始文本当作输入，再由内部规则完成训练和切分。

很多现代多语言模型都使用 SentencePiece 或与其类似的方案。

### 6.4 这一节应该记住什么

真正需要记住的不是每个算法的全部细节，而是：

- 现代 LLM 主流是 subword tokenizer
- BPE 是最值得先掌握的基础方法
- 多语言场景下，不依赖空格的 tokenizer 往往更自然

## 7. 从一句文本到训练样本

这一节是本章最重要的部分。

因为从第 7 章的 Mini-GPT 视角看，我们最终关心的不是“tokenizer 的定义”，而是：

**一段文本怎样变成 next-token prediction 的训练样本。**

### 7.1 从 raw text 到 token ids

假设我们有一句文本：

```text
I love deep learning.
```

第一步是文本预处理。可能包括：

- 去掉明显脏数据
- 统一换行
- 统一空白符
- 按需要保留或删除特殊符号

第二步是 tokenizer 编码：

```text
text
-> tokens
-> token ids
```

例如：

```text
["I", " love", " deep", " learning", "."]
-> [40, 1842, 2387, 7123, 13]
```

### 7.2 从 token ids 到训练窗口

语言模型并不是一次训练整本书，而是把长文本切成一个个有限长度的训练窗口。

假设最大上下文长度是 `block_size = 8`，一段更长的文本 ids 为：

```text
[5, 9, 12, 7, 20, 4, 8, 10, 6]
```

那么一个可能的训练窗口是：

```text
input_ids = [5, 9, 12, 7, 20, 4, 8, 10]
labels    = [9, 12, 7, 20, 4, 8, 10, 6]
```

你会发现：

**labels 本质上就是 input 向左错开一位后的结果。**

这正是 causal language modeling 的训练目标：

给定前文，预测下一个 token。

### 7.3 为什么要 shift

在位置 `t` 上，模型输入看到的是：

$$
x_1, x_2, \dots, x_t
$$

它要预测的是：

$$
x_{t+1}
$$

所以训练时常写成：

$$
\mathcal{L} = - \sum_{t=1}^{T-1} \log p(x_{t+1} \mid x_{\le t})
$$

这就是第 9 章会正式展开的预训练目标。

第 8 章先把数据层面的“shift”看懂，后面你就不会在 loss 计算那里突然断掉。

## 8. padding、truncation、attention mask、packing

这几个概念经常一起出现，但很容易混淆。

### 8.1 为什么需要 padding

现实里的文本长度不一样。

例如一条样本长度是 32，另一条是 127。

但 GPU 上做 batch 计算时，通常希望它们组成规则张量，所以需要把较短样本补到统一长度：

```text
[5, 9, 12]
-> [5, 9, 12, PAD, PAD]
```

这就是 padding。

### 8.2 什么是 truncation

如果一条样本太长，超过模型允许的最大长度，就要截断：

```text
[x1, x2, ..., x4096]
-> keep first 1024 or last 1024
```

这就是 truncation。

截断策略不是一个无关紧要的小细节，因为它会直接影响：

- 模型能看到哪些上下文
- 信息是否被截掉
- 长文本训练样本如何构造

### 8.3 attention mask 是什么

有了 padding 之后，模型不能把 `PAD` 当成真实内容去关注。

所以我们需要一个 attention mask 来告诉模型：

- 哪些位置是真实 token
- 哪些位置只是补齐占位

例如：

```text
input_ids      = [5, 9, 12, PAD, PAD]
attention_mask = [1, 1, 1, 0, 0]
```

这里的 `1` 表示可用位置，`0` 表示 padding 位置。

### 8.4 attention mask 和 causal mask 不是一回事

这是面试和工程里都非常常见的混淆点。

padding attention mask 解决的是：

**哪些位置根本不是有效文本。**

causal mask 解决的是：

**当前 token 能不能偷看未来 token。**

前者是“哪些位置存在”，后者是“存在的位置里哪些允许被看见”。

### 8.5 什么是 packing

如果训练数据里有大量短文本，直接每条单独 padding 会浪费很多 token budget。

例如：

- 样本 A 长度 20
- 样本 B 长度 18
- 最大长度设成 128

那么大量位置都在浪费。

packing 的思路是：

**把多条较短文本拼接进同一个训练序列里，提高有效 token 占比。**

这在大规模预训练里很重要，因为训练成本往往按 token 计。

换句话说，数据处理并不只是“准备一下输入”，它直接影响训练效率。

## 9. 中文、英文和代码为什么会切得不一样

如果只看英文例子，很容易误以为 tokenizer 的问题很简单。

但一进入中文、多语言和代码场景，很多现象就会变得明显。

### 9.1 中文

中文天然没有空格分词边界。

例如：

```text
我喜欢机器学习
```

它可以被切成：

- 字级片段
- 多字词片段
- 更灵活的 subword 组合

这也是为什么很多中文或多语言模型更适合使用 SentencePiece 一类不依赖空格的方案。

### 9.2 英文

英文虽然有空格，但“空格分开”并不等于“最适合建模的 token 边界”。

例如：

- 词根和后缀可以拆分
- 低频长词可以拆成多个 subword
- 标点和前后空格也可能被编码进 token 规则里

所以很多英文 tokenizer 输出里，你会看到像 `" hello"` 这样带前导空格的 token。

### 9.3 代码

代码和自然语言又不一样。

代码里大量有意义的模式来自：

- 缩进
- 括号
- 运算符
- 下划线
- 驼峰命名
- 重复的 API 名称

因此代码模型往往对 tokenizer 设计更敏感，因为一个糟糕的切分方式会破坏程序结构信号。

这也是为什么“同样的模型结构”，面对代码任务时 tokenizer 往往会明显影响效果。

## 10. 一个最小代码实验

这一节不追求工业级完整训练，只做两个最小实验。

### 10.1 用现成 tokenizer 看分词结果

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "I love deep learning. 我也喜欢大模型。"
encoded = tokenizer(text)

print(tokenizer.tokenize(text))
print(encoded["input_ids"])
```

你可以观察几件事：

- 英文 token 往往不是按“完整单词”切
- 中英文混合时，token 长度分布并不均匀
- `input_ids` 才是后续模型真正消费的输入

### 10.2 训练一个最小 BPE tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=2000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)

files = ["corpus.txt"]
tokenizer.train(files, trainer)

print(tokenizer.encode("I love large language models").tokens)
```

这个实验的价值不在于训出一个强 tokenizer，而在于你会第一次真正看到：

- 词表是怎么从语料长出来的
- 高频片段为什么会逐渐形成稳定 token
- tokenizer 不是固定天降的，它本身也是训练出来的

## 11. 如何把它接到 Mini-GPT

到了这里，第 7 章的 Mini-GPT 就终于能真正“接数据”了。

最小数据集类通常要做的事情是：

1. 读取原始文本
2. 用 tokenizer 编码成 ids
3. 切成固定长度窗口
4. 返回 `input_ids` 和 `labels`

一个最小示意代码如下：

```python
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        chunk = self.token_ids[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
```

这段代码虽然很短，却连接了前后两章：

- 第 7 章负责定义模型
- 第 8 章负责把文本变成训练样本
- 第 9 章负责把样本真正送进训练循环并优化参数

## 12. 常见误区

### 误区 1：tokenizer 就是分词工具

这不够准确。

Tokenizer 不只是“把一句话切开”，它还定义了：

- 词表
- 特殊 token
- 文本到 id 的映射方式
- 后续数据格式

它是整个训练系统的一部分。

### 误区 2：一个 token 大致就等于一个词

这通常不成立。

一个 token 可能是：

- 半个词
- 一个前缀
- 一个标点
- 一个字节片段

所以 token 数和字数、词数不能直接画等号。

### 误区 3：attention mask 和 causal mask 是同一个东西

不是。

padding mask 处理的是无效位置，causal mask 处理的是未来信息泄漏。

两者经常同时存在，但含义完全不同。

### 误区 4：词表越大越好

词表太小会导致序列变长，词表太大又会增加 embedding 和输出层规模。

词表大小是一个工程权衡，不是越大越先进。

## 13. 面试问题

### Q1：为什么现代 LLM 大多采用 subword tokenizer？

因为它兼顾了词级方法的语义完整性和字符级方法的开放词表能力，是词表大小、OOV 风险和序列长度之间的工程折中。

### Q2：BPE 的核心思想是什么？

从较小粒度开始，反复合并语料中高频共现的 token 对，逐步构造出更适合语料分布的 subword 词表。

### Q3：padding mask 和 causal mask 的区别是什么？

padding mask 用来屏蔽补齐位置，causal mask 用来防止当前位置看到未来 token。一个解决“哪些位置有效”，一个解决“有效位置中哪些允许被看见”。

### Q4：为什么 tokenizer 会影响代码模型效果？

因为代码里有大量结构化符号和重复模式。若切分方式破坏了变量名、运算符和语法边界，模型就更难捕捉程序结构。

## 14. 本章小结

这一章最核心的收获，不是记住几个 tokenizer 名字，而是建立下面这条完整主线：

$$
\text{raw text}
\rightarrow \text{tokenizer}
\rightarrow \text{token ids}
\rightarrow \text{training samples}
\rightarrow \text{Mini-GPT input}
$$

从这里开始，我们已经具备了进入预训练的最后一块基础。

下一章会正式回答：

**当模型、tokenizer 和训练样本都准备好以后，一个最小的 GPT 预训练流程到底怎样跑起来？**
