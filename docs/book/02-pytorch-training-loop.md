# 第 2 章：PyTorch 与训练循环

## 1. 本章要解决的问题

第 1 章里，我们已经从概念上建立了训练的最小闭环：

`数据 -> 模型 -> 预测 -> loss -> 梯度 -> 参数更新 -> 验证`

但当你第一次真正打开一个 PyTorch 训练脚本时，还是很容易卡在几个地方：

- Tensor 和普通数组到底有什么区别
- 为什么 `loss.backward()` 一调用，梯度就“自动出来了”
- `nn.Module`、Dataset、DataLoader 分别负责什么
- 为什么训练时要 `model.train()`，验证时要 `model.eval()`
- 一个标准训练循环到底长什么样

这一章的目标，就是把“训练的理论闭环”翻译成“训练脚本里的具体对象和步骤”。

你不需要在这一章掌握 PyTorch 的全部 API，但应该做到两件事：

- 能看懂一个最小训练脚本到底在做什么
- 能自己写出一个标准的 train / eval loop

这也是为什么这一章放在第 1 章后面、第 7 章前面。

- 第 1 章解决“为什么要这样训练”
- 第 2 章解决“这些动作在 PyTorch 里怎么落地”
- 第 7 章再把这些部件拼成一个 Mini-GPT

如果这一章吃透，后面你看到的预训练、SFT、LoRA，本质上都只是这个训练循环的不同工程变体。

## 2. 你学完后应该会什么

- 能理解 Tensor、shape、dtype、device 的基本作用
- 能解释 `requires_grad`、autograd 和 `backward()` 的关系
- 能看懂 `nn.Module` 的基本写法
- 能理解 Dataset 与 DataLoader 如何组织 batch 数据
- 能独立写出标准的训练与验证循环
- 能理解 checkpoint 为什么要保存，以及最小保存内容是什么

## 3. 先把训练脚本里的角色认清楚

如果只记一件事，那就是：

PyTorch 不是在发明新的训练逻辑，它只是把第 1 章里的抽象概念变成了代码对象。

可以先做一张最小映射表：

- 数据样本，对应 Dataset 里的单条样本
- 一批样本，对应 DataLoader 产出的一个 batch
- 模型，对应一个 `nn.Module`
- 参数 \( \theta \)，对应 `model.parameters()`
- 预测结果，对应 forward 的输出
- loss，对应一个标量 Tensor
- 梯度，对应参数上的 `.grad`
- 参数更新，对应 `optimizer.step()`
- 验证流程，对应 `model.eval()` 下的前向计算

这张映射表很重要，因为后面你读任何训练代码，本质上都在找这几个角色。

### 一个最小类比

可以把 PyTorch 训练理解成一条装配线：

- Dataset 负责从仓库里拿单件原料
- DataLoader 负责把原料打包成一箱一箱的 batch
- `nn.Module` 负责加工
- loss 函数负责质检
- autograd 负责追踪“是哪一步加工出了问题”
- optimizer 负责根据问题调整机器参数

这样你就能更自然地理解：训练脚本虽然看起来有很多组件，但其实都是在服务同一件事，让 loss 下降。

## 4. Tensor：深度学习里最基本的数据单位

在 PyTorch 里，大部分东西最终都是 Tensor。

你可以先把 Tensor 粗略理解成“带了更多深度学习语义的多维数组”。

例如：

- 一个标量：shape 是 `[]`
- 一个向量：shape 可能是 `[10]`
- 一个 batch 的二维特征：shape 可能是 `[32, 128]`
- 一批图像：shape 可能是 `[32, 3, 224, 224]`

### 4.1 为什么 shape 这么重要

深度学习里最常见的 bug 之一，不是公式错了，而是 shape 不对。

因为模型的每一层都在假设输入张量满足某种结构。

例如：

- batch 维度通常放在最前面
- 全连接层常见输入是 `[batch_size, hidden_dim]`
- 分类 logits 常见输出是 `[batch_size, num_classes]`

如果 shape 没对齐，模型要么直接报错，要么更麻烦：不报错但语义错了。

所以读代码时，你应该养成一个很强的习惯：

每经过一个关键步骤，都问一句“这个 Tensor 的 shape 现在是什么”。

### 4.2 dtype：它里面存的是什么类型

Tensor 不只是有 shape，还有 dtype。

常见的 dtype 包括：

- `torch.float32`：最常见的浮点数类型
- `torch.float64`：精度更高，但更慢、更占内存
- `torch.int64`：常用于类别 id、token id、索引
- `torch.bool`：布尔 mask

很多初学者一开始不太重视 dtype，但它会直接影响：

- 能不能参与某些运算
- 结果精度够不够
- 显存和内存占用大小

例如，embedding 的输入通常必须是整数索引，而不是浮点数。

### 4.3 device：数据和模型放在哪里算

device 解决的是“这些 Tensor 在 CPU 还是 GPU 上”。

最常见的两种情况：

- `cpu`
- `cuda`

这一点看起来像工程细节，但其实非常关键，因为模型参数和输入数据必须在同一个 device 上。

如果模型在 GPU 上、数据还在 CPU 上，运算就会报错。

所以后面你经常会看到：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

这几行虽然朴素，却是训练脚本中最基本的设备对齐步骤。

## 5. Autograd：为什么 `backward()` 能自动求梯度

第 1 章里我们说过，训练需要知道 loss 对参数的梯度。

在 PyTorch 里，这件事主要由 autograd 完成。

### 5.1 `requires_grad=True` 表示什么

如果一个 Tensor 需要参与梯度计算，那么它通常会带上：

```python
requires_grad=True
```

这表示 PyTorch 需要跟踪围绕这个 Tensor 的运算过程，以便后面反向传播时计算梯度。

最典型的就是模型参数。

你通常不需要手动给每个参数都写 `requires_grad=True`，因为 `nn.Module` 里的可训练参数默认就是要参与梯度更新的。

### 5.2 前向传播时，PyTorch 在做什么

当前向计算发生时，PyTorch 不只是算出一个结果，它还会顺手记录这一路上的运算关系。

可以把它想成：

- 你做了一连串数学操作
- PyTorch 一边算结果，一边记下“这个结果是怎么来的”
- 等你最后调用 `loss.backward()` 时，它再顺着这条图反向把梯度传回去

所以，`backward()` 并不是凭空“魔法求导”，而是因为前面已经构建了计算图。

### 5.3 一个最小例子

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
loss = y

loss.backward()
print(x.grad)  # tensor(7.)
```

这里：

- `y = x^2 + 3x`
- 当 `x = 2` 时，导数是 `2x + 3 = 7`

所以 `x.grad` 是 `7`。

这个例子非常简单，但已经包含了训练的核心思想：

- 先前向算出一个标量结果
- 再让 autograd 根据计算图自动求梯度

### 5.4 为什么每轮都要 `zero_grad()`

很多初学者第一次看到训练循环时，都会困惑：

为什么每次反向传播前要先清空梯度？

因为在 PyTorch 里，梯度默认是累加的，不会自动清零。

也就是说，如果你连续两次调用 `backward()`，参数上的 `.grad` 会把两次结果加起来。

所以标准训练循环里通常会有：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

顺序也很重要：

- 先清空旧梯度
- 再计算当前 batch 的梯度
- 最后更新参数

## 6. `nn.Module`：把模型组织起来

从工程角度看，`nn.Module` 的核心作用是：

把模型里的层、参数和前向逻辑收纳进一个统一对象。

一个最小例子如下：

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
```

这里有两个关键点。

### 6.1 `__init__` 负责定义层

这一部分是在声明模型里有哪些可训练组件，例如：

- `nn.Linear`
- `nn.Embedding`
- `nn.LayerNorm`
- `nn.Dropout`

这些层里的参数会自动被 PyTorch 注册到当前模块中。

### 6.2 `forward()` 负责定义数据怎么流过模型

`forward()` 定义的是前向传播逻辑，也就是：

输入一个 Tensor，经过哪些变换，得到输出。

比如：

- 先过线性层
- 再过激活函数
- 再过输出层

后面我们写：

```python
logits = model(x)
```

本质上就是在调用 `forward()`。

### 6.3 为什么 optimizer 能拿到模型参数

因为 `nn.Module` 会自动追踪它内部注册的参数，所以你可以直接写：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

这句代码的意思就是：

把模型中所有需要训练的参数交给优化器管理。

## 7. Dataset 与 DataLoader：训练数据怎么进模型

训练脚本不只是模型，还要考虑数据怎么组织。

PyTorch 一般把这部分拆成两个角色：

- Dataset：定义“单条样本怎么取”
- DataLoader：定义“多条样本怎么拼成 batch”

### 7.1 Dataset 负责单样本

一个最小 Dataset 通常要实现两件事：

- `__len__()`：数据集有多大
- `__getitem__(idx)`：第 `idx` 条样本怎么返回

例如，一个二分类 toy dataset 可以返回：

- 输入特征 `x`
- 对应标签 `y`

### 7.2 DataLoader 负责 batch、shuffle 和迭代

有了 Dataset 以后，DataLoader 会进一步帮你做这些事：

- 每次取出一批样本
- 控制 batch size
- 在训练时打乱样本顺序
- 按迭代器方式持续提供 batch

例如：

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

这句代码可以理解成：

“把训练集打包成每批 32 条，并在每个 epoch 开始前重新打乱。”

### 7.3 为什么 batch 这么重要

理论上你可以每次只用一个样本更新参数，但那样效率低，梯度噪声也大。

batch 的好处在于：

- 更容易利用并行计算
- 梯度更稳定
- 吞吐量更高

当然，batch 也不是越大越好，因为它会受到内存和优化动态的限制。

但在当前阶段，你只需要先理解：

现代深度学习训练基本都不是“单条样本更新”，而是“按 batch 更新”。

## 8. 一个标准训练循环到底在做什么

这一节是全章最核心的部分。

先给出一个高度概括的版本：

1. 从 DataLoader 里取一个 batch
2. 把数据放到正确的 device 上
3. 前向计算得到预测
4. 用预测和标签计算 loss
5. 清空旧梯度
6. 反向传播得到新梯度
7. 用 optimizer 更新参数
8. 记录 loss 和指标

如果把这个流程理解透，后面无论你看到的是图像分类、文本分类还是语言模型训练，本质都大同小异。

### 8.1 最小训练循环模板

```python
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这 6 行基本就是深度学习训练的最小骨架。

### 8.2 每一步分别对应什么含义

#### 第一步：取一个 batch

```python
for x, y in train_loader:
```

表示从训练集里按 batch 取数据。

#### 第二步：前向传播

```python
logits = model(x)
```

表示把输入送进模型，得到预测结果。

对于分类问题，`logits` 一般是还没过 softmax 的分数。

#### 第三步：计算 loss

```python
loss = criterion(logits, y)
```

这一步把“预测结果”和“正确答案”比较起来，量化模型这次错得有多严重。

#### 第四步：清空旧梯度

```python
optimizer.zero_grad()
```

避免和上一个 batch 的梯度混在一起。

#### 第五步：反向传播

```python
loss.backward()
```

autograd 根据当前 loss，把梯度一路传回各层参数。

#### 第六步：参数更新

```python
optimizer.step()
```

优化器读取各参数的 `.grad`，再按照自己的更新规则修改参数。

这就完成了一次完整的训练 step。

## 9. 为什么训练和验证要分开

如果只看训练集 loss，不少时候会产生错觉：

- 训练 loss 在降
- 所以模型应该越来越好

但这并不总是成立，因为模型可能只是越来越会记住训练数据，而不是越来越会泛化。

所以标准流程里，一般要把训练和验证分开。

- 训练集：用来更新参数
- 验证集：用来评估当前模型的泛化能力

### 9.1 `model.train()` 和 `model.eval()` 的作用

这两个调用不是“装饰动作”，而是真的会影响某些层的行为。

例如：

- Dropout 在训练时会随机丢弃部分神经元
- BatchNorm 在训练和推理时统计方式不同

所以通常写法是：

```python
model.train()
```

表示进入训练模式。

而在验证时：

```python
model.eval()
```

表示进入评估模式。

### 9.2 为什么验证时还要 `torch.no_grad()`

验证和推理阶段通常不需要梯度，因为我们不打算更新参数。

所以常见写法是：

```python
model.eval()
with torch.no_grad():
    logits = model(x)
```

这样做的好处是：

- 节省显存
- 降低额外计算开销
- 避免误构建计算图

### 9.3 一个最小验证循环

```python
model.eval()
val_loss = 0.0

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        val_loss += loss.item()
```

注意这里没有：

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

因为验证的目标不是学习，而是测量。

## 10. 从概念到代码：一个最小可运行样板

下面给一个不依赖复杂数据集的最小示例。

这个例子做的是：

- 输入二维点坐标
- 判断它属于 0 类还是 1 类
- 用一个简单 MLP 完成分类

这个任务的好处是足够简单，你可以把注意力集中在训练循环本身，而不是数据处理细节上。

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 构造一个简单的二分类 toy dataset
n = 400
x = torch.randn(n, 2)
y = (x[:, 0] + x[:, 1] > 0).long()

train_x, val_x = x[:320], x[320:]
train_y, val_y = y[:320], y[320:]

train_dataset = TensorDataset(train_x, train_y)
val_dataset = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 2. 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)


# 3. 训练和验证
for epoch in range(20):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            val_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    print(
        f"epoch={epoch:02d} "
        f"train_loss={train_loss / len(train_loader):.4f} "
        f"val_loss={val_loss / len(val_loader):.4f} "
        f"val_acc={correct / total:.4f}"
    )
```

### 10.1 这段代码最值得你观察什么

如果你第一次自己跑，建议重点观察下面几件事：

- `xb` 的 shape 是不是 `[batch_size, 2]`
- `logits` 的 shape 是不是 `[batch_size, 2]`
- `yb` 的 shape 是不是 `[batch_size]`
- `CrossEntropyLoss` 输入的是 logits，不是先 softmax 后的概率
- 训练阶段和验证阶段的代码结构到底差在哪里

只要这些点看明白，你对标准训练循环就已经建立了很稳的骨架。

## 11. checkpoint：为什么训练不能只看当前内存

训练稍微变复杂一点后，你就不能只依赖“程序当前跑着”这一件事了。

因为现实里经常会发生：

- 训练中断
- 机器重启
- 想回到某个历史最好结果
- 想拿某个 epoch 的模型继续微调

这就需要 checkpoint。

### 11.1 最小 checkpoint 通常保存什么

最常见的是保存：

- `model_state_dict`
- `optimizer_state_dict`
- 当前 epoch
- 当前 loss 或最佳指标

例如：

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "checkpoint.pt",
)
```

### 11.2 为什么不只保存模型权重

如果你只想做推理，保存模型权重通常已经够了。

但如果你想恢复训练，只保存权重往往不够，因为优化器内部也有状态。

例如 AdamW 会维护动量相关统计量。

如果这些状态丢了，你虽然还能继续训练，但训练轨迹已经不是原来的那条了。

所以从工程角度看：

- 推理场景：只存模型权重也可以
- 恢复训练场景：最好同时保存 optimizer 状态

## 12. 实验观察：跑这个最小样板时你应该看到什么

如果上面的 toy example 正常运行，你大概率会观察到：

- 训练 loss 逐步下降
- 验证 loss 通常也先下降
- 验证准确率会明显高于随机猜测

这说明模型已经学会了一条简单分类边界。

如果结果不符合预期，可以优先检查：

- 学习率是不是过大
- 输入和标签 shape 是否匹配
- 标签 dtype 是否正确，是否为整数类别 id
- 训练和验证模式有没有切换
- 是否错误地在验证阶段也做了反向传播

这类检查习惯很重要，因为以后模型一旦变大，排查 bug 的方式并不会变，只是对象更复杂。

## 13. 常见误区

### 误区 1：`backward()` 会自动更新参数

不会。

`loss.backward()` 只负责计算梯度，真正更新参数的是 `optimizer.step()`。

### 误区 2：验证时只要不调用 `step()` 就行，不用 `eval()`

不对。

有些层在训练态和评估态行为不同，所以验证前应该显式调用 `model.eval()`。

### 误区 3：loss 降了，模型一定泛化更好了

不一定。

训练集 loss 下降只能说明模型更适应训练数据，不代表对未见数据一定更好，所以必须看验证集表现。

### 误区 4：所有输出都应该先过 softmax 再喂给交叉熵

不对。

在 PyTorch 里，`nn.CrossEntropyLoss()` 通常直接接收 logits，它内部已经包含了相应处理。

### 误区 5：DataLoader 只是“读数据的工具”，不重要

不对。

batch 的构造方式、shuffle、数据加载效率，都会直接影响训练行为和实验速度。

## 14. 面试问题

### Q1：`loss.backward()` 和 `optimizer.step()` 有什么区别

`loss.backward()` 用来计算梯度，把梯度写到参数的 `.grad` 上；`optimizer.step()` 读取这些梯度，并根据优化算法更新参数。前者负责“求导”，后者负责“更新”。

### Q2：为什么训练时要先 `zero_grad()`

因为 PyTorch 默认会累加梯度，不手动清零的话，当前 batch 的梯度会和之前的梯度混在一起，导致更新结果不符合预期。

### Q3：`model.train()` 和 `model.eval()` 的核心区别是什么

它们会切换模型中部分层的行为，尤其是 Dropout 和 BatchNorm。训练时应使用 `train()`，验证和推理时应使用 `eval()`。

### Q4：Dataset 和 DataLoader 分别解决什么问题

Dataset 负责定义单条样本如何读取，DataLoader 负责把样本组织成 batch，并处理 shuffle、迭代和加载流程。

### Q5：为什么 checkpoint 里最好保存 optimizer 状态

因为恢复训练时不只是要恢复模型参数，还要恢复优化器内部状态。否则虽然能继续训练，但训练动态可能已经改变。

## 15. 本章小结

这一章最核心的目标，不是让你记住多少 API，而是让你把训练脚本读成一个完整闭环。

你现在应该能把这些对象连起来看：

- Tensor 是数据载体
- autograd 负责梯度计算
- `nn.Module` 负责组织模型
- Dataset / DataLoader 负责组织数据
- train loop 负责更新参数
- eval loop 负责验证泛化
- checkpoint 负责让训练过程可恢复

如果说第 1 章解决的是“为什么训练会让模型变好”，那么这一章解决的是“这个过程在代码里到底怎么发生”。

有了这层理解，后面进入 NLP、语言模型、Transformer 和 Mini-GPT 时，你就不会只是在记结构图，而是在看一个更复杂版本的训练系统。

## 16. 延伸阅读

- PyTorch 官方文档：Tensor、autograd、`nn.Module`、Dataset / DataLoader
- 《Build a Large Language Model (From Scratch)》里关于训练循环和 GPT 实现的相关章节
- Stanford CS336 中关于语言模型训练基础设施的课程材料
