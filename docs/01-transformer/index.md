# 01 · Transformer 底层精通

!!! abstract "本章目标"
    深入理解 Transformer 的每一个组件，能够从零手写实现，并掌握现代大模型对原始架构的关键改进。

## 本章内容

- [Attention 机制](attention.md) — Self-Attention 数学推导与代码实现
- [位置编码](positional-encoding.md) — 从 Sinusoidal 到 RoPE / ALiBi
- [KV Cache](kv-cache.md) — 推理加速的核心机制
- [主流架构对比](architectures.md) — GPT / LLaMA / Mistral / Qwen 的设计选择
- [动手实现 miniGPT](minigpt.md) — 完整代码 + 训练实验

## 前置知识

- [x] 熟悉 PyTorch 张量操作
- [x] 了解基础神经网络（MLP、CNN、RNN）
- [x] 了解反向传播原理

## 推荐资源

| 资源 | 类型 | 说明 |
|------|------|------|
| [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) | 视频 | 强烈推荐，从零实现 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 论文 | 原始论文，必读 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 博客 | Jay Alammar 图解，帮助建立直觉 |

## 学习时间参考

> 有深度学习基础：约 **1 周**（精读 + 动手实现）
