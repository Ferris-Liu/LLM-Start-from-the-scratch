# 机器于此理解语言

> 一个AI炼丹师的大模型学习笔记。

[![Deploy Docs](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

在线阅读：[ferris-liu.github.io/LLM-Start-from-the-scratch](https://ferris-liu.github.io/LLM-Start-from-the-scratch)

仓库展示名：`then-language-generates-by-computer`

---

## 这本笔记在写什么

这不是一份停留在 API 调用层的速成手册，而是一条从语言模型底层结构一路走到工程实践的学习主线。

主线大致是：

`数据 -> Tokenizer -> Transformer -> 预训练 -> 微调 -> RAG -> Agent -> LLMOps / 部署`

我希望把每一章都写成三件事同时成立：

- 能建立直觉
- 能看到公式和代码
- 能沉淀成可复现的实验与项目

## 写作定位

这套笔记面向希望真正理解语言模型内部机制，而不只停留在工具调用层的学习者。

如果你在意的是下面这些问题，那它就是写给你的：

- Transformer 为什么这样设计
- 预训练到底在学什么
- 微调、RAG、Agent 分别解决什么问题
- 一个大模型项目怎样从原理走到工程落地

## 主线参考

这套笔记主要沿着三条主线组织：

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)

它们分别对应三种能力：

- `CS336`：理解语言模型的完整开发流程
- `Hugging Face`：掌握 Transformers、数据处理、微调与工具链
- `From Scratch`：从零实现 GPT 风格模型，理解训练与推理细节

## 全书结构

### Part 0：学习路线与岗位定位

- 第 0 章：大模型岗位地图与学习路线

### Part 1：基础补齐

- 第 1 章：机器学习与深度学习最小必要基础
- 第 2 章：PyTorch 与训练循环

### Part 2：NLP 与语言模型基础

- 第 3 章：NLP 基础与文本表示
- 第 4 章：从 n-gram 到神经语言模型

### Part 3：Transformer 核心

- 第 5 章：Attention 机制
- 第 6 章：Transformer 架构
- 第 7 章：从零实现一个 Mini-GPT

### Part 4：现代 LLM 训练流程

- 第 8 章：Tokenizer 与数据处理
- 第 9 章：预训练 Pretraining
- 第 10 章：Scaling Law 与大模型为什么变大
- 第 11 章：LLM 评估

### Part 5：后训练与微调

- 第 12 章：Supervised Fine-Tuning，SFT
- 第 13 章：参数高效微调：LoRA / QLoRA / PEFT
- 第 14 章：偏好对齐：RLHF / DPO / RLAIF

### Part 6：LLM 应用工程

- 第 15 章：Embedding、向量数据库与语义检索
- 第 16 章：RAG 从原理到项目
- 第 17 章：Agent 与 Tool Calling
- 第 18 章：LLMOps、部署与生产化

完整提纲和阶段安排见 [docs/roadmap.md](docs/roadmap.md)。

## 项目路线

这套笔记会尽量沉淀成几类可展示项目：

1. `Mini-GPT from Scratch`
2. `LoRA Fine-tuning 中文任务`
3. `Course Notes RAG Assistant`
4. `LLM Evaluation Benchmark`

目标不是只会“学过”，而是能把学习过程整理成：

- 代码仓库
- README / 技术文档
- 实验记录
- 博客或复盘材料

## 仓库结构

```text
then-language-generates-by-computer/
├── README.md
├── docs/
│   ├── index.md
│   ├── roadmap.md
│   ├── book/
│   │   ├── index.md
│   │   ├── 00-role-roadmap.md
│   │   ├── 01-ml-dl-basics.md
│   │   ├── ...
│   │   └── 18-llmops-deployment.md
│   └── assets/
├── mkdocs.yml
└── requirements-docs.txt
```

## 当前状态

- 排版优化中
- 第一章码字ing


## License

MIT © FerrisLIU
