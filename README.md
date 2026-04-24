# LLM 从零到求职：面向机器学习基础学生的大模型学习笔记

> 一份面向硕士求职与工程落地的开源 LLM 学习路线图：从机器学习基础、Transformer 和 GPT from scratch，到微调、RAG、Agent、部署与面试准备。

[![Deploy Docs](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

在线阅读：[ferris-liu.github.io/LLM-Start-from-the-scratch](https://ferris-liu.github.io/LLM-Start-from-the-scratch)

---

## 写作定位

这套笔记面向已经具备机器学习基础、希望进入大模型相关方向的学生或初级工程师。

目标不是教你“怎么调一个 API”，而是把大模型完整链路串起来：

`数据 -> Tokenizer -> Transformer -> 预训练 -> SFT/LoRA/DPO -> Embedding/RAG/Agent -> LLMOps/部署 -> 项目展示与求职`

我会尽量把每一章写成三件事同时成立：

- 能建立直觉
- 能看到公式和代码
- 能沉淀成项目与面试表达

## 这套笔记适合谁

- 已学过机器学习、深度学习，想系统进入 LLM 方向的同学
- 想补齐 Transformer、预训练、微调、RAG、Agent 全链路的人
- 想把学习过程整理成 GitHub 项目、博客、简历素材的人

## 主线参考

这套笔记会主要沿着三条主线组织：

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch)

它们分别对应三种能力：

- `CS336`：理解语言模型完整开发流程
- `Hugging Face`：掌握 Transformers、数据处理、微调与工具链
- `Raschka`：从零实现 GPT-style LLM、预训练和微调

## 学习路线

```text
机器学习基础
-> 深度学习基础
-> NLP 与语言模型
-> Attention 与 Transformer
-> GPT / LLM from scratch
-> Tokenizer 与预训练
-> SFT / LoRA / RLHF / DPO
-> Embedding / RAG / Agent
-> LLMOps / 部署 / 面试
```

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

### Part 7：综合项目与求职准备

- 附录 A：推荐项目路线
- 附录 B：每章统一写作模板
- 附录 C：推荐学习顺序

完整提纲和阶段安排见 [docs/roadmap.md](docs/roadmap.md)。

## 项目产出路线

这套笔记会尽量沉淀成 4 个可展示项目：

1. `Mini-GPT from Scratch`
   证明你理解 Transformer 和语言模型底层原理。
2. `LoRA Fine-tuning 中文任务`
   证明你掌握 Hugging Face、PEFT 与参数高效微调。
3. `Course Notes RAG Assistant`
   证明你能构建一个完整可演示的 LLM 应用。
4. `LLM Evaluation Benchmark`
   证明你具备评估、对比、错误分析和实验设计能力。

## 仓库结构

```text
llm-start-from-scratch/
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

当前正文会直接按“18 章主线式笔记”继续写，不再保留旧的 9 个模块式目录。

## 每章统一写法

每章尽量遵循同一套结构：

1. 本章要解决的问题
2. 核心直觉
3. 数学定义
4. 代码实现
5. 实验观察
6. 常见误区
7. 面试问题
8. 延伸阅读

这样做的目标是让每篇内容既能当学习笔记，也能直接转化为：

- GitHub README
- 项目文档
- 面试复盘材料
- 技术博客提纲

## 本地预览

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

然后访问 `http://127.0.0.1:8000`。

## 当前更新重点

- 先把首页、路线图和整体提纲统一到“从零到求职”这条主线
- 优先打磨 Transformer 和 Mini-GPT 相关章节，作为全书样板
- 后续逐步补齐微调、RAG、Agent、评估和部署部分

## 贡献

欢迎提 Issue 或 PR。

如果你发现：

- 概念解释不清楚
- 公式有错误
- 代码跑不通
- 某个岗位能力拆解不合理
- 有更好的课程、论文或项目参考

都很欢迎直接交流。

## License

MIT © FerrisLIU
