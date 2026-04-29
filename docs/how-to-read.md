# 如何阅读这本书

这本笔记不是按“扫一遍目录就结束”的方式设计的，它更像一本需要边读边跑、边记边改的技术书。

## 如果你是第一次来

推荐先按这个顺序走：

1. [作者介绍](about-author.md)：先知道这本笔记是从什么视角写的。
2. [前言](preface.md)：理解为什么这本笔记会这样展开。
3. [全书提纲与学习路线](roadmap.md)：建立全局地图。
4. [第 1 章：机器学习与深度学习最小必要基础](book/01-ml-dl-basics.md)：把训练闭环补稳。
5. [第 5 章：Attention 机制](book/05-attention.md)：进入 Transformer 主线。

## 三种读法

### 1. 从零跟读

适合刚开始系统学习 LLM 的读者。

建议路线：

- Part 1 基础补齐
- Part 2 NLP 与语言模型基础
- Part 3 Transformer 核心
- Part 4 现代 LLM 训练流程
- Part 5 / Part 6 再进入应用与工程

### 2. 先啃核心

适合已经有一点基础、但想尽快抓住关键的读者。

建议先看：

- [第 5 章：Attention 机制](book/05-attention.md)
- [第 6 章：Transformer 架构](book/06-transformer-architecture.md)
- [第 7 章：从零实现一个 Mini-GPT](book/07-mini-gpt.md)

### 3. 先看应用链路

适合已经了解基础概念，想先建立工程直觉的读者。

建议先看：

- [第 15 章：Embedding、向量数据库与语义检索](book/15-embedding-retrieval.md)
- [第 16 章：RAG 从原理到项目](book/16-rag.md)
- [第 17 章：Agent 与 Tool Calling](book/17-agent-tool-calling.md)

## 怎么读效果最好

我更推荐你把它当成一本“要动手的书”来用：

- 遇到公式，自己推一遍
- 遇到代码，亲手跑一遍
- 遇到实验，改几个参数看看结果
- 每学完一章，写一页自己的复盘

这样知识才会从“看过”变成“真的会”。

## 你最终应该带走什么

理想情况下，读完这本笔记，你不只是多了很多术语，而是能逐步沉淀出这些东西：

- 一套更完整的 LLM 知识地图
- 几个能拿得出手的 from-scratch 或应用项目
- 一批可以复用的技术文档、README 和博客素材
- 一种更稳定的学习和表达方式
