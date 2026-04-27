# 全书提纲与学习路线


目标不是零散记录知识点，而是围绕“学习闭环 + 项目闭环 + 求职闭环”组织整套内容。

## 写作定位

- 面向已经具备机器学习基础、希望进入大模型方向的学生或初级工程师
- 强调从原理到工程实现，而不是只停留在 API 使用
- 每一章尽量同时覆盖直觉、数学、代码、实验、面试表达

## 主线参考资源

- Stanford CS336：适合按语言模型完整开发流程学习
- Hugging Face LLM Course：适合掌握 Transformers、数据处理、微调和训练工具链
- Sebastian Raschka《Build a Large Language Model From Scratch》：适合从零实现 GPT-style LLM、预训练和微调
- Stanford CS224N：补 NLP 基础和经典任务背景
- Speech and Language Processing：作为 NLP 参考书
- Full Stack LLM Bootcamp：补应用与工程实践
- Berkeley LLM Agents：补 Agent 理解与项目思路

## 推荐学习顺序

### 第 0 阶段：搭建仓库

- 创建 GitHub repo
- 确定目录结构
- 写 README
- 建立 `notes / code / projects / papers` 文件夹

### 第 1 阶段：基础补齐

- 第 1 章：机器学习与深度学习最小必要基础
- 第 2 章：PyTorch 与训练循环

### 第 2 阶段：NLP 与语言模型

- 第 3 章：NLP 基础与文本表示
- 第 4 章：从 n-gram 到神经语言模型

### 第 3 阶段：Transformer 核心

- 第 5 章：Attention 机制
- 第 6 章：Transformer 架构
- 第 7 章：从零实现一个 Mini-GPT

### 第 4 阶段：现代 LLM 训练

- 第 8 章：Tokenizer 与数据处理
- 第 9 章：预训练
- 第 10 章：Scaling Law
- 第 11 章：LLM 评估

### 第 5 阶段：后训练与微调

- 第 12 章：SFT
- 第 13 章：LoRA / QLoRA / PEFT
- 第 14 章：RLHF / DPO / RLAIF

### 第 6 阶段：LLM 应用工程

- 第 15 章：Embedding 与语义检索
- 第 16 章：RAG
- 第 17 章：Agent 与 Tool Calling
- 第 18 章：LLMOps 与部署

### 第 7 阶段：求职强化

- 整理项目 README
- 准备面试题
- 写技术博客
- 做项目 demo
- 准备简历项目描述

## 全书 18 章提纲

| Part | 章节 | 主题 |
|------|------|------|
| Part 0 | 第 0 章 | 大模型岗位地图与学习路线 |
| Part 1 | 第 1 章 | 机器学习与深度学习最小必要基础 |
| Part 1 | 第 2 章 | PyTorch 与训练循环 |
| Part 2 | 第 3 章 | NLP 基础与文本表示 |
| Part 2 | 第 4 章 | 从 n-gram 到神经语言模型 |
| Part 3 | 第 5 章 | Attention 机制 |
| Part 3 | 第 6 章 | Transformer 架构 |
| Part 3 | 第 7 章 | 从零实现一个 Mini-GPT |
| Part 4 | 第 8 章 | Tokenizer 与数据处理 |
| Part 4 | 第 9 章 | 预训练 Pretraining |
| Part 4 | 第 10 章 | Scaling Law 与大模型为什么变大 |
| Part 4 | 第 11 章 | LLM 评估 |
| Part 5 | 第 12 章 | Supervised Fine-Tuning，SFT |
| Part 5 | 第 13 章 | 参数高效微调：LoRA / QLoRA / PEFT |
| Part 5 | 第 14 章 | 偏好对齐：RLHF / DPO / RLAIF |
| Part 6 | 第 15 章 | Embedding、向量数据库与语义检索 |
| Part 6 | 第 16 章 | RAG 从原理到项目 |
| Part 6 | 第 17 章 | Agent 与 Tool Calling |
| Part 6 | 第 18 章 | LLMOps、部署与生产化 |

## 推荐项目路线

### 项目 1：Mini-GPT from Scratch

目标：

- 证明你理解 Transformer 和语言模型底层原理

技术点：

- PyTorch
- Transformer
- Causal LM
- Tokenizer
- Training Loop
- Text Generation

产出：

- GitHub 代码
- README
- 模型结构图
- loss 曲线
- 生成样例
- 实验分析

### 项目 2：LoRA Fine-tuning 中文任务

目标：

- 证明你掌握 Hugging Face 和微调技术

技术点：

- Transformers
- Datasets
- PEFT
- LoRA
- QLoRA
- SFT
- Evaluation

产出：

- 微调脚本
- 数据集说明
- 训练日志
- base model 与 fine-tuned model 对比
- 错误案例分析

### 项目 3：Course Notes RAG Assistant

目标：

- 证明你能构建真实 LLM 应用

技术点：

- PDF parsing
- Chunking
- Embedding
- Vector Database
- Retrieval
- Reranking
- Generation
- Citation
- RAG Evaluation

产出：

- 可运行 demo
- README
- 系统架构图
- 检索评估
- 回答质量评估

### 项目 4：LLM Evaluation Benchmark

目标：

- 证明你具备模型评估能力

技术点：

- evaluation set
- task-specific metric
- LLM-as-a-judge
- RAG evaluation
- bad case analysis
- A/B testing

产出：

- benchmark 数据
- eval 脚本
- 对比报告
- 错误分析

## 每章统一写作模板

```markdown
# 第 X 章：标题
## 1. 本章要解决的问题
本章试图回答什么问题？
为什么这个问题重要？
它和 LLM 有什么关系？

## 2. 核心直觉
用通俗语言解释本章核心概念。
尽量使用类比、图示或简单例子。

## 3. 数学定义
列出必要公式。
解释公式中每个符号的含义。
避免堆砌过多数学细节。

## 4. 代码实现
给出最小可运行代码。
优先使用 PyTorch / Hugging Face。
代码需要配注释。

## 5. 实验观察
记录输入、输出、loss、metric。
分析实验现象。
说明哪些结果符合预期，哪些不符合预期。

## 6. 常见误区
列出初学者容易混淆的地方。

## 7. 面试问题
整理 5-10 个高频问题。
每个问题给出简明答案。

## 8. 延伸阅读
列出课程、论文、博客、官方文档。
```

## 当前仓库说明

- 仓库正文将直接按新版 18 章结构继续写作
- `docs/book/` 是当前主要章节区
- 首页、路线图和导航已经统一到“18 章 + 求职路线”结构

如果你是第一次读，建议按章节顺序推进，并优先补齐 Part 1 到 Part 4。
