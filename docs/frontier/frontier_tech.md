# LLM 前沿技术与新概念补充

> 本文档用于记录大模型相关前沿技术、新工具、新框架等，
> 便于后续补充到 18 章笔记中，并保持知识体系统一。

---

## 使用说明

1. 每当遇到新技术，直接按下面的模板新增一个条目。
2. 在“对应章节”里标记建议归档位置，先写章节号或 TODO 即可。
3. 每隔一段时间，把这里的内容同步到正式章节里：
   - 如果是核心概念，升级为章节内的小节
   - 如果更偏 demo、工具或案例，放入扩展阅读 / tips
4. 尽量保持字段统一，后续做索引和章节映射会更方便。

---

## 技术条目模板

### 技术名：`<填写技术名称>`

- **类型**：Model / Training / Agent / Tool / Skill / Dataset / Framework
- **发布时间 / 来源**：论文、blog、GitHub 或官方文档链接
- **原理概述**：
  - 核心原理（尽量 2-4 行）
  - 核心组件 / 算法 / 模块
- **典型应用 / Demo**：
  - 可运行示例或典型场景
  - 示例代码片段或 repo 链接
- **对应章节**：
  - 建议放置的 18 章笔记章节
- **笔记状态**：
  - TODO / 写完 / 待更新
- **备注**：
  - 适合初学者理解的重点
  - 注意事项或坑

---

## 已收录条目

### 技术名：OpenClaw

- **类型**：LLM Skill / Tool
- **发布时间 / 来源**：[GitHub/OpenClaw](https://github.com/openclaw)
- **原理概述**：
  - 通过 LLM 调用外部工具或 API，把“生成回答”扩展为“执行任务”。
  - 常见实现思路是让模型先决定动作，再把工具执行结果回填给模型继续推理。
  - 可以和 chain-of-thought、tool calling、工作流编排结合，用于构造更完整的 Agent loop。
- **典型应用 / Demo**：
  - 自动化办公流程
  - 数据分析助手
  - 可进一步结合 RAG 或 skills 机制做复合任务
- **对应章节**：
  - 第 17 章 Agent 与 Tool Calling
- **笔记状态**：
  - TODO
- **备注**：
  - 初学者可以先理解 tool calling 的基本流程：定义工具、生成参数、执行工具、回填结果。
  - 如果后续要正式写进章节，建议把它当作“工具使用能力”案例，而不是单独抽象成全新理论。

### 技术名：Hermes Agent

- **类型**：Multi-agent / LLM Agent Framework
- **发布时间 / 来源**：[GitHub/Hermes](https://github.com/hermes-agent)
- **原理概述**：
  - 强调分层 agent、任务拆解、记忆与规划，让复杂任务不必由单个 prompt 一次性完成。
  - 常见系统模块包括 planning、reflection、memory、tool use，以及多 agent 间的职责分工。
  - 它适合拿来理解“Agent 不只是一次工具调用，而是一个可迭代推进的任务系统”。
- **典型应用 / Demo**：
  - 复杂任务自动化
  - 科学研究 / 数据分析 multi-agent
  - 多角色协作式任务拆解与执行
- **对应章节**：
  - 第 17 章 Agent 与 Tool Calling
- **笔记状态**：
  - TODO
- **备注**：
  - 初学者先抓住核心思想：任务拆解、角色分工、状态记忆、结果反思。
  - 做 demo 时不必一开始就上多 agent，可以先用 1-2 个工具做最小闭环。

---

## 技术映射表

| 技术名称 | 类型 | 对应章节 | 笔记状态 | 来源 / 链接 |
|----------|------|----------|----------|-------------|
| OpenClaw | Skill / Tool | 17 Agent | TODO | GitHub/OpenClaw |
| Hermes Agent | Multi-agent | 17 Agent | TODO | GitHub/Hermes |

---

## 后续可补充方向

- 推理增强：test-time scaling、self-reflection、tree/graph search
- Agent 工程：memory、planning、multi-agent orchestration、tool sandbox
- 训练范式：合成数据、post-training 新配方、agentic finetuning
- 应用框架：skills、workflow engine、browser/computer use、长期记忆系统
