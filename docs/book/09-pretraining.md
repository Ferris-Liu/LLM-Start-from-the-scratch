# 第 9 章：预训练 Pretraining

## 本章目标

- 理解 causal language modeling 的训练目标
- 理解数据、算力、模型规模之间的关系
- 看懂一个最小预训练流程

## 当前内容安排

- 这一章会承接 Mini-GPT 项目，进入更完整的预训练流程
- 后续会补齐 from-scratch 训练和工具链训练两条线

## 推荐补充方向

- shift logits / shift labels
- 数据清洗与 sequence packing
- warmup、cosine decay、mixed precision、checkpointing

## 当前状态

> 过渡页。后续会把 from-scratch 训练和 Hugging Face 训练流程衔接起来。
