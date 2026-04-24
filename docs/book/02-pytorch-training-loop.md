# 第 2 章：PyTorch 与训练循环

## 本章目标

- 掌握 Tensor、Autograd、`nn.Module`、Dataset 与 DataLoader
- 看懂标准训练循环和验证流程
- 为后续从零实现 Mini-GPT 做准备

## 建议覆盖内容

- Tensor 基础与 device / dtype
- `requires_grad` 与 `backward`
- `model.train()` 和 `model.eval()`
- 训练、验证、checkpoint 保存

## 当前内容安排

- 这一章后续会作为从基础进入 Mini-GPT 实现前的桥梁章
- 重点是先把训练循环、验证流程和 checkpoint 机制讲透

## 当前状态

> 草稿。后续会补充一个独立的 PyTorch 训练循环样板章。
