# 从零学习大模型工程

> 一个工程师系统自学 LLM 的过程记录：从 Transformer 到 RAG、Agent、微调和部署。

[![Deploy Docs](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ferris-Liu/LLM-Start-from-the-scratch/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

📖 **在线阅读**：[ferris-liu.github.io/LLM-Start-from-the-scratch](https://ferris-liu.github.io/LLM-Start-from-the-scratch)

---

## 这是什么

这是一份边学边写的 LLM 工程学习笔记，不是已经完成的权威教材。我会把学习过程中的直觉、公式、代码实验、工程坑和参考资料整理出来，尽量让每一章都能读、能跑、能复现。

内容会持续更新，也欢迎指出错误、补充资料或一起讨论。

## 📚 内容结构

| 模块 | 主题 | 方向 | 状态 |
|------|------|------|------|
| 01 | Transformer 底层精通 | 全方向 | 🚧 进行中 |
| 02 | HuggingFace 生态全掌握 | 应用 / 部署 | 📝 计划中 |
| 03 | RAG 系统完整实现 | 应用 | 📝 计划中 |
| 04 | Agent 与工具调用 | 应用 | 📝 计划中 |
| 05 | 微调与对齐全流程 | 研究 / 应用 | 📝 计划中 |
| 06 | 推理部署与量化 | 部署 | 📝 计划中 |
| 07 | 评测体系与 Prompt 工程 | 应用 / 研究 | 📝 计划中 |
| 08 | 大模型系统设计 | 部署 / 应用 | 📝 计划中 |
| 09 | 研究前沿追踪 | 研究 | 📝 计划中 |

## ✍️ 写作方式

每篇笔记会尽量遵循同一套结构：

1. 核心一句话：这个技术解决什么问题
2. 背景与动机：为什么需要它
3. 直觉解释：先不用公式建立理解
4. 数学定义：讲清楚关键公式和张量形状
5. 最小代码：用可运行实验验证直觉
6. 工程注意事项：记录真实开发里的坑
7. 延伸阅读：论文、博客、视频和源码

## 🗂 仓库结构

```
LLM-Start-from-the-scratch/
├── docs/                 # 所有笔记（Markdown）
│   ├── 01-transformer/   # 每个模块一个目录
│   └── ...
├── notebooks/            # 配套 Jupyter Notebook（规划中）
├── projects/             # 完整动手项目（规划中）
└── assets/               # 图片、图表
```

## 🚀 本地预览

```bash
pip install -r requirements-docs.txt
mkdocs serve
# 访问 http://127.0.0.1:8000
```

## 🧭 当前优先级

1. 打磨 `01-transformer`，把 Attention、位置编码、KV Cache 和 miniGPT 写成第一章样板。
2. 补充可运行 Notebook，让核心概念不只停留在文字层面。
3. 做一个 RAG 最小闭环项目，从文档切分、向量检索到答案生成和评测。

## 🤝 贡献

欢迎 Issue 和 PR。如果你发现内容错误、公式不清楚、代码跑不通，或者有更好的参考资料，欢迎直接提出。

## 📄 License

MIT © FerrisLIU
