<section class="llm-hero llm-hero--book">
  <div class="llm-hero__eyebrow">Then Language Generates By Computer</div>
  <h1 class="llm-hero__title">机器于此理解语言</h1>
  <p class="llm-hero__subtitle">
    一个AI炼丹师的大模型学习笔记。
    从语言模型的底层结构，到训练、微调、RAG、Agent 与部署实践，
    把抽象理论压进可运行的实验、项目与长期写作里。
  </p>
  <div class="llm-hero__meta">
    <span class="llm-pill">AI Alchemist</span>
    <span class="llm-pill">Transformer</span>
    <span class="llm-pill">Pretraining</span>
    <span class="llm-pill">Finetuning</span>
    <span class="llm-pill">RAG</span>
    <span class="llm-pill">Agent</span>
    <span class="llm-pill">LLMOps</span>
  </div>
</section>

<div class="llm-highlight-grid">
  <article class="llm-highlight">
    <div class="llm-highlight__label">写作定位</div>
    <p>面向希望真正理解语言模型内部机制，而不只停留在工具调用层的学习者。</p>
  </article>
  <article class="llm-highlight">
    <div class="llm-highlight__label">学习目标</div>
    <p>把 LLM 的训练链路、应用链路与工程链路串成一条能复现、能解释、能展示的主线。</p>
  </article>
  <article class="llm-highlight">
    <div class="llm-highlight__label">主线参考</div>
    <p>以 from-scratch 视角组织 CS336、Hugging Face LLM Course 与真实项目实践。</p>
  </article>
</div>

## 开始阅读

<div class="llm-card-grid llm-card-grid--entry">
  <article class="llm-card">
    <h3><a href="book/01-ml-dl-basics/">如果你是初学者</a></h3>
    <p>先把训练闭环、损失函数、梯度、优化器这些地基补稳，再进入大模型主线。</p>
  </article>
  <article class="llm-card">
    <h3><a href="book/05-attention/">如果你想直进 Transformer</a></h3>
    <p>从 Attention 开始，抓住语言模型最核心的结构直觉与计算方式。</p>
  </article>
  <article class="llm-card">
    <h3><a href="roadmap/">如果你想先看全书地图</a></h3>
    <p>先浏览整条学习路线、章节安排和项目路线，再决定自己的阅读顺序。</p>
  </article>
</div>

## 学习主线

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

## 章节地图

<div class="llm-path-grid">
  <article class="llm-path-card">
    <h3>Part 0 · 学习路线与岗位定位</h3>
    <p>第 0 章：大模型岗位地图与学习路线</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 1 · 基础补齐</h3>
    <p>第 1 章：机器学习与深度学习最小必要基础</p>
    <p>第 2 章：PyTorch 与训练循环</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 2 · NLP 与语言模型基础</h3>
    <p>第 3 章：NLP 基础与文本表示</p>
    <p>第 4 章：从 n-gram 到神经语言模型</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 3 · Transformer 核心</h3>
    <p>第 5 章：Attention 机制</p>
    <p>第 6 章：Transformer 架构</p>
    <p>第 7 章：从零实现一个 Mini-GPT</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 4 · 现代 LLM 训练流程</h3>
    <p>第 8 章：Tokenizer 与数据处理</p>
    <p>第 9 章：预训练 Pretraining</p>
    <p>第 10 章：Scaling Law 与大模型为什么变大</p>
    <p>第 11 章：LLM 评估</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 5 · 后训练与微调</h3>
    <p>第 12 章：SFT</p>
    <p>第 13 章：LoRA / QLoRA / PEFT</p>
    <p>第 14 章：RLHF / DPO / RLAIF</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 6 · LLM 应用工程</h3>
    <p>第 15 章：Embedding、向量数据库与语义检索</p>
    <p>第 16 章：RAG 从原理到项目</p>
    <p>第 17 章：Agent 与 Tool Calling</p>
    <p>第 18 章：LLMOps、部署与生产化</p>
  </article>
  <article class="llm-path-card">
    <h3>Part 7 · 综合项目与求职准备</h3>
    <p>附录 A：推荐项目路线</p>
    <p>附录 B：每章统一写作模板</p>
    <p>附录 C：推荐学习顺序</p>
  </article>
</div>

## 项目路线

<div class="llm-card-grid llm-card-grid--projects">
  <article class="llm-card">
    <h3>项目 1 · Mini-GPT from Scratch</h3>
    <p>从零实现 decoder-only GPT，沉淀模型结构图、训练曲线、生成样例和实验分析。</p>
  </article>
  <article class="llm-card">
    <h3>项目 2 · LoRA Fine-tuning 中文任务</h3>
    <p>使用 Transformers、Datasets、PEFT 完成一个可展示的参数高效微调项目。</p>
  </article>
  <article class="llm-card">
    <h3>项目 3 · Course Notes RAG Assistant</h3>
    <p>做出一个可运行、可评估、可展示引用来源的课程资料问答系统。</p>
  </article>
  <article class="llm-card">
    <h3>项目 4 · LLM Evaluation Benchmark</h3>
    <p>自己设计 evaluation set、对比报告和 bad case analysis，补齐评估能力。</p>
  </article>
</div>

## 推荐先读

<div class="llm-card-grid llm-card-grid--featured">
  <article class="llm-card">
    <h3><a href="book/05-attention/">样板章 · Attention 机制</a></h3>
    <p>适合快速判断这本笔记是否合你胃口：原理、公式、代码表达会集中体现在这里。</p>
  </article>
  <article class="llm-card">
    <h3><a href="book/07-mini-gpt/">样板章 · 从零实现一个 Mini-GPT</a></h3>
    <p>如果你偏爱 from-scratch 学习法，这一章会是全书最能体现项目感的入口。</p>
  </article>
</div>

## 适合怎么读

!!! tip "建议阅读方式"
    推荐按 Part 顺序推进。每学完一个阶段，就同步整理代码实验、笔记、README 和一轮复盘，这样知识才会真正沉淀。

## 当前说明

- 当前内容将按新版 18 章主线持续展开。
- 当前优先会先打磨 Transformer 和 Mini-GPT 相关内容，作为全书样板章。
- 完整章节提纲、写作模板和推荐顺序见 [全书提纲与学习路线](roadmap.md)。

---

*最后更新：{{ git_revision_date_localized }}*
