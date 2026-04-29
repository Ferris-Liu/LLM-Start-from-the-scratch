# DeepSeek-V4 技术分析报告

> 基于 DeepSeek-V4 技术报告整理，重点分析其在百万 token 上下文、高效注意力、MoE 训练稳定性、低精度推理和后训练范式上的技术意义。

---

## 1. 基本信息

- **名称**：DeepSeek-V4
- **类型**：MoE / Reasoning Model / Long-Context Model / Agent Model
- **发布时间**：2026 年 4 月 24 日技术报告版本
- **来源**：DeepSeek-V4 技术报告、官方 Hugging Face 模型集合
- **核心模型**：
  - DeepSeek-V4-Pro：1.6T 总参数，49B 激活参数
  - DeepSeek-V4-Flash：284B 总参数，13B 激活参数
- **上下文长度**：原生支持 1M tokens

## 2. 为什么值得关注

DeepSeek-V4 的重点不是简单扩大参数规模，而是试图突破超长上下文的效率瓶颈。传统 Transformer 注意力在序列长度增加时会带来极高的计算与 KV Cache 存储成本，这限制了 test-time scaling、长程推理、跨文档分析和复杂 Agent 工作流。

DeepSeek-V4 用混合注意力架构、MoE 稀疏激活、FP4 低精度、训练稳定性技巧和推理系统优化组成了一条完整路线：让百万 token 上下文从展示性能力变成更接近日常可用的模型能力。

它值得纳入这套笔记体系，是因为它同时连接了多个主线主题：

- Attention 从全量计算走向压缩、稀疏和分层检索。
- MoE 从“扩大参数量”走向“控制激活成本和训练稳定性”。
- 后训练从单一 SFT/RL 流程走向多领域专家训练和统一蒸馏。
- Agent 和长上下文正在推动模型架构本身发生变化。
- 推理部署的瓶颈从纯算力扩展到 KV Cache、低精度、缓存复用和长序列调度。

## 3. 核心技术拆解

### 架构层

DeepSeek-V4 继承了 DeepSeekMoE 和 Multi-Token Prediction，同时引入三项关键变化：混合注意力、Manifold-Constrained Hyper-Connections，以及 Muon 优化器。

最关键的是混合注意力架构。DeepSeek-V4 将 CSA 与 HCA 交替使用：

- **CSA（Compressed Sparse Attention）**：先把多个 token 的 KV 条目压缩成一个条目，再用稀疏选择机制只关注 top-k 个压缩 KV 条目。它适合在超长上下文中进行全局信息定位。
- **HCA（Heavily Compressed Attention）**：采用更高压缩率，将更长片段压缩成单个 KV 条目，但保留密集注意力。它适合用较低成本维持远距离信息覆盖。
- **滑动窗口分支**：由于压缩块可能损失局部细节，模型额外保留最近 token 的未压缩 KV 条目，用于增强局部依赖建模。

在 1M token 场景下，报告称 DeepSeek-V4-Pro 相比 DeepSeek-V3.2 只需要约 27% 的单 token 推理 FLOPs 和 10% 的 KV Cache；DeepSeek-V4-Flash 则降低到约 10% FLOPs 和 7% KV Cache。

### 训练层

DeepSeek-V4-Pro 使用 33T tokens 预训练，DeepSeek-V4-Flash 使用 32T tokens 预训练。训练序列长度从 4K 逐步扩展到 16K、64K，最终扩展到 1M。

训练中的几个关键点：

- 大部分参数使用 **Muon optimizer**，embedding、prediction head、RMSNorm 等模块保留 AdamW。
- 训练早期先使用 dense attention 预热，再在更长序列阶段引入 sparse attention。
- 使用辅助 loss-free load balancing，并保留小权重 balance loss，防止单序列内专家负载极端不均。
- 使用 MTP loss 辅助训练，多数训练阶段权重为 0.3，学习率衰减阶段降到 0.1。

DeepSeek-V4 还明确讨论了万亿参数 MoE 训练中的不稳定性，并采用两类技巧：

- **Anticipatory Routing**：用历史参数提前计算路由索引，降低路由变化与主干网络同步更新带来的 loss spike。
- **SwiGLU Clamping**：限制 SwiGLU 中部分激活值范围，抑制 MoE 层异常值扩散。

这说明大规模 MoE 的难点不只是专家数量和通信效率，还包括路由机制、异常值和训练动态之间的耦合。

### 推理 / Agent 层

DeepSeek-V4 的百万 token 上下文直接服务于长程推理和 Agent 任务。报告中特别强调，长上下文让模型可以在工具调用失败、状态丢失或多轮任务推进时，从更完整的历史中恢复问题求解状态。

后训练采用两阶段范式：

1. 先针对数学、代码、Agent、指令跟随等领域分别训练专家模型。
2. 再通过 on-policy distillation 将多个专家能力整合到统一模型。

这是一种值得关注的后训练路线：不是试图让一个模型在单一 RL 阶段中同时学会所有能力，而是先让领域专家充分发展，再把专家策略蒸馏给统一学生模型。

### 工程层

DeepSeek-V4 的工程优化覆盖训练、推理和低精度：

- MoE 模块使用融合 kernel，重叠计算、通信和内存访问。
- 使用 TileLang 提升 kernel 开发效率和运行性能。
- 使用 batch-invariant 和 deterministic kernel，保证训练和推理的 bitwise reproducibility。
- 对 MoE expert weights 和 indexer QK path 使用 FP4 量化感知训练。
- KV Cache 使用 BF16 与 FP8 混合存储，并设计异构 KV Cache 结构。
- 支持 on-disk KV Cache 存储和共享前缀复用，降低长上下文服务成本。

这些工程设计说明，百万 token 上下文不是单靠模型结构解决的能力，而是架构、训练框架、低精度数值格式、缓存系统和 serving 共同作用的结果。

## 4. 与已有知识体系的映射

- **第 5 章：Attention 机制**
  - CSA、HCA、稀疏注意力、压缩 KV、滑动窗口注意力。
- **第 6 章：Transformer 架构**
  - mHC 对残差连接的增强，MoE Transformer 的结构变化。
- **第 9 章：预训练 Pretraining**
  - 长上下文 curriculum、dense-to-sparse attention 训练策略、训练稳定性。
- **第 10 章：Scaling Law 与大模型为什么变大**
  - 从参数扩展转向 test-time scaling、上下文扩展和激活参数效率。
- **第 11 章：LLM 评估**
  - 长上下文、Agent、代码、推理、知识评测的综合比较。
- **第 14 章：偏好对齐**
  - GRPO、RL、on-policy distillation、多专家后训练合并。
- **第 17 章：Agent 与 Tool Calling**
  - 长程任务、工具调用状态恢复、Agentic Search、代码 Agent。
- **第 18 章：LLMOps、部署与生产化**
  - FP4、KV Cache 管理、on-disk cache、推理 FLOPs 与服务成本优化。

**归类判断**：DeepSeek-V4 不是单个全新概念，而是多个已有趋势的系统性组合。它更适合作为“下一代长上下文 MoE 推理模型”的综合案例。

## 5. 值得写进主线笔记的内容

稳定、可复用的核心原理：

- 长上下文的核心瓶颈是 attention FLOPs 与 KV Cache，而不只是窗口长度。
- 压缩注意力和稀疏注意力可以组合使用：先降低序列长度，再选择关键块。
- 近端信息和远端信息需要不同机制处理，滑动窗口分支是重要补偿。
- MoE 的训练稳定性与路由机制强相关，路由异常会放大激活异常。
- 低精度不是部署阶段才考虑的优化，可能需要从训练阶段做 quantization-aware training。
- 后训练可以拆成“领域专家训练”和“统一模型蒸馏”两个阶段。

更适合放在扩展阅读或案例中的内容：

- mHC 的完整数学推导。
- Muon 的 hybrid Newton-Schulz 迭代细节。
- TileLang、确定性 kernel、on-disk KV Cache 的具体实现。
- 各个 benchmark 的详细分数对比。

## 6. 初学者应该重点理解什么

如果只讲 3 个点，建议抓住：

1. **百万 token 上下文不是简单把 RoPE 外推到 1M**  
   真正的难点是每生成一个 token 都要面对超长历史带来的计算和 KV Cache 压力。

2. **DeepSeek-V4 的注意力是“压缩 + 稀疏 + 局部窗口”的组合**  
   远处信息用压缩表示，重要远程块用稀疏选择，最近信息用滑动窗口保真。

3. **大模型能力提升正在从参数规模转向系统效率**  
   MoE、FP4、KV Cache、test-time scaling、Agent 长程任务会一起决定模型是否真的可用。

容易被宣传带偏的地方：

- 不能只看“1M context”这个数字，要看 1M 下的推理成本、检索准确性和长任务稳定性。
- 不能只看总参数量，MoE 模型更应关注激活参数、路由质量和专家负载。
- 不能把 benchmark 领先等同于所有真实任务领先，尤其是复杂多轮写作、Agent 可靠性和低延迟交互仍需独立验证。

## 7. 争议点与待验证问题

官方报告认为 DeepSeek-V4-Pro-Max 在开放模型中重新定义了 SOTA，并在部分长上下文任务上超过 Gemini-3.1-Pro。这个判断在技术报告内部有评测支持，但仍需要社区复现和第三方 benchmark 验证。

待验证问题包括：

- CSA/HCA 在不同类型长上下文任务上的召回损失有多大。
- 压缩 KV 对细粒度引用、法律/代码/表格类任务是否存在系统性弱点。
- Anticipatory Routing 与 SwiGLU Clamping 的理论机制尚不清晰，是否能泛化到其他 MoE 架构。
- FP4 在更多硬件平台上的实际收益，是否能达到报告中的理论预期。
- 1M 上下文在真实服务中如何平衡延迟、成本和用户体验。

我的判断是：DeepSeek-V4 最值得学习的不是某个单独模块，而是它把长上下文建模、稀疏计算、低精度训练、后训练专家蒸馏和推理系统放在同一设计空间里考虑。这类系统性工程路线，很可能比单点模型技巧更能代表下一阶段 LLM 竞争。

## 8. 可补充材料

- **论文 / 技术报告**：本地文件 `/Users/ferrisliu/Downloads/DeepSeek_V4.pdf`
- **官方模型集合**：`https://huggingface.co/collections/deepseek-ai/deepseek-v4`
- **官方推理实现**：`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
- **建议后续补充**：
  - 第三方评测结果
  - 社区复现体验
  - 长上下文真实任务案例
  - FP4 推理在具体硬件上的成本数据

## 9. 笔记状态

- **状态**：已完成初版
- **是否需要回写正式章节**：是
- **优先回写章节**：
  - 第 5 章 Attention 机制
  - 第 6 章 Transformer 架构
  - 第 17 章 Agent 与 Tool Calling
  - 第 18 章 LLMOps、部署与生产化
- **下次更新时重点补充什么**：
  - 社区第三方评测
  - 与同代模型在真实长上下文任务上的对比
  - CSA/HCA 的图示化解释
  - KV Cache 成本计算示例
