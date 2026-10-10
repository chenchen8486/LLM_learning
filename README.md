# LLM 机制学习手册

　　一套可离线浏览的中文学习手册，目标是系统讲清大语言模型（LLM）从数据到部署的完整机制：训练数据、分词、词向量与位置编码、Transformer 架构、预训练、中间训练、后训练（SFT / 偏好对齐 / 推理能力训练）、推理与后处理、推理系统与部署、应用范式、评测。

　　讲解对标高校公开课水平：先直觉类比再上公式，关键术语逐个讲透，抽象机制配原创示意图（页面内 SVG），全部公式用 KaTeX 渲染。

## 直接阅读（无需构建）

　　成品页面已随库发布：克隆后用浏览器打开 `output/html/ch0_1_what_is_llm.html`（第 0 章第 1 节），从侧边栏逐章阅读即可。公式与字体走 CDN（KaTeX、Noto Sans SC），首次打开需联网一次。

## 重新构建（仅改动内容后需要）

```bash
python output/html/build.py
```

　　页面源文件在 `output/html/pages/`，构建产物（`output/html/ch*.html` 与配图副本）也提交在库中，构建只是重新生成它们。

## 目录结构

```
LLM_learning/
├── AGENTS.md               # 工程协作基准
├── docs/                   # 开发方案与 HTML 编写规范
├── ref_paper/              # 参考论文（只读素材）
├── minimind-master/        # 可读代码参考（Apache 2.0，上游 jingyaogong/minimind）
├── temp/                   # 归档与临时文件
└── output/html/
    ├── build.py            # 构建脚本
    ├── template.html       # 页面骨架
    ├── static/global.css   # 全局样式
    └── pages/              # 页面片段（手册正文源文件）
```

## 内容骨架

```
第 0 章  全景导论：LLM 流水线总览
第 1 章  训练数据：收集、清洗、去重、配比与合成数据
第 2 章  分词：文本如何变成 Token
第 3 章  词向量与位置编码（含长度外推）
第 4 章  Transformer 架构精讲
第 5 章  现代架构变体与效率改进（RMSNorm / SwiGLU / GQA / MLA / MoE 等）
第 6 章  预训练：目标、优化与 Scaling Law
第 7 章  预训练工程：分布式并行、混合精度、稳定性与算力估算
第 8 章  中间训练（Mid-training）：长上下文扩展与领域适配
第 9 章  后训练 I：监督微调与参数高效微调（SFT / LoRA）
第 10 章 后训练 II：偏好对齐（RLHF / PPO / DPO 家族 / 安全对齐）
第 11 章 后训练 III：推理能力训练（RLVR / GRPO / 长思维链）
第 12 章 推理机制与后处理：解码策略、停止条件、结构化输出
第 13 章 推理系统与部署（KV Cache、量化、蒸馏、投机采样、推理时扩展）
第 14 章 应用范式（Prompt / CoT / RAG / Function Calling / Agent）
第 15 章 多模态扩展（ViT、CLIP、连接器、VLM 训练流水线、统一架构）
附录     数学与深度学习基础速览（A.1–A.13）
```

## 素材与出处

- 第 4 章以 `ref_paper/2016_attention_is_all_you_need/` 的论文译文为唯一依据。
- 正文中的代码对照引用自 minimind（Apache 2.0），引用处均注明文件与行号。
- 其余章节内容以原始论文（arXiv）与官方技术报告为依据，每页末尾附参考来源。

## 鸣谢

　　本手册的代码对照离不开开源项目 [minimind](https://github.com/jingyaogong/minimind)（作者 jingyaogong，Apache 2.0 许可证）。minimind 用极简代码完整实现了从分词、RoPE、RMSNorm、SwiGLU、GQA 到预训练与微调的全流程，本手册多章以其源码作为机制描述的工程印证，引用处均在页面内注明出处文件。`minimind-master/` 目录为其上游仓库的只读副本，保留原始 `LICENSE`，一切权利归原作者所有。在此向作者及项目贡献者致谢。

## 许可证

　　手册正文为个人学习笔记。`minimind-master/` 目录遵循其原有 Apache 2.0 许可证（详见该目录下的 `LICENSE` 文件）。
