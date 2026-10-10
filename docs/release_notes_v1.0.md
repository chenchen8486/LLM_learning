# v1.0 发布说明（Release Notes）

　　以下文字可直接粘贴到 GitHub 的 Release 页面（Releases → Draft a new release → 选择 `v1.0` 标签）。

## 标题

　　v1.0：LLM 全链路中文手册首版

## 正文

　　首个完整版本。15 章正文加 13 节数学附录，覆盖 LLM 从数据到部署的完整链路，共 95 个页面，全部可离线直接打开。

### 覆盖内容

- 第 0–5 章：全景导论、训练数据、分词、词向量与位置编码、Transformer 架构精讲、现代架构变体。
- 第 6–8 章：预训练目标与 Scaling Law、预训练工程、中间训练与长上下文。
- 第 9–11 章：SFT 与 LoRA、偏好对齐（RLHF / DPO）、推理能力训练（RLVR / GRPO）。
- 第 12–14 章：解码与后处理、推理系统与部署（KV Cache / 量化 / 蒸馏 / 投机采样）、应用范式（Prompt / CoT / RAG / Function Calling / Agent）。
- 第 15 章：多模态扩展（ViT / CLIP / 连接器 / VLM 训练 / 统一架构）。
- 附录 A.1–A.13：向量与旋转、点积与余弦相似度、softmax 与温度、交叉熵与困惑度、梯度与反向传播、激活函数与归一化、幂律与对数坐标、残差连接等。

### 特点

- 每条机制先直觉类比再上公式，关键公式逐符号注解。
- 全站 200 余张原创 SVG 示意图，覆盖矩阵运算、注意力、KV Cache、量化、投机采样等。
- 机制描述对照开源项目 minimind 的 PyTorch 源码，引用处标注文件与行号。
- 每页末尾附参考来源，列明 arXiv 编号与官方文档。

### 在线阅读

　　<https://chenchen8486.github.io/LLM_learning/>

### 许可

　　正文 CC BY-NC-SA 4.0。`minimind-master/` 目录遵循其原始 Apache 2.0 许可证。

## 打标签命令（供参考）

```bash
git tag -a v1.0 -m "v1.0：LLM 全链路中文手册首版（15 章 + 附录，95 页）"
git push origin v1.0
```
