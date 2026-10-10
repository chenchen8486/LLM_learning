# 第 13 章 推理系统与部署 · 分页规格

> 章定位：训练完成后的部署侧优化——同一个模型，怎么服务得更快、更便宜。
> 生命周期定位条：hero = P8（推理与部署），已生成至 temp/strips/ch13_*_strip.html。
> 主线数字（minimind 实测，禁止改动）：hidden 768、8 层、词表 6400、8 头 / KV 头 4、head_dim 96、FFN 2432、约 63.9M 参数。

## ch13_1_inference_cost.html 推理的成本结构
- 标题建议：13.1 推理的成本结构：显存、算力、带宽谁先成为瓶颈
- h2 固定：13.1.1 训练和推理不是一回事（训练批量并行 vs 推理逐 token 串行）；13.1.2 一次解码的账：参数量、显存占用、FLOPs 怎么算；13.1.3 真正的瓶颈是带宽（memory-bound，arithmetic intensity 概念）；13.1.4 优化全景图（本章后续节预告：KV Cache/批处理/量化/蒸馏/投机采样）
- 要点：用 minimind 63.9M 参数手算 FP16 显存 ≈ 128MB；解码每步都要把全部权重从显存搬一遍 → 带宽决定 tok/s；GPU 算力空转
- 图：至少 2 个 SVG（成本三角示意图 / 带宽搬运示意）
- 论文：Pope et al. 2211.05102（MLSys 2023，decode 带宽约束的系统分析）；可提 Vaswani 1706.03762 的增量解码设定

## ch13_2_kv_cache.html KV Cache
- 标题建议：13.2 KV Cache：用显存换速度的核心缓存
- h2 固定：13.2.1 没有缓存时每步重复算什么（手算例：3 个 token 生成第 4 个，再生成第 5 个时的重复计算）；13.2.2 缓存机制：只算新 token 的 Q，K/V 从缓存读；13.2.3 minimind 代码印证（Attention 121–123 行 cat past、generate 269–270 行 past_len 切片、286 行缓存回写）；13.2.4 缓存的显存代价与 GQA 的省法
- 显存公式（必须推一遍）：每层每 token 缓存 = 2（K、V）× kv头数 × head_dim × 2 字节（FP16）= 2×4×96×2 = 1536 B；8 层 = 12 KB/token；上下文 8192 → 约 100 MB
- 图：至少 2 个 SVG（重复计算对比图 / 缓存读写流程）
- 论文：KV Cache 无专门提出论文（工程共识，Fairseq/HF use_cache 传统）；压缩动机引 Shazeer MQA 1911.02150；显存分析引 Pope 2211.05102

## ch13_3_serving_systems.html 服务化系统
- 标题建议：13.3 服务化：连续批处理与 PagedAttention
- h2 固定：13.3.1 静态批处理的浪费（长短不齐、等最慢的）；13.3.2 连续批处理：token 级调度，谁结束谁走；13.3.3 KV Cache 的显存碎片问题与 PagedAttention（操作系统分页类比）；13.3.4 vLLM 的实测声明
- 要点：操作系统虚拟内存分页类比必须讲透；吞吐提升数字以 vLLM 论文为准
- 图：至少 2 个 SVG（静态 vs 连续批处理时间线 / 分页内存示意）
- 论文：Orca（OSDI 2022，USENIX 官网，无 arXiv；175B 同延迟吞吐 36.9x FasterTransformer）；vLLM/PagedAttention 2309.06180（SOSP 2023；KV 显存浪费 60-80%，吞吐 2-4x；注：24x 是博客口径非论文，须分开标注）

## ch13_4_quantization.html 量化
- 标题建议：13.4 量化：FP16 到 INT4 的压缩术
- h2 固定：13.4.1 什么是量化（浮点 → 整数的映射，手算例：一组 FP16 权重变成 INT8）；13.4.2 精度损失的来源与离群值问题；13.4.3 代表方法：SmoothQuant / GPTQ / AWQ 各解决什么；13.4.4 量化在部署中的位置（W8A8、W4A16 记号约定）
- 要点：量化只压权重还是激活也压的区分；minimind 无量化实现，纯论文路线
- 图：至少 2 个 SVG（数值映射手算图 / 方法对比表）
- 论文：SmoothQuant 2211.10438（ICML 2023，W8A8，1.56x 加速、显存减半）；GPTQ 2210.17323（ICLR 2023，3-4 bit 精度损失可忽略，175B 单 GPU 可推理，加速 3.25x A100）；AWQ 2306.00978（MLSys 2024 最佳论文，保护 1% 显著权重，TinyChat 3x+）；逐 benchmark 困惑度差值引原文表格，不自编数字

## ch13_5_distillation.html 蒸馏
- 标题建议：13.5 蒸馏：大模型教小模型
- h2 固定：13.5.1 为什么蒸馏比直接训小模型好（暗知识、软标签里的类间关系）；13.5.2 Hinton 原始机制：温度 T 与 KL 散度，T² 因子；13.5.3 minimind 代码印证（train_distillation.py：T²×KL 25–36 行、总损失 α·CE+(1-α)·KL 90–91 行、默认 α=0.5 / T=1.5 163–164 行、教师 logits 截断到学生词表 63–64 行）；13.5.4 LLM 时代的蒸馏（DeepSeek R1 蒸馏小模型等）
- 图：至少 2 个 SVG（软标签 vs 硬标签对比 / 蒸馏流程）
- 论文：Hinton 1503.02531（软标签温度 T、暗知识、T² 因子）；DistilBERT 1910.01108（小 40%、保留 97%、快 60%）；DeepSeek-R1 2501.12948（蒸馏 80 万条轨迹灌小模型，Distill-Qwen-32B AIME 72.6% vs 直接 RL 47.0%，「对小模型蒸馏比 RL 更有效」）

## ch13_6_speculative_scaling.html 投机采样与推理时扩展
- 标题建议：13.6 投机采样与推理时扩展
- h2 固定：13.6.1 投机采样：小模型起草、大模型验收；13.6.2 无损性：为什么输出分布不变；13.6.3 推理时扩展：用更多计算换更好答案（self-consistency 多数投票、思维链预算、o1 范式）；13.6.4 本章总回顾
- 要点：投机采样加速来自「一次前向验证多个 token」；无损性论证只讲直觉不证定理
- 图：至少 2 个 SVG（起草-验收时间线 / 推理时扩展曲线示意）
- 论文：投机采样 Leviathan 2211.17192（ICML 2023，T5-XXL 2-3x，identical outputs）与 Chen/DeepMind 2302.01318（Chinchilla 70B 2-2.5x，分布不变）；self-consistency 2203.11171（ICLR 2023，GSM8K +17.9%）；o1 官方博客 2024-09-12（train-time / test-time compute 双扩展，AIME 74%/83%/93% 阶梯，注明博客口径、无技术报告）

## 通用约束（每页必读）
- 开头 doc-meta-card + 综述段；结尾 socratic-card 要点回顾 + 参考来源区块
- 图 1 = 定位条（插入 temp/strips/ 对应片段内容，caption 按规范）；后续图从图 2 起编号
- 公式 \( \) / \[ \]，禁 $；公式符号给注解（参照 ch3_1 公式 2 样式）；公式内 <>& 写实体
- 自然段段首两个全角空格；卡片表格列表内不缩进
- 矩阵画法与尺寸语义一致（扁长条=列多，高瘦条=行多，方阵才画方）
- 注释卡用橙色 highlight-box info，前缀「注：」；先文字后图
- minimind 引用注行号出处；理论表述以论文为准，不拿实现反推理论
