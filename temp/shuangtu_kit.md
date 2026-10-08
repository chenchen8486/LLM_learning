# 定位双图批量补齐工具包（campaign 执行手册）

> 目标：给 §6C 适用范围内尚未加双图的 41 个页面统一补「图一 全链路定位条 + 图二 局部放大图」。
> 已完成页面（不要动）：ch2_2、ch3_1、ch3_2、ch3_4、ch4_2、ch4_3。第 0 章与附录（ch90）不加。

## 1. 插入规则（每页相同）

1. 定位插入点：导读路线图 `</ol>` 之后可能有 1 个收尾 `<p>`；双图插在「导读全部内容之后、第一个 `<h2>` 之前」。
2. 插入顺序固定为四件套：
   - 过渡段 1（`<p>`，段首两个全角空格）：句式参考 ch3_1——「逐节展开之前，先看一眼本节内容在整条链路中的位置……心里有这张地图，后面每一节都知道自己站在哪」。按本页内容改写，不要照抄。
   - 图一：读 `temp/strips/{本页文件名去 .html}_strip.html`，把整块 `<div class="diagram">` 拷入，并把其中的 `__CAPTION__` 替换为本页专属图注。
   - 过渡段 2（`<p>`，两全角空格）：句式参考——「这个紫框放大之后长什么样？它的输入和输出各是什么形状？把 X 单独拆开看」。
   - 图二：按下表设计现画（或按 §3 复用规则移动页内已有图）。
3. 图一图注要点：一句话走完链路，点名本节在哪一站、这一站只管什么；句式参照 ch3_1 图一 caption。
4. 除上述插入与必要的交叉引用修正外，严禁改动页面任何已有内容。文件保持 UTF-8（无 BOM）。不要运行 build.py。

## 2. 图二绘制纪律

- `.diagram` 容器 + `.diagram-tag`「原创示意图 · 非论文原图」+ `viewBox` 自适应（宽 720–960）+ `font-family="Noto Sans SC, sans-serif"`。
- 配色：紫 `#6d28d9`、青 `#0891b2`、红 `#dc2626`、琥珀 `#fef3c7`、灰 `#9ca3af`；箭头用 `<defs><marker>` 复用。
- 必须标全输入形状与输出形状；矩阵画成带格子的块并标行 × 列。
- 字号不小于 10.5；标签与箭头、边框、其他标签不重叠；任何元素不超出 viewBox。
- 数字优先级：页面正文已有的数字 > minimind 源码/实测 > 不标数字只画结构。禁止编造。
- 分词器实测命令（工程根下跑）：
  `PYTHONIOENCODING=utf-8 python -c "from tokenizers import Tokenizer; tok=Tokenizer.from_file('minimind-master/model/tokenizer.json'); print(tok.encode('文本').ids)"`
- minimind 关键配置（已核准）：hidden 768、8 层、词表 6400、8 头、kv 头 4、head_dim 96、FFN intermediate 2432、MoE 4 专家 top-1、辅助损失系数 5e-4、全量参数约 63.9M、权重共享 embedding/lm_head。

## 3. 复用规则（仅这些页）

页内已有一张语义等同于图二的内容 SVG：把它整块移动到图二位置（过渡段 2 之后），caption 改写成「局部放大」框架，并修正原位置正文里的「下图/这张图」等交叉引用。原位置不再保留副本。

- ch9_2：移动「conversations JSON → chat template → token 流」流水线 SVG
- ch9_3：移动「token 序列损失掩码」SVG
- ch9_4：移动「LoRA 旁路结构」SVG
- ch9_5：移动「选型决策树」SVG
- ch8_1：不移动（页内三阶段图留在原地讲分界线），新画一张中间训练内部放大图

## 4. 每页图二设计规格

### 前向链路条（图一 = 分词/Embedding/位置编码/Transformer 那条链）

| 页面 | 图二放大对象与必备元素 |
|------|------|
| ch2_1_why_tokenization | 分词总览：字符串「我爱中国」→ 分词器 → token ID [463, 2043, 1405]（实测）→ 词表 6400 条对照。输入一段文本，输出 3 个 ID |
| ch2_3_wordpiece_unigram | 同一英文词在 WordPiece（贪心最长匹配）与 Unigram（概率选路）下的两条切分路径对比。标注：算法行为示意（minimind 用的是 BPE） |
| ch2_4_sentencepiece | SentencePiece 训练流程：语料 → 统计相邻对频次 → 反复合并 → 词表；▁ 表示词首空格。流程示意 |
| ch2_5_vocab_special_tokens | 词表解剖：6400 行分普通 token 区 + 特殊 token 区（bos/eos/pad 等，以页面正文为准）。标 minimind 词表 6400 |
| ch2_6_tokenizer_in_practice | 多语言实测对比：中文「我爱中国」[463,2043,1405] 一字一 token，对照英文或 emoji 拆得更碎（用上面的命令实测后再画） |
| ch3_3_sinusoidal_learned | 正弦位置编码矩阵热力波纹（位置 × 维度，d=768）：不同维度频率递减；旁注可学习方案是一张大表 |
| ch3_5_length_extrapolation | 长度外推示意：横轴位置、训练区间内平稳、区间外失控；标 minimind max_seq_len 340（预训练）/768（SFT） |
| ch4_1_architecture_overview | 端到端前向：token ID [463,2043,1405] → Embedding 3×768 → Transformer 块 ×8 → RMSNorm → lm_head（与 embedding 共享权重）→ logits 3×6400 → 取最后一位预测下一个 token。全 minimind 数字 |
| ch4_4_ffn_residual_norm | 块内部：x(3×768) → RMSNorm → 注意力 → +（残差）→ RMSNorm → FFN 768→2432→768（SwiGLU）→ +（残差）→ 输出 3×768，形状全程保持 |
| ch4_5_decoder_only | 因果掩码：4×4 可见性矩阵（下三角可见、上三角被 −∞ 掩掉），以页面正文口径为准 |
| ch4_6_forward_walkthrough | 一条序列的逐步形状流：3 个 ID → 3×768 → 块内各步 → 3×6400 logits → 取最后位置 → 下一个 token（比 4.1 更细，跟正文步骤对得上） |
| ch5_1_norm_activation | RMSNorm 计算流：向量 → 除以 RMS → ×γ（可学增益）；旁注 SwiGLU 三路投影 768↔2432。以页面正文为准 |
| ch5_2_mqa_gqa | MHA / GQA / MQA 三栏：8Q-8KV、8Q-4KV（minimind 默认）、8Q-1KV；每 token 的 KV cache 存 2×768 vs 2×384 vs 2×96 个数 |
| ch5_3_mla | MLA：KV 先压缩成低秩潜向量再展开；形状数字以页面正文（DeepSeek-V2）为准，不编造 |
| ch5_4_moe | MoE：router Linear(768→4) 打分 → top-1 选专家（minimind 默认 4 选 1）→ 只过一个专家的 FFN；对照 dense 每 token 过全部参数；aux loss 系数 5e-4 |
| ch5_5_modern_recap | 现代块全家福：RMSNorm + RoPE + GQA + SwiGLU（+MoE）标注各组件来自哪一节，汇总 minimind 默认配置 |

### 生命周期条（图一 = 数据→预训练→训练工程→中间训练→SFT→偏好对齐→推理训练→推理部署）

| 页面 | 图二放大对象与必备元素 |
|------|------|
| ch1_1_corpus_sources | 语料来源构成：网页/书籍/代码/论文/对话等类别构成（页面有数字用数字，没有就只画类别示意不标百分比） |
| ch1_2_quality_filtering | 质量过滤漏斗：原始爬取 → 规则启发式 → 分类器打分 → 精选，逐级收窄 |
| ch1_3_dedup | 去重流水线：段落 → 指纹/MinHash → 重复簇 → 只留一份 |
| ch1_4_data_mixture | 配比与采样：多领域配比堆叠条 → 按配比采样混成训练流 |
| ch1_5_synthetic_data | 合成数据循环：种子数据 → 教师模型生成 → 质量过滤 → 入库（带回环箭头） |
| ch6_1_pretraining_objective | next-token 目标：序列逐位预测下一个 token，每个位置一个 softmax 交叉熵，最后求平均 |
| ch6_2_optimizer_hyperparams | 学习率曲线：短 warmup + 长 cosine 衰减（与页面正文口径一致） |
| ch6_3_training_loop | 训练循环：batch → forward → loss → backward → clip(1.0) → step → zero_grad 环形图 |
| ch6_4_scaling_law | Scaling Law：log-log 坐标下 loss 随算力直线下降（幂律）；数字以页面正文为准 |
| ch6_5_emergence | 涌现曲线：横轴规模、纵轴能力，小模型贴随机线、过阈值陡升 |
| ch7_1_scaling_wall | 显存墙分解：参数 + 梯度 + Adam 状态（约 2×参数）+ 激活堆叠，标精度倍数（以正文为准） |
| ch7_2_data_parallel_zero | DP vs ZeRO：4 卡各存完整副本 vs 参数/梯度/优化器状态切分到各卡 |
| ch7_3_tensor_pipeline_parallel | TP + PP：矩阵按列切到两卡（TP）；8 层分两段流水（PP） |
| ch7_4_mixed_precision_stability | 混合精度流：fp32 主权重 ↔ bf16 计算 + loss scaling（scaler.scale → unscale → step，minimind 训练脚本） |
| ch7_5_compute_cost | 算力账：C ≈ 6ND 分解（前向 2ND + 反向 4ND）；代入数字以正文为准 |
| ch8_1_midtraining_overview | 中间训练内部放大：base → 换数据配方（高质量 + 长文档 + 领域加重）→ 拉长序列 → 增强版 base。与页内三阶段图分工，不重复 |
| ch8_2_rope_scaling | RoPE 缩放/位置插值示意：以正文口径为准 |
| ch8_3_long_context_training | 窗口阶段式拉长（如 4K→32K→128K）+ 注意力成本随长度平方增长曲线；数字以正文为准 |
| ch8_4_domain_adapt_anneal | 退火：lr 衰减尾段 + 高质量数据集中喂入 |
| ch8_5_midtraining_practice | 中间训练配方要素汇总：数据 / 长度 / lr 三轴 |
| ch9_1_why_sft | SFT 阶段内部：base model + 指令对 → 交叉熵训练 → instruct model，输入输出标清 |
| ch9_2 ~ ch9_5 | 按 §3 复用规则移动页内已有图 |

## 5. 验图手法（每页必做）

1. 从改后的页面片段里抽出全部 `<svg.*?</svg>` 写进 `temp/svgprobe_{页名}.html`（加 `<meta charset="utf-8">`）。
2. Edge headless 截图（工程根下）：
   `"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless=new --disable-gpu --screenshot="D:/project/python_release/LLM_learning/temp/svgprobe_{页名}.png" --window-size=W,H "file:///D:/project/python_release/LLM_learning/temp/svgprobe_{页名}.html"`
   W 取 760–1100；H 公式：svg 高度 × (W/720) 之和 + 100（window-size 高度含约 100px 浏览器 chrome）。
3. 用 ReadMediaFile 验看：标签不重叠、不越界、箭头可见、形状标注齐全。不合格改了再验。
4. 验完删除 probe html/png。

## 6. 自检清单（每页交付前）

- [ ] 插入位置正确（导读之后、第一个 h2 之前，四件套顺序对）
- [ ] 图一 `__CAPTION__` 已替换，图注点名本节位置
- [ ] 图二标全输入/输出形状，数字可追溯
- [ ] 页面 `<h1>` 仍唯一；`<svg` 数量 ≥ 2（图一 + 图二）
- [ ] 无 `$` 公式混入；未改动其他正文
- [ ] 渲染验看通过，probe 文件已删
