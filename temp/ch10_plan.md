# 第 10 章 后训练 II：偏好对齐 — 分页规格

> 供 5 个并行子代理使用。每页一节，文件名即页面片段名（output/html/pages/ 下）。
> 章标题（build.py 已有）：第 10 章 后训练 II：偏好对齐。
> 图一定位条：temp/strips/{页名}_strip.html 已生成，生命周期条 hero=P6 偏好对齐，按 shuangtu_kit.md 规则内联。
> 全章统一叙事线：SFT 之后模型会模仿但不会分辨好坏 → 先造偏好数据与奖励模型（10.2）→ 经典路线 RLHF/PPO（10.3）→ 简化路线 DPO 家族（10.4）→ 安全与权衡收束（10.5）。

## 已核准事实弹药（minimind 实测，可直接引用，注明 Apache 2.0 + 文件：行号）

- DPO 损失实现：trainer/train_dpo.py:34-50，`loss = -logsigmoid(β·(π对数比差 − ref对数比差))`，chosen 是 batch 前半、rejected 后半（63-65 行 cat 后切分）。
- β 默认 0.15；学习率 4e-8（注释原文：建议<=5e-8避免遗忘）；batch 4、epoch 1、max_seq_len 1024。
- 参考模型 = full_sft 权重复制一份并冻结（train_dpo.py:176-180）。
- DPODataset：chosen/rejected 各为对话 list，loss mask 只罩 assistant 回答区间（dataset/lm_dataset.py:125-178、180 起 generate_loss_mask）。
- minimind 另有 trainer/train_ppo.py、train_grpo.py，写 PPO 页时先读 train_ppo.py 再引用。
- 模型底座数字（如引用）：hidden 768、8 层、词表 6400、全量约 63.9M 参数。

## 论文事实（写前必须用 WebSearch / FetchURL 核对 arxiv.org/abs/{id}，核对不了的标待查证，禁止编造）

- InstructGPT：arXiv:2203.02155（SFT 约 13k prompt；40 名标注员；RM 6B；PPO-ptx；1.3B 对齐版人类偏好超过 175B GPT-3；KL 惩罚系数等细节以论文为准）。
- PPO：arXiv:1707.06347（裁剪代理目标 clip 机制）。
- DPO：arXiv:2305.18290（Bradley-Terry 下的闭式最优解、隐式奖励、β 的作用）。
- IPO：arXiv:2310.12036（指出 DPO 易过拟合，改进目标）。
- KTO：arXiv:2402.01306（Kahneman-Tversky 前景理论，单条好/坏标签，无需成对）。
- SimPO：arXiv:2405.14734（无参考模型，长度归一化，可选引用）。
- HHH 标准：Askell et al. arXiv:2112.00861（Helpful、Honest、Harmless）。
- Constitutional AI：arXiv:2212.08073（原则 critique-revision + RLAIF）。
- Llama 2：arXiv:2307.09288（安全 RLHF、红队实践）。
- LIMA：arXiv:2305.11206（表层对齐假说，第 9 章已引用，可呼应）。

---

## ch10_1_alignment_gap — 10.1 偏好差距：为什么 SFT 之后还要对齐

固定 h2：
- 10.1.1 SFT 的上限：模仿训练缺了什么
- 10.1.2 什么是「好回答」：Helpful、Honest、Harmless
- 10.1.3 从「标答案」到「标偏好」：对齐的信号来源
- 10.1.4 两条对齐路线：RLHF 与直接偏好优化

内容要点：
- SFT 三短板：只能模仿示范（学不到「哪个更好」）；示范数据贵且质量封顶于标注员水平；同一个问题存在多种合格回答，模仿单一路径浪费（可用 LIMA 表层对齐假说呼应第 9 章）。
- 生活化类比：SFT 像照字帖练字，对齐像老师打分圈点；偏好标注像「盲测两杯咖啡选一杯」，比「写出满分配方」容易得多。
- HHH 三条逐条一句话定义 + 一个微型例子。
- 路线对比表：RLHF（训 RM + PPO）vs DPO 家族（直接偏好损失），各列代表论文、信号、模型数量、稳定性，为 10.3/10.4 埋伏笔。
- 图二（局部放大）：生命周期「偏好对齐」节点放大——SFT 模型 →（偏好信号）→ 对齐模型，标出两条路线分叉。

## ch10_2_reward_model — 10.2 偏好数据与奖励模型

固定 h2：
- 10.2.1 偏好数据：chosen / rejected 成对样本
- 10.2.2 Bradley-Terry：把「谁更好」变成可微分的分数
- 10.2.3 奖励模型的训练实践（InstructGPT 参照系）
- 10.2.4 奖励模型的盲区：reward hacking 与分布外失效

内容要点：
- 数据结构：同一个 prompt 两个回答 + 哪个更好；贴 minimind DPODataset 字段结构（lm_dataset.py:125-141）当真实样例；K 个回答排序可拆成 C(K,2) 对。
- Bradley-Terry：P(胜) = σ(r好 − r差)，RM 训练损失 = −log σ(r_chosen − r_rejected)，公式带符号注解。
- InstructGPT 实践：RM 规模 6B、标注员 40 人、标注员间一致率等数字以论文核对为准。
- 盲区：RM 只是人类偏好的代理，代理必有漏洞 → reward hacking（长度偏见是经典案例）；分布外 prompt 上分数不可信。为 10.3 的 KL 罚项和 10.5 的权衡埋伏笔。
- 图二（局部放大）：偏好数据 → RM 打分流水线放大（prompt+两回答 → RM → 两个标量分数 → 比较）。

## ch10_3_ppo_rlhf — 10.3 RLHF：PPO 强化学习对齐

固定 h2：
- 10.3.1 四模型同台：策略、参考、奖励、价值
- 10.3.2 奖励信号设计：RM 分数减 KL 罚项
- 10.3.3 PPO 裁剪目标：小步更新，防止一步跨太远
- 10.3.4 工程代价：RLHF 为什么又贵又脆

内容要点：
- 先读 minimind trainer/train_ppo.py 再动笔，引用其结构注明出处。
- 四模型内存账：策略模型 + 冻结参考模型 + 冻结 RM + 价值模型（critic），显存翻倍是 RLHF 贵的直观原因。
- 单步奖励公式：r = RM(prompt, 回答) − β·KL(π‖π_ref)，KL 罚项防模型为了高分跑飞（呼应 10.2 reward hacking）。
- PPO clip：ratio = π_new/π_old，目标 min(ratio·A, clip(ratio, 1±ε)·A)，给公式 + 符号注解 + 一个数值小例（ε=0.2 时 ratio=1.3 被截断）。
- InstructGPT 的 PPO-ptx：混入预训练梯度防遗忘。
- 脆：超参敏感、实现细节多、训练曲线抖动——顺势引出 10.4 DPO。
- 图二（局部放大）：RLHF 一轮训练循环（采样回答 → RM 打分 → 算 KL → PPO 更新策略）。

## ch10_4_dpo_family — 10.4 DPO 家族：绕过奖励模型的直接偏好优化

固定 h2：
- 10.4.1 从 RLHF 到 DPO：一条数学捷径
- 10.4.2 DPO 损失逐项拆解：含 minimind 实现对照
- 10.4.3 β 与参考模型：DPO 的两个旋钮
- 10.4.4 家族图谱：IPO / KTO / SimPO

内容要点：
- 捷径直觉：KL 约束下最优策略有闭式解，把「奖励」反解成「策略与参考模型的对数比」，于是不再需要显式 RM 和 RL 循环。只讲逻辑链，深推导注明见原论文。
- 损失公式：L = −log σ(β·[(log π(y好|x) − log π_ref(y好|x)) − (log π(y差|x) − log π_ref(y差|x))])，带符号注解；与 Bradley-Terry 损失形状呼应 10.2。
- 对照 minimind train_dpo.py:34-50 逐行讲（log_softmax → gather 取目标 token → mask 只算回答区 → 前后半切分 chosen/rejected → logsigmoid），注明 Apache 2.0。
- 两个旋钮：β 大 = 贴着参考模型走（保守），β 小 = 敢跑远（易退化）；参考模型通常就是 SFT 模型本身（minimind 冻结 full_sft，lr 4e-8 防遗忘）。
- 家族对比表：DPO / IPO / KTO / SimPO 四列：要解决什么痛点、数据形态、是否需参考模型、代表论文。
- 图二（局部放大）：DPO 前向——同一 prompt 的 chosen/rejected 分别过策略模型与冻结参考模型，四个对数概率汇进一个损失。

## ch10_5_safety_alignment — 10.5 安全对齐与实践权衡

固定 h2：
- 10.5.1 安全偏好数据：拒答、红队与安全边界
- 10.5.2 Constitutional AI 与 RLAIF：让 AI 来当评委
- 10.5.3 对齐税：安全与能力的拉锯
- 10.5.4 落地权衡：RLHF 还是 DPO，对齐做到哪一步

内容要点：
- 安全偏好数据：危险问题的 chosen 是「礼貌拒答 + 给出安全替代」，不是不回答；红队（red team）主动攻击找漏洞，Llama 2 实践为参照。
- Constitutional AI 两步：按原则自我批评改写（SL-CAI）→ 用 AI 反馈训偏好模型（RLAIF），减少对人类标注的依赖；原则清单给两三个例子。
- 对齐税：过度拒答让模型变「傻白甜」，有用性下降；用「过度谨慎的客服」类比；说明现代做法是分级拒答。
- 权衡收束全章：RLHF（效果上限高、贵且脆）vs DPO（便宜稳定、上限受偏好数据质量封顶）；奖励 hacking、偏好数据质量、评估困难三个未解问题，为第 11 章 RLVR 埋一句引子（可验证奖励绕开 RM 盲区）。
- 图二（局部放大）：安全对齐流水线——红队攻击样本 → 安全偏好对 → 对齐训练 → 拒答行为，标注「AI 反馈」替代人工的环节。
