# 第 12 章 推理机制与后处理 — 分页规格

> 供 5 个并行子代理使用。每页一节，文件名即页面片段名（output/html/pages/ 下）。
> 章标题（build.py 已有）：第 12 章 推理机制与后处理。
> 图一定位条：temp/strips/{页名}_strip.html 已生成，生命周期条 hero=P8 推理与部署，按 shuangtu_kit.md 规则内联。
> 全章叙事线：训练完得到的是「会打分的模型」，本章讲怎么把分数变成文字——主循环（12.1）→ 采样三件套（12.2）→ 停止与流式（12.3）→ 结构化约束（12.4）→ 调参与陷阱收束（12.5）。系统工程（KV Cache 细节、量化、投机采样）一律留给第 13 章，本章只预告不展开。
> 衔接约定：logits/softmax/温度数学引附录 A.3、A.4；交叉熵训练目标引第 6 章；采样时自回归回路引第 0、4 章。

## 已核准事实弹药（minimind 实测，可直接引用，注明 Apache 2.0 + 文件：行号）

- 解码主循环：model/model_minimind.py:261-293 的 generate。逐步结构：
  - 只把新 token 送进前向：`input_ids[:, past_len:]`（269-270 行），past_key_values 复用（286 行）——KV Cache 的教学版，细节留给第 13 章。
  - 温度：`logits / temperature`（272 行）。
  - 重复惩罚：已出现过的 token，正分除以 rp、负分乘以 rp（273-275 行）。
  - top_k：低于第 k 名的 logit 置 −∞（276-277 行）。
  - top_p：排序后累计概率超过 p 的置 −∞，掩码右移一格保住阈值线上的第一个（278-282 行）。
  - 选词：`do_sample=True` 用 `torch.multinomial` 按概率抽，否则 argmax（283 行）。
  - 停止：已完成的序列强制补 eos，`finished.all()` 才整体 break（284、288-290 行）。
- 默认值：generate 签名 temperature=0.85、top_p=0.85、top_k=50、max_new_tokens=8192、eos_token_id=2（262 行）；eval_llm.py:42-44 命令行默认 temperature 0.85、top_p 0.95、max_new_tokens 8192。
- 流式输出：eval_llm.py:65、84-90，TextStreamer 边生成边打印（skip_prompt=True, skip_special_tokens=True）。
- 模型底座：hidden 768、8 层、词表 6400、约 63.9M 参数。

## 论文事实（写前必须用 WebSearch / FetchURL 核对 arxiv.org/abs/{id}，核对不了的标待查证，禁止编造）

- top-p（nucleus sampling）：Holtzman et al.，arXiv:1904.09751（The Curious Case of Neural Text Degeneration：beam/贪心退化、nucleus 提出）。
- top-k：Fan et al.，arXiv:1805.04833（分层故事生成，top-k 采样）。
- 重复惩罚：CTRL（Keskar et al.，arXiv:1909.05858）训练期外的惩罚项用法。
- 对比搜索 contrastive search：Su et al.，arXiv:2202.06417（可选）。
- min-p 采样：arXiv:2407.01082（可选）。
- 约束解码/结构化输出：Outlines（Willard & Louf，arXiv:2307.09709，FSM 压缩词表）；llama.cpp 的 GBNF 语法约束（项目文档，非论文）；OpenAI Structured Outputs（2024-08 官方博客，无 arXiv，注明为博客）。
- 温度与 softmax 数学引本手册附录 A.3，不必引外部论文。

---

## ch12_1_decoding_loop — 12.1 解码主循环：从 logits 到文字

固定 h2：
- 12.1.1 一步解码的解剖：logits → 修正 → softmax → 选词
- 12.1.2 贪心与采样：argmax 为什么不够用
- 12.1.3 自回归循环：每步只算新 token（KV Cache 预告）
- 12.1.4 minimind 的 generate 逐行走读

内容要点：
- 一步解码四环节：模型前向出 logits（词表 6400 维）→ 温度/惩罚/top-k/top-p 修正 → softmax 成概率 → 抽样或 argmax 取一个 token。
- 贪心的问题：每步取最大 → 容易复读、文本干瘪；引 nucleus 论文对 beam/贪心退化的结论。
- 循环结构：新 token 接回输入再算下一步（呼应第 0、4 章自回归）；点出 minimind 每步只送新 token（269-270 行）就是 KV Cache，细节预告第 13 章。
- minimind generate 走读：按 261-293 行分「循环外准备 / 前向 / 修正 / 选词 / 停止」五段对照。
- 图二（局部放大）：推理部署节点放大——prompt token 进模型 → logits → 修正链 → 选词 → 接回输入的红色自回归回路，标注 minimind 默认参数（T=0.85、top_p=0.85、top_k=50）。

## ch12_2_sampling — 12.2 采样三件套：temperature、top-k、top-p

固定 h2：
- 12.2.1 温度：调节分布的锐度（数学见附录 A.3）
- 12.2.2 top-k 与 top-p：截断尾巴的两种刀法
- 12.2.3 三件套的联合使用：顺序与相互作用
- 12.2.4 进阶抽样：min-p、重复惩罚与对比搜索

内容要点：
- 温度：T<1 变尖锐（保守）、T>1 变平坦（奔放）、T→0 退化为 argmax；给一个 4 词概率分布的数值小例（T=0.5/1/2 下概率变化，可用附录 A.3 口径）。
- top-k：固定名额截断，问题是分布平坦时误伤、尖锐时放行尾巴；top-p：按累计概率动态名额，解决 top-k 的两个毛病——nucleus 论文动机。
- 联合顺序用 minimind 实际顺序讲：温度 → 重复惩罚 → top-k → top-p → multinomial（272-283 行），说明顺序会影响结果。
- 进阶：min-p（按最大概率的比例设阈值，2407.01082）、重复惩罚机制（正分除负分乘，273-275 行）、contrastive search 一句话带过（2202.06417）。
- 图二（局部放大）：一条修正链流水线——原始 logits 条形图逐站变化（温度变形 → top-k 砍尾 → top-p 再收 → 最终抽样），每站配小分布图，数字自洽可复算。

## ch12_3_stopping — 12.3 停止条件与流式输出

固定 h2：
- 12.3.1 EOS：模型怎么学会「说完就停」
- 12.3.2 max_tokens 与 stop 序列：三道刹车
- 12.3.3 流式输出：逐 token 推送的工程意义
- 12.3.4 minimind 的停止逻辑：finished 掩码

内容要点：
- EOS 是训练出来的：训练数据每条样本以 eos 收尾（第 6、9 章数据格式），推理时模型自己产出 eos 即停；不会停的模型是没训好或分布外。
- 三道刹车：EOS 主动停、max_new_tokens 硬上限（minimind 默认 8192，注释提醒「并非模型实际长文本能力」）、stop 字符串（API 层概念，模型外拦截）。
- 截断风险：撞上限被砍断的回答是半个句子，结构化输出场景尤其致命（为 12.4 埋伏笔）。
- 流式：TextStreamer 逐 token 吐出（eval_llm.py:65、84-90）；体验意义（首 token 延迟）与工程意义（SSE 推送），吞吐细节留第 13 章。
- minimind 停止逻辑：finished 掩码让已停序列补 eos 占位、全部停才 break（284、288-290 行）——批量推理的对齐技巧。
- 图二（局部放大）：一条生成时间线——逐 token 推进，标出首 token 延迟点、EOS 命中点、max_tokens 硬刹车点三种结局。

## ch12_4_structured_output — 12.4 结构化输出：让模型按格式说话

固定 h2：
- 12.4.1 为什么需要约束：JSON 错一个逗号就全盘崩
- 12.4.2 约束解码原理：每步给词表加掩码
- 12.4.3 工程方案：Outlines、GBNF 与 Structured Outputs
- 12.4.4 代价与边界：约束会不会让模型变笨

内容要点：
- 场景：API 返回 JSON、函数调用参数、固定模板；prompt 请求格式是「软约束」，约束解码是「硬约束」。
- 原理：把格式编译成状态机（FSM）或上下文无关文法（CFG），解码每步只保留语法合法的 token，其余 logit 置 −∞——与 top-p 的置 −∞ 是同一招（呼应 12.2），只是掩码来自语法而不是概率。
- 方案对照表：Outlines（FSM，arXiv:2307.09709）、llama.cpp GBNF（语法文件）、OpenAI Structured Outputs（JSON Schema，官方博客注明）；token 对齐问题（一个语法符号可能跨多个 token）要点破。
- 代价：约束本身不改概率分布的相对形状（只是屏蔽），但格式死板可能挤压表达空间；给「该约束什么、不该约束什么」的实践建议（约束骨架、放开内容）。
- 图二（局部放大）：约束解码一帧——当前状态机停在「等待键名字符串」状态，词表 6400 个 token 被分成合法（绿）与非法（红）两堆，softmax 只在合法堆里做。

## ch12_5_decoding_practice — 12.5 解码实践与陷阱

固定 h2：
- 12.5.1 退化三兄弟：重复、空转、跑题
- 12.5.2 参数速查：不同任务怎么调三件套
- 12.5.3 解码与训练的关系：采样参数补不了能力的课
- 12.5.4 接力第 13 章：从解码机制到推理系统

内容要点：
- 退化三兄弟逐个给成因与对策：重复（贪心/低温 + 无惩罚 → 重复惩罚、升温）、空转绕圈（分布平坦 + 长上下文稀释）、跑题（高温乱抽样）；nucleus 论文对退化的分析作依据。
- 速查表：任务类型 × temperature/top_p/重复惩罚建议（事实问答低温、创意写作高温、代码低温 + 结构化约束、数学推理低温或自一致性投票——自一致性留一句指向 14 章）。
- 观点：采样参数只改变「从分布里怎么挑」，不改变分布本身——分布的形状是训练定的（呼应第 6-11 章）；烂模型调不出好回答，好模型可以被烂参数毁掉。
- 接力：KV Cache、量化、投机采样、批量调度都在第 13 章；应用层组合（CoT、RAG、Agent）在第 14 章。
- 图二（局部放大）：「参数调参台」示意——三个滑块（温度/top_p/重复惩罚）各配一条分布曲线小图，标注不同任务的推荐档位。
