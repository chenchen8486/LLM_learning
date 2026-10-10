# 第 14 章 应用范式 · 分页规格

> 章定位：模型训练部署完毕（第 12–13 章收尾），本章回答「同一个模型怎么用得更好」——不改权重，只在模型外面搭脚手架。
> 生命周期定位条：hero = P8。注意图注必须写清：应用范式不是生命周期的新一站，而是架在 P8 推理部署之上的使用层；P8 高亮表示「模型本体到此为止，本章在它之上做文章」。已生成至 temp/strips/ch14_*_strip.html。
> minimind 锚点：eval_llm.py 的 apply_chat_template（76 行）与 base 模型裸拼 bos_token+prompt（74 行）——chat template 就是 prompt 结构化的代码实现。其余内容 minimind 无对应实现，走论文路线。

## ch14_1_prompt_engineering.html Prompt 工程与上下文学习
- 标题建议：14.1 Prompt 工程：不改权重的编程方式
- h2 固定：14.1.1 Prompt 是什么：模型唯一的输入通道（系统提示、用户消息、chat template 的结构化，对照 eval_llm.py 74/76 行两种拼法）；14.1.2 上下文学习：few-shot 为什么不改权重却有效（GPT-3 的定义与效果）；14.1.3 实用技巧清单（角色设定、指令具体化、输出格式约束、分隔符），每条说清为什么有效（往模型的训练分布上靠）；14.1.4 Prompt 的边界：它是请求不是约束（呼应 12.4 节的结构化输出）
- 论文：GPT-3 2005.14165（NeurIPS 2020，zero/one/few-shot 定义，1750 亿参数，无梯度更新现场学任务）；prompt 综述一新一旧：2107.13586（Pre-train Prompt Predict，ACM CSUR 2023）与 2402.07927（2024 现代技巧综述）；可选 2302.11382 模式目录（注意 2302.11373 是物理论文，别写错）

## ch14_2_cot.html 思维链
- 标题建议：14.2 思维链：让模型把中间步骤写出来
- h2 固定：14.2.1 一步到位的答案为什么容易错（算力分配直觉：答案 token 只有几个，中间过程没有计算量）；14.2.2 Few-shot CoT：示例里带上推理过程；14.2.3 Zero-shot CoT：「Let's think step by step」一句话的威力；14.2.4 变体与演进：自洽性多数投票、ToT 树搜索、到第 11 章 RLVR 的内化
- 要点：CoT 本质是给模型「计算草稿纸」，把推理时算力摊到更多 token 上（呼应 13.6 推理时扩展）
- 论文：CoT 2201.11903（NeurIPS 2022，GSM8K PaLM 540B 17.9%→58.1%）；Zero-shot CoT 2205.11916（MultiArith 17.7%→78.7%）；ToT 2305.10601（Game of 24：CoT 4% → ToT 74%）；自洽性 2203.11171（GSM8K +17.9 个点，13.6 节已展开，此处引用即可）

## ch14_3_rag.html RAG
- 标题建议：14.3 RAG：给模型外挂一个资料库
- h2 固定：14.3.1 为什么需要外挂：参数记忆的三个短板（知识截止、私有知识、幻觉）；14.3.2 RAG 原论文结构：检索器 DPR + 生成器，问题编码 → 检索 → 拼接进上下文；14.3.3 工程链路：切块、嵌入、向量库、召回、重排、塞进 prompt；14.3.4 RAG 的失效模式（检索不到、切坏上下文、噪声混入）
- 要点：RAG 没改模型一个字，只是把「查资料」这一步前置到推理之前；嵌入向量呼应 3.1 节、相似度呼应 A.2
- 论文：RAG 2005.11401（NeurIPS 2020，参数记忆+非参数记忆，检索器 DPR+生成器 BART-large，RAG-Sequence/RAG-Token 两形式）；DPR 2004.04906（EMNLP 2020，双编码器，top-20 命中率超 BM25 9-19 个点）

## ch14_4_function_calling.html Function Calling
- 标题建议：14.4 Function Calling：让模型学会「打电话」
- h2 固定：14.4.1 模型的短板：不会算、不知今夕、够不着外部世界；14.4.2 机制拆解：工具 schema 进 prompt → 模型输出结构化调用请求（呼应 12.4 约束解码）→ 外部执行 → 结果回注上下文；14.4.3 训练侧：Toolformer 自学调工具、Gorilla/ToolLLM 微调路线；14.4.4 边界与安全：模型只发请求，执行权在外部
- 要点：模型本身永远不会「执行」任何操作，它只生成文本；是脚手架把特定文本格式解释成调用
- 论文：OpenAI 官方博客 Function calling（2023-06 随 0613 模型引入，JSON Schema 描述函数，含提示注入风险警告）；Toolformer 2302.04761（NeurIPS 2023，自监督学调 API）；Gorilla 2305.15334（LLaMA 微调+文档检索，API 调用准确率超 GPT-4）；ToolLLM 2307.16789（ICLR 2024 Spotlight，ToolBench 16464 个真实 API）

## ch14_5_agent.html Agent
- 标题建议：14.5 Agent：规划、工具与多步执行循环
- h2 固定：14.5.1 从单次问答到多步闭环：Agent = LLM + 规划 + 工具 + 记忆；14.5.2 ReAct：推理与行动交替（Thought/Action/Observation 循环）；14.5.3 Reflexion 与自我纠错；14.5.4 Agent 的可靠性与局限：错误累积、上下文膨胀、评估难
- 要点：Agent 没有新机制，是把第 12 章解码、14.2 CoT、14.3 RAG、14.4 工具调用组装成一个循环
- 论文：ReAct 2210.03629（ICLR 2023，Thought/Action/Observation 循环，ALFWorld/WebShop 超基线 34%/10%）；Reflexion 2303.11366（NeurIPS 2023，文字反思存情景记忆，HumanEval 91%）；Agent 综述 2309.07864（复旦，大脑/感知/行动三组件框架）

## 通用约束（每页必读）
- 开头 doc-meta-card + 综述段；结尾 socratic-card 要点回顾 + 参考来源区块
- 图 1 = 定位条（插入 temp/strips/ 对应片段内容，caption 按上方「章定位」要求写）；后续图从图 2 起编号
- 公式 \( \) / \[ \]，禁 $；公式符号给注解；公式内 <>& 写实体
- 自然段段首两个全角空格；卡片表格列表内不缩进
- 注释卡用橙色 highlight-box info，前缀「注释：」；先文字后图
- minimind 引用注行号出处；理论表述以论文为准
- 拿不准的标待查证，绝不编造
- 不设「动手验证」区块
