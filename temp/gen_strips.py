#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为定位双图 campaign 批量生成每页的图一（全链路定位条）SVG 片段。

输出到 temp/strips/{page}_strip.html，内容是一个完整 .diagram 块（SVG 已按页面状态着色），
正文处的 __CAPTION__ 占位符由撰写代理替换为逐页图注。

状态语义（与已确认的 6 页一致）：
  hero  = 紫底白字，本节主角，第三行标签改为（本节主角）
  solid = 浅紫实线，已讲章节
  dash  = 灰底虚线，后续章节
"""

from pathlib import Path

OUT = Path(__file__).parent / "strips"
OUT.mkdir(exist_ok=True)

MARKER = """    <defs>
      <marker id="arrS{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#6d28d9"/>
      </marker>
      <marker id="arrG{uid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#9ca3af"/>
      </marker>
    </defs>
"""

# 三种状态的样式参数：(rect fill, stroke, dash, 标题色, 说明色, 章节号色)
STYLE = {
    "hero": ("#6d28d9", "#4c1d95", None, "#ffffff", "#ede9fe", "#d8b4fe"),
    "solid": ("#f5f3fe", "#6d28d9", None, "#4c1d95", "#555555", "#888888"),
    "dash": ("#fafafa", "#9ca3af", 'stroke-dasharray="5 4"', "#6b7280", "#6b7280", "#9ca3af"),
}


def box(x, y, title, desc, ref, state):
    fill, stroke, dash, c1, c2, c3 = STYLE[state]
    dash_attr = f" {dash}" if dash else ""
    sw = "2" if state == "hero" else "1.5"
    if state == "hero":
        ref = "（本节主角）"
    return (
        f'    <rect x="{x}" y="{y}" width="150" height="80" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>\n'
        f'    <text x="{x+75}" y="{y+28}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="{c1}">{title}</text>\n'
        f'    <text x="{x+75}" y="{y+50}" text-anchor="middle" font-size="11" fill="{c2}">{desc}</text>\n'
        f'    <text x="{x+75}" y="{y+68}" text-anchor="middle" font-size="10" fill="{c3}">{ref}</text>\n'
    )


def arrow(x1, y1, x2, y2, solid_arrow, uid):
    if solid_arrow:
        return f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrS{uid})"/>\n'
    return f'    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9ca3af" stroke-width="2" stroke-dasharray="4 3" marker-end="url(#arrG{uid})"/>\n'


def legend(kind):
    """kind: 'start'（输入起点）| 'normal'（已讲章节）| 'hero_only'（本章最前，无已讲框）"""
    if kind == "start":
        left = "实线框：输入起点"
    elif kind == "hero_only":
        left = "紫底框：本节主角"
    else:
        left = "实线框：已讲章节"
    return (
        f'    <rect x="200" y="292" width="18" height="11" rx="2" fill="#f5f3fe" stroke="#6d28d9" stroke-width="1.2"/>\n'
        f'    <text x="226" y="302" font-size="10.5" fill="#555">{left}</text>\n'
        f'    <rect x="412" y="292" width="18" height="11" rx="2" fill="#fafafa" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="4 3"/>\n'
        f'    <text x="438" y="302" font-size="10.5" fill="#6b7280">虚线框：后续章节</text>\n'
    )


# ── 前向链路条（第 2、3、4、5 章）──
# 行1：A 原始文本(20) → B 分词器(200) → C Embedding(380) → D 位置编码(560)
# 行2（右→左）：E Transformer 层(560) → F 输出投影(380) → G 采样(200)，回路 G → A
FWD_BOXES = {
    "A": (20, 36, "原始文本：", "「我爱中国」", "（输入起点）"),
    "B": (200, 36, "分词器：", "切成 token ID 序列", "（第 2 章）"),
    "C": (380, 36, "Embedding 层：", "查表：ID → 词向量", "（3.1 节）"),
    "D": (560, 36, "位置编码：", "注入「第几格」", "（3.2–3.5 节）"),
    "E": (560, 152, "Transformer 层：", "注意力与前馈 ×N", "（第 4 章）"),
    "F": (380, 152, "输出投影：", "算各 token 的概率", "（第 4 章）"),
    "G": (200, 152, "采样：", "吐出下一个 token", "（第 12 章）"),
}
FWD_FLOW = ["A", "B", "C", "D", "E", "F", "G"]  # 箭头方向按此顺序


def forward_strip(uid, hero, legend_kind):
    states = {k: "dash" for k in FWD_BOXES}
    for k in FWD_FLOW[: FWD_FLOW.index(hero[0])]:
        states[k] = "solid"
    for h in hero:
        states[h] = "hero"
    parts = [f'  <svg viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans SC, sans-serif">\n']
    parts.append(MARKER.format(uid=uid))
    for k, (x, y, t, d, r) in FWD_BOXES.items():
        parts.append(box(x, y, t, d, r, states[k]))
    # 行1 箭头 A→B→C→D（y=76）
    for i, (a, b) in enumerate([("A", "B"), ("B", "C"), ("C", "D")]):
        x1 = FWD_BOXES[a][0] + 150
        x2 = FWD_BOXES[b][0] - 4
        ok = states[a] != "dash" and states[b] != "dash"
        parts.append(arrow(x1, 76, x2, 76, ok, uid))
    # 右侧折向下行 D→E
    ok = states["D"] != "dash" and states["E"] != "dash"
    parts.append(arrow(635, 116, 635, 148, ok, uid))
    # 行2 箭头 E→F→G（右→左，y=192）
    for a, b in [("E", "F"), ("F", "G")]:
        x1 = FWD_BOXES[a][0]
        x2 = FWD_BOXES[b][0] + 150 + 4
        ok = states[a] != "dash" and states[b] != "dash"
        parts.append(arrow(x1, 192, x2, 192, ok, uid))
    # 自回归回路 G → A（采样属第 12 章，始终灰虚）
    parts.append(
        f'    <path d="M 200 192 C 110 200, 95 170, 95 118" fill="none" stroke="#9ca3af" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#arrG{uid})"/>\n'
        f'    <text x="140" y="248" text-anchor="middle" font-size="10.5" fill="#6b7280">生成的新 token 接回输入，继续循环（自回归，第 12 章）</text>\n'
    )
    parts.append(legend(legend_kind))
    parts.append("  </svg>\n")
    return "".join(parts)


# ── 生命周期条（第 1、6、7、8、9 章）──
# 行1：P1 训练数据(20) → P2 预训练(200) → P3 训练工程(380) → P4 中间训练(560)
# 行2（右→左）：P5 SFT(560) → P6 偏好对齐(380) → P7 推理训练(200) → P8 推理与部署(20)
LIFE_BOXES = {
    "P1": (20, 36, "训练数据：", "收集 · 清洗 · 配比", "（第 1 章）"),
    "P2": (200, 36, "预训练：", "海量语料自监督", "（第 6 章）"),
    "P3": (380, 36, "训练工程：", "并行 · 精度 · 稳定", "（第 7 章）"),
    "P4": (560, 36, "中间训练：", "收口 · 长上下文", "（第 8 章）"),
    "P5": (560, 152, "SFT：", "学会对话格式", "（第 9 章）"),
    "P6": (380, 152, "偏好对齐：", "RLHF / DPO", "（第 10 章）"),
    "P7": (200, 152, "推理训练：", "RLVR / GRPO", "（第 11 章）"),
    "P8": (20, 152, "推理与部署：", "解码 · 加速", "（第 12–13 章）"),
}
LIFE_FLOW = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


def lifecycle_strip(uid, hero):
    states = {k: "dash" for k in LIFE_BOXES}
    for k in LIFE_FLOW[: LIFE_FLOW.index(hero)]:
        states[k] = "solid"
    states[hero] = "hero"
    parts = [f'  <svg viewBox="0 0 720 330" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans SC, sans-serif">\n']
    parts.append(MARKER.format(uid=uid))
    for k, (x, y, t, d, r) in LIFE_BOXES.items():
        parts.append(box(x, y, t, d, r, states[k]))
    for a, b in [("P1", "P2"), ("P2", "P3"), ("P3", "P4")]:
        x1 = LIFE_BOXES[a][0] + 150
        x2 = LIFE_BOXES[b][0] - 4
        ok = states[a] != "dash" and states[b] != "dash"
        parts.append(arrow(x1, 76, x2, 76, ok, uid))
    ok = states["P4"] != "dash" and states["P5"] != "dash"
    parts.append(arrow(635, 116, 635, 148, ok, uid))
    for a, b in [("P5", "P6"), ("P6", "P7"), ("P7", "P8")]:
        x1 = LIFE_BOXES[a][0]
        x2 = LIFE_BOXES[b][0] + 150 + 4
        ok = states[a] != "dash" and states[b] != "dash"
        parts.append(arrow(x1, 192, x2, 192, ok, uid))
    parts.append(legend("hero_only" if hero == "P1" else "normal"))
    parts.append("  </svg>\n")
    return "".join(parts)


WRAP = """<div class="diagram">
  <span class="diagram-tag">原创示意图 · 非论文原图</span>
{svg}  <div class="diagram-caption">__CAPTION__</div>
</div>
"""

# 每页的图一配置：(文件名, 链路, hero, legend)
PAGES = {
    # 前向链路
    "ch2_1_why_tokenization": ("fwd", ["B"], "start"),
    "ch2_3_wordpiece_unigram": ("fwd", ["B"], "start"),
    "ch2_4_sentencepiece": ("fwd", ["B"], "start"),
    "ch2_5_vocab_special_tokens": ("fwd", ["B"], "start"),
    "ch2_6_tokenizer_in_practice": ("fwd", ["B"], "start"),
    "ch3_3_sinusoidal_learned": ("fwd", ["D"], "normal"),
    "ch3_5_length_extrapolation": ("fwd", ["D"], "normal"),
    "ch4_1_architecture_overview": ("fwd", ["E", "F"], "normal"),
    "ch4_4_ffn_residual_norm": ("fwd", ["E"], "normal"),
    "ch4_5_decoder_only": ("fwd", ["E"], "normal"),
    "ch4_6_forward_walkthrough": ("fwd", ["E", "F"], "normal"),
    "ch5_1_norm_activation": ("fwd", ["E"], "normal"),
    "ch5_2_mqa_gqa": ("fwd", ["E"], "normal"),
    "ch5_3_mla": ("fwd", ["E"], "normal"),
    "ch5_4_moe": ("fwd", ["E"], "normal"),
    "ch5_5_modern_recap": ("fwd", ["E"], "normal"),
    # 生命周期
    "ch1_1_corpus_sources": ("life", "P1"),
    "ch1_2_quality_filtering": ("life", "P1"),
    "ch1_3_dedup": ("life", "P1"),
    "ch1_4_data_mixture": ("life", "P1"),
    "ch1_5_synthetic_data": ("life", "P1"),
    "ch6_1_pretraining_objective": ("life", "P2"),
    "ch6_2_optimizer_hyperparams": ("life", "P2"),
    "ch6_3_training_loop": ("life", "P2"),
    "ch6_4_scaling_law": ("life", "P2"),
    "ch6_5_emergence": ("life", "P2"),
    "ch7_1_scaling_wall": ("life", "P3"),
    "ch7_2_data_parallel_zero": ("life", "P3"),
    "ch7_3_tensor_pipeline_parallel": ("life", "P3"),
    "ch7_4_mixed_precision_stability": ("life", "P3"),
    "ch7_5_compute_cost": ("life", "P3"),
    "ch8_1_midtraining_overview": ("life", "P4"),
    "ch8_2_rope_scaling": ("life", "P4"),
    "ch8_3_long_context_training": ("life", "P4"),
    "ch8_4_domain_adapt_anneal": ("life", "P4"),
    "ch8_5_midtraining_practice": ("life", "P4"),
    "ch9_1_why_sft": ("life", "P5"),
    "ch9_2_sft_data": ("life", "P5"),
    "ch9_3_sft_mechanics": ("life", "P5"),
    "ch9_4_lora": ("life", "P5"),
    "ch9_5_peft_family": ("life", "P5"),
    "ch10_1_alignment_gap": ("life", "P6"),
    "ch10_2_reward_model": ("life", "P6"),
    "ch10_3_ppo_rlhf": ("life", "P6"),
    "ch10_4_dpo_family": ("life", "P6"),
    "ch10_5_safety_alignment": ("life", "P6"),
    "ch11_1_why_rlvr": ("life", "P7"),
    "ch11_2_rlvr_loop": ("life", "P7"),
    "ch11_3_grpo": ("life", "P7"),
    "ch11_4_long_cot": ("life", "P7"),
    "ch11_5_reasoning_pitfalls": ("life", "P7"),
    "ch12_1_decoding_loop": ("life", "P8"),
    "ch12_2_sampling": ("life", "P8"),
    "ch12_3_stopping": ("life", "P8"),
    "ch12_4_structured_output": ("life", "P8"),
    "ch12_5_decoding_practice": ("life", "P8"),
}


def main():
    for name, cfg in PAGES.items():
        uid = name.replace("ch", "").replace("_", "")
        if cfg[0] == "fwd":
            svg = forward_strip(uid, cfg[1], cfg[2])
        else:
            svg = lifecycle_strip(uid, cfg[1])
        (OUT / f"{name}_strip.html").write_text(WRAP.format(svg=svg), encoding="utf-8")
    print(f"生成 {len(PAGES)} 个图一片段 -> {OUT}")


if __name__ == "__main__":
    main()
