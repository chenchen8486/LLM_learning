# -*- coding: utf-8 -*-
"""ch4_4 图组重排：页首图 3 移入 4.4.4，两塔放大图统一自下而上，顺序对齐图 2 左右。"""
import io, re

P = 'output/html/pages/ch4_4_ffn_residual_norm.html'
s = io.open(P, encoding='utf-8').read()
orig = s

# ── 1) 图 2：注释、图例、图注换向（紫框=图3编码塔，橙框=图4解码塔）──
for a, b in [
    ('（紫色粗边框 = 图 4 展开范围）', '（紫色粗边框 = 图 3 展开范围）'),
    ('（橙色边框 = 图 3 展开范围）', '（橙色边框 = 图 4 展开范围）'),
    ('紫色边框的两个子层组（自注意力 + FFN）= 图 4 展开范围（4.4.4）',
     '紫色边框的两个子层组（自注意力 + FFN）= 图 3 展开范围（4.4.4）'),
    ('橙色边框的两个子层组（子层 1 + 子层 3）= 图 3 展开的范围',
     '橙色边框的两个子层组（子层 1 + 子层 3）= 图 4 展开的范围'),
]:
    assert a in s, a
    s = s.replace(a, b)

# 图 2 图注：改写展开关系描述
old_cap2 = '两塔的子层组是同构的——编码塔的子层组（自注意力 + FFN，紫色边框）由图 4 按原论文 Post-LN 顺序展开；今天的 decoder-only LLM 相当于只保留解码塔、去掉编码器-解码器注意力子层，图 3 展开的就是解码塔中橙色边框圈出的两个子层组（子层 1 掩码自注意力 + 子层 3 FFN）的现代形态。本节讲的 FFN、残差、归一化三件套对两塔一视同仁，差别只在顺序与掩码。'
new_cap2 = '两塔的子层组是同构的：图 3 放大左侧编码塔紫色边框圈出的两个子层组（自注意力 + FFN）按原论文 Post-LN 顺序的完整数据流；图 4 放大右侧解码塔橙色边框圈出的两个子层组（子层 1 掩码自注意力 + 子层 3 FFN）在现代 decoder-only 模型里的形态（Pre-Norm）。今天的 LLM 相当于只保留解码塔、去掉编码器-解码器注意力子层。本节讲的 FFN、残差、归一化三件套对两塔一视同仁，差别只在顺序与掩码。'
assert old_cap2 in s
s = s.replace(old_cap2, new_cap2)

# ── 2) 页首：删掉现代拼装图（旧图 3），换成过渡段 ──
new_trans = ('<p>　　三张图的分工先立清楚：图 2 是两塔总览（数据自下而上），紫框圈出编码塔一层的两个子层组，橙框圈出解码塔一层的三个子层组（其中子层 2「编码器-解码器注意力」在 decoder-only 模型里没有对应物，后面不再出现）。本节讲到 4.4.4 拼装时，再把这两个框各自放大一层：图 3 放大编码塔（原论文的顺序），图 4 放大解码塔（今天模型的做法）。先把三件套逐件讲透，拼装留到最后。\n\n')
s, n1 = re.subn(r'<p>　　对应关系先立清楚.*?(?=<h2>4\.4\.1 FFN)', new_trans, s, flags=re.S)
assert n1 == 1, '页首图块删除失败'

# ── 3) 4.4.4：把原图 4（编码塔 Post-LN）重绘为自下而上并改编号为图 3 ──
FIG3 = '''<p>　　把套路画出来。图 3 是图 2 左侧编码塔一层的放大，自下而上读：每个子层先计算（注意力或 FFN），再被 dropout、残差相加、层归一化依次包裹，数据流方向与图 2 一致：</p>

<div class="diagram">
  <span class="diagram-tag">原创示意图 · 非论文原图</span>
  <svg viewBox="0 0 760 660" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans SC, sans-serif">
    <defs>
      <marker id="arrE44" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#6d28d9"/>
      </marker>
      <marker id="arrEC44" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#0891b2"/>
      </marker>
    </defs>
    <text x="40" y="30" font-size="13.5" font-weight="bold" fill="#4c1d95">局部放大：编码塔一层（原论文 Post-LN 顺序，512 维）</text>
    <text x="40" y="50" font-size="11" fill="#64748b">来自图 2 左侧紫色边框；数据自下而上，每个子层被 dropout、残差相加、层归一化依次包裹</text>

    <rect x="270" y="64" width="220" height="42" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="380" y="82" text-anchor="middle" font-size="14" fill="#333">输出</text>
    <text x="380" y="99" text-anchor="middle" font-size="11" fill="#666">形状不变：(n, 512)</text>
    <line x1="380" y1="124" x2="380" y2="108" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="124" width="220" height="42" rx="10" fill="#ecfeff" stroke="#0891b2" stroke-width="1.5"/>
    <text x="380" y="143" text-anchor="middle" font-size="13.5" fill="#155e75">层归一化 LayerNorm</text>
    <text x="380" y="159" text-anchor="middle" font-size="10.5" fill="#666">标准化 + 缩放平移</text>
    <line x1="380" y1="184" x2="380" y2="168" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <circle cx="380" cy="199" r="15" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="380" y="205" text-anchor="middle" font-size="18" fill="#92400e">+</text>
    <line x1="380" y1="232" x2="380" y2="216" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="232" width="220" height="34" rx="10" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>
    <text x="380" y="253" text-anchor="middle" font-size="12.5" fill="#555">Dropout 随机丢弃（p=0.1）</text>
    <line x1="380" y1="284" x2="380" y2="268" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="284" width="220" height="44" rx="10" fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>
    <text x="380" y="303" text-anchor="middle" font-size="14" fill="#333">FFN 前馈网络</text>
    <text x="380" y="320" text-anchor="middle" font-size="10.5" fill="#666">512 → 2048 → 512，逐位置加工</text>
    <line x1="380" y1="346" x2="380" y2="330" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="346" width="220" height="42" rx="10" fill="#ecfeff" stroke="#0891b2" stroke-width="1.5"/>
    <text x="380" y="365" text-anchor="middle" font-size="13.5" fill="#155e75">层归一化 LayerNorm</text>
    <text x="380" y="381" text-anchor="middle" font-size="10.5" fill="#666">标准化 + 缩放平移</text>
    <line x1="380" y1="406" x2="380" y2="390" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <circle cx="380" cy="421" r="15" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="380" y="427" text-anchor="middle" font-size="18" fill="#92400e">+</text>
    <line x1="380" y1="454" x2="380" y2="438" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="454" width="220" height="34" rx="10" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>
    <text x="380" y="475" text-anchor="middle" font-size="12.5" fill="#555">Dropout 随机丢弃（p=0.1）</text>
    <line x1="380" y1="506" x2="380" y2="490" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="506" width="220" height="44" rx="10" fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>
    <text x="380" y="525" text-anchor="middle" font-size="14" fill="#333">多头自注意力</text>
    <text x="380" y="542" text-anchor="middle" font-size="10.5" fill="#666">token 间交流（4.2 节）</text>
    <line x1="380" y1="568" x2="380" y2="552" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrE44)"/>

    <rect x="270" y="568" width="220" height="46" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="380" y="588" text-anchor="middle" font-size="14" fill="#333">输入 x</text>
    <text x="380" y="605" text-anchor="middle" font-size="11" fill="#666">n 个 token，各 512 维</text>

    <path d="M490,586 H585 V421 H397" fill="none" stroke="#0891b2" stroke-width="2" marker-end="url(#arrEC44)"/>
    <text x="498" y="578" text-anchor="start" font-size="11.5" fill="#0891b2">残差旁路：输入直达</text>
    <path d="M270,360 H175 V199 H363" fill="none" stroke="#0891b2" stroke-width="2" marker-end="url(#arrEC44)"/>
    <text x="182" y="352" text-anchor="start" font-size="11.5" fill="#0891b2">残差旁路</text>

    <text x="380" y="642" text-anchor="middle" font-size="12.5" fill="#666">每个子层都被同一套流程包裹：计算 → dropout → 残差相加 → 层归一化</text>
  </svg>
  <div class="diagram-caption">图 3：原论文 Post-LN 顺序下，一层编码器的完整数据流（图 2 左侧紫色边框的放大，方向与图 2 一致：自下而上）。两个子层各自被三件套包裹，张量形状全程 (n, 512) 不变。与图 4（解码塔现代 Pre-Norm 形态）对照看：这里是「相加后归一化」，图 4 是「归一化后计算再相加」。</div>
</div>
'''

s, n2 = re.subn(r'<div class="diagram">\s*<span class="diagram-tag">原创示意图 · 非论文原图</span>\s*<svg viewBox="0 0 760 640".*?</svg>\s*<div class="diagram-caption">图 4：[^<]*</div>\s*</div>', FIG3.rstrip('\n'), s, flags=re.S)
assert n2 == 1, '编码塔图块替换失败'

# 「注意图 4 里」→「注意图 3 里」
assert '注意图 4 里一个贯穿始终的事实' in s
s = s.replace('注意图 4 里一个贯穿始终的事实', '注意图 3 里一个贯穿始终的事实')

# ── 4) 在「演进预告」框之后插入过渡段 + 图 4（解码塔、现代 Pre-Norm、自下而上）──
FIG4 = '''
<p>　　现代模型对图 3 这套流程做了三处改动：归一化挪到子层之前、LN 换成 RMSNorm、FFN 换成 SwiGLU。图 4 是图 2 右侧解码塔一层的放大，用的就是改动后的现代形态，与图 3 对照着看，改动位置一清二楚：</p>

<div class="diagram">
  <span class="diagram-tag">原创示意图 · 非论文原图</span>
  <svg viewBox="0 0 760 620" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans SC, sans-serif">
    <defs>
      <marker id="arrD44" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#6d28d9"/>
      </marker>
      <marker id="arrDC44" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="#0891b2"/>
      </marker>
    </defs>
    <text x="40" y="30" font-size="13.5" font-weight="bold" fill="#4c1d95">局部放大：解码塔一层（现代 Pre-Norm 顺序，minimind 数字）</text>
    <text x="40" y="50" font-size="11" fill="#64748b">来自图 2 右侧橙色边框；对照图 3 的原论文顺序——归一化挪到子层之前，LN 换成 RMSNorm（第 5 章细讲）</text>

    <rect x="270" y="64" width="220" height="46" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="380" y="84" text-anchor="middle" font-size="14" fill="#333">输出</text>
    <text x="380" y="101" text-anchor="middle" font-size="11" fill="#666">3 × 768：形状与输入完全相同</text>
    <line x1="380" y1="128" x2="380" y2="112" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <circle cx="380" cy="143" r="15" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="380" y="149" text-anchor="middle" font-size="18" fill="#92400e">+</text>
    <line x1="380" y1="176" x2="380" y2="160" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <rect x="270" y="176" width="220" height="48" rx="10" fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>
    <text x="380" y="196" text-anchor="middle" font-size="14" fill="#333">FFN（SwiGLU 门控）</text>
    <text x="380" y="215" text-anchor="middle" font-size="10.5" fill="#666">768 → 2432 → 768，逐位置（5.1 节）</text>
    <line x1="380" y1="242" x2="380" y2="226" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <rect x="270" y="242" width="220" height="42" rx="10" fill="#ecfeff" stroke="#0891b2" stroke-width="1.5"/>
    <text x="380" y="261" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#155e75">RMSNorm</text>
    <text x="380" y="277" text-anchor="middle" font-size="10.5" fill="#666">3 × 768 → 3 × 768</text>
    <line x1="380" y1="302" x2="380" y2="286" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <circle cx="380" cy="317" r="15" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="380" y="323" text-anchor="middle" font-size="18" fill="#92400e">+</text>
    <line x1="380" y1="350" x2="380" y2="334" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <rect x="270" y="350" width="220" height="52" rx="10" fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>
    <text x="380" y="371" text-anchor="middle" font-size="14" fill="#333">掩码多头自注意力</text>
    <text x="380" y="390" text-anchor="middle" font-size="10.5" fill="#666">GQA：8 个 Q 头 / 4 组 K/V（5.2 节）</text>
    <line x1="380" y1="420" x2="380" y2="404" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <rect x="270" y="420" width="220" height="42" rx="10" fill="#ecfeff" stroke="#0891b2" stroke-width="1.5"/>
    <text x="380" y="439" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#155e75">RMSNorm</text>
    <text x="380" y="455" text-anchor="middle" font-size="10.5" fill="#666">3 × 768 → 3 × 768</text>
    <line x1="380" y1="480" x2="380" y2="464" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrD44)"/>

    <rect x="270" y="480" width="220" height="46" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
    <text x="380" y="500" text-anchor="middle" font-size="14" fill="#333">输入 x</text>
    <text x="380" y="517" text-anchor="middle" font-size="11" fill="#666">3 个 token × 768 维</text>

    <path d="M490,498 H585 V317 H397" fill="none" stroke="#0891b2" stroke-width="2" marker-end="url(#arrDC44)"/>
    <text x="498" y="490" text-anchor="start" font-size="11.5" fill="#0891b2">残差旁路：输入直达</text>
    <path d="M270,434 H175 V143 H363" fill="none" stroke="#0891b2" stroke-width="2" marker-end="url(#arrDC44)"/>
    <text x="182" y="426" text-anchor="start" font-size="11.5" fill="#0891b2">残差旁路</text>

    <text x="380" y="556" text-anchor="middle" font-size="12.5" fill="#666">每个子层都是「先 RMSNorm、再计算、最后残差相加」；形状全程 3 × 768</text>
    <text x="380" y="580" text-anchor="middle" font-size="12.5" fill="#666">同样的 Block 叠 8 层（minimind）</text>
  </svg>
  <div class="diagram-caption">图 4：图 2 右侧解码塔中橙色边框圈出的两个子层组（子层 1 掩码自注意力 + 子层 3 FFN）在现代 decoder-only 模型里的展开形态（Pre-Norm 顺序，数字取自 minimind：768 维主干、FFN 中间层 2432、GQA 8 头 Q / 4 组 KV），方向与图 2、图 3 一致：自下而上。两个子层各自先过 RMSNorm 再计算，输入沿残差旁路原样直达加法点；张量形状从头到尾是 3 × 768，所以 8 层可以逐层接续。与图 3（原论文 Post-LN）对照：顺序从「相加后归一化」变成「归一化后计算再相加」，LN 换成 RMSNorm，FFN 换成 SwiGLU，原因在第 5 章。</div>
</div>
'''

anchor = '5.1 节细讲。\n</div>\n'
assert anchor in s, '演进预告框锚点缺失'
s = s.replace(anchor, anchor + FIG4, 1)

assert s != orig
io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('done; chars', len(s))
