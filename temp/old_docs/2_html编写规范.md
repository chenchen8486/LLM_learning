# Transformer 学习手册 — HTML 编写规范

> 本文档完整记录当前手册所有页面的结构格式约定，作为编写新章节片段的硬性参照。
> 与 `开发方案.md` 第 7 节互为补充：本文件更细、含可直接复制的代码骨架。
> 创建时间：2026-06-17　|　基线版本：第 1 章定稿格式

---

## 0. 速查：一节页面的标准骨架

```html
<article class="content">
  <h1>3 模型架构</h1>
  <p class="subtitle">对应论文：第 3 节 模型架构（含 3.1–3.5）　|　整理时间：2026-06-22</p>

  <!-- ① 快速浏览卡片 -->
  <div class="doc-meta-card"> … </div>

  <!-- ② 分级讲解：h2 / h3 必须使用与论文一致的阿拉伯数字标号 -->
  <h2>一、…</h2>
  <p>…</p>

  <!-- ③ 公式 / 配图 / 表格 / 重点框 按需穿插 -->

  <!-- ④ 末尾要点回顾卡片 -->
  <div class="socratic-card"> … </div>
</article>
```

页面的导航侧边栏、面包屑、上下节翻页、`<head>`（含 KaTeX/字体/全局 CSS）均由 `build.py` 自动注入，**片段中不要手写**。

---

## 1. 文件与命名约定

### 1.1 章节组织原则

- **每个大章节一个 HTML 文件**：HTML 页面的拆分严格遵循原始论文的顶级章节（如 `3 模型架构`、`5 训练`、`6 结果`）。该章节下的所有子章节（`3.1`、`3.2`、`3.2.1` 等）全部写在该同一个 HTML 文件中。
- **References、致谢不解构**：参考文献与致谢部分不单独生成页面。
- **章节标题与原始论文文档一致**：页面 `<h1>` 使用论文原始章节标题，不自行创造新标题。

### 1.2 文件名格式

文件名统一采用：

```
{年份}{论文缩写}_{章节序号}_{英文短名}.html
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `{年份}{论文缩写}` | 论文发表年份 + 缩写，必须包含年份以便清晰识别 | `2016Attention`、`2020ViT`、`2021DINO`、`2021Swin`、`2022MAE` |
| `{章节序号}` | 对应原始论文的顶级章节序号 | `0`=摘要，`1`=第 1 节，…，`8`=附录/可视化 |
| `{英文短名}` | 该章节内容的英文短标识 | `abstract`、`introduction`、`model_architecture` |

**示例：**

- `2016Attention_0_abstract.html` —— 摘要
- `2016Attention_1_introduction.html` —— 1 引言
- `2016Attention_3_model_architecture.html` —— 3 模型架构（含 3.1–3.5 全部子节）
- `2016Attention_5_training.html` —— 5 训练（含 5.1–5.4 全部子节）
- `2016Attention_8_attention_visualization.html` —— 注意力可视化（附录）

### 1.3 其他约定

| 项目 | 约定 |
|------|------|
| 片段存放目录 | `output/html/pages/` |
| 文件编码 | `utf-8-sig`（带 BOM，跨平台兼容） |
| 唯一 h1 | 每个片段必须且只有一个 `<h1>`，build 据此提取页面标题 |
| 旧格式兼容 | `build.py` 仍保留对旧 `ch{章}_{节}_{短名}.html` 格式的解析，但新页面统一使用新格式 |

---

## 2. 顶部固定结构

### 2.1 标题与副标题

```html
<h1>3 模型架构</h1>
<p class="subtitle">对应论文：第 3 节 模型架构（含 3.1–3.5）　|　整理时间：2026-06-22</p>
```

- `subtitle` 用于溯源，统一格式「对应论文：<章节> | 整理时间：<日期>」。

### 2.2 快速浏览卡片 `.doc-meta-card`

每节开头必备，给出本节的「电梯概览」。固定用一个 `.doc-meta` 表格，最后一行放关键词标签。

```html
<div class="doc-meta-card">
  <h3 class="doc-meta-title">快速浏览</h3>
  <table class="doc-meta">
    <tbody>
      <tr><td>核心问题</td><td>……</td></tr>
      <tr><td>本文主张</td><td>……</td></tr>
      <tr><td>一句话结论</td><td>……</td></tr>
      <tr><td>关键词</td><td>
        <div class="keywords"><span>标签A</span><span>标签B</span><span>标签C</span></div>
      </td></tr>
    </tbody>
  </table>
</div>
```

- 左列为固定宽度的标签名（如「核心问题/本文主张/关键词」），右列为内容。
- 行数不固定，按内容增减；`.keywords` 内每个 `<span>` 渲染为一枚胶囊标签。

---

## 3. 正文排版约定

| 元素 | 用法 |
|------|------|
| `<h2>` | 论文二级章节标题，**必须使用阿拉伯数字且与原文标号完全一致**，如 `3.1 编码器和解码器堆栈`、`5.2 硬件与时间安排` |
| `<h3>` | 论文三级章节标题，同样使用阿拉伯数字且与原文一致，如 `3.2.1 缩放点积注意力` |
| `<h4>` | 四级描述性小标题（大写灰色小标题，少用）；若原文未编号，则不得自行添加数字 |
| `<p>` | 普通段落 |
| `<ul>/<ol>` | 列表，关键步骤用有序列表 |
| `<blockquote>` | 引用论文原话 / 举例句子 |
| `<strong>` | 关键术语、结论加粗强调 |
| `<code>` | 行内代码 / 变量名（非数学公式） |
| `<pre><code>` | 代码块 / 数据流示意（深色背景） |

- **编号铁律**：标题编号必须与原始论文文档**百分百对应**。原始论文有编号的（如 3.1、3.2.1）必须保留；原始论文未编号的段落，使用纯描述性标题，**禁止自行使用「一、二、三…」中文序号或虚构 4.1/4.2 等标号**。
- **讲解风格**：先给直觉或类比，再上公式；重要结论用重点框（见 §5）突出。
- **承上启下**：适当点出与前作 / 后续论文的关系，呼应「演进主线」。

---

## 4. 公式（KaTeX）

| 类型 | 定界符 | 示例 |
|------|--------|------|
| 行内公式 | `\( … \)` | 维度 `\(d_k\)`、序列长度 `\(n\)` |
| 行间公式 | `\[ … \]` | 居中独立成行 |

- 论文 md 中的 `$…$` / `$$…$$` 必须转换为上述定界符，**不要直接用 `$`**（KaTeX 未配置美元定界符）。
- **公式卡片** `.formula-card`：行间公式的推荐容器，公式下方用 `.formula-note` 写一句话解读。

```html
<div class="formula-card">
  \[ \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V \tag{1} \]
  <div class="formula-note">公式 (1)：Transformer 的心脏。</div>
</div>
```

- 多行公式可在一个 `.formula-card` 内连写多个 `\[ … \]`。
- 编号用 `\tag{1}`。

> ⚠️ 已知坑：KaTeX 定界符在 `template.html` 中写作 `'\\('`（HTML 源码两个反斜杠），不可多写转义。此为模板维护项，编写片段时无需关心。

---

## 5. 重点框 `.highlight-box`

四种语义色，通过附加 class 切换：

```html
<div class="highlight-box">……</div>            <!-- 绿色：核心要点 / 正面结论 -->
<div class="highlight-box info">……</div>        <!-- 蓝色：补充说明 / 背景 -->
<div class="highlight-box warn">……</div>        <!-- 黄色：注意 / 矛盾点 -->
<div class="highlight-box danger">……</div>      <!-- 红色：易错 / 关键约束 -->
```

| 变体 | 颜色 | 适用 |
|------|------|------|
| （默认） | 绿 | 核心要点、正面结论、关键直觉 |
| `info` | 蓝 | 背景补充、定义解释 |
| `warn` | 黄 | 注意事项、问题所在、矛盾 |
| `danger` | 红 | 易错点、强约束（如掩码、维度限制） |

---

## 6. 概念卡片网格 `.concept-grid`

并列展示同级概念（如 Query / Key / Value），自适应换行。

```html
<div class="concept-grid">
  <div class="concept-card"><h4>Query（查询）</h4><p>当前位置「想找什么」。</p></div>
  <div class="concept-card"><h4>Key（键）</h4><p>每个位置「能提供什么标签」。</p></div>
  <div class="concept-card"><h4>Value（值）</h4><p>每个位置「真正的内容」。</p></div>
</div>
```

---

## 6A. 术语详解卡片 `.term-card`（初学者友好）

> 面向初学者，把专业词汇「掰开揉碎」：中文名 + 英文词源 + 通俗解释 + 生活化类比。每出现一个对初学者陌生的关键术语，优先用此卡片讲透。

```html
<div class="term-card">
  <div class="term-name">
    <span class="term-zh">编码器 Encoder</span>
    <span class="term-en">编码 = 把信息压缩成一种内部表示</span>
  </div>
  <div class="term-body">
    <p>编码器负责<strong>"读懂"输入</strong>……</p>
    <div class="term-analogy">就像一位同声传译员，先把整句外语在脑子里<strong>理解</strong>成"意思"。</div>
  </div>
</div>
```

- `.term-zh`：中文术语名（橙色加粗）；`.term-en`：英文原词 / 词源解释（灰斜体）。
- `.term-body`：通俗讲解段落。
- `.term-analogy`：生活化类比，自动加前缀「打个比方」。**强烈建议每个术语卡片都配一个类比。**

---

## 6B. 原创示意图 `.diagram`（自制 SVG）

> 当论文原图过于学术、或某流程/结构需要更直观呈现时，用内联 SVG 手绘示意图。**必须明确标注「原创示意图·非论文原图」以区别于论文配图。**

```html
<div class="diagram">
  <span class="diagram-tag">原创示意图 · 非论文原图</span>
  <svg viewBox="0 0 720 300" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans SC, sans-serif">
    <!-- 用 rect / text / line / path + marker 绘制流程 -->
  </svg>
  <div class="diagram-caption">图示：……一句话说明这张图在讲什么。</div>
</div>
```

- `.diagram-tag`：左上角紫色胶囊标签，固定写「原创示意图 · 非论文原图」。
- SVG 用 `viewBox` 自适应缩放，`font-family` 设为 `Noto Sans SC` 保持中文一致。
- 配色沿用主题：编码器/主结构用紫 `#6d28d9`，解码器/对比用青 `#0891b2`，强调/警示用红 `#dc2626`，语义数据用琥珀 `#fef3c7`。
- 箭头用 `<defs><marker>` 定义复用。
- `.diagram-caption`：图下说明。

---

## 7. 配图（论文原图）

> 图片名照抄 `ref_paper/<论文>/images/` 中的原始文件名；`src` 统一写 `images/ch{N}/<图片名>`，build.py 会自动从 ref_paper 复制同名图片到 `output/html/images/chN/`。

### 7.1 单图 `.img-block`

```html
<div class="img-block">
  <img src="images/ch1/f7896a22….jpg" alt="Transformer 模型架构图">
  <div class="img-caption">图 1：Transformer 整体架构。</div>
</div>
```

- 所有 `.img-block img` 支持**点击放大**（滚轮缩放 / 拖拽平移 / 点击关闭），由模板全局脚本处理。

### 7.2 多图并排 `.img-row`

```html
<div class="img-row">          <!-- 默认两列；三列加 .cols-3 -->
  <div class="img-block"> … </div>
  <div class="img-block"> … </div>
</div>
```

- `.img-row` 默认 2 列，`.img-row.cols-3` 为 3 列；窄屏自动单列。

---

## 8. 表格 `.table-wrap`

所有 `<table>` 用 `.table-wrap` 包裹，支持窄屏横向滚动。表头深紫渐变，悬停行高亮。

```html
<div class="table-wrap">
  <table>
    <thead><tr><th>层类型</th><th>每层复杂度</th><th>顺序操作数</th></tr></thead>
    <tbody>
      <tr><td>自注意力</td><td>\(O(n^2\cdot d)\)</td><td>\(O(1)\)</td></tr>
    </tbody>
  </table>
</div>
```

- 表格单元格内可写 KaTeX 行内公式。

---

## 9. 要点回顾卡片 `.socratic-card`

每节**末尾**必备，用 3 组「问题 + 关键补充」做轻量小结（自问自答，非互动）。

```html
<div class="socratic-card">
  <div class="sc-head"><span class="sc-icon">🧭</span><span>本节要点回顾</span></div>
  <div class="socratic-qa">
    <div class="sc-q">为什么 RNN 难以并行？</div>
    <div class="sc-key">因为 \(h_t\) 依赖 \(h_{t-1}\)，计算被强制串行……</div>
  </div>
  <!-- 重复 2-3 组 socratic-qa -->
</div>
```

- `.sc-q` 自动加前缀「问 」；`.sc-key` 自动加前缀「关键补充 · 」。
- 另有 `.sc-mine`（前缀「我的理解 · 」）为预留样式，当前模式下一般不用。

---

## 10. 不要手写的部分（由 build.py 注入）

| 部分 | 说明 |
|------|------|
| `<!DOCTYPE>` / `<head>` / `<body>` | 模板提供 |
| KaTeX / 字体 / Mermaid 引用 | 模板 `<head>` 注入 |
| 全局 CSS | `static/global.css` 注入 |
| 侧边栏导航 `.sidebar` | 按文件名自动生成 |
| 面包屑 `.breadcrumb` | 自动生成 |
| 上下节翻页 `.footer-nav` | 自动生成 |
| 图片放大脚本 | 模板全局脚本 |

> 片段只需写 `<article class="content"> … </article>`，外加可选的页面级 `<style>`、`<script>`（会被 build 提取注入）。

---

## 11. 完整性验证（强制红线）

> ⭐ **最高质量红线**：每编写完一章，必须对比原始 `ref_paper` 文档逐项核对，**公式与图片不得有任何缺失**。论文的图、表、公式是论证骨架，漏一个都会破坏理解链条。

### 11.1 验证范围

| 核对项 | 标准 |
|--------|------|
| **图片** | 原始 md 中所有 `![](images/xxx.jpg)` 引用的图片，必须全部在本章某个 html 中出现，且被 build.py 实际复制到 `images/{年份}{论文缩写}/` 目录下 |
| **行间公式** | 原始 md 中所有 `$$ … $$` 公式块，必须全部转写为 `\[ … \]` 出现在对应小节 |
| **编号公式** | 原始 md 中的 `\tag{n}` 编号公式必须保留编号 |
| **数据表** | 原始 md 中的关键 `<table>`（如结果表、消融表、复杂度表）不得遗漏 |
| **行内变量** | 正文涉及的关键变量/符号（如 \(d_k, n, h\)）应以公式形式呈现，不丢信息 |

### 11.2 验证操作（编写后必做）

1. **清点原始引用**：
   ```bash
   cd ref_paper/<论文>
   grep -oE '!\[\]\(images/[^)]+\)' *_zh.md     # 列出所有图片
   grep -c '^\$\$' *_zh.md                        # 行间公式块计数
   grep -oE '\\tag\{[0-9]+\}' *_zh.md             # 编号公式
   ```
2. **核对 html 产出**：对上一步每个图片名 / 公式，在本章 `{年份}{论文缩写}_*.html` 中 `grep -l` 确认存在。
3. **核对复制落地**：`ls output/html/images/{年份}{论文缩写}/ | wc -l` 应与引用图片数一致。
4. **发现缺失立即补全**，再重新构建。
5. 只有「图片数一致 + 公式逐条命中」后，该章才算可交付。

> 注：原始论文中纯参考文献、致谢等非内容性段落可不逐字搬运，但**正文中的图、表、公式一律不得省略**。

---

## 12. 变更记录

| 时间 | 内容 |
|------|------|
| 2026-06-17 | 初版：依据第 1 章定稿格式，完整记录页面骨架、命名、各组件类名与用法、公式/配图/表格规范。 |
| 2026-06-17 | 新增 §11 完整性验证红线：编写每章后必须对比 ref_paper 核对公式与图片零缺失，附验证操作命令。 |
| 2026-06-22 | 新增 §3 编号铁律：标题必须使用阿拉伯数字且与原始论文章节标号百分百对应；禁止自行使用中文序号或虚构标号。同步修正 §0 骨架注释。 |
