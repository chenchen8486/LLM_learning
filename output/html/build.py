#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 机制学习手册 HTML 构建脚本

用法:
    python output/html/build.py

说明:
    1. 扫描 output/html/pages/ 目录下的页面片段
    2. 自动提取标题、正文、样式和脚本
    3. 根据文件名自动推导章节层级，生成导航侧边栏与面包屑
    4. 自动复制 pages 中引用的 ref_paper 配图到 output/html/images/{章标识}/
    5. 将结果写入 output/html/ 目录（覆盖原有 HTML）

新增页面流程:
    1. 在 output/html/pages/ 下新建页面片段（如 pages/ch2_1_bpe.html）
    2. 运行 python output/html/build.py
    3. 自动在 output/html/ 生成完整页面（含导航、公式、配图）
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────
# 可配置：章节与页面标题
# ──────────────────────────────────────────
# 三级小节分组标题（ch{章}_{节} 分组），当前命名规则下一般不使用
NAV_GROUPS: Dict[str, str] = {}

# 章号 → 顶层导航/面包屑标题（LLM 生命周期主线）
CHAPTER_TITLES: Dict[int, str] = {
    0: "第 0 章 全景导论",
    1: "第 1 章 训练数据",
    2: "第 2 章 分词",
    3: "第 3 章 词向量与位置编码",
    4: "第 4 章 Transformer 架构精讲",
    5: "第 5 章 现代架构变体与效率改进",
    6: "第 6 章 预训练",
    7: "第 7 章 预训练工程",
    8: "第 8 章 中间训练",
    9: "第 9 章 后训练 I：监督微调",
    10: "第 10 章 后训练 II：偏好对齐",
    11: "第 11 章 后训练 III：推理能力训练",
    12: "第 12 章 推理机制与后处理",
    13: "第 13 章 推理系统与部署",
    14: "第 14 章 应用范式",
    15: "第 15 章 评测与局限",
    16: "第 16 章 前沿趋势与展望",
    90: "附录 数学与深度学习基础速览",
}

PAGE_TITLE_FALLBACKS: Dict[str, str] = {}

# ──────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────
HTML_DIR = Path(__file__).parent.resolve()
PAGES_DIR = HTML_DIR / "pages"
TEMPLATE_FILE = HTML_DIR / "template.html"
GLOBAL_CSS_FILE = HTML_DIR / "static" / "global.css"
IMAGES_DIR = HTML_DIR / "images"
REF_PAPER_DIR = HTML_DIR.parent.parent / "ref_paper"


def parse_page(page_path: Path) -> Tuple[str, str, str, str]:
    """解析页面片段，返回 (page_styles, content, scripts, title)。"""
    text = page_path.read_text(encoding="utf-8-sig")

    page_styles = "\n".join(re.findall(r"<style>.*?</style>", text, re.DOTALL))
    scripts = "\n".join(re.findall(r"<script>.*?</script>", text, re.DOTALL))
    content = re.sub(r"<style>.*?</style>", "", text, flags=re.DOTALL)
    content = re.sub(r"<script>.*?</script>", "", content, flags=re.DOTALL)
    content = content.strip()

    h1_match = re.search(r"<h1>(.*?)</h1>", content, re.DOTALL)
    if h1_match:
        title = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
    else:
        title = PAGE_TITLE_FALLBACKS.get(page_path.name, page_path.stem)

    return page_styles, content, scripts, title


def parse_page_key(filename: str) -> Tuple[str, int, int, Optional[int]]:
    """解析文件名，返回 (paper_key, chapter, section, subsection)。

    命名格式：ch{章}_{节}_{短名}.html（如 ch2_1_bpe.html）
    兼容四级格式：ch{章}_{节}_{子节}_{短名}.html
    """
    m = re.match(r"^ch(\d+)_(\d+)(?:_(\d+))?_(.+)\.html$", filename)
    if m:
        chapter = int(m.group(1))
        section = int(m.group(2))
        subsection = int(m.group(3)) if m.group(3) else None
        return f"ch{chapter}", chapter, section, subsection

    print(
        f"[警告] 文件名 '{filename}' 不符合命名规范，将被排在导航末尾",
        file=sys.stderr,
    )
    return "unknown", 99, 99, 99


def parse_chapter_key(filename: str) -> Tuple[int, int, Optional[int]]:
    """从文件名解析章节号（兼容旧接口）。"""
    _, chapter, section, subsection = parse_page_key(filename)
    return chapter, section, subsection


def get_paper_key(filename: str) -> str:
    """从文件名解析论文标识（用于图片目录）。"""
    paper_key, _, _, _ = parse_page_key(filename)
    return paper_key


def get_group_key(filename: str) -> Optional[str]:
    """返回该文件所属的分组键，若为一节条目则返回 None。"""
    chapter, section, subsection = parse_chapter_key(filename)
    if subsection is not None:
        return f"ch{chapter}_{section}"
    return None


def build_nav_tree(pages_info: List[Tuple[Path, str, str]]) -> List[dict]:
    """构建导航树结构。"""
    items = []
    for path, title, filename in pages_info:
        chapter, section, subsection = parse_chapter_key(filename)
        group_key = get_group_key(filename)
        items.append(
            {
                "path": path,
                "title": title,
                "filename": filename,
                "chapter": chapter,
                "section": section,
                "subsection": subsection,
                "group_key": group_key,
                "sort_key": (chapter, section, subsection or 0, filename),
            }
        )

    items.sort(key=lambda x: x["sort_key"])

    groups: Dict[Optional[str], List[dict]] = {}
    for item in items:
        groups.setdefault(item["group_key"], []).append(item)

    tree_nodes = []

    if None in groups:
        for item in groups[None]:
            tree_nodes.append(
                (
                    item["sort_key"],
                    {
                        "type": "leaf",
                        "title": item["title"],
                        "href": item["filename"],
                    },
                )
            )

    for gk in groups:
        if gk is None:
            continue
        children = groups[gk]
        group_title = NAV_GROUPS.get(gk, gk)
        g_chapter, g_section, _ = parse_chapter_key(gk + "_dummy.html")
        group_sort_key = (g_chapter, g_section, 0, "")
        tree_nodes.append(
            (
                group_sort_key,
                {
                    "type": "group",
                    "title": group_title,
                    "children": [
                        {"type": "leaf", "title": c["title"], "href": c["filename"]}
                        for c in children
                    ],
                },
            )
        )

    tree_nodes.sort(key=lambda x: x[0])
    return [node for _, node in tree_nodes]


def get_chapter_title(nav_tree: List[dict], current_href: str) -> str:
    """根据当前页面动态推导顶层章节标题。"""
    current_chapter, _, _ = parse_chapter_key(current_href)
    if current_chapter != 99:
        return CHAPTER_TITLES.get(current_chapter, f"第 {current_chapter} 章")

    chapters = set()
    for node in nav_tree:
        if node["type"] == "leaf":
            chapter, _, _ = parse_chapter_key(node["href"])
            if chapter != 99:
                chapters.add(chapter)
        elif node["type"] == "group":
            for child in node["children"]:
                chapter, _, _ = parse_chapter_key(child["href"])
                if chapter != 99:
                    chapters.add(chapter)

    if not chapters:
        return "文档导航"

    min_chapter = min(chapters)
    return CHAPTER_TITLES.get(min_chapter, f"第 {min_chapter} 章")


def render_sidebar(nav_tree: List[dict], current_href: str) -> str:
    """渲染侧边栏 HTML。"""

    def render_leaf(node: dict, level: int = 0) -> str:
        cls = "nav-link"
        if node["href"] == current_href:
            cls += " active"
        return f'<a class="{cls}" href="{node["href"]}">{node["title"]}</a>'

    def render_group(node: dict, level: int = 0) -> str:
        has_active = any(
            child.get("href") == current_href for child in node["children"]
        )
        toggle_cls = "nav-toggle"
        children_cls = "nav-children"
        if not has_active:
            toggle_cls += " collapsed"
            children_cls += " collapsed"

        children_html = "\n".join(render_leaf(c, level + 1) for c in node["children"])

        return (
            f'<div class="nav-group">\n'
            f'  <div class="{toggle_cls}" onclick="toggleNav(this)">'
            f'<span class="arrow">▼</span><span>{node["title"]}</span></div>\n'
            f'  <div class="{children_cls}">\n    {children_html}\n  </div>\n'
            f'</div>'
        )

    def get_node_chapter(node: dict) -> int:
        if node["type"] == "leaf":
            chapter, _, _ = parse_chapter_key(node["href"])
            return chapter
        elif node["type"] == "group" and node.get("children"):
            chapter, _, _ = parse_chapter_key(node["children"][0]["href"])
            return chapter
        return 99

    chapter_nodes: Dict[int, List[dict]] = {}
    for node in nav_tree:
        ch = get_node_chapter(node)
        chapter_nodes.setdefault(ch, []).append(node)

    current_chapter, _, _ = parse_chapter_key(current_href)

    chapter_htmls = []
    for ch in sorted(chapter_nodes.keys()):
        nodes = chapter_nodes[ch]
        inner_html = "\n".join(
            render_group(n, 0) if n["type"] == "group" else render_leaf(n, 0)
            for n in nodes
        )

        chapter_title = CHAPTER_TITLES.get(ch, f"第 {ch} 章")
        is_current = ch == current_chapter

        toggle_cls = "nav-toggle"
        children_cls = "nav-children"
        if not is_current:
            toggle_cls += " collapsed"
            children_cls += " collapsed"

        chapter_htmls.append(
            f'<div class="nav-group">\n'
            f'  <div class="{toggle_cls}" onclick="toggleNav(this)">'
            f'<span class="arrow">▼</span><span>{chapter_title}</span></div>\n'
            f'  <div class="{children_cls}">\n    {inner_html}\n  </div>\n'
            f'</div>'
        )

    top_level_html = "\n".join(chapter_htmls)

    sidebar_html = (
        '<aside class="sidebar">\n'
        '  <div class="sidebar-header">'
        '<h1>LLM 机制学习手册</h1><p>从数据到部署 · 全链路拆解</p></div>\n'
        '  <nav class="nav-tree">\n'
        f'    {top_level_html}\n'
        '  </nav>\n'
        '</aside>'
    )
    return sidebar_html


def render_breadcrumb(nav_tree: List[dict], current_href: str) -> str:
    """渲染面包屑。"""
    current_title = current_href
    parent_title = None

    for node in nav_tree:
        if node["type"] == "leaf" and node["href"] == current_href:
            current_title = node["title"]
            break
        elif node["type"] == "group":
            for child in node["children"]:
                if child["href"] == current_href:
                    current_title = child["title"]
                    parent_title = node["title"]
                    break
            if parent_title:
                break

    chapter_title = get_chapter_title(nav_tree, current_href)

    parts = ['<a href="#">首页</a>']
    parts.append('<span class="breadcrumb-sep">/</span>')
    parts.append(f'<a href="#">{chapter_title}</a>')
    if parent_title:
        parts.append('<span class="breadcrumb-sep">/</span>')
        parts.append(f'<a href="#">{parent_title}</a>')
    parts.append('<span class="breadcrumb-sep">/</span>')
    parts.append(f'<span>{current_title}</span>')

    return '<div class="breadcrumb">\n    ' + "\n    ".join(parts) + "\n  </div>"


def render_footer_nav(pages_info: List[Tuple[Path, str, str]], current_href: str) -> str:
    """渲染页面底部上一篇/下一篇导航。"""
    items = sorted(
        [(parse_chapter_key(fn), fn, title) for _, title, fn in pages_info],
        key=lambda x: x[0],
    )
    hrefs = [fn for _, fn, _ in items]
    titles = {fn: title for _, fn, title in items}

    idx = hrefs.index(current_href) if current_href in hrefs else -1
    prev_link = ""
    next_link = ""

    if idx > 0:
        prev_href = hrefs[idx - 1]
        prev_link = f'<a class="prev" href="{prev_href}" title="{titles[prev_href]}">← 上一节：{titles[prev_href]}</a>'
    else:
        prev_link = '<span class="prev disabled">← 无上一节</span>'

    if 0 <= idx < len(hrefs) - 1:
        next_href = hrefs[idx + 1]
        next_link = f'<a class="next" href="{next_href}" title="{titles[next_href]}">下一节：{titles[next_href]} →</a>'
    else:
        next_link = '<span class="next disabled">无下一节 →</span>'

    return f'<div class="footer-nav">{prev_link}{next_link}</div>'


def collect_and_copy_images(content: str, paper_key: str) -> List[str]:
    """从内容中解析图片引用，自动从 ref_paper 复制到 images/{paper_key}/。

    返回实际可用的图片路径列表（不存在的源图片会打印警告）。
    """
    if not REF_PAPER_DIR.exists():
        print(f"[警告] 未找到 ref_paper 目录: {REF_PAPER_DIR}", file=sys.stderr)
        return []

    pattern = re.compile(r'src=["\'](images/[^/"\']+/[^"\']+)["\']')
    refs = pattern.findall(content)
    if not refs:
        return []

    target_dir = IMAGES_DIR / paper_key
    target_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for ref in refs:
        # ref 形如 images/ch2/xxx.jpg（按章分目录）
        parts = ref.split("/")
        if len(parts) != 3:
            continue
        img_name = parts[2]

        # 在所有 ref_paper/*/images/ 中查找同名图片
        src_candidates = list(REF_PAPER_DIR.glob(f"*/images/{img_name}"))
        if not src_candidates:
            print(f"[警告] 未找到图片源文件: {img_name}", file=sys.stderr)
            continue

        src = src_candidates[0]
        dst = target_dir / img_name
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(HTML_DIR)))

    return copied


def build() -> None:
    if not TEMPLATE_FILE.exists():
        print(f"[错误] 模板文件不存在: {TEMPLATE_FILE}", file=sys.stderr)
        sys.exit(1)

    if not PAGES_DIR.exists():
        PAGES_DIR.mkdir(parents=True)

    template = TEMPLATE_FILE.read_text(encoding="utf-8-sig")

    global_styles = ""
    if GLOBAL_CSS_FILE.exists():
        global_css = GLOBAL_CSS_FILE.read_text(encoding="utf-8-sig")
        global_styles = f"<style>\n{global_css}\n</style>"
    else:
        print(f"[警告] 全局 CSS 文件不存在: {GLOBAL_CSS_FILE}", file=sys.stderr)

    page_files = sorted(PAGES_DIR.glob("*.html"))
    if not page_files:
        print("[警告] pages/ 目录下未找到任何 HTML 文件", file=sys.stderr)
        sys.exit(0)

    pages_info = []
    for pf in page_files:
        _, _, _, title = parse_page(pf)
        pages_info.append((pf, title, pf.name))

    nav_tree = build_nav_tree(pages_info)

    built_count = 0
    for pf, title, filename in pages_info:
        page_styles, content, scripts, _ = parse_page(pf)
        paper_key = get_paper_key(filename)
        chapter, _, _ = parse_chapter_key(filename)

        copied_images = collect_and_copy_images(content, paper_key)
        if copied_images:
            print(f"  [复制] {len(copied_images)} 张配图到 images/{paper_key}/")

        sidebar = render_sidebar(nav_tree, filename)
        breadcrumb = render_breadcrumb(nav_tree, filename)
        footer_nav = render_footer_nav(pages_info, filename)

        page_html = (
            template.replace("{{TITLE}}", title)
            .replace("{{GLOBAL_STYLES}}", global_styles)
            .replace("{{PAGE_STYLES}}", page_styles)
            .replace("{{SIDEBAR}}", sidebar)
            .replace("{{BREADCRUMB}}", breadcrumb)
            .replace("{{CONTENT}}", content)
            .replace("{{FOOTER_NAV}}", footer_nav)
            .replace("{{BODY_SCRIPT}}", scripts)
        )

        out_path = HTML_DIR / filename
        out_path.write_text(page_html, encoding="utf-8-sig")
        built_count += 1
        print(f"  [生成] {filename}")

    print(f"\n完成！共生成 {built_count} 个页面。")


if __name__ == "__main__":
    build()
