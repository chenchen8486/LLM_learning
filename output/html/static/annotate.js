/* ============================================================
 * annotate.js — 页面标注工具（荧光马克笔）
 * 数据存浏览器 localStorage，键 llm-anno:<页面文件名>，不进仓库。
 * 锚点双保险：节点路径优先，原文+上下文兜底（页面重建后仍能找回）。
 * 依赖：无第三方库。由 template.html 以 defer 引入。
 * ============================================================ */
(function () {
  'use strict';

  var ARTICLE = document.querySelector('.content');
  if (!ARTICLE) return;

  var PAGE = (location.pathname.split('/').pop() || 'index.html');
  var KEY = 'llm-anno:' + PAGE;
  var STORE_PREFIX = 'llm-anno:';

  // 马克笔四色（半透明）
  var COLORS = {
    yellow: 'rgba(255, 230, 0, 0.35)',
    green:  'rgba(74, 222, 128, 0.32)',
    pink:   'rgba(244, 114, 182, 0.30)',
    blue:   'rgba(96, 165, 250, 0.32)',
    red:    'rgba(255, 0, 0, 0.36)'
  };
  var COLOR_NAMES = { yellow: '黄', green: '绿', pink: '粉', blue: '蓝', red: '红' };

  /* ---------- 存储 ---------- */

  function loadAnnos() {
    try {
      var raw = localStorage.getItem(KEY);
      var arr = raw ? JSON.parse(raw) : [];
      return Array.isArray(arr) ? arr : [];
    } catch (e) { return []; }
  }
  function saveAnnos() {
    try { localStorage.setItem(KEY, JSON.stringify(annos)); } catch (e) {}
  }
  var annos = loadAnnos();

  /* ---------- 锚点：节点路径 ---------- */

  // 文本节点 -> 相对 ARTICLE 的 childNodes 索引路径
  function nodePath(node) {
    var path = [];
    while (node && node !== ARTICLE) {
      var parent = node.parentNode;
      if (!parent) return null;
      var idx = Array.prototype.indexOf.call(parent.childNodes, node);
      path.unshift(idx);
      node = parent;
    }
    return node === ARTICLE ? path : null;
  }
  function pathToNode(path) {
    var node = ARTICLE;
    for (var i = 0; i < path.length; i++) {
      if (!node.childNodes || path[i] >= node.childNodes.length) return null;
      node = node.childNodes[path[i]];
    }
    return node;
  }

  /* ---------- 锚点：全文搜索兜底 ---------- */

  // 把全文字符偏移映射回（文本节点, 节点内偏移）
  function offsetToPoint(globalIdx) {
    var walker = document.createTreeWalker(ARTICLE, NodeFilter.SHOW_TEXT);
    var node, acc = 0;
    while ((node = walker.nextNode())) {
      if (acc + node.length >= globalIdx) {
        return { node: node, offset: globalIdx - acc };
      }
      acc += node.length;
    }
    return null;
  }
  function findByQuote(a) {
    var full = ARTICLE.textContent;
    var needle = a.prefix + a.text + a.suffix;
    var idx = full.indexOf(needle);
    if (idx >= 0) idx += a.prefix.length;
    else {
      idx = full.indexOf(a.text);
      if (idx < 0) return null;
    }
    var p1 = offsetToPoint(idx);
    var p2 = offsetToPoint(idx + a.text.length);
    if (!p1 || !p2) return null;
    var range = document.createRange();
    range.setStart(p1.node, p1.offset);
    range.setEnd(p2.node, p2.offset);
    return range;
  }

  /* ---------- 渲染：用 <mark> 包区间 ---------- */

  function inForbidden(node) {
    // 公式、SVG、脚本内不标
    var el = node.parentElement;
    while (el && el !== ARTICLE) {
      var cls = el.className || '';
      if (typeof cls === 'string' && cls.indexOf('katex') >= 0) return true;
      if (el.tagName === 'svg' || el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return true;
      el = el.parentElement;
    }
    return false;
  }
  function textNodesInRange(range) {
    var walker = document.createTreeWalker(ARTICLE, NodeFilter.SHOW_TEXT);
    var node, out = [];
    while ((node = walker.nextNode())) {
      if (!range.intersectsNode(node)) continue;
      if (inForbidden(node)) continue;
      if (!node.textContent.trim()) continue;
      out.push(node);
    }
    return out;
  }
  function wrapRange(range, id, color) {
    textNodesInRange(range).forEach(function (n) {
      var start = (n === range.startContainer) ? range.startOffset : 0;
      var end = (n === range.endContainer) ? range.endOffset : n.length;
      if (start >= end) return;
      var r = document.createRange();
      r.setStart(n, start);
      r.setEnd(n, end);
      var m = document.createElement('mark');
      m.className = 'anno';
      m.dataset.annoId = id;
      m.style.background = COLORS[color] || COLORS.yellow;
      try { r.surroundContents(m); } catch (e) {}
    });
  }
  function unwrap(id) {
    document.querySelectorAll('mark.anno[data-anno-id="' + id + '"]').forEach(function (m) {
      var p = m.parentNode;
      while (m.firstChild) p.insertBefore(m.firstChild, m);
      p.removeChild(m);
      if (p.normalize) p.normalize();
    });
  }
  function renderOne(a) {
    var range = null;
    if (a.sp && a.ep) {
      var sn = pathToNode(a.sp), en = pathToNode(a.ep);
      if (sn && en && sn.nodeType === 3 && en.nodeType === 3 &&
          a.so <= sn.length && a.eo <= en.length) {
        range = document.createRange();
        range.setStart(sn, a.so);
        range.setEnd(en, a.eo);
        // 路径命中但文本变了 -> 弃用，走文本兜底
        if (range.toString() !== a.text) range = null;
      }
    }
    if (!range) range = findByQuote(a);
    if (range) wrapRange(range, a.id, a.color);
    return !!range;
  }
  function renderAll() {
    var orphans = 0;
    annos.forEach(function (a) { if (!renderOne(a)) orphans++; });
    return orphans;
  }

  /* ---------- 创建 / 删除 ---------- */

  var uid = 1;
  function newId() { return 'a' + Date.now().toString(36) + (uid++); }

  function createAnno(range, color) {
    var text = range.toString();
    if (!text.trim()) return null;
    var a = {
      id: newId(),
      color: color,
      text: text,
      prefix: '',
      suffix: '',
      sp: nodePath(range.startContainer),
      ep: nodePath(range.endContainer),
      so: range.startOffset,
      eo: range.endOffset
    };
    // 上下文各取 20 字，用于跨版本兜底
    var full = ARTICLE.textContent;
    var at = full.indexOf(text);
    if (at >= 0) {
      a.prefix = full.slice(Math.max(0, at - 20), at);
      a.suffix = full.slice(at + text.length, at + text.length + 20);
    }
    wrapRange(range, a.id, a.color);
    annos.push(a);
    saveAnnos();
    return a;
  }
  function removeAnno(id) {
    unwrap(id);
    annos = annos.filter(function (a) { return a.id !== id; });
    saveAnnos();
  }
  function recolor(id, color) {
    annos.forEach(function (a) {
      if (a.id === id) a.color = color;
    });
    document.querySelectorAll('mark.anno[data-anno-id="' + id + '"]').forEach(function (m) {
      m.style.background = COLORS[color];
    });
    saveAnnos();
  }

  /* ---------- 悬浮工具条 ---------- */

  var bar = document.createElement('div');
  bar.id = 'anno-bar';
  bar.style.display = 'none';
  document.body.appendChild(bar);

  var barMode = 'create';   // 'create' | 'edit'
  var barTargetId = null;
  var savedRange = null;

  function dot(color) {
    var b = document.createElement('button');
    b.className = 'anno-dot';
    b.title = COLOR_NAMES[color];
    b.style.background = COLORS[color].replace(/[\d.]+\)$/, '0.85)');
    b.dataset.color = color;
    return b;
  }
  function buildBar() {
    bar.innerHTML = '';
    Object.keys(COLORS).forEach(function (c) { bar.appendChild(dot(c)); });
    if (barMode === 'edit') {
      var del = document.createElement('button');
      del.id = 'anno-del';
      del.textContent = '删除';
      bar.appendChild(del);
    }
  }
  function showBar(rect, mode, id) {
    barMode = mode;
    barTargetId = id || null;
    buildBar();
    bar.style.display = 'flex';
    var x = Math.max(8, rect.left + rect.width / 2 - bar.offsetWidth / 2);
    var y = rect.top - bar.offsetHeight - 8;
    if (y < 8) y = rect.bottom + 8;
    bar.style.left = x + 'px';
    bar.style.top = y + 'px';
  }
  function hideBar() { bar.style.display = 'none'; savedRange = null; barTargetId = null; }

  bar.addEventListener('mousedown', function (e) { e.preventDefault(); });
  bar.addEventListener('click', function (e) {
    var t = e.target;
    if (t.classList.contains('anno-dot')) {
      var color = t.dataset.color;
      if (barMode === 'create' && savedRange) {
        createAnno(savedRange, color);
        var sel = window.getSelection();
        if (sel) sel.removeAllRanges();
      } else if (barMode === 'edit' && barTargetId) {
        recolor(barTargetId, color);
      }
      hideBar();
    } else if (t.id === 'anno-del' && barTargetId) {
      removeAnno(barTargetId);
      hideBar();
    }
  });

  // 选中文字 -> 弹创建条
  document.addEventListener('mouseup', function (e) {
    if (bar.contains(e.target)) return;
    if (e.target.closest && e.target.closest('mark.anno')) return; // 点在已有标注上，交给 click 弹编辑条
    setTimeout(function () {
      var sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) { if (!bar.contains(e.target)) hideBar(); return; }
      var range = sel.getRangeAt(0);
      var box = range.commonAncestorContainer;
      var el = box.nodeType === 1 ? box : box.parentElement;
      if (!el || !ARTICLE.contains(el) || el.closest && el.closest('svg')) { hideBar(); return; }
      if (!range.toString().trim()) { hideBar(); return; }
      savedRange = range.cloneRange();
      showBar(range.getBoundingClientRect(), 'create');
    }, 0);
  });

  // 点已有标注 -> 弹编辑条（换色/删除）
  ARTICLE.addEventListener('click', function (e) {
    var m = e.target.closest ? e.target.closest('mark.anno') : null;
    if (!m) return;
    e.stopPropagation();
    showBar(m.getBoundingClientRect(), 'edit', m.dataset.annoId);
  });
  document.addEventListener('mousedown', function (e) {
    if (!bar.contains(e.target) && !(e.target.closest && e.target.closest('mark.anno'))) hideBar();
  });

  /* ---------- 右下角控制面板 ---------- */

  var panel = document.createElement('div');
  panel.id = 'anno-panel';
  var btnToggle = document.createElement('button');
  btnToggle.textContent = '隐藏标注';
  var btnExport = document.createElement('button');
  btnExport.textContent = '导出标注';
  var btnImport = document.createElement('button');
  btnImport.textContent = '导入标注';
  var btnClear = document.createElement('button');
  btnClear.textContent = '清空标注';
  var fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = '.json';
  fileInput.style.display = 'none';
  panel.appendChild(btnToggle);
  panel.appendChild(btnExport);
  panel.appendChild(btnImport);
  panel.appendChild(btnClear);
  document.body.appendChild(panel);
  document.body.appendChild(fileInput);

  btnToggle.addEventListener('click', function () {
    var hidden = document.body.classList.toggle('anno-hidden');
    btnToggle.textContent = hidden ? '显示标注' : '隐藏标注';
  });
  btnExport.addEventListener('click', function () {
    var dump = {};
    try {
      for (var i = 0; i < localStorage.length; i++) {
        var k = localStorage.key(i);
        if (k && k.indexOf(STORE_PREFIX) === 0) dump[k] = JSON.parse(localStorage.getItem(k));
      }
    } catch (e) {}
    var blob = new Blob([JSON.stringify(dump, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'llm-annotations.json';
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  });
  btnImport.addEventListener('click', function () { fileInput.click(); });
  fileInput.addEventListener('change', function () {
    var f = fileInput.files[0];
    if (!f) return;
    var rd = new FileReader();
    rd.onload = function () {
      try {
        var dump = JSON.parse(rd.result);
        var n = 0;
        Object.keys(dump).forEach(function (k) {
          if (k.indexOf(STORE_PREFIX) === 0 && Array.isArray(dump[k])) {
            localStorage.setItem(k, JSON.stringify(dump[k]));
            n++;
          }
        });
        alert('已导入 ' + n + ' 个页面的标注，页面将刷新。');
        location.reload();
      } catch (e) { alert('导入失败：文件格式不对。'); }
    };
    rd.readAsText(f);
    fileInput.value = '';
  });
  btnClear.addEventListener('click', function () {
    if (!confirm('确定清空当前页的所有标注？此操作不可撤销。')) return;
    try { localStorage.removeItem(KEY); } catch (e) {}
    if (confirm('要连同其他页面的标注一起清空吗？（选"取消"则只清当前页）')) {
      try {
        var ks = [];
        for (var i = 0; i < localStorage.length; i++) {
          var k = localStorage.key(i);
          if (k && k.indexOf(STORE_PREFIX) === 0) ks.push(k);
        }
        ks.forEach(function (k) { localStorage.removeItem(k); });
      } catch (e) {}
    }
    location.reload();
  });

  /* ---------- 样式 ---------- */

  var css = [
    'mark.anno{padding:0.05em 0;border-radius:2px;cursor:pointer;color:inherit;}',
    'body.anno-hidden mark.anno{background:transparent !important;}',
    'body.anno-hidden #anno-panel #anno-bar{display:none;}',
    '#anno-bar{position:fixed;z-index:9999;background:#fff;border:1px solid #e5e7eb;border-radius:10px;',
    '  box-shadow:0 4px 16px rgba(76,29,149,.18);padding:6px 8px;display:flex;gap:7px;align-items:center;}',
    '.anno-dot{width:18px;height:18px;border-radius:50%;border:1px solid rgba(0,0,0,.12);cursor:pointer;padding:0;transition:transform .12s;}',
    '.anno-dot:hover{transform:scale(1.2);}',
    '#anno-del{font-size:12px;color:#b91c1c;background:#fff;border:1px solid #fecaca;border-radius:6px;padding:2px 8px;cursor:pointer;}',
    '#anno-del:hover{background:#fef2f2;}',
    '#anno-panel{position:fixed;right:14px;bottom:14px;z-index:9990;display:flex;flex-direction:column;gap:6px;}',
    '#anno-panel button{font-size:12px;color:#4c1d95;background:rgba(255,255,255,.92);border:1px solid #ddd6fe;',
    '  border-radius:8px;padding:4px 10px;cursor:pointer;box-shadow:0 2px 8px rgba(76,29,149,.10);}',
    '#anno-panel button:hover{background:#f5f3fe;}'
  ].join('\n');
  var style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);

  /* ---------- 启动 ---------- */

  renderAll();

  // 调试与测试钩子
  window.__anno = {
    create: function (range, color) { return createAnno(range, color); },
    remove: removeAnno,
    list: function () { return annos.slice(); }
  };
})();
