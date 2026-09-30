"use strict";

/* ============ PhysicsRAG frontend ============ */

const $ = (sel) => document.querySelector(sel);

/* ---------- LaTeX rendering (KaTeX, vendored offline) ---------- */

function katexHtml(tex, displayMode) {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,   // render bad TeX in red instead of crashing
      strict: "ignore",
    });
  } catch (e) {
    return `<code>${esc(tex)}</code>`;
  }
}

/* Pull math out BEFORE markdown processing so $...$, \(...\), \[...\],
   $$...$$ survive escaping; re-inserted as rendered HTML at the end. */
function extractMath(src) {
  const math = [];
  const stash = (tex, displayMode) => {
    math.push(katexHtml(tex, displayMode));
    return `\x00M${math.length - 1}\x00`;
  };
  src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(tex, true));
  src = src.replace(/\\\[([\s\S]+?)\\\]/g, (_, tex) => stash(tex, true));
  src = src.replace(/\\\(([\s\S]+?)\\\)/g, (_, tex) => stash(tex, false));
  src = src.replace(/\$(?!\s)([^$\n]+?)(?<!\s)\$/g, (_, tex) => stash(tex, false));
  return { src, math };
}

/* ---------- tiny markdown renderer (no CDN, offline-safe) ---------- */

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(src) {
  const fences = [];
  src = src.replace(/```[a-zA-Z0-9]*\n?([\s\S]*?)```/g, (_, code) => {
    fences.push(`<pre class="code"><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\x00F${fences.length - 1}\x00`;
  });

  let math = [];
  ({ src, math } = extractMath(src));

  const inline = (t) => esc(t)
    .replace(/\[(\d{1,2})\]/g, '<sup class="cite">[$1]</sup>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\*([^*]+)\*/g, "<i>$1</i>");

  const blocks = src.split(/\n{2,}/).map((block) => {
    const t = block.trim();
    if (!t) return "";
    if (/^#{1,6}\s/.test(t)) {
      const level = Math.min(t.match(/^#+/)[0].length + 2, 6);
      const text = inline(t.replace(/^#+\s*/, ""));
      return `<h${level}>${text}</h${level}>`;
    }
    if (/^([-*]|\d+\.)\s/.test(t)) {
      const items = t.split("\n").filter(Boolean).map((l) =>
        `<li>${inline(l.replace(/^([-*]|\d+\.)\s*/, ""))}</li>`).join("");
      return `<ul>${items}</ul>`;
    }
    return `<p>${inline(t).replace(/\n/g, "<br>")}</p>`;
  });

  let html = blocks.join("\n");
  html = html.replace(/\x00F(\d+)\x00/g, (_, i) => fences[Number(i)]);
  html = html.replace(/\x00M(\d+)\x00/g, (_, i) => math[Number(i)] ?? "");
  return html;
}

/* ---------- api helper ---------- */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* no body */ }
  if (!res.ok) {
    const detail = body && body.detail ? body.detail : res.statusText;
    throw new Error(detail);
  }
  return body;
}

/* ---------- stats / provider chip ---------- */

async function loadHealth() {
  try {
    const h = await api("/api/health");
    $("#stat-papers").textContent = h.papers;
    $("#stat-chunks").textContent = h.chunks;
    $("#stat-topics").textContent = h.topics;
    $("#provider-chip").textContent = `${h.provider} · ${h.model}`;
  } catch (e) {
    $("#provider-chip").textContent = "api offline";
  }
}

/* ---------- view switching ---------- */

function showView(name) {
  document.querySelectorAll(".nav-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "papers") loadPapers();
  if (name === "research") loadTree();
}

/* ---------- chat ---------- */

function addUserMsg(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="msg-body">${esc(text)}</div>`;
  $("#chat").appendChild(el);
}

function addAssistantCard() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="msg-body"><span class="caret"></span></div><div class="sources"></div>`;
  $("#chat").appendChild(el);
  return el;
}

function renderSources(container, sources) {
  container.innerHTML = "";
  if (!sources || !sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "source-list";
  for (const s of sources) {
    const a = document.createElement("a");
    a.className = "source";
    a.href = s.arxiv_url;
    a.target = "_blank";
    a.rel = "noopener";
    a.title = s.section ? `${s.section} · page ${s.page}` : "";
    a.innerHTML = `<b>[${s.n}]</b> ${esc(s.title || s.paper_id)}` +
      (s.section ? ` <em>· ${esc(s.section)}${s.page ? ", p." + s.page : ""}</em>` : "");
    wrap.appendChild(a);
  }
  container.appendChild(wrap);
}

async function askQuestion(question) {
  addUserMsg(question);
  const card = addAssistantCard();
  const bodyEl = card.querySelector(".msg-body");
  const sourcesEl = card.querySelector(".sources");
  let text = "";

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = rawEvent.split("\n").find((l) => l.startsWith("data: "));
        if (!dataLine) continue;
        let event;
        try { event = JSON.parse(dataLine.slice(6)); } catch { continue; }

        if (event.type === "citations") {
          renderSources(sourcesEl, event.sources);
        } else if (event.type === "token") {
          text += event.text;
          bodyEl.innerHTML = renderMarkdown(text);
        } else if (event.type === "error") {
          bodyEl.insertAdjacentHTML("beforeend",
            `<div class="error-box">${esc(event.error)}</div>`);
        } else if (event.type === "done") {
          if (event.citations && event.citations.length) {
            renderSources(sourcesEl, event.citations);
          }
        }
      }
    }
    if (!text.trim() && !bodyEl.querySelector(".error-box")) {
      bodyEl.innerHTML = `<div class="error-box">Empty response from the provider.</div>`;
    }
  } catch (e) {
    bodyEl.innerHTML = `<div class="error-box">${esc(String(e.message || e))}</div>`;
  }
  const caret = bodyEl.querySelector(".caret");
  if (caret) caret.remove();
  $("#chat").scrollTop = $("#chat").scrollHeight;
}

/* ---------- papers ---------- */

async function loadPapers() {
  try {
    const papers = await api("/api/papers");
    window.__papers = papers;
    const subtitle = $("#papers-subtitle");
    if (subtitle) {
      subtitle.textContent =
        `${papers.length} papers downloaded and indexed by the pipeline.`;
    }
    renderPaperList();
  } catch (e) {
    $("#paper-list").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  }
}

function renderPaperList() {
  const filter = ($("#paper-filter").value || "").toLowerCase();
  const papers = (window.__papers || []).filter((p) =>
    !filter ||
    (p.title || "").toLowerCase().includes(filter) ||
    (p.paper_id || "").includes(filter) ||
    (p.authors || []).some((a) => a.toLowerCase().includes(filter)));
  const list = $("#paper-list");
  list.innerHTML = "";
  for (const p of papers) {
    const authors = p.authors || [];
    const authorStr = authors.length > 3
      ? authors.slice(0, 3).join(", ") + " et al."
      : authors.join(", ");
    const el = document.createElement("article");
    el.className = "paper-card";
    el.innerHTML = `
      <h3><a href="https://arxiv.org/abs/${esc(p.paper_id)}" target="_blank" rel="noopener">${esc(p.title)}</a></h3>
      <div class="paper-meta">
        <span class="mono">${esc(p.paper_id)}</span>
        <span>${esc(authorStr)}</span>
        <span>${esc(p.published || "")}</span>
        <span>${p.chunk_count} chunks</span>
      </div>
      <p class="paper-abstract">${esc((p.abstract || "").slice(0, 260))}${(p.abstract || "").length > 260 ? "…" : ""}</p>
      <div class="chips">${(p.categories || []).map((c) => `<span class="chip">${esc(c)}</span>`).join("")}</div>`;
    list.appendChild(el);
  }
  if (!papers.length) list.innerHTML = `<p class="muted">No papers match.</p>`;
}

/* ---------- research runs ---------- */

/* Experimental: graphical topic -> papers tree. A horizontal tidy-tree
   (root left, growth right): topic cards and paper leaves positioned by a
   leaf-allocation layout, connected by SVG bezier edges. Positions animate
   via rAF so expand/collapse glides and edges follow the nodes. */

const ptreeState = { collapsed: new Set() };
const ptreeDom = {
  nodes: new Map(),   // key -> element (persistent across renders)
  pos: new Map(),     // key -> {x, y} current animated position
  edges: new Map(),   // "a->b" -> path element
  scale: 1,
};

const PTREE = {
  COL: 270,           // horizontal distance between depth levels
  NODE_W: 224,        // topic card width
  PAPER_W: 208,       // paper card width
  TOPIC_H: 50,        // leaf-row height for a topic
  PAPER_H: 34,        // leaf-row height for a paper
  GAP: 10,            // vertical gap between sibling subtrees
};

function ptreeReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/* ---- layout: returns [{key, node, kind, depth, x, y, parentKey}] ---- */

function ptreeSubHeight(node) {
  const papers = node.papers || [];
  const kids = node.children || [];
  if (!papers.length && !kids.length) return PTREE.TOPIC_H;
  const collapsed = ptreeState.collapsed.has(node.name);
  if (collapsed || (!papers.length && !kids.length)) return PTREE.TOPIC_H;
  let total = 0;
  for (const paper of papers) total += PTREE.PAPER_H;
  for (const child of kids) total += ptreeSubHeight(child) + PTREE.GAP;
  return total - PTREE.GAP;
}

function ptreeLayout(nodes) {
  const out = [];
  let y = 8;
  const walk = (node, depth, parentKey) => {
    const papers = node.papers || [];
    const kids = node.children || [];
    const hasBody = (papers.length || kids.length) &&
      !ptreeState.collapsed.has(node.name);
    const x = 12 + depth * PTREE.COL + PTREE.NODE_W / 2;
    if (!hasBody) {
      out.push({ key: node.name, node, kind: "topic", depth, x, y: y + PTREE.TOPIC_H / 2, parentKey });
      y += PTREE.TOPIC_H + PTREE.GAP;
      return;
    }
    const startY = y;
    for (const paper of papers) {
      out.push({ key: `paper:${paper.paper_id}`, node: paper, kind: "paper", depth: depth + 1,
                 x: 12 + (depth + 1) * PTREE.COL + PTREE.NODE_W / 2, y: y + PTREE.PAPER_H / 2,
                 parentKey: node.name });
      y += PTREE.PAPER_H;
    }
    for (const child of kids) walk(child, depth + 1, node.name);
    const centerY = startY + (y - PTREE.GAP - startY) / 2;
    out.push({ key: node.name, node, kind: "topic", depth, x, y: centerY, parentKey });
  };
  for (const root of nodes) walk(root, 0, "");
  return out;
}

/* ---- rendering ---- */

function ptreeNodeEl(key, item) {
  let el = ptreeDom.nodes.get(key);
  if (el) return el;
  if (item.kind === "paper") {
    const paper = item.node;
    const failed = paper.status === "failed";
    el = document.createElement("a");
    el.className = "gnode paper" + (failed ? " failed" : "");
    el.href = `https://arxiv.org/abs/${encodeURIComponent(paper.paper_id)}`;
    el.target = "_blank";
    el.rel = "noopener";
    el.title = paper.title;
    el.innerHTML =
      `<span class="gnode-doc">${failed ? "&#9888;" : "&#128196;"}</span>` +
      `<span class="gnode-title">${esc(paper.title)}</span>` +
      `<span class="gnode-id mono">${esc(paper.paper_id)}</span>`;
  } else {
    const topic = item.node;
    const papers = topic.papers || [];
    const kids = topic.children || [];
    const hasBody = (papers.length || kids.length) > 0;
    el = document.createElement("div");
    el.className = "gnode topic" + (hasBody ? " has-body" : "");
    el.title = topic.name;
    el.innerHTML =
      `<div class="gnode-row">` +
      `<span class="ptree-toggle">${hasBody ? "&#9656;" : ""}</span>` +
      `<span class="gnode-name">${esc(topic.name)}</span>` +
      `</div>` +
      `<div class="gnode-meta">` +
      `<span class="chip neutral mono">c${topic.cycle}</span>` +
      (papers.length ? `<span class="chip">${papers.length} ${papers.length === 1 ? "paper" : "papers"}</span>` : "") +
      (kids.length ? `<span class="ptree-subcount">${kids.length} sub${kids.length === 1 ? "" : "s"}</span>` : "") +
      `</div>`;
    if (hasBody) {
      el.addEventListener("click", (ev) => {
        ev.preventDefault();
        if (ptreeState.collapsed.has(topic.name)) ptreeState.collapsed.delete(topic.name);
        else ptreeState.collapsed.add(topic.name);
        renderTopicTree(window.__treeNodes || []);
      });
    }
  }
  ptreeDom.nodes.set(key, el);
  return el;
}

function ptreeEdgeD(x1, y1, x2, y2) {
  const dx = Math.max(30, (x2 - x1) * 0.5);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function ptreeAnimate(items) {
  // items: [{key, el, x, y, w, h, parentKey, kind}]  (y = vertical center)
  const canvas = document.getElementById("topic-ptree");
  const svg = document.getElementById("tree-edges");
  const targets = new Map(items.map((it) => [it.key, it]));
  const duration = ptreeReducedMotion() ? 0 : 360;

  // Fade out nodes that disappeared from the target set.
  for (const [key, el] of [...ptreeDom.nodes]) {
    if (!targets.has(key)) {
      ptreeDom.nodes.delete(key);
      ptreeDom.pos.delete(key);
      el.classList.add("gnode-out");
      setTimeout(() => el.remove(), 280);
    }
  }
  // Drop stale edges.
  const wantedEdges = new Set(
    items.filter((it) => it.parentKey).map((it) => `${it.parentKey}->${it.key}`));
  for (const [key, pathEl] of [...ptreeDom.edges]) {
    if (!wantedEdges.has(key)) {
      ptreeDom.edges.delete(key);
      pathEl.remove();
    }
  }

  for (const it of items) {
    const el = ptreeNodeEl(it.key, it);
    if (el.parentNode !== canvas) canvas.appendChild(el);
    const from = ptreeDom.pos.get(it.key) ||
      { x: it.x, y: it.y, o: 0 };   // new nodes fade in at their target spot
    ptreeDom.pos.set(it.key, { ...from, kind: it.kind });
    if (it.kind === "topic" && it.node.children) {
      el.classList.toggle("open", !ptreeState.collapsed.has(it.node.name));
    }

    // Edge element (under the nodes).
    if (it.parentKey) {
      const ekey = `${it.parentKey}->${it.key}`;
      let pathEl = ptreeDom.edges.get(ekey);
      if (!pathEl) {
        pathEl = document.createElementNS("http://www.w3.org/2000/svg", "path");
        pathEl.setAttribute("class", it.kind === "paper" ? "gedge paper" : "gedge");
        svg.appendChild(pathEl);
        ptreeDom.edges.set(ekey, pathEl);
      }
    }
  }

  const t0 = performance.now();
  const frame = (now) => {
    const t = duration === 0 ? 1 : Math.min(1, (now - t0) / duration);
    const ease = 1 - Math.pow(1 - t, 3);   // cubic ease-out
    let maxY = 0, maxX = 0;
    for (const it of items) {
      const from = ptreeDom.pos.get(it.key) || { x: it.x, y: it.y };
      const cx = from.x + (it.x - from.x) * ease;
      const cy = from.y + (it.y - from.y) * ease;
      ptreeDom.pos.set(it.key, { x: cx, y: cy });
      const el = ptreeDom.nodes.get(it.key);
      if (!el) continue;
      const w = it.kind === "paper" ? PTREE.PAPER_W : PTREE.NODE_W;
      const h = it.kind === "paper" ? PTREE.PAPER_H : PTREE.TOPIC_H;
      el.style.transform = `translate(${cx - w / 2}px, ${cy - h / 2}px)`;
      el.style.width = w + "px";
      el.style.opacity = String(from.o + (1 - from.o) * ease);
      if (t >= 1) el.style.opacity = "";
      maxX = Math.max(maxX, cx + w / 2);
      maxY = Math.max(maxY, cy + h / 2);
    }
    for (const [ekey, pathEl] of ptreeDom.edges) {
      const [pk, ck] = ekey.split("->");
      const p = ptreeDom.pos.get(pk);
      const c = ptreeDom.pos.get(ck);
      if (!p || !c) { pathEl.remove(); ptreeDom.edges.delete(ekey); continue; }
      const cw = (c.kind === "paper" ? PTREE.PAPER_W : PTREE.NODE_W) / 2;
      pathEl.setAttribute("d", ptreeEdgeD(p.x + PTREE.NODE_W / 2, p.y, c.x - cw, c.y));
    }
    canvas.style.width = (maxX + PTREE.COL / 2) + "px";
    canvas.style.height = (maxY + 40) + "px";
    svg.setAttribute("width", canvas.style.width);
    svg.setAttribute("height", canvas.style.height);
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function renderTopicTree(nodes) {
  if (!nodes || !nodes.length) {
    ptreeDom.nodes.forEach((el) => el.remove());
    ptreeDom.nodes.clear(); ptreeDom.pos.clear();
    ptreeDom.edges.forEach((el) => el.remove()); ptreeDom.edges.clear();
    document.getElementById("tree-edges").setAttribute("width", "0");
    document.getElementById("tree-edges").setAttribute("height", "0");
    document.getElementById("topic-ptree").innerHTML =
      `<p class="muted ptree-empty">No topics yet &mdash; start a research run.</p>`;
    return;
  }
  const canvas = document.getElementById("topic-ptree");
  canvas.querySelectorAll(".ptree-empty").forEach((el) => el.remove());
  ptreeAnimate(ptreeLayout(nodes));
}

async function loadTree() {
  try {
    const data = await api("/api/topics/tree");
    $("#topic-tree").textContent = data.tree || "(no topics yet)";
    window.__treeNodes = data.nodes;
    renderTopicTree(data.nodes);
  } catch (e) {
    const box = $("#topic-ptree");
    if (box) box.innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  }
}

async function pollRun(runId) {
  try {
    const run = await api(`/api/research/runs/${runId}`);
    $("#run-dot").className = `run-dot ${run.status === "running" || run.status === "queued" ? "running" : run.status}`;
    $("#run-title").textContent = run.topic;
    $("#run-meta").textContent =
      `status: ${run.status} · cycle ${run.cycles_done}/${run.max_cycles}` +
      (run.error ? ` · ${run.error}` : "");
    const logEl = $("#run-log");
    const stick = logEl.scrollTop + logEl.clientHeight >= logEl.scrollHeight - 30;
    logEl.textContent = run.log || "";
    if (stick) logEl.scrollTop = logEl.scrollHeight;
    if (run.tree) $("#topic-tree").textContent = run.tree;
    if (run.status === "completed" || run.status === "failed") {
      loadHealth();
      loadTree();
      $("#run-btn").disabled = false;
      state.pollTimer = null;
      return;
    }
    state.pollTimer = setTimeout(() => pollRun(runId), 1500);
  } catch (e) {
    $("#run-meta").textContent = `poll failed: ${e.message}`;
    $("#run-btn").disabled = false;
  }
}

const state = { pollTimer: null, papers: [] };

/* ---------- settings ---------- */

async function loadSettings() {
  try {
    const s = await api("/api/settings");
    const select = $("#set-provider");
    select.innerHTML = s.providers
      .map((p) => `<option value="${p}" ${p === s.provider ? "selected" : ""}>${p}</option>`)
      .join("");
    $("#set-model").value = s.model || "";
    $("#set-base-url").value = s.base_url || "";
    $("#set-key-state").textContent = s.api_key_set ? "(a key is saved)" : "(no key saved)";
  } catch (e) {
    const note = $("#settings-note");
    if (note) note.textContent = `failed to load settings: ${e.message}`;
  }
}

/* ---------- wiring ---------- */

function init() {
  document.querySelectorAll(".nav-btn").forEach((btn) =>
    btn.addEventListener("click", () => showView(btn.dataset.view)));

  $("#ask-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const input = $("#ask-input");
    const question = input.value.trim();
    if (question.length < 3) return;
    input.value = "";
    askQuestion(question);
  });

  // Enter sends, Shift+Enter adds a newline (like every modern chat).
  $("#ask-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      $("#ask-form").requestSubmit();
    }
  });

  $("#paper-filter").addEventListener("input", renderPaperList);

  // Topics & papers graph: collapse/expand everything at once.
  $("#tree-toggle-all").addEventListener("click", (ev) => {
    const topics = [];
    (function collect(ns) {
      for (const n of ns) {
        if ((n.children || []).length || (n.papers || []).length) topics.push(n.name);
        collect(n.children || []);
      }
    })(window.__treeNodes || []);
    const anyOpen = topics.some((name) => !ptreeState.collapsed.has(name));
    ptreeState.collapsed.clear();
    if (anyOpen) topics.forEach((name) => ptreeState.collapsed.add(name));
    renderTopicTree(window.__treeNodes || []);
    ev.target.textContent = anyOpen ? "Expand all" : "Collapse all";
  });

  // Graph zoom (CSS zoom keeps layout + scroll bounds correct).
  const setZoom = (z) => {
    ptreeDom.scale = Math.min(1.5, Math.max(0.5, z));
    $("#tree-plane").style.zoom = ptreeDom.scale;
    $("#tree-zoom-reset").textContent = Math.round(ptreeDom.scale * 100) + "%";
  };
  $("#tree-zoom-in").addEventListener("click", () => setZoom(ptreeDom.scale + 0.15));
  $("#tree-zoom-out").addEventListener("click", () => setZoom(ptreeDom.scale - 0.15));
  $("#tree-zoom-reset").addEventListener("click", () => setZoom(1));

  $("#run-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const topic = $("#run-topic").value.trim();
    if (topic.length < 3) return;
    const btn = $("#run-btn");
    btn.disabled = true;
    $("#run-status").classList.remove("hidden");
    $("#run-log").textContent = "";
    $("#run-meta").textContent = "starting…";
    try {
      const run = await api("/api/research/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, max_cycles: Number($("#run-cycles").value) }),
      });
      $("#run-title").textContent = run.topic;
      pollRun(run.run_id);
    } catch (e) {
      $("#run-meta").textContent = e.message;
      btn.disabled = false;
    }
  });

  $("#settings-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const payload = {
      provider: $("#set-provider").value,
      model: $("#set-model").value.trim(),
      base_url: $("#set-base-url").value.trim(),
    };
    const key = $("#set-key").value.trim();
    if (key) payload.api_key = key;
    try {
      await api("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      $("#set-key").value = "";
      await loadSettings();
      await loadHealth();
      $("#set-key-state").textContent = "(saved)";
    } catch (e) {
      $("#set-key-state").textContent = `save failed: ${e.message}`;
    }
  });

  loadHealth();
  loadSettings();
  loadTree();
}

init();