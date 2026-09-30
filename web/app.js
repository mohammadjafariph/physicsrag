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
    $("#papers-subtitle").textContent =
      `${papers.length} papers downloaded and indexed by the pipeline.`;
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

async function loadTree() {
  try {
    const data = await api("/api/topics/tree");
    $("#topic-tree").textContent = data.tree || "(no topics yet)";
  } catch (e) {
    $("#topic-tree").textContent = `error: ${e.message}`;
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
    $("#settings-note").textContent = `failed to load settings: ${e.message}`;
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

  $("#paper-filter").addEventListener("input", renderPaperList);

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