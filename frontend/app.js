/* AI Dataset Cleaner - front-end (vanilla JS, no build step) */
(() => {
  "use strict";

  // ------------------------------------------------------------------ state
  const S = {
    datasetId: null, dataset: null, profile: null, pass: 1,
    jobId: null, poll: null, plan: null, actions: [], toolbox: [],
    result: null, settings: null, ollama: null, selectedColumn: null,
    sevFilter: { high: true, medium: true, low: true },
  };
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtNum = (n) => (n === null || n === undefined) ? "–" : (typeof n === "number" ? (Number.isInteger(n) ? n.toLocaleString() : n.toLocaleString(undefined, { maximumFractionDigits: 3 })) : esc(n));
  const pct = (n) => `${(Math.round(n * 10) / 10).toFixed(1)}%`;

  function toast(msg, isError = false) {
    const t = $("toast"); t.textContent = msg; t.classList.toggle("error", isError); t.classList.remove("hidden");
    clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), isError ? 6000 : 3200);
  }
  // ---------------- API base (same origin, config.js, or ?api= override) ----------------
  const API_BASE = (() => {
    try {
      const q = new URLSearchParams(location.search).get("api");
      if (q) { localStorage.setItem("api_base", q.replace(/\/+$/, "")); history.replaceState(null, "", location.pathname); }
      const saved = localStorage.getItem("api_base");
      if (saved) return saved;
    } catch (_) { /* ignore */ }
    return ((window.APP_CONFIG && window.APP_CONFIG.API_BASE) || "").replace(/\/+$/, "");
  })();
  // ---------------- anonymous session token (kept in this browser only) ----------------
  function sessionToken() {
    let t = null;
    try { t = localStorage.getItem("session_token"); } catch (_) { /* ignore */ }
    if (!t || !/^[A-Za-z0-9_-]{16,64}$/.test(t)) {
      t = (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "") : Math.random().toString(36).slice(2) + Date.now().toString(36)).slice(0, 48);
      try { localStorage.setItem("session_token", t); } catch (_) { /* ignore */ }
    }
    return t;
  }
  async function api(path, opts = {}) {
    const headers = Object.assign({ "X-Session-Token": sessionToken() }, opts.headers || {});
    let r;
    try { r = await fetch(API_BASE + path, { ...opts, headers }); }
    catch (e) { throw new Error(API_BASE ? `Cannot reach the backend at ${API_BASE} (is it awake / is the URL right?)` : "Cannot reach the backend"); }
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { const j = await r.json(); msg = j.detail || JSON.stringify(j); } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    return r.json();
  }
  const postJson = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  async function downloadFile(path) {
    const r = await fetch(API_BASE + path, { headers: { "X-Session-Token": sessionToken() } });
    if (!r.ok) { let msg = r.statusText; try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ } throw new Error(msg); }
    const cd = r.headers.get("Content-Disposition") || "";
    const m = /filename="?([^";]+)"?/.exec(cd);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = m ? m[1] : "download"; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ------------------------------------------------------------------ stepper
  function showStep(n) {
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    $("view" + n).classList.remove("hidden");
    document.querySelectorAll(".step").forEach((b) => {
      const k = +b.dataset.step;
      b.classList.toggle("active", k === n);
      b.classList.toggle("done", k < n);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  function enableStep(n) { const b = document.querySelector(`.step[data-step="${n}"]`); if (b) b.disabled = false; }
  document.querySelectorAll(".step").forEach((b) => b.addEventListener("click", () => !b.disabled && showStep(+b.dataset.step)));

  // ------------------------------------------------------------------ Ollama status + settings
  async function refreshOllama() {
    try {
      const st = await api("/api/llm/status");
      S.ollama = st;
      const pill = $("ollamaPill"), txt = $("ollamaText");
      pill.className = "pill";
      const name = st.provider === "ollama" ? "Ollama" : (st.provider || "AI");
      if (!st.reachable) { pill.classList.add("pill-bad"); txt.textContent = `${name}: ${st.error || "not reachable"}`; }
      else if (!st.model_available) { pill.classList.add("pill-warn"); txt.textContent = `${name} up · model ${st.model} not available`; }
      else { pill.classList.add("pill-ok"); txt.textContent = `${name} · ${st.model}`; }
      return st;
    } catch (e) { $("ollamaText").textContent = "backend offline"; $("ollamaPill").className = "pill pill-bad"; return null; }
  }
  function planOpts() {
    let o = {};
    try { o = JSON.parse(localStorage.getItem("plan_opts") || "{}"); } catch (_) { /* ignore */ }
    return o;
  }
  async function loadSettings() {
    S.settings = await api("/api/settings");
    const s = S.settings, hosted = !!s.hosted, po = planOpts();
    document.querySelectorAll(".local-only").forEach((el) => el.classList.toggle("hidden", hosted));
    $("hostedNote").classList.toggle("hidden", !hosted);
    $("setProvider").innerHTML = (s.providers || []).map((p) => `<option value="${p}" ${p === s.provider ? "selected" : ""}>${p}</option>`).join("");
    $("setUrl").value = s.base_url || ""; $("setModel").value = s.model || ""; $("setKey").value = ""; $("setKey").placeholder = s.api_key ? "(key is set - leave blank to keep)" : "paste API key";
    $("setTemp").value = s.temperature; $("setCtx").value = s.num_ctx; $("setTimeout").value = s.request_timeout;
    $("setCpc").value = po.columns_per_call ?? s.columns_per_call;
    $("setClean").checked = po.include_clean_columns ?? s.include_clean_columns; $("setSummary").checked = po.summarize_plan ?? s.summarize_plan;
    $("setFallback").checked = s.allow_rule_fallback;
    $("hostedProvider").textContent = `${s.provider} · ${s.model}`;
    const sel = $("setModelSelect"); sel.innerHTML = `<option value="">(pick an available model)</option>`;
    (S.ollama?.models || []).forEach((m) => { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); });
    $("keyRow").classList.toggle("hidden", hosted || s.provider === "ollama");
    $("ctxRow").classList.toggle("hidden", hosted || s.provider !== "ollama");
  }
  $("setModelSelect").addEventListener("change", (e) => { if (e.target.value) $("setModel").value = e.target.value; });
  $("setProvider").addEventListener("change", () => { const p = $("setProvider").value; $("keyRow").classList.toggle("hidden", p === "ollama"); $("ctxRow").classList.toggle("hidden", p !== "ollama"); $("setModel").value = ""; $("setUrl").value = ""; $("setModel").placeholder = "(provider default)"; $("setUrl").placeholder = "(provider default)"; });
  $("btnSettings").addEventListener("click", async () => { await refreshOllama(); await loadSettings(); $("settingsModal").classList.remove("hidden"); $("settingsStatus").textContent = ""; });
  $("btnCloseSettings").addEventListener("click", () => $("settingsModal").classList.add("hidden"));
  function savePlanOpts() {
    const po = { columns_per_call: +$("setCpc").value || 1, include_clean_columns: $("setClean").checked, summarize_plan: $("setSummary").checked };
    try { localStorage.setItem("plan_opts", JSON.stringify(po)); } catch (_) { /* ignore */ }
    return po;
  }
  $("btnSaveSettings").addEventListener("click", async () => {
    savePlanOpts();
    if (S.settings?.hosted) { $("settingsModal").classList.add("hidden"); toast("Planning options saved"); return; }
    try {
      const body = {
        provider: $("setProvider").value, base_url: $("setUrl").value.trim(), model: $("setModel").value.trim(), temperature: +$("setTemp").value,
        num_ctx: +$("setCtx").value, request_timeout: +$("setTimeout").value, columns_per_call: +$("setCpc").value,
        include_clean_columns: $("setClean").checked, summarize_plan: $("setSummary").checked, allow_rule_fallback: $("setFallback").checked,
      };
      if ($("setKey").value.trim()) body.api_key = $("setKey").value.trim();
      S.settings = await postJson("/api/settings", body);
      await refreshOllama(); $("settingsModal").classList.add("hidden"); toast("Settings saved");
    } catch (e) { $("settingsStatus").textContent = "Error: " + e.message; }
  });
  $("btnTestOllama").addEventListener("click", async () => {
    $("settingsStatus").textContent = "Testing…";
    try {
      if (!S.settings?.hosted) {
        const body = { provider: $("setProvider").value, base_url: $("setUrl").value.trim(), model: $("setModel").value.trim() };
        if ($("setKey").value.trim()) body.api_key = $("setKey").value.trim();
        await postJson("/api/settings", body);
      }
      const st = await refreshOllama(); await loadSettings();
      $("settingsStatus").textContent = st.reachable ? `Connected to ${st.provider}. ${st.models?.length ? "Models: " + st.models.slice(0, 12).join(", ") + (st.models.length > 12 ? "…" : "") + ". " : ""}${st.model_available ? "Selected model is available." : "Selected model is NOT available: " + (st.error || st.model)}` : "Not reachable: " + (st.error || "");
    } catch (e) { $("settingsStatus").textContent = "Error: " + e.message; }
  });

  // ------------------------------------------------------------------ STEP 1: upload
  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) uploadFile(f); });
  $("btnBrowse").addEventListener("click", () => $("fileInput").click());
  dz.addEventListener("click", (e) => { if (e.target === dz || e.target.classList.contains("dz-title") || e.target.classList.contains("dz-icon")) $("fileInput").click(); });
  $("fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) uploadFile(f); e.target.value = ""; });
  $("btnSample").addEventListener("click", async () => {
    setUploadStatus("Loading sample dataset…");
    try { onDatasetLoaded(await postJson("/api/sample", {})); } catch (e) { setUploadStatus("Error: " + e.message, true); }
  });

  function setUploadStatus(msg, isError = false) {
    const el = $("uploadStatus"); el.textContent = msg; el.classList.toggle("error", isError); el.classList.toggle("hidden", !msg);
  }
  async function uploadFile(file) {
    setUploadStatus(`Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)…`);
    const fd = new FormData(); fd.append("file", file);
    try {
      const data = await api("/api/upload", { method: "POST", body: fd });
      onDatasetLoaded(data);
    } catch (e) { setUploadStatus("Upload failed: " + e.message, true); }
  }
  function onDatasetLoaded(data) {
    S.datasetId = data.dataset_id; S.dataset = data; S.profile = null; S.plan = null; S.actions = []; S.result = null; S.pass = 1;
    setUploadStatus("");
    $("datasetCard").classList.remove("hidden");
    $("dsName").textContent = data.meta.file_name;
    const m = data.meta;
    $("dsMeta").textContent = `${data.shape.rows.toLocaleString()} rows × ${data.shape.columns} columns` +
      (m.encoding ? ` · encoding ${m.encoding}` : "") + (m.delimiter ? ` · delimiter "${m.delimiter === "\t" ? "\\t" : m.delimiter}"` : "") + (m.sheet !== undefined ? ` · sheet ${m.sheet}` : "");
    const sw = $("sheetWrap");
    if (m.sheets && m.sheets.length > 1) {
      sw.classList.remove("hidden"); const sel = $("sheetSelect"); sel.innerHTML = "";
      m.sheets.forEach((s) => { const o = document.createElement("option"); o.value = s; o.textContent = s; o.selected = s === m.sheet; sel.appendChild(o); });
    } else sw.classList.add("hidden");
    renderPreview($("previewTable"), data.preview);
    document.querySelectorAll(".step").forEach((b) => { if (+b.dataset.step > 1) b.disabled = true; });
    loadMyDatasets();
    showStep(1);
    $("datasetCard").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  $("sheetSelect").addEventListener("change", async (e) => {
    try { onDatasetLoaded(await postJson(`/api/datasets/${S.datasetId}/sheet`, { sheet: e.target.value })); } catch (err) { toast(err.message, true); }
  });
  function renderPreview(container, p) {
    const head = p.columns.map((c) => `<th title="${esc(p.dtypes?.[c] || "")}">${esc(c)}</th>`).join("");
    const rows = p.rows.map((r) => `<tr>${r.map((v) => v === null ? `<td class="null">null</td>` : `<td class="${typeof v === "number" ? "num" : ""}">${esc(v)}</td>`).join("")}</tr>`).join("");
    container.innerHTML = `<table class="tbl"><thead><tr>${head}</tr></thead><tbody>${rows || `<tr><td class="empty" colspan="${p.columns.length}">No rows</td></tr>`}</tbody></table>`;
  }

  // ------------------------------------------------------------------ STEP 2: analysis
  $("btnAnalyze").addEventListener("click", () => analyze(false));
  async function analyze(force) {
    const btn = $("btnAnalyze"); btn.disabled = true; btn.textContent = "Analyzing…";
    try {
      const data = await api(`/api/datasets/${S.datasetId}/analyze${force ? "?force=1" : ""}`, { method: "POST" });
      S.profile = data.profile; S.pass = data.pass;
      renderAnalysis(); enableStep(2); showStep(2);
    } catch (e) { toast("Analysis failed: " + e.message, true); }
    finally { btn.disabled = false; btn.textContent = "Analyze dataset →"; }
  }
  function scoreColor(s) { return s >= 90 ? "var(--low)" : s >= 75 ? "#4caf50" : s >= 60 ? "var(--medium)" : "var(--high)"; }
  function tile(k, v, s = "") { return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`; }
  function scoreTile(label, score, extra = "") {
    return `<div class="tile score"><div class="ring" style="--pct:${score.score};--ring-color:${scoreColor(score.score)}"><span>${score.grade}</span></div><div><div class="k">${label}</div><div class="v">${score.score}</div><div class="s">${extra || "out of 100"}</div></div></div>`;
  }
  function renderAnalysis() {
    const p = S.profile, d = p.dataset, ic = d.issue_counts || {};
    $("passBadge").textContent = S.pass > 1 ? `pass ${S.pass}` : "";
    $("statTiles").innerHTML = scoreTile("Quality score", p.score) +
      tile("Rows", d.rows.toLocaleString(), `${d.empty_rows || 0} empty · ${d.mostly_empty_rows || 0} mostly empty`) +
      tile("Columns", d.columns, `${(d.empty_columns || []).length} empty · ${(d.constant_columns || []).length} constant`) +
      tile("Missing cells", pct(d.missing_pct), `${d.missing_cells.toLocaleString()} incl. placeholders`) +
      tile("Duplicate rows", d.duplicate_rows ?? 0, `${d.duplicate_rows_normalized ?? 0} ignoring case/space`) +
      tile("Issues", p.issues.length, `<span class="badge high">${ic.high || 0}</span> <span class="badge medium">${ic.medium || 0}</span> <span class="badge low">${ic.low || 0}</span>`);
    renderIssues($("issueList"), p.issues, true); $("issueCount").textContent = p.issues.length;
    renderColumnTable(); $("colDetail").classList.add("hidden"); S.selectedColumn = null;
  }
  document.querySelectorAll("#sevFilters .chip").forEach((c) => c.addEventListener("click", () => {
    S.sevFilter[c.dataset.sev] = !S.sevFilter[c.dataset.sev]; c.classList.toggle("active", S.sevFilter[c.dataset.sev]);
    renderIssues($("issueList"), S.profile.issues, true);
  }));
  function renderIssues(container, issues, filter) {
    const list = issues.filter((i) => !filter || S.sevFilter[i.severity]).filter((i) => !S.selectedColumn || i.column === S.selectedColumn || (filter && false));
    if (!list.length) { container.innerHTML = `<div class="empty">No issues ${S.selectedColumn ? "for this column" : "to show"} 🎉</div>`; return; }
    container.innerHTML = list.map((i) => `
      <div class="issue ${i.severity}">
        <div class="head"><span class="badge ${i.severity}">${i.severity}</span>
          <span class="badge col">${esc(i.column || "dataset")}</span><span class="title">${esc(i.title)}</span></div>
        <div class="detail">${esc(i.detail)}</div>
        ${(i.suggestions?.length || Object.keys(i.evidence || {}).length) ? `<details><summary>Evidence & suggested fix</summary>
          ${i.suggestions?.length ? `<div class="chips">${i.suggestions.map((s, k) => `<span class="c" title="${esc(s.reason || "")}">${k ? "or " : ""}${esc(s.action)} ${esc(shortParams(s.params))}</span>`).join("")}</div>` : ""}
          ${Object.keys(i.evidence || {}).length ? `<pre class="evidence">${esc(JSON.stringify(i.evidence, null, 1))}</pre>` : ""}</details>` : ""}
      </div>`).join("");
  }
  function shortParams(p) {
    if (!p || !Object.keys(p).length) return "";
    const q = { ...p }; if (q.mapping) q.mapping = `${q.mapping.length} pairs`;
    if (q.tokens && q.tokens.length > 4) q.tokens = q.tokens.slice(0, 4).concat(["…"]);
    const s = JSON.stringify(q); return s.length > 90 ? s.slice(0, 88) + "…" : s;
  }
  function renderColumnTable() {
    const cols = S.profile.columns;
    $("colCount").textContent = `${cols.length} columns · click a row for details`;
    $("colTable").innerHTML = `<thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Issues</th></tr></thead><tbody>` +
      cols.map((c) => {
        const n = (c.issue_ids || []).length;
        const sev = S.profile.issues.filter((i) => i.column === c.name).map((i) => i.severity);
        const worst = sev.includes("high") ? "high" : sev.includes("medium") ? "medium" : sev.length ? "low" : "";
        return `<tr class="clickable ${S.selectedColumn === c.name ? "selected" : ""}" data-col="${esc(c.name)}"><td class="mono">${esc(c.name)}</td><td><span class="badge type">${esc(c.inferred_type)}</span></td><td class="num">${pct(c.missing_pct)}</td><td class="num">${c.unique}</td><td>${n ? `<span class="badge ${worst}">${n}</span>` : `<span class="muted">0</span>`}</td></tr>`;
      }).join("") + "</tbody>";
    $("colTable").querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => selectColumn(tr.dataset.col)));
  }
  function selectColumn(name) {
    S.selectedColumn = S.selectedColumn === name ? null : name;
    renderColumnTable();
    const box = $("colDetail");
    if (!S.selectedColumn) { box.classList.add("hidden"); renderIssues($("issueList"), S.profile.issues, true); return; }
    const c = S.profile.columns.find((x) => x.name === name);
    const kv = (k, v) => `<div><span>${k}</span>${v}</div>`;
    let html = `<div class="row gap" style="justify-content:space-between"><h3 style="margin:0;font-family:var(--mono)">${esc(c.name)}</h3><button class="btn btn-ghost" id="btnClearCol">✕ show all issues</button></div><div class="kv" style="margin-top:8px">` +
      kv("type", `<span class="badge type">${esc(c.inferred_type)}</span> <span class="muted">${esc(c.pandas_dtype)}</span>`) +
      kv("missing", `${c.missing} (${pct(c.missing_pct)})`) + kv("placeholders", c.missing_disguised) + kv("unique", `${c.unique} (${pct(c.unique_pct)})`);
    if (c.numeric) { const n = c.numeric; html += kv("min / max", `${fmtNum(n.min)} / ${fmtNum(n.max)}`) + kv("mean / median", `${fmtNum(n.mean)} / ${fmtNum(n.median)}`) + kv("std / skew", `${fmtNum(n.std)} / ${fmtNum(n.skew)}`) + kv("outliers (IQR)", `${n.outliers_iqr} (${n.outliers_extreme} extreme)`) + kv("negatives / zeros", `${n.negatives} / ${n.zeros}`); }
    if (c.datetime) { const dt = c.datetime; html += kv("date range", `${esc(dt.min || "?")} → ${esc(dt.max || "?")}`) + kv("formats", esc(JSON.stringify(dt.formats || {}))) + kv("future / <1900", `${dt.future_count || 0} / ${dt.before_1900_count || 0}`); }
    if (c.text) { const t = c.text; html += kv("whitespace issues", (t.leading_trailing_whitespace || 0) + (t.internal_double_spaces || 0)) + kv("mojibake", t.mojibake || 0) + kv("length min/avg/max", `${t.min_len}/${t.avg_len}/${t.max_len}`); }
    html += `</div>`;
    if (c.top_values?.length) html += `<div class="muted" style="margin-top:10px">Top values</div><div class="chips">${c.top_values.map(([v, n]) => `<span class="c">${v === null ? "<i>null</i>" : esc(v)}<b>×${n}</b></span>`).join("")}</div>`;
    if (c.disguised_tokens && Object.keys(c.disguised_tokens).length) html += `<div class="muted" style="margin-top:10px">Placeholder values</div><div class="chips">${Object.entries(c.disguised_tokens).map(([v, n]) => `<span class="c">${esc(JSON.stringify(v))}<b>×${n}</b></span>`).join("")}</div>`;
    const groups = [...(c.text?.case_variant_groups || []), ...(c.text?.fuzzy_groups || [])];
    if (groups.length) html += `<div class="muted" style="margin-top:10px">Variant groups (canonical ← variants)</div><div class="chips">${groups.slice(0, 12).map((g) => `<span class="c" title="${esc(g.kind)}">${esc(g.canonical)} ← ${g.variants.filter((v) => v !== g.canonical).map((v) => esc(JSON.stringify(v))).join(", ")}</span>`).join("")}</div>`;
    box.innerHTML = html; box.classList.remove("hidden");
    $("btnClearCol").addEventListener("click", () => selectColumn(name));
    renderIssues($("issueList"), S.profile.issues, true);
  }

  // ------------------------------------------------------------------ STEP 3: planning
  $("btnAiPlan").addEventListener("click", startAiPlan);
  $("btnRePlan").addEventListener("click", startAiPlan);
  $("btnRulesPlan").addEventListener("click", async () => {
    try { const d = await postJson(`/api/datasets/${S.datasetId}/plan/rules`, {}); setPlan(d.plan); enableStep(3); showStep(3); }
    catch (e) { toast(e.message, true); }
  });
  async function startAiPlan() {
    const st = await refreshOllama();
    if (st && (!st.reachable || !st.model_available) && !(S.settings?.allow_rule_fallback ?? true)) {
      toast(st.reachable ? `Model ${st.model} is not available at ${st.provider}` : `${st.provider || "AI provider"} is not reachable: ${st.error || ""}`, true); return;
    }
    try {
      const d = await postJson(`/api/datasets/${S.datasetId}/plan`, planOpts());
      S.jobId = d.job_id; S.plan = null; S.actions = [];
      $("planBody").classList.add("hidden"); $("planProgress").classList.remove("hidden"); $("btnApply").disabled = true;
      $("planLog").textContent = ""; $("progressBar").style.width = "0%"; $("progressTitle").textContent = "Starting…";
      enableStep(3); showStep(3);
      clearInterval(S.poll); S.poll = setInterval(pollJob, 1200); pollJob();
    } catch (e) { toast(e.message, true); }
  }
  $("btnCancelPlan").addEventListener("click", async () => { if (S.jobId) { await postJson(`/api/jobs/${S.jobId}/cancel`, {}); toast("Cancelling after the current model call…"); } });
  async function pollJob() {
    if (!S.jobId) return;
    let j; try { j = await api(`/api/jobs/${S.jobId}`); } catch (e) { return; }
    const pr = j.progress || {};
    $("progressElapsed").textContent = `${Math.round(j.elapsed)}s`;
    const frac = pr.total ? pr.done / pr.total : 0; $("progressBar").style.width = `${Math.max(3, frac * 100)}%`;
    const stage = pr.stage === "dataset" ? "Reasoning about the dataset structure" : pr.stage === "column" ? `Thinking about column "${pr.current}"` : pr.stage === "summary" ? "Writing the summary" : j.status === "done" ? "Done" : "Starting";
    $("progressTitle").textContent = stage; $("progressSub").textContent = pr.total ? `${pr.done} of ${pr.total} steps · ${(j.partial_actions || []).length} actions proposed so far` : "";
    const log = $("planLog"); log.innerHTML = (j.log || []).map((l) => `<span class="t">${l.t.toFixed(1).padStart(6)}s</span> <span class="${l.level}">${esc(l.msg)}</span>`).join("\n"); log.scrollTop = log.scrollHeight;
    if (j.status === "done") { clearInterval(S.poll); S.poll = null; $("planProgress").classList.add("hidden"); setPlan(j.plan); }
    else if (j.status === "error") { clearInterval(S.poll); S.poll = null; $("progressTitle").textContent = "Failed"; $("progressSub").textContent = j.error || ""; document.querySelector("#planProgress .spinner").style.display = "none"; toast("Planning failed: " + j.error, true); }
  }
  function setPlan(plan) {
    S.plan = plan; S.actions = (plan.actions || []).map((a) => ({ ...a, params: a.params || {}, enabled: a.enabled !== false }));
    $("planBody").classList.remove("hidden"); $("planProgress").classList.add("hidden");
    $("planMeta").textContent = `${plan.source && plan.source !== "rules" && plan.source !== "user" ? plan.source + " · " + plan.model : "rule-based planner"} · ${S.actions.length} actions`;
    $("planSummary").textContent = plan.summary || (plan.source === "rules" ? "This plan was produced by the rule engine (no AI). One default action per detected issue." : "No summary.");
    $("datasetThinking").textContent = plan.dataset_thinking || "(none)"; $("datasetThinkingWrap").classList.toggle("hidden", !plan.dataset_thinking);
    const w = $("planWarnings"); w.classList.toggle("hidden", !(plan.warnings || []).length); w.innerHTML = (plan.warnings || []).map((x) => `<div>⚠ ${esc(x)}</div>`).join("");
    renderPlan();
  }
  function groupKey(a) { return a.scope === "column" ? a.column : "__dataset__"; }
  function renderPlan() {
    const groups = new Map();
    S.actions.forEach((a) => { const k = groupKey(a); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(a); });
    const order = ["__dataset__", ...(S.profile?.columns || []).map((c) => c.name)];
    const keys = [...groups.keys()].sort((x, y) => (order.indexOf(x) === -1 ? 999 : order.indexOf(x)) - (order.indexOf(y) === -1 ? 999 : order.indexOf(y)));
    const thoughts = S.plan?.column_thoughts || {};
    $("planGroups").innerHTML = keys.map((k) => {
      const list = groups.get(k);
      const title = k === "__dataset__" ? "🗂 Whole dataset" : k;
      const think = k !== "__dataset__" && thoughts[k] ? `<div class="think">🧠 ${esc(thoughts[k])}</div>` : "";
      return `<div class="group"><div class="group-head"><h3>${esc(title)}</h3><span class="count">${list.length}</span>${think}</div>${list.map(renderAction).join("")}</div>`;
    }).join("") || `<div class="card empty">The plan is empty - the data looks clean, or add a manual action.</div>`;
    $("planGroups").querySelectorAll("[data-toggle]").forEach((cb) => cb.addEventListener("change", (e) => { const a = S.actions.find((x) => x.id === cb.dataset.toggle); a.enabled = e.target.checked; cb.closest(".action").classList.toggle("off", !a.enabled); updateSelectedCount(); }));
    $("planGroups").querySelectorAll("[data-edit]").forEach((b) => b.addEventListener("click", () => toggleEdit(b.dataset.edit)));
    $("planGroups").querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => { S.actions = S.actions.filter((x) => x.id !== b.dataset.remove); renderPlan(); }));
    $("planGroups").querySelectorAll("[data-params]").forEach((ta) => ta.addEventListener("change", () => {
      const a = S.actions.find((x) => x.id === ta.dataset.params);
      try { a.params = JSON.parse(ta.value || "{}"); ta.style.borderColor = ""; a.source = a.source === "ai" ? "ai" : a.source; }
      catch (e) { ta.style.borderColor = "var(--high)"; toast("Invalid JSON in parameters", true); }
    }));
    updateSelectedCount();
  }
  function renderAction(a) {
    const spec = S.toolbox.find((t) => t.name === a.action) || {};
    const risk = spec.risk === "high" ? `<span class="badge risk-high" title="deletes rows or columns">deletes data</span>` : spec.risk === "medium" ? `<span class="badge risk-medium">modifies values</span>` : "";
    const conf = a.confidence !== null && a.confidence !== undefined ? `<div class="conf" title="confidence ${Math.round(a.confidence * 100)}%"><div style="width:${Math.round(a.confidence * 100)}%"></div></div>` : "";
    const params = Object.keys(a.params || {}).length ? `<code title="${esc(JSON.stringify(a.params))}">${esc(shortParams(a.params))}</code>` : `<code class="muted">no parameters</code>`;
    const target = a.scope === "column" ? "" : (a.params?.columns ? `<span class="badge col">${esc(a.params.columns.join(", "))}</span>` : a.params?.subset ? `<span class="badge col">by ${esc(a.params.subset.join(", "))}</span>` : "");
    return `<div class="action ${a.enabled ? "" : "off"}" id="act-${esc(a.id)}">
      <input type="checkbox" data-toggle="${esc(a.id)}" ${a.enabled ? "checked" : ""}>
      <div>
        <div class="row gap wrap"><span class="name">${esc(a.action)}</span>${target}<span class="badge ${a.source}">${a.source === "ai" ? "AI" : a.source}</span>${risk}</div>
        <div class="reason">${esc(a.reason || spec.description || "")}</div>
        <div class="params">${params}<div class="tools"><button data-edit="${esc(a.id)}">edit params</button><button data-remove="${esc(a.id)}">remove</button></div></div>
        <textarea data-params="${esc(a.id)}" class="hidden">${esc(JSON.stringify(a.params || {}, null, 1))}</textarea>
        ${spec.params && Object.keys(spec.params).length ? `<small class="hidden param-help">${esc(Object.entries(spec.params).map(([k, v]) => `${k}: ${v}`).join(" · "))}</small>` : ""}
      </div>
      <div class="meta">${conf}</div>
    </div>`;
  }
  function toggleEdit(id) {
    const box = document.getElementById("act-" + id); const ta = box.querySelector("textarea"); const help = box.querySelector(".param-help");
    ta.classList.toggle("hidden"); if (help) help.classList.toggle("hidden", ta.classList.contains("hidden"));
    if (!ta.classList.contains("hidden")) ta.focus();
    else { const a = S.actions.find((x) => x.id === id); box.querySelector(".params code").textContent = shortParams(a.params) || "no parameters"; }
  }
  function updateSelectedCount() {
    const n = S.actions.filter((a) => a.enabled).length;
    $("selectedCount").textContent = `${n} of ${S.actions.length} actions selected`;
    $("btnApply").disabled = n === 0; $("btnApply").textContent = `Apply ${n} action${n === 1 ? "" : "s"} →`;
  }
  $("btnSelectAll").addEventListener("click", () => { S.actions.forEach((a) => (a.enabled = true)); renderPlan(); });
  $("btnSelectNone").addEventListener("click", () => { S.actions.forEach((a) => (a.enabled = false)); renderPlan(); });
  $("btnDeselectRisky").addEventListener("click", () => { S.actions.forEach((a) => { if ((S.toolbox.find((t) => t.name === a.action) || {}).risk === "high") a.enabled = false; }); renderPlan(); });

  // manual action modal
  $("btnAddAction").addEventListener("click", () => {
    const sel = $("addAction"); sel.innerHTML = S.toolbox.map((t) => `<option value="${esc(t.name)}">${esc(t.name)} (${t.scope})</option>`).join("");
    const cols = $("addColumn"); cols.innerHTML = `<option value="">(dataset level)</option>` + (S.profile?.columns || []).map((c) => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
    updateAddHelp(); $("addParams").value = "{}"; $("addReason").value = ""; $("addStatus").textContent = ""; $("addModal").classList.remove("hidden");
  });
  function updateAddHelp() { const t = S.toolbox.find((x) => x.name === $("addAction").value) || {}; $("addActionDesc").textContent = t.description || ""; $("addParamsHelp").textContent = Object.entries(t.params || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "no parameters"; }
  $("addAction").addEventListener("change", updateAddHelp);
  $("btnCloseAdd").addEventListener("click", () => $("addModal").classList.add("hidden"));
  $("btnConfirmAdd").addEventListener("click", () => {
    let params; try { params = JSON.parse($("addParams").value || "{}"); } catch (e) { $("addStatus").textContent = "Parameters must be valid JSON"; return; }
    const t = S.toolbox.find((x) => x.name === $("addAction").value);
    const col = $("addColumn").value || null;
    if (t.scope === "column" && !col) { $("addStatus").textContent = "This action needs a column"; return; }
    if (t.name === "drop_columns" && !params.columns && col) params.columns = [col];
    if (t.name === "drop_rows_with_missing" && !params.columns && col) params.columns = [col];
    const id = "user_" + Math.random().toString(36).slice(2, 8);
    S.actions.push({ id, action: t.name, column: t.scope === "column" ? col : null, params, reason: $("addReason").value || "Added manually", confidence: null, source: "user", enabled: true, scope: t.scope, risk: t.risk, priority: t.priority });
    if (!S.plan) S.plan = { source: "user", summary: "", dataset_thinking: "", column_thoughts: {}, warnings: [] };
    $("addModal").classList.add("hidden"); $("planBody").classList.remove("hidden"); renderPlan(); toast("Action added");
  });

  // ------------------------------------------------------------------ STEP 4: apply
  $("btnApply").addEventListener("click", applyPlan);
  async function applyPlan() {
    const selected = S.actions.filter((a) => a.enabled);
    if (!selected.length) return;
    const btn = $("btnApply"); btn.disabled = true; btn.textContent = "Applying…";
    try {
      const res = await postJson(`/api/datasets/${S.datasetId}/apply`, { actions: selected, plan: { ...S.plan, actions: undefined } });
      S.result = res; S.profile = res.profile_after;
      renderResults(res); enableStep(4); showStep(4);
    } catch (e) { toast("Apply failed: " + e.message, true); }
    finally { updateSelectedCount(); }
  }
  function renderResults(res) {
    const b = res.profile_before, a = res.profile_after;
    const delta = Math.round((a.score.score - b.score.score) * 10) / 10;
    $("resultTiles").innerHTML = scoreTile("Score before", b.score) + scoreTile("Score after", a.score, `<span class="${delta >= 0 ? "delta-up" : "delta-down"}">${delta >= 0 ? "+" : ""}${delta}</span> points`) +
      tile("Rows", `${res.shape.rows.toLocaleString()}`, `was ${b.dataset.rows.toLocaleString()}`) + tile("Columns", res.shape.columns, `was ${b.dataset.columns}`) +
      tile("Missing cells", pct(a.dataset.missing_pct), `was ${pct(b.dataset.missing_pct)}`) + tile("Issues left", a.issues.length, `was ${b.issue_count}`);
    const base = `/api/datasets/${S.datasetId}`;
    const dl = (id, path) => { const el = $(id); el.href = "#"; el.onclick = (e) => { e.preventDefault(); downloadFile(path).catch((err) => toast(err.message, true)); }; };
    dl("dlCsv", `${base}/download?fmt=csv`); dl("dlXlsx", `${base}/download?fmt=xlsx`); dl("dlJson", `${base}/download?fmt=json`);
    dl("dlReport", `${base}/report?fmt=md`); dl("dlReportJson", `${base}/report?fmt=json`);
    const log = res.log || [];
    $("logCount").textContent = `${log.filter((l) => l.status === "applied").length} applied · ${log.filter((l) => l.status !== "applied").length} skipped`;
    $("logTable").innerHTML = `<thead><tr><th>#</th><th>Action</th><th>Column</th><th>Status</th><th>Result</th><th>Changed</th><th>Examples</th></tr></thead><tbody>` + log.map((e, i) => {
      const changed = e.rows_after !== undefined && e.rows_after !== e.rows_before ? `${e.rows_before.toLocaleString()} → ${e.rows_after.toLocaleString()} rows` : e.cols_after !== undefined && e.cols_after !== e.cols_before ? `${e.cols_before} → ${e.cols_after} cols` : e.cells_changed !== undefined ? `${e.cells_changed} cells${e.nulls_before !== e.nulls_after ? ` · nulls ${e.nulls_before}→${e.nulls_after}` : ""}${e.dtype_before !== e.dtype_after ? ` · ${esc(e.dtype_before)}→${esc(e.dtype_after)}` : ""}` : "";
      const ex = (e.examples || []).slice(0, 3).map(([x, y]) => `<span>${x === null ? "<i>null</i>" : esc(x)} → ${y === null ? "<i>null</i>" : esc(y)}</span>`).join("<br>");
      return `<tr><td>${i + 1}</td><td class="mono">${esc(e.action)}</td><td class="mono">${esc(e.column || (e.params?.columns || []).join(", ") || "–")}</td><td class="status-${e.status}">${e.status}</td><td style="white-space:normal;max-width:380px">${esc(e.message)}</td><td style="white-space:normal">${changed}</td><td class="examples">${ex}</td></tr>`;
    }).join("") + "</tbody>";
    $("remainCount").textContent = a.issues.length; S.selectedColumn = null; S.sevFilter = { high: true, medium: true, low: true };
    renderIssues($("remainList"), a.issues, false);
    $("cleanShape").textContent = `${res.shape.rows.toLocaleString()} rows × ${res.shape.columns} columns (first 25)`;
    renderPreview($("cleanPreview"), res.preview);
  }
  $("btnAnotherPass").addEventListener("click", async () => {
    try {
      const data = await api(`/api/datasets/${S.datasetId}/analyze`, { method: "POST" });
      S.profile = data.profile; S.pass = data.pass; S.plan = null; S.actions = [];
      document.querySelectorAll("#sevFilters .chip").forEach((c) => c.classList.add("active")); S.sevFilter = { high: true, medium: true, low: true };
      renderAnalysis(); showStep(2); toast(`Pass ${S.pass}: analysing the cleaned data`);
    } catch (e) { toast(e.message, true); }
  });
  $("btnReset").addEventListener("click", async () => {
    if (!confirm("Discard all cleaning and go back to the original upload?")) return;
    try { onDatasetLoaded(await postJson(`/api/datasets/${S.datasetId}/reset`, {})); toast("Reset to original"); } catch (e) { toast(e.message, true); }
  });

  // ------------------------------------------------------------------ previous datasets (this browser session)
  async function loadMyDatasets() {
    try {
      const d = await api("/api/datasets");
      const list = d.datasets || [];
      $("myDatasets").classList.toggle("hidden", !list.length);
      $("myDatasetList").innerHTML = list.map((x) => `<div class="ds-item" data-id="${esc(x.dataset_id)}">
          <div><b>${esc(x.file_name)}</b><div class="muted">${x.rows.toLocaleString()} rows × ${x.columns} cols · ${x.passes} pass${x.passes === 1 ? "" : "es"}${x.score !== null && x.score !== undefined ? " · score " + x.score : ""} · ${esc((x.updated || "").replace("T", " "))}</div></div>
          <div class="row gap"><button class="btn btn-outline" data-open="${esc(x.dataset_id)}">Open</button><button class="btn btn-ghost" data-del="${esc(x.dataset_id)}" title="Delete">🗑</button></div></div>`).join("");
      $("myDatasetList").querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", async () => {
        try {
          const data = await api(`/api/datasets/${b.dataset.open}`); onDatasetLoaded(data);
          if (data.has_profile) {
            const a = await api(`/api/datasets/${data.dataset_id}/analyze`, { method: "POST" });
            S.profile = a.profile; S.pass = a.pass; renderAnalysis(); enableStep(2);
            if (a.last_plan && a.last_plan.actions?.length) { setPlan(a.last_plan); enableStep(3); }
            showStep(2);
          }
        } catch (e) { toast(e.message, true); }
      }));
      $("myDatasetList").querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
        if (!confirm("Delete this dataset?")) return;
        try { await api(`/api/datasets/${b.dataset.del}`, { method: "DELETE" }); loadMyDatasets(); } catch (e) { toast(e.message, true); }
      }));
    } catch (e) { /* backend offline or no datasets */ }
  }

  // ------------------------------------------------------------------ init
  (async () => {
    const wake = setTimeout(() => setUploadStatus("Waking up the server… free hosting sleeps when idle, this can take up to a minute."), 2500);
    try { S.toolbox = await api("/api/toolbox"); } catch (e) { /* ignore */ }
    clearTimeout(wake); setUploadStatus("");
    await refreshOllama(); await loadSettings(); await loadMyDatasets();
    setInterval(refreshOllama, 60000);
  })();
})();
