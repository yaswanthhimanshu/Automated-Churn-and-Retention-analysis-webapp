// static/main.js
// Frontend wiring for Churn and Retention analysis single-page webapp.

(() => {
  "use strict";

  // --- Helpers & selectors ---
  const DEFAULT_THRESHOLD = 0.35;  // must match core.py DEFAULT_THRESHOLD
  const SID = window.SESSION_ID || document.getElementById("sessionId")?.textContent || "";
  const $ = (sel) => document.querySelector(sel);
  const el = (sel) => $(sel);
  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function setResult(sel, html) { const n = el(sel); if (n) n.innerHTML = html; }
  function showToast(msg) {
    const t = document.createElement("div");
    t.className = "toast";
    t.style.position = "fixed";
    t.style.right = "18px";
    t.style.bottom = "18px";
    t.style.background = "rgba(0,0,0,0.6)";
    t.style.color = "#fff";
    t.style.padding = "8px 12px";
    t.style.borderRadius = "8px";
    t.style.zIndex = "99999";
    t.innerText = msg;
    document.body.appendChild(t);
    setTimeout(() => t.style.opacity = "0", 2800);
    setTimeout(() => t.remove(), 3200);
  }
  function setLoading(btnSelector, isLoading, text) {
    const btn = el(btnSelector);
    if (!btn) return;
    if (isLoading) {
      btn.dataset._orig = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = (text || "Working...") + " ⏳";
    } else {
      btn.disabled = false;
      btn.innerHTML = btn.dataset._orig || btn.innerHTML;
    }
  }


  function updateTrainTargetUI(columns, likelyTargets) {
  if (!Array.isArray(columns) || !columns.length) return;
  const trainPanel = document.querySelector("#train");
  if (!trainPanel) return;

  // Find existing text input (target_col)
  const existing = trainPanel.querySelector("input[name='target_col']");

  // Create <select> element
  const select = document.createElement("select");
  select.name = "target_col";
  select.setAttribute("aria-label", "Target column");
  select.style.width = "100%";
  select.style.padding = "10px";
  select.style.borderRadius = "8px";
  select.style.border = "1px solid rgba(255,255,255,0.04)";

  // Placeholder
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "-- select target column --";
  select.appendChild(placeholder);

  // Add all column options
  columns.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    select.appendChild(opt);
  });

  // Auto-select likely target if detected
  if (Array.isArray(likelyTargets) && likelyTargets.length > 0) {
    const first = likelyTargets.find(t => columns.includes(t)) || likelyTargets[0];
    if (first) select.value = first;
  }

  // Replace existing input with dropdown
  if (existing) {
    existing.parentNode.replaceChild(select, existing);
  } else {
    const form = document.getElementById("trainForm");
    if (form) {
      const wrapper = document.createElement("div");
      wrapper.className = "form-row";
      const label = document.createElement("label");
      label.style.flex = "1";
      const caption = document.createElement("div");
      caption.className = "small muted";
      caption.textContent = "Target column";
      label.appendChild(caption);
      label.appendChild(select);
      wrapper.appendChild(label);
      form.insertBefore(wrapper, form.firstChild);
    }
  }
}


  // stores current session id globally for updates
  let CURRENT_SID = SID;

  // updates hidden inputs + window var when server provides new session id
  function updateSessionId(newSid) {
    if (!newSid) return;
    CURRENT_SID = newSid;
    window.SESSION_ID = newSid;
    const badge = document.getElementById("sessionId");
    if (badge) badge.textContent = newSid;
    const els = document.querySelectorAll("input[name='session_id']");
    els.forEach(i => i.value = newSid);
  }

  // small pretty JSON
  function jsonPretty(obj) {
    return "<pre style='white-space:pre-wrap;margin:0;font-size:13px;color:var(--muted)'>" + escapeHtml(JSON.stringify(obj, null, 2)) + "</pre>";
  }

  // --- Dashboard render helpers  ---
  function formatCurrency(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
    return Number(v).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
  }

  function renderKpiCard(title, value, hint) {
    return `<div style="min-width:160px;padding:12px;border-radius:10px;background:rgba(255,255,255,0.02);margin-right:8px;color:#e6eef6">
              <div style="font-size:12px;color:var(--muted)">${escapeHtml(title)}</div>
              <div style="font-weight:700;font-size:16px;margin-top:6px">${value}</div>
              ${hint ? `<div style="font-size:12px;color:var(--muted);margin-top:6px">${escapeHtml(hint)}</div>` : ""}
            </div>`;
  }
  

  function renderProgressBar(pct, msg) {
      const p = Math.max(0, Math.min(100, pct));
      const isError = pct === -1;
      // Use --danger from style.css if it exists, otherwise a fallback red
      const barColor = isError ? "var(--danger, #ef4444)" : "linear-gradient(90deg, var(--accent), var(--accent-2))";
      const textColor = isError ? "var(--danger, #ef4444)" : "var(--text, #e6eef6)";
      return `<div style="margin-top:10px;padding:8px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid ${isError ? 'var(--danger, #ef4444)' : 'rgba(255,255,255,0.04)'}">
                <div style="font-size:13px;font-weight:600;color:${textColor};margin-bottom:6px">${escapeHtml(msg)} (${p}%)</div>
                <div style="height:8px;border-radius:4px;background:rgba(255,255,255,0.05);overflow:hidden">
                  <div id="trainProgressFill" style="height:100%;width:${p}%;background:${barColor};transition:width 0.5s ease"></div>
                </div>
              </div>`;
  }

  // Helper function to render the final train dashboard
  function renderTrainDashboard(data) {
      let html = `<div style="margin-bottom:8px"><strong style="color:var(--text)">Training Dashboard</strong></div>`;
      
      // KPI cards
      if (data.meta && data.meta.n_rows) {
          html += "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px'>";
          html += renderKpiCard("Churn rate", ((data.kpis?.churn_rate || 0) * 100).toFixed(2) + "%", "From training labels");
          html += renderKpiCard("Rows (labelled)", data.meta.n_rows, "Rows used to train");
          html += "</div>";
      }

      // Model metrics
      if (data.meta && data.meta.metrics) {
          html += `<div style="margin-top:6px"><strong style="color:var(--text)">Model metrics</strong></div>`;
          html += renderMetricList(data.meta.metrics);
          // Threshold info
          html += `<div style="margin-top:6px;font-size:12px;color:var(--muted)">Prediction threshold: <strong style="color:var(--accent)">${DEFAULT_THRESHOLD}</strong></div>`;
          html += `<div style="font-size:11px;color:var(--muted);margin-top:3px;font-style:italic">Lower threshold (${DEFAULT_THRESHOLD}) is used to increase recall and detect more churn customers.</div>`;
          // Best metric highlight
          if (data.meta.best_metric && data.meta.best_score != null) {
            const bm = data.meta.best_metric.toUpperCase().replace("_"," ");
            const bs = (data.meta.best_score * 100).toFixed(2);
            html += `<div style="margin-top:6px;font-size:13px;"><span class="best-metric">Best Metric: ${escapeHtml(bm)} (${bs}%)</span></div>`;
            html += `<div style="font-size:11px;color:var(--muted);margin-top:2px;">Model evaluated using priority: ROC-AUC &gt; F1 &gt; Accuracy</div>`;
          }
          // Recall warning
          const recall = data.meta.metrics.recall;
          if (recall != null) {
            if (recall < 0.5)
              html += `<div class="recall-warning recall-low">⚠️ Low Recall (${(recall*100).toFixed(1)}%): Model is missing many churn customers</div>`;
            else if (recall < 0.7)
              html += `<div class="recall-warning recall-medium">⚠️ Moderate Recall (${(recall*100).toFixed(1)}%): Can be improved</div>`;
            else
              html += `<div class="recall-warning recall-good">✅ Good Recall (${(recall*100).toFixed(1)}%): Model detects most churn customers</div>`;
          }

          // --- Fit Status block ---
          const fitStatus = data.meta.fit_status;
          const fitReason = data.meta.fit_reason;
          if (fitStatus && fitStatus !== "N/A") {
            const fitCls = getFitClass(fitStatus);
            const trainScoreStr = data.meta.train_score != null ? (data.meta.train_score * 100).toFixed(2) + "%" : "—";
            const testScoreStr  = data.meta.test_score  != null ? (data.meta.test_score  * 100).toFixed(2) + "%" : "—";
            html += `<div class="fit-status-block" style="margin-top:10px;padding:12px 14px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06)">
              <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px">
                <span style="font-size:13px;color:var(--muted)">Model Fit:</span>
                <span class="${fitCls}" style="font-size:14px">● ${escapeHtml(fitStatus)}</span>
                <span style="font-size:12px;color:var(--muted)">— ${escapeHtml(fitReason)}</span>
              </div>
              <div style="display:flex;gap:16px;font-size:13px;flex-wrap:wrap">
                <span style="color:var(--muted)">Train Score (F1): <strong style="color:var(--text)">${trainScoreStr}</strong></span>
                <span style="color:var(--muted)">Test Score (F1): <strong style="color:var(--text)">${testScoreStr}</strong></span>
              </div>
            </div>`;
          } else if (fitStatus === "N/A") {
            html += `<div style="margin-top:8px;font-size:12px;color:var(--muted);font-style:italic">Fit analysis not available for full-data training mode (no holdout set).</div>`;
          }

          // --- Suspicious performance warning ---
          const _m = data.meta.metrics;
          if (
            _m.accuracy  != null && _m.accuracy  >= 0.98 &&
            _m.f1        != null && _m.f1        >= 0.98 &&
            _m.recall    != null && _m.recall    >= 0.98 &&
            _m.precision != null && _m.precision >= 0.98
          ) {
            html += `<div style="margin-top:10px;padding:12px 14px;border-radius:10px;
                       background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
                       display:flex;align-items:flex-start;gap:10px">
              <span style="font-size:18px;line-height:1">⚠️</span>
              <div>
                <div style="font-weight:700;color:#f59e0b;font-size:13px;margin-bottom:3px">
                  Suspiciously perfect performance detected
                </div>
                <div style="font-size:12px;color:var(--muted);line-height:1.5">
                  Accuracy, F1, Recall, and Precision are all ≥ 98%. This may indicate
                  <strong style="color:#f59e0b">data leakage</strong>,
                  <strong style="color:#f59e0b">duplicate rows</strong>, or
                  <strong style="color:#f59e0b">overly simple data</strong>.
                  Verify your dataset and feature engineering before deploying this model.
                </div>
              </div>
            </div>`;
          }
      }

      // SHAP summary (top features)
      if (data.shap && data.shap.top_features) {
          html += `<div style="margin-top:10px"><strong style="color:var(--text)">Top features (SHAP)</strong></div>`;
          html += "<div style='margin-top:8px;display:flex;flex-direction:column;gap:6px'>";
          data.shap.top_features.slice(0,12).forEach(f => {
            html += `<div style="display:flex;justify-content:space-between;align-items:center">
                      <div style="font-size:13px;color:var(--muted)">${escapeHtml(f.name)}</div>
                      <div style="font-weight:700">${(f.mean_abs_shap || 0).toFixed(4)}</div>
                    </div>`;
          });
          html += "</div>";
      }
      
      // Fallback: raw meta
      if (!data.kpis && (!data.meta || !data.meta.metrics) && !data.shap) {
        html += "<div class='muted'>Training finished — no KPI/metric data returned.</div>";
      }

      setResult("#trainResult", html);
  }
  
  function metricClass(v) {
    if (typeof v !== "number" || v == null) return "";
    if (v >= 0.8) return "metric-high";
    if (v >= 0.6) return "metric-medium";
    return "metric-low";
  }

  // Map fit status label to CSS class
  function getFitClass(status) {
    if (!status) return "";
    const s = status.toLowerCase().replace(/\s+/g, "");
    if (s === "goodfit")          return "fit-good";
    if (s === "acceptable")       return "fit-acceptable";
    if (s === "mildoverfitting")  return "fit-mild";
    if (s === "overfitting")      return "fit-over";
    if (s === "severeoverfitting") return "fit-severe";
    if (s === "underfitting")     return "fit-under";
    return "muted";
  }

  function renderMetricList(metrics) {
    if (!metrics) return "";
    const METRIC_KEYS = ["accuracy","precision","recall","f1","roc_auc"];
    let html = "<div style='margin-top:10px;display:flex;gap:8px;flex-wrap:wrap'>";
    for (const k of Object.keys(metrics)) {
      const v = metrics[k];
      const isRate = k.toLowerCase().includes("rate") || k.toLowerCase().includes("recall") || k.toLowerCase().includes("precision");
      const formatted = typeof v === "number" ? (Math.round(v*10000)/100) + (isRate ? "%" : "") : String(v);
      const cls = METRIC_KEYS.includes(k) ? metricClass(v) : "";
      html += renderKpiCard(k, `<span class="${cls}">${escapeHtml(String(formatted))}</span>`, null);
    }
    html += "</div>";
    return html;
  }

  function riskClass(prob) {
    if (prob >= 0.7) return "risk-high";
    if (prob >= DEFAULT_THRESHOLD) return "risk-medium";
    return "risk-low";
  }

  // Churn-probability color: high probability = red (bad), low = green (safe)
  function probClass(prob) {
    if (prob >= 0.7)                             return "metric-low";    // red
    if (prob >= DEFAULT_THRESHOLD)               return "metric-medium"; // yellow
    return "metric-high";                                                // green
  }

  function renderPredictionTable(rows, isModal = false) {
    if (!rows || !rows.length) return "<div class='muted'>No sample rows</div>";
    // Already sorted by backend; keep order
    const sorted = [...rows].sort((a, b) =>
      (parseFloat(b.churn_probability) || 0) - (parseFloat(a.churn_probability) || 0)
    );
    const limit = isModal ? sorted.length : 100;
    // Exclude prediction columns from regular columns; add row index
    const cols = Object.keys(sorted[0]).filter(k => k !== "churn_probability" && k !== "predicted_churn");

    let html = `<div style="margin-top:12px">`;

    // Filter toolbar
    html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">`;
    html += `<span style="font-size:12px;color:var(--muted)">${isModal ? sorted.length : Math.min(limit, sorted.length)} rows · sorted by churn probability</span>`;
    html += `<div style="display:flex;gap:4px;margin-left:auto">`;
    html += `<button class="ghost small pred-filter-btn active" data-filter="all">All</button>`;
    html += `<button class="ghost small pred-filter-btn" data-filter="high" style="color:#ef4444">High Risk</button>`;
    html += `<button class="ghost small pred-filter-btn" data-filter="medium" style="color:#f59e0b">Medium Risk</button>`;
    html += `<button class="ghost small pred-filter-btn" data-filter="low" style="color:#10b981">Low Risk</button>`;
    html += `</div></div>`;

    html += `<div style="overflow-x:auto"><table class="pred-table">`;
    html += "<thead><tr>";
    html += `<th style="color:var(--muted)">#</th>`;
    cols.slice(0, 8).forEach(k => html += `<th>${escapeHtml(k)}</th>`);
    html += "<th>Churn Probability</th><th>Risk</th><th>Prediction</th></tr></thead><tbody>";

    sorted.slice(0, limit).forEach((r, idx) => {
      const prob = parseFloat(r.churn_probability) || 0;
      const cls  = riskClass(prob);
      const riskLabel = prob >= 0.7 ? "High" : prob >= DEFAULT_THRESHOLD ? "Medium" : "Low";
      const riskStyle = prob >= 0.7
        ? "color:#ef4444;font-weight:600"
        : prob >= DEFAULT_THRESHOLD
          ? "color:#f59e0b;font-weight:600"
          : "color:#10b981;font-weight:600";
      const predLabel = r.predicted_churn == 1 ? "Will Churn" : "Will Stay";
      const predStyle = r.predicted_churn == 1
        ? "color:#ef4444;font-weight:600"
        : "color:#10b981;font-weight:600";
      html += `<tr class="pred-row ${cls}" data-prob="${prob}">`;
      html += `<td class="pred-row-id">${idx + 1}</td>`;
      cols.slice(0, 8).forEach(k => html += `<td>${escapeHtml(String(r[k] === undefined ? "" : r[k]))}</td>`);
      html += `<td><span class="${probClass(prob)}">${(prob * 100).toFixed(2)}%</span></td>`;
      html += `<td><span style="${riskStyle}">${riskLabel}</span></td>`;
      html += `<td><span style="${predStyle}">${predLabel}</span></td>`;
      html += `</tr>`;
    });
    html += "</tbody></table></div></div>";
    return html;
  }

  function renderSmallTable(rows) {
    if (!rows || !rows.length) return "<div class='muted'>No sample rows</div>";
    const keys = Object.keys(rows[0]);
    let html = "<div style='margin-top:10px;max-height:220px;overflow:auto;background:rgba(255,255,255,0.01);padding:8px;border-radius:8px;'>";
    html += "<table style='width:100%;font-size:12px;border-collapse:collapse'><thead><tr>";
    keys.slice(0,12).forEach(k => html += `<th style='text-align:left;padding:6px;color:var(--muted)'>${escapeHtml(k)}</th>`);
    html += "</tr></thead><tbody>";
    rows.slice(0,20).forEach(r => {
      html += "<tr>";
      keys.slice(0,12).forEach(k => html += `<td style='padding:6px;border-top:1px solid rgba(255,255,255,0.02)'>${escapeHtml(String(r[k]===undefined?"":r[k]))}</td>`);
      html += "</tr>";
    });
    html += "</tbody></table></div>";
    return html;
  }

  function renderFeatureContributions(contribs) {
    if (!contribs) return "<div class='muted'>No feature contributions available.</div>";
    const items = Object.entries(contribs).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,30);
    let html = "<div style='margin-top:8px;display:flex;flex-direction:column;gap:6px;'>";
    const maxAbs = Math.max(...items.map(i=>Math.abs(i[1])||0), 1e-9);
    items.forEach(([k,v])=>{
      const pct = Math.min(100, Math.round((Math.abs(v)/maxAbs)*100));
      const sign = v >= 0 ? "+" : "-";
      html += `<div style="display:flex;align-items:center;gap:8px">
                <div style="width:160px;font-size:13px;color:var(--muted)">${escapeHtml(k)}</div>
                <div style="flex:1;background:rgba(255,255,255,0.03);height:14px;border-radius:8px;overflow:hidden">
                  <div style="height:100%;width:${pct}%;background:linear-gradient(90deg,var(--accent),#7c3aed)"></div>
                </div>
                <div style="width:60px;text-align:right;font-size:13px;color:var(--muted)">${sign}${Math.abs(v).toFixed(3)}</div>
               </div>`;
    });
    html += "</div>";
    return html;
  }

  // --- Column selector (rendered after upload, used at train time) ---
  let columns_to_drop = [];   // module-level: read by train submit handler

  const _DROP_SUGGEST_KEYWORDS = ["id", "number", "code", "name", "surname", "rownum"];

  function _shouldSuggestDrop(colName) {
    const lower = colName.toLowerCase();
    return _DROP_SUGGEST_KEYWORDS.some(kw => lower.includes(kw));
  }

  function renderColumnSelector(columns) {
    // Remove old selector if dataset is re-uploaded
    const existing = document.getElementById("colSelectorPanel");
    if (existing) existing.remove();

    if (!columns || !columns.length) return;

    // Build panel — insert it between the Train panel and the Segment section
    const panel = document.createElement("div");
    panel.id = "colSelectorPanel";
    panel.className = "panel";
    panel.style.marginTop = "0";   // gap handled by parent grid gap

    panel.innerHTML = `
      <h2 style="margin-bottom:6px">Column Selection <span style="font-size:14px;font-weight:400;color:var(--muted)">(Optional but Recommended)</span></h2>
      <p class="small muted" style="margin-bottom:12px">
        Remove identifier columns like <strong>CustomerId</strong>, <strong>RowNumber</strong>, or customer names.
        These do not help the model and can hurt accuracy.
        Pre-checked columns are auto-suggested based on their name.
      </p>
      <div id="colCheckboxGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;max-height:220px;overflow-y:auto;padding:4px 0"></div>
      <div style="margin-top:12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button id="colSelAllBtn"    class="ghost"    type="button" style="font-size:13px">Check all</button>
        <button id="colSelNoneBtn"   class="ghost"    type="button" style="font-size:13px">Uncheck all</button>
        <button id="colSelConfirmBtn" class="primary" type="button" style="font-size:13px">Confirm Selection</button>
        <span id="colSelCount" class="small muted" style="margin-left:4px"></span>
      </div>
      <div id="colSelMsg" style="margin-top:8px;font-size:13px;min-height:20px"></div>
    `;

    // Populate checkboxes
    const grid = panel.querySelector("#colCheckboxGrid");
    columns.forEach(col => {
      const suggest = _shouldSuggestDrop(col);
      const label = document.createElement("label");
      label.style.display = "flex";
      label.style.alignItems = "center";
      label.style.gap = "8px";
      label.style.padding = "6px 8px";
      label.style.borderRadius = "8px";
      label.style.background = suggest ? "rgba(245,158,11,0.07)" : "rgba(255,255,255,0.02)";
      label.style.border = suggest ? "1px solid rgba(245,158,11,0.2)" : "1px solid rgba(255,255,255,0.03)";
      label.style.cursor = "pointer";
      label.style.fontSize = "13px";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = col;
      cb.checked = suggest;
      cb.dataset.col = col;
      cb.addEventListener("change", _syncDropList);

      const span = document.createElement("span");
      span.textContent = col;
      span.style.color = suggest ? "#f59e0b" : "var(--text)";
      span.title = suggest ? "Auto-suggested: likely an identifier column" : "";

      label.appendChild(cb);
      label.appendChild(span);
      grid.appendChild(label);
    });

    // Check-all / Uncheck-all buttons
    panel.querySelector("#colSelAllBtn").addEventListener("click", () => {
      panel.querySelectorAll("#colCheckboxGrid input[type=checkbox]").forEach(cb => cb.checked = true);
      _syncDropList();
    });
    panel.querySelector("#colSelNoneBtn").addEventListener("click", () => {
      panel.querySelectorAll("#colCheckboxGrid input[type=checkbox]").forEach(cb => cb.checked = false);
      _syncDropList();
    });

    // Confirm Selection — UX only, no backend call
    panel.querySelector("#colSelConfirmBtn").addEventListener("click", () => {
      _syncDropList();
      const msgEl = panel.querySelector("#colSelMsg");
      if (msgEl) {
        const count = columns_to_drop.length;
        msgEl.innerHTML = `<span style="color:#10b981;font-weight:600">&#10004; Selection saved.</span>`
          + ` <span style="color:var(--muted)">Changes will be applied when you click Train.`
          + (count ? ` (${count} column(s) excluded)` : " (no columns excluded)")
          + `</span>`;
      }
    });

    // Insert: immediately after the Upload section
    const uploadSection = document.getElementById("upload");
    if (uploadSection && uploadSection.nextSibling) {
      uploadSection.parentNode.insertBefore(panel, uploadSection.nextSibling);
    } else if (uploadSection) {
      uploadSection.parentNode.appendChild(panel);
    } else {
      // Fallback: append to main
      document.querySelector("main").appendChild(panel);
    }

    _syncDropList();   // initialise count from pre-checked boxes
  }

  function _syncDropList() {
    const checked = document.querySelectorAll("#colCheckboxGrid input[type=checkbox]:checked");
    columns_to_drop = Array.from(checked).map(cb => cb.value);
    const countEl = document.getElementById("colSelCount");
    if (countEl) {
      countEl.textContent = columns_to_drop.length
        ? `${columns_to_drop.length} column(s) will be excluded from training`
        : "No columns excluded — all will be used for training";
    }
  }

  // --- Upload handling  ---

  const uploadForm = el("#uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fileEl = el("#fileInput");
      const file = fileEl && fileEl.files && fileEl.files[0];
      if (!file) { showToast("Choose a CSV or XLSX file"); return; }

      const fd = new FormData();
      fd.append("session_id", CURRENT_SID || "");
      fd.append("file", file);

      setLoading("#uploadBtn", true, "Uploading...");
      try {
        const resp = await fetch("/upload", { method: "POST", body: fd });
        if (!resp.ok) {
          const t = await resp.json().catch(()=>({error:"upload failed"}));
          throw new Error(t.error || "upload failed");
        }
        const data = await resp.json();
        if (data.session_id) updateSessionId(data.session_id);

        setResult("#uploadResult", data.preview_html || "<div class='muted'>Preview not available.</div>");
        setResult("#edaSummary", data.eda ? jsonPretty(data.eda) : "<div class='muted'>EDA not available</div>");

        if (data.eda_full) {
          try {
            window.CHURN_EDA_FULL_JSON = data.eda_full;
            if (window.CHI_charts && typeof window.CHI_charts.renderEdaFull === "function") {
              window.CHI_charts.renderEdaFull(data.eda_full);
            } else {
              setResult("#edaFullResult", jsonPretty(data.eda_full));
            }
          } catch (e) {
            setResult("#edaFullResult", jsonPretty(data.eda_full));
          }
        }

        if (data.columns && Array.isArray(data.columns)) {
          console.log("columns:", data.columns);
          if (data.likely_targets && data.likely_targets.length) {
            showToast("Likely target columns: " + data.likely_targets.join(", "));
            // Auto-populate target input if empty
            const targetInput = el("input[name='target_col']");
            if (targetInput && !targetInput.value) targetInput.value = data.likely_targets[0];
          }
          updateTrainTargetUI(data.columns, data.likely_targets || []);
          renderColumnSelector(data.columns);
        }

        showToast("Upload complete");
        loadTimeChurnCandidates();
      } catch (err) {
        console.error("Upload error", err);
        showToast("Upload failed: " + (err.message || ""));
        setResult("#uploadResult", "<div class='muted'>Upload failed: " + escapeHtml(err.message || "") + "</div>");
      } finally {
        setLoading("#uploadBtn", false);
      }
    });
  }

  // Clear upload input
  const clearUploadBtn = el("#clearUploadBtn");
  if (clearUploadBtn) {
    clearUploadBtn.addEventListener("click", () => {
      const fileEl = el("#fileInput"); if (fileEl) fileEl.value = "";
      setResult("#uploadResult", "<div class='muted'>Cleared upload input.</div>");
      setResult("#edaSummary", "<div class='muted'>Upload a dataset to see EDA summary.</div>");
      setResult("#edaFullResult", "");
      const oldPanel = document.getElementById("colSelectorPanel");
      if (oldPanel) oldPanel.remove();
      columns_to_drop = [];
    });
  }

  // Download Preview as CSV (client-side, from rendered table)
  const downloadPreviewBtn = el("#downloadPreviewBtn");
  if (downloadPreviewBtn) {
    downloadPreviewBtn.addEventListener("click", () => {
      const resultEl = el("#uploadResult");
      if (!resultEl) { showToast("No preview to download"); return; }
      const table = resultEl.querySelector("table");
      if (!table) { showToast("Upload a dataset first"); return; }
      try {
        const rows = Array.from(table.querySelectorAll("tr"));
        const csv = rows.map(row =>
          Array.from(row.querySelectorAll("th,td"))
            .map(cell => {
              const v = cell.innerText.replace(/"/g, '""');
              return `"${v}"`;
            })
            .join(",")
        ).join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "preview.csv";
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        showToast("Download failed: " + e.message);
      }
    });
  }

  // Full EDA click 
  const fullEdaBtn = el("#fullEdaBtn");
  if (fullEdaBtn) {
    fullEdaBtn.addEventListener("click", async (ev) => {
      ev.preventDefault();
      setLoading("#fullEdaBtn", true, "Computing full EDA...");
      try {
        const fd = new FormData();
        fd.append("session_id", CURRENT_SID || "");
        const res = await fetch("/eda_full", { method: "POST", body: fd });
        if (!res.ok) throw res;
        const data = await res.json();
        if (data.session_id) updateSessionId(data.session_id);
        if (data.eda_full) {
          window.CHURN_EDA_FULL_JSON = data.eda_full;
          if (window.CHI_charts && typeof window.CHI_charts.renderEdaFull === "function") {
            window.CHI_charts.renderEdaFull(data.eda_full);
          } else {
            setResult("#edaFullResult", jsonPretty(data.eda_full));
          }
        } else {
          setResult("#edaFullResult", "<div class='muted'>No full EDA returned.</div>");
        }
        showToast("Full EDA ready");
      } catch (err) {
        console.error("EDA error", err);
        setResult("#edaFullResult", "<div class='muted'>Full EDA failed</div>");
        showToast("Full EDA failed");
      } finally {
        setLoading("#fullEdaBtn", false);
      }
    });
  }

  // --- Train  ---
  // --- Train  ---
  const trainForm = el("#trainForm");
  let trainingInterval = null; 

  function stopPolling() {
      if (trainingInterval) {
          clearInterval(trainingInterval);
          trainingInterval = null;
          setLoading("#trainBtn", false);
      }
  }

  async function startPolling(sid) {
      if (trainingInterval) return; 

      const pollStatus = async () => {
          try {
              const res = await fetch(`/train_status?session_id=${sid}`);
              if (!res.ok) {
                  stopPolling();
                  throw new Error("Failed to fetch training status.");
              }
              const data = await res.json();
              updateSessionId(data.session_id);
              
              const progressHtml = renderProgressBar(data.progress, data.message);
              setResult("#trainResult", progressHtml);

              if (data.status === "completed") {
                  stopPolling();
                  renderTrainDashboard(data); 
                  showToast("Training finished!");
                  fetchAndRenderModelComparison();
              } else if (data.status === "failed") {
                  stopPolling();
                  setResult("#trainResult", `<div style='color:var(--danger,#ef4444);padding:8px'>Training failed: ${escapeHtml(data.message || "Unknown error")}</div>`);
                  showToast("Training failed.");
              }
              
          } catch (err) {
              stopPolling();
              console.error("Polling error", err);
              setResult("#trainResult", `<div class='muted' style='color:var(--danger, #ef4444)'>Training polling failed: ${escapeHtml(err.message || "")}</div>`);
              showToast("Training polling failed.");
          }
      };
      
      trainingInterval = setInterval(pollStatus, 1000); 
      pollStatus(); 
  }

  if (trainForm) {
    trainForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(trainForm);
      fd.set("session_id", CURRENT_SID || "");
      // Attach columns_to_drop (empty string = no exclusions)
      fd.set("columns_to_drop", (typeof columns_to_drop !== 'undefined' ? columns_to_drop : []).join(","));
      const target_col = fd.get("target_col");
      if (!target_col) { showToast("Enter target column name"); return; }
      
      stopPolling(); 

      setLoading("#trainBtn", true, "Starting...");
      setResult("#trainResult", renderProgressBar(0, "Submitting training job...")); 
      
      try {
          const res = await fetch("/train", { method: "POST", body: fd });

          if (res.status === 409) { 
               showToast("Training is already running for this session.");
               setLoading("#trainBtn", false);
               startPolling(CURRENT_SID); 
               return;
          }

          if (!res.ok) {
              const t = await res.json().catch(()=>({error:"training failed"}));
              throw new Error(t.error || "training failed");
          }
          
          const data = await res.json();
          if (data.session_id) updateSessionId(data.session_id);
          
          if (data.status === "training_started") {
            startPolling(data.session_id || CURRENT_SID);
          } else {
            renderTrainDashboard(data);
            setLoading("#trainBtn", false);
          }

      } catch (err) {
          console.error("Train start error", err);
          setResult("#trainResult", "<div class='muted'>Training failed to start: " + escapeHtml(err.message || "") + "</div>");
          setLoading("#trainBtn", false);
          showToast("Training failed to start.");
      }
    });
  }

  const retrainBtn = el("#retrainBtn");
  if (retrainBtn) retrainBtn.addEventListener("click", () => trainForm && trainForm.requestSubmit());

  // --- Predict  ---
  const predictForm = el("#predictForm");
  if (predictForm) {
    predictForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(predictForm);
      fd.set("session_id", CURRENT_SID || "");

      setLoading("#predictBtn", true, "Predicting...");
      try {
        // Requests JSON dashboard first by setting Accept header
        const res = await fetch("/predict", { method: "POST", body: fd, headers: { "Accept": "application/json" } });

        if (!res.ok) {
          const txt = await res.text().catch(()=>"");
          throw new Error(txt || "Prediction failed");
        }

        const ct = res.headers.get("content-type") || "";
        if (ct.includes("application/json")) {
          const data = await res.json();
          if (data.session_id) updateSessionId(data.session_id);

          // Build prediction dashboard
          let html = `<div style="margin-bottom:8px"><strong style="color:#fff">Prediction Dashboard</strong></div>`;

          // KPI row 1 — summary
          html += "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px'>";
          html += renderKpiCard("Rows scored", data.n_rows || 0, "total customers");
          html += renderKpiCard("Predicted churn", ((data.churn_rate||0)*100).toFixed(2) + "%", "at threshold " + DEFAULT_THRESHOLD);
          if (data.revenue_summary) {
            html += renderKpiCard("Avg revenue", formatCurrency(data.revenue_summary.avg_revenue), "per customer");
            try {
              const churnCount = Math.round((data.churn_rate||0) * (data.n_rows||0));
              const revenueLost = data.revenue_summary.avg_revenue * churnCount;
              html += renderKpiCard("Est. revenue at risk", formatCurrency(revenueLost), `${churnCount} churners`);
            } catch(e){}
          }
          html += "</div>";

          // KPI row 2 — risk segments (from full data counts)
          if (data.risk_counts) {
            const rc = data.risk_counts;
            html += "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px'>";
            html += renderKpiCard(
              "🔴 High Risk",
              `<span class="metric-low">${rc.high}</span>`,
              "prob ≥ 0.70"
            );
            html += renderKpiCard(
              "🟡 Medium Risk",
              `<span class="metric-medium">${rc.medium}</span>`,
              "0.35 ≤ prob < 0.70"
            );
            html += renderKpiCard(
              "🟢 Low Risk",
              `<span class="metric-high">${rc.low}</span>`,
              "prob < 0.35"
            );
            html += "</div>";
          }

          // Action buttons row
          html += `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px">`;
          html += `<button class="ghost small" id="predictExpandBtn" type="button">🔍 Expand Full View</button>`;
          html += `<button class="ghost small" id="predictDownloadBtn" type="button">⬇ Download CSV</button>`;
          html += `</div>`;

          // Prediction table (top 100)
          if (data.rows_sample && data.rows_sample.length) {
            html += renderPredictionTable(data.rows_sample);
          } else if (data.preview_html) {
            html += `<div style="margin-top:10px">${data.preview_html}</div>`;
          }

          setResult("#predictResult", html);

          // Wire up risk filter buttons
          document.querySelectorAll(".pred-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
              document.querySelectorAll(".pred-filter-btn").forEach(b => b.classList.remove("active"));
              btn.classList.add("active");
              const filter = btn.dataset.filter;
              document.querySelectorAll(".pred-row").forEach(tr => {
                const prob = parseFloat(tr.dataset.prob || "0");
                let show = true;
                if (filter === "high")   show = prob >= 0.7;
                if (filter === "medium") show = prob >= DEFAULT_THRESHOLD && prob < 0.7;
                if (filter === "low")    show = prob < DEFAULT_THRESHOLD;
                tr.style.display = show ? "" : "none";
              });
            });
          });

          // Download CSV
          const dlBtn = document.getElementById("predictDownloadBtn");
          if (dlBtn && data.full_data) {
            dlBtn.addEventListener("click", () => {
              const rows = data.full_data;
              if (!rows || !rows.length) return;
              const keys = Object.keys(rows[0]);
              const csv = [keys.join(","), ...rows.map(r =>
                keys.map(k => { const v = r[k]; return `"${String(v ?? "").replace(/"/g, '""')}"`; }).join(",")
              )].join("\n");
              const blob = new Blob([csv], { type: "text/csv" });
              const url  = URL.createObjectURL(blob);
              const a    = document.createElement("a");
              a.href = url; a.download = "churn_predictions.csv"; a.click();
              URL.revokeObjectURL(url);
            });
          }

          // Expand modal
          const expandBtn = document.getElementById("predictExpandBtn");
          if (expandBtn && data.full_data) {
            expandBtn.addEventListener("click", () => {
              const modal = document.getElementById("predictModal");
              const body  = document.getElementById("predictModalBody");
              if (!modal || !body) return;
              body.innerHTML = renderPredictionTable(data.full_data, true);
              modal.style.display = "flex";
              // wire filters inside modal too
              body.querySelectorAll(".pred-filter-btn").forEach(btn => {
                btn.addEventListener("click", () => {
                  body.querySelectorAll(".pred-filter-btn").forEach(b => b.classList.remove("active"));
                  btn.classList.add("active");
                  const filter = btn.dataset.filter;
                  body.querySelectorAll(".pred-row").forEach(tr => {
                    const prob = parseFloat(tr.dataset.prob || "0");
                    let show = true;
                    if (filter === "high")   show = prob >= 0.7;
                    if (filter === "medium") show = prob >= DEFAULT_THRESHOLD && prob < 0.7;
                    if (filter === "low")    show = prob < DEFAULT_THRESHOLD;
                    tr.style.display = show ? "" : "none";
                  });
                });
              });
            });
          }

          showToast("Prediction dashboard ready");
        } 
      } catch (err) {
        console.error("Predict error", err);
        setResult("#predictResult", "<div class='muted'>Prediction failed: " + escapeHtml(err.message || "") + "</div>");
        showToast("Prediction failed");
      } finally {
        setLoading("#predictBtn", false);
      }
    });
  }

  // --- Time-based Churn Analysis ---
const timeChurnForm = document.getElementById("timeChurnForm");
const timeColumnSelect = document.getElementById("timeColumnSelect");
if (timeColumnSelect) {
  timeColumnSelect.addEventListener("change", (e) => {
    const selectedColumn = e.target.value;
    _updateRangeVisibility(timeColumnSelect);
    loadUniqueValues(selectedColumn);
  });
}
document.querySelectorAll("input[name='analysisType']").forEach(radio => {
  radio.addEventListener("change", (e) => {
    const isRange = e.target.value === "range";

    const rangeBox = document.getElementById("rangeInputs");
    const valueBox = document.getElementById("timeValueContainer");

    if (rangeBox) rangeBox.style.display = isRange ? "flex" : "none";
    if (valueBox) valueBox.style.display = isRange ? "none" : "block";
  });
});

async function loadUniqueValues(column) {
  try {
    const fd = new FormData();
    fd.append("session_id", CURRENT_SID || "");
    fd.append("column", column);

    const res = await fetch("/get_unique_values", {
      method: "POST",
      body: fd
    });

    if (!res.ok) return;

    const data = await res.json();

    const container = document.getElementById("timeValueContainer");
    if (!container) return;

    container.innerHTML = "";

    const select = document.createElement("select");
    select.id = "timeValue";
    select.required = true;

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "-- Select value --";
    select.appendChild(placeholder);

    data.values.forEach(v => {
      const value = (typeof v === "object") ? v.value : v;

      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value;
      select.appendChild(opt);
    });

    container.appendChild(select);

  } catch (err) {
    console.error("unique values failed", err);
  }
}

function loadTimeChurnCandidates() {
  try {
    const columns = window.CHURN_EDA_FULL_JSON?.columns;

    if (!columns || !columns.length) return;

    const select = document.getElementById("timeColumnSelect");
    if (!select) return;

    select.innerHTML = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "-- Select column --";
    select.appendChild(placeholder);

    columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col.name;
      opt.textContent = col.name;
      if (col.dtype === "object" || col.dtype === "category") {
        opt.dataset.type = "period";
      } else {
        opt.dataset.type = "duration";
      }
      select.appendChild(opt);
    });

    // Re-apply visibility for the currently selected column
    _updateRangeVisibility(select);

  } catch (err) {
    console.error("Failed to load columns", err);
  }
}

// Shows/hides range inputs based on whether the selected column is categorical
function _updateRangeVisibility(selectEl) {
  const selectedOpt = selectEl?.selectedOptions[0];
  const colType = selectedOpt?.dataset?.type;
  const isCategorical = colType === "period";

  const rangeRadios = document.querySelectorAll("input[name='analysisType']");
  const rangeLabel = document.querySelector("label:has(input[value='range'])") ||
    (() => { // fallback for browsers without :has support
      let l = null;
      rangeRadios.forEach(r => { if (r.value === "range") l = r.closest("label"); });
      return l;
    })();

  if (isCategorical) {
    // Categorical: range makes no sense — force single and hide range option
    rangeRadios.forEach(r => { if (r.value === "single") r.checked = true; });
    if (rangeLabel) rangeLabel.style.opacity = "0.3";
    if (rangeLabel) rangeLabel.style.pointerEvents = "none";
    const rangeBox = document.getElementById("rangeInputs");
    const valueBox = document.getElementById("timeValueContainer");
    if (rangeBox) rangeBox.style.display = "none";
    if (valueBox) valueBox.style.display = "block";
  } else {
    // Numeric: allow both
    if (rangeLabel) rangeLabel.style.opacity = "";
    if (rangeLabel) rangeLabel.style.pointerEvents = "";
  }
}

if (timeChurnForm) {
  timeChurnForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();

    const colSelect = document.getElementById("timeColumnSelect");
    const timeColumn = colSelect?.value;
    const timeType = colSelect?.selectedOptions[0]?.dataset?.type;
    const userValue = document.getElementById("timeValue")?.value;

    const analysisType = document.querySelector("input[name='analysisType']:checked")?.value || "single";
    if (!timeColumn) {
      showToast("Select a column first");
      return;
    }

    if (analysisType === "single" && !userValue) {
      showToast("Select a value");
      return;
    }

    if (analysisType === "range") {
      const minVal = document.getElementById("minValue")?.value;
      const maxVal = document.getElementById("maxValue")?.value;

      if (!minVal || !maxVal) {
        showToast("Enter both minimum and maximum values");
        return;
      }
    }



    const fd = new FormData();
    fd.append("session_id", CURRENT_SID || "");
    fd.append("time_column", timeColumn);
    fd.append("time_type", timeType);
    fd.append("analysis_type", analysisType);

    if (analysisType === "range") {
      const minVal = document.getElementById("minValue")?.value;
      const maxVal = document.getElementById("maxValue")?.value;

      fd.append("min_value", minVal);
      fd.append("max_value", maxVal);
    } else {
      fd.append("user_value", userValue);
    }

    setLoading("#timeChurnBtn", true, "Analyzing...");
    try {
      const res = await fetch("/time_churn", {
        method: "POST",
        body: fd
      });

      if (!res.ok) {
        // Surface backend error message (e.g. "train model first")
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      const r = data.time_churn;

      if (!r) {
        setResult("#timeChurnResult", "<div class='muted'>No result returned from server.</div>");
        return;
      }

      let html = `<div style="background:rgba(255,255,255,0.02);padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,0.04)">
        <div style="margin-bottom:10px"><strong style="color:var(--text)">Segment-based Churn Result</strong></div>`;

      if (r.mode === "groupby") {
        // --- Categorical column: render segment table ---
        const segs = r.result && r.result.segments;
        html += `<div style="margin-bottom:8px;font-size:13px;color:var(--muted)">
          Column: <strong style="color:#fff">${escapeHtml(r.column)}</strong>
          &nbsp;·&nbsp; ${segs ? segs.length : 0} segments
        </div>`;
        if (segs && segs.length) {
          html += `<div style="overflow:auto;max-height:320px">
            <table style="width:100%;font-size:13px;border-collapse:collapse">
              <thead><tr>
                <th style="padding:8px;text-align:left;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Value</th>
                <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Customers</th>
                <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Churned</th>
                <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Retained</th>
                <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Churn Rate</th>
              </tr></thead><tbody>`;
          segs.forEach(s => {
            const rate = (s.churn_rate * 100).toFixed(2);
            const rateColor = s.churn_rate > 0.3 ? "#ef4444" : s.churn_rate > 0.15 ? "#f59e0b" : "#10b981";
            html += `<tr>
              <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.03)">${escapeHtml(s.value)}</td>
              <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${s.customers}</td>
              <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${s.churned}</td>
              <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${s.not_churned}</td>
              <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03);font-weight:700;color:${rateColor}">${rate}%</td>
            </tr>`;
          });
          html += `</tbody></table></div>`;
        } else {
          html += `<div class="muted">No segments returned.</div>`;
        }

      } else {
        // --- Numeric column: render KPI cards ---
        html += `<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:4px">`;
        html += renderKpiCard("Customers", r.time_filter.customers_after_filter, `Filtered by ${r.time_filter.column}`);
        html += renderKpiCard("Churn rate", (r.observed_churn.churn_rate * 100).toFixed(2) + "%", `${r.observed_churn.churned} churned`);
        html += renderKpiCard("Retained", r.observed_churn.not_churned, "customers stayed");
        if (r.predicted_churn) {
          html += renderKpiCard("Predicted churn", r.predicted_churn.predicted_churn, `${r.predicted_churn.high_risk || 0} high risk`);
        }
        html += `</div>`;
        html += `<div style="margin-top:10px;font-size:13px;color:var(--muted)">
          Filter: <strong style="color:#fff">${escapeHtml(String(r.time_filter.column))}</strong>
          &nbsp;≤&nbsp; <strong style="color:#fff">${escapeHtml(String(r.time_filter.value))}</strong>
        </div>`;
      }

      html += `</div>`;

      // --- Lifecycle risk section (if returned) ---
      const lc = data.lifecycle_risk;
      if (lc) {
        html += `<div style="margin-top:12px;background:rgba(255,255,255,0.02);padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,0.04)">`;
        html += `<div style="margin-bottom:10px"><strong style="color:var(--text)">Lifecycle Risk Analysis</strong></div>`;

        if (lc.error) {
          // No valid lifecycle column found — surface the backend message clearly
          html += `<div style="display:flex;align-items:flex-start;gap:10px;padding:10px;border-radius:8px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2)">
            <div style="font-size:18px;line-height:1">&#8505;&#65039;</div>
            <div>
              <div style="font-size:13px;color:#f59e0b;font-weight:600;margin-bottom:4px">Lifecycle analysis not available</div>
              <div style="font-size:13px;color:var(--muted);line-height:1.5">${escapeHtml(lc.error)}</div>
            </div>
          </div>`;

        } else if (lc.segments) {
          // Valid result — show column label + interpretation + stage cards
          html += `<div style="margin-bottom:10px">
            <div style="font-size:13px;color:var(--muted);margin-bottom:4px">
              Based on customer lifecycle column:
              <strong style="color:#fff">${escapeHtml(lc.column_used || "")}</strong>
            </div>
            <div style="font-size:12px;color:var(--muted);font-style:italic">
              ${escapeHtml(lc.interpretation || "Lifecycle calculated using duration-based column (e.g., tenure in months).")}
            </div>
          </div>`;

          const lcSegs = lc.segments;
          const highlight = lc.highest_risk_segment;
          const stageLabels = {
            early_stage: "Early  (\u22643 months)",
            mid_stage:   "Mid  (4\u201312 months)",
            late_stage:  "Late  (>12 months)"
          };

          html += `<div style="display:flex;gap:10px;flex-wrap:wrap">`;
          Object.entries(lcSegs).forEach(([stage, stats]) => {
            const isHighest = stage === highlight;
            const rateStr   = stats.customers > 0 ? (stats.churn_rate * 100).toFixed(2) + "%" : "N/A";
            const borderColor = isHighest ? "rgba(239,68,68,0.4)" : "rgba(255,255,255,0.04)";
            const rateColor   = isHighest ? "#ef4444" : "#fff";
            html += `<div style="min-width:160px;padding:12px;border-radius:8px;background:rgba(255,255,255,0.01);border:1px solid ${borderColor}">
              <div style="font-size:12px;color:var(--muted);margin-bottom:4px">${escapeHtml(stageLabels[stage] || stage)}</div>
              <div style="font-weight:700;font-size:18px;color:${rateColor}">${rateStr}</div>
              <div style="font-size:12px;color:var(--muted);margin-top:4px">${stats.customers} customers</div>
              ${isHighest ? `<div style="font-size:11px;color:#ef4444;margin-top:6px;font-weight:600">&#9888; Highest churn risk</div>` : ""}
            </div>`;
          });
          html += `</div>`;
        }

        html += `</div>`;
      }

      setResult("#timeChurnResult", html);
      showToast("Analysis ready");

    } catch (err) {
      console.error(err);
      setResult("#timeChurnResult", `<div style="color:var(--danger,#ef4444);padding:8px">Analysis failed: ${escapeHtml(err.message || "Unknown error")}</div>`);
      showToast("Analysis failed: " + (err.message || ""));
    } finally {
      setLoading("#timeChurnBtn", false);
    }
  });
}

  // --- Explain  ---
  const explainForm = el("#explainForm");
  if (explainForm) {
    explainForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(explainForm);
      fd.set("session_id", CURRENT_SID || "");
      setLoading("#explainBtn", true, "Explaining...");
      try {
        const rowIndex = fd.get("row_index");
        let res;
        if (rowIndex !== null && rowIndex !== "") {
          const params = new URLSearchParams({ session_id: CURRENT_SID || "", row_index: rowIndex });
          res = await fetch("/explain?" + params.toString(), { method: "GET", headers: { "Accept": "application/json" }});
        } else {
          res = await fetch("/explain", { method: "POST", body: fd, headers: { "Accept": "application/json" }});
        }
        if (!res.ok) throw new Error("Explain endpoint not available or failed");
        const data = await res.json();
        if (data.session_id) updateSessionId(data.session_id);

        //explanation
        const ex = data.explanation || data;
        let html = `<div style="margin-bottom:8px"><strong style="color:#fff">Per-customer Explanation</strong></div>`;
        if (ex.predicted_churn !== undefined) {
          html += "<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px'>";
          html += renderKpiCard("Predicted churn", ex.predicted_churn, "");
          if (ex.churn_probability !== undefined) {
            html += renderKpiCard("Probability", ((ex.churn_probability||0)*100).toFixed(2) + "%", "");
          }
          html += "</div>";
        }
        if (ex.feature_contributions) {
          html += "<div style='margin-top:8px'><strong style='color:#fff'>Feature contributions</strong></div>";
          html += renderFeatureContributions(ex.feature_contributions);
        } else {
          // fallback to raw JSON
          html += "<div style='margin-top:8px'>" + jsonPretty(ex) + "</div>";
        }

        setResult("#explainResult", html);
        showToast("Explanation ready");
      } catch (err) {
        console.error("Explain error", err);
        setResult("#explainResult", "<div class='muted'>Explain failed: " + escapeHtml(err.message || "") + "</div>");
        showToast("Explain failed");
      } finally {
        setLoading("#explainBtn", false);
      }
    });
  }

  // --- Simulation
  const simulateForm = el("#simulateForm");
  if (simulateForm) {
    simulateForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const fd = new FormData(simulateForm);
      fd.set("session_id", CURRENT_SID || "");
      setLoading("#simulateBtn", true, "Simulating...");
      try {
        const res = await fetch("/simulate", { method: "POST", body: fd });
        if (!res.ok) throw res;
        const data = await res.json();
        if (data.session_id) updateSessionId(data.session_id);
        if (data.simulate) {
          const s = data.simulate;
          let html = `<div style="margin-bottom:8px"><strong style="color:#fff">Simulation Results</strong></div>`;
          html += `<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px">`;
          html += renderKpiCard("Churn before", ((s.before_churn_rate||0)*100).toFixed(2)+"%", "baseline");
          html += renderKpiCard("Churn after", ((s.after_churn_rate||0)*100).toFixed(2)+"%", "post-action");
          html += renderKpiCard("Retained customers", s.retained_customers ?? "-", "saved");
          html += renderKpiCard("Revenue saved", "$" + formatCurrency(s.revenue_saved), "estimated");
          html += renderKpiCard("Action cost", "$" + formatCurrency(s.action_cost), `${s.n_targeted_customers ?? "-"} targeted`);
          html += renderKpiCard("ROI", (s.roi != null ? s.roi.toFixed(2) + "%" : "-"), "return on investment");
          html += `</div>`;
          html += `<div style="font-size:12px;color:var(--muted)">Reduction factor applied: ${(s.reduction_factor||0).toFixed(2)}%</div>`;
          setResult("#simulateResult", html);
        } else {
          setResult("#simulateResult", "<div class='muted'>No simulation returned</div>");
        }
        showToast("Simulation complete");
      } catch (err) {
        console.error("Simulate error", err);
        setResult("#simulateResult", "<div class='muted'>Simulation failed</div>");
        showToast("Simulation failed");
      } finally {
        setLoading("#simulateBtn", false);
      }
    });
  }

  // --- Model Comparison Table ---
  async function fetchAndRenderModelComparison() {
    try {
      const res = await fetch(`/model_history?session_id=${CURRENT_SID || ""}`);
      if (!res.ok) return;
      const data = await res.json();
      if (data.session_id) updateSessionId(data.session_id);
      renderModelComparisonTable(data.history || []);
    } catch (e) {
      console.warn("Model history fetch failed", e);
    }
  }

  // Returns true when all four key metrics are suspiciously perfect (>= 0.98)
  function _isSuspicious(e) {
    return (e.accuracy  ?? 0) >= 0.98 &&
           (e.f1        ?? 0) >= 0.98 &&
           (e.recall    ?? 0) >= 0.98 &&
           (e.precision ?? 0) >= 0.98;
  }

  // Combined score: F1 heaviest, then recall, precision, accuracy, roc_auc
  function _compScore(e) {
    return (0.30 * (e.f1        ?? 0))
         + (0.25 * (e.recall    ?? 0))
         + (0.20 * (e.precision ?? 0))
         + (0.15 * (e.accuracy  ?? 0))
         + (0.10 * (e.roc_auc   ?? 0));
  }

  // Color-code a 0-1 metric using CSS classes
  function _mc(val, isPercent = true) {
    if (val == null) return "-";
    const cls = val >= 0.8 ? "metric-high" : val >= 0.6 ? "metric-medium" : "metric-low";
    const str = isPercent ? (val * 100).toFixed(2) + "%" : val.toFixed(4);
    return `<span class="${cls}">${str}</span>`;
  }

  function renderModelComparisonTable(history) {
    const container = document.getElementById("modelComparisonPanel");
    if (!container) return;
    if (!history || history.length < 2) { container.style.display = "none"; return; }
    container.style.display = "block";

    // Mark which entries are suspicious
    const suspiciousFlags = history.map(_isSuspicious);
    const allSuspicious   = suspiciousFlags.every(Boolean);

    // Pick best model: prefer non-suspicious models; fall back to all if every model is suspicious
    let bestIdx   = 0;
    let bestScore = -Infinity;
    history.forEach((entry, i) => {
      // Skip suspicious models unless every model is suspicious (fallback)
      if (!allSuspicious && suspiciousFlags[i]) return;
      const s = _compScore(entry);
      if (s > bestScore) { bestScore = s; bestIdx = i; }
    });
    const latestIdx = history.length - 1;

    let html = `<div style="margin-bottom:10px;display:flex;justify-content:space-between;align-items:center">
      <strong style="color:var(--text)">Model Comparison</strong>
      <span class="small muted">${history.length} model(s) trained this session</span>
    </div>
    <div style="overflow-x:auto"><table style="width:100%;font-size:13px;border-collapse:collapse">
      <thead><tr>
        <th style="padding:8px;text-align:left;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">#</th>
        <th style="padding:8px;text-align:left;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Model</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Accuracy</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Recall</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">F1 Score</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">ROC-AUC</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Precision</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Train Score</th>
        <th style="padding:8px;text-align:right;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Test Score</th>
        <th style="padding:8px;text-align:left;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Fit Status</th>
        <th style="padding:8px;text-align:left;color:var(--muted);border-bottom:1px solid rgba(255,255,255,0.06)">Mode</th>
      </tr></thead><tbody>`;

    history.forEach((entry, i) => {
      const isLatest    = i === latestIdx;
      const isBest      = i === bestIdx;
      const isSuspect   = suspiciousFlags[i];
      const latestBadge = isLatest
        ? `<span style="margin-left:6px;font-size:11px;padding:2px 6px;border-radius:4px;background:rgba(6,182,212,0.15);color:#06b6d4">latest</span>`
        : "";
      const bestBadge = isBest
        ? `<span class="badge-best">BEST MODEL</span>`
        : "";
      const suspectBadge = isSuspect
        ? `<span style="margin-left:6px;font-size:11px;padding:2px 6px;border-radius:4px;background:rgba(245,158,11,0.15);color:#f59e0b;border:1px solid rgba(245,158,11,0.25)" title="Accuracy, F1, Recall &amp; Precision are all ≥ 98% — possible data leakage or duplicate rows">⚠️ suspicious</span>`
        : "";
      const rowClass = isBest ? "best-model-row" : "";
      const rowBg    = isSuspect
        ? "background:rgba(245,158,11,0.04);"
        : (!isBest && isLatest) ? "background:rgba(6,182,212,0.04);" : "";
      html += `<tr class="${rowClass}" style="${rowBg}outline-offset:-1px">
        <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.03);color:var(--muted)">${i + 1}</td>
        <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.03)">${escapeHtml(entry.model_type)}${latestBadge}${bestBadge}${suspectBadge}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${_mc(entry.accuracy)}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${_mc(entry.recall)}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${_mc(entry.f1)}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${_mc(entry.roc_auc, false)}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${_mc(entry.precision)}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${entry.train_score != null ? _mc(entry.train_score) : '<span style="color:var(--muted)">—</span>'}</td>
        <td style="padding:8px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.03)">${entry.test_score  != null ? _mc(entry.test_score)  : '<span style="color:var(--muted)">—</span>'}</td>
        <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.03)">${entry.trained_on === "full" ? `<span style="font-size:11px;color:#f59e0b" title="Model was trained on the full dataset — no holdout test set was used, so metrics may not reflect real-world performance">Using full training, fit status cannot be determined</span>` : `<span class="${getFitClass(entry.fit_status)}" title="${escapeHtml(entry.fit_reason || '')}">${escapeHtml(entry.fit_status || '—')}</span>`}</td>
        <td style="padding:8px;border-bottom:1px solid rgba(255,255,255,0.03);color:var(--muted)">${escapeHtml(entry.trained_on || "")}</td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
    container.innerHTML = html;
  }

  // --- Chat 
  const chatForm = el("#chatForm");
  const chatWindow = el("#chatWindow");
  function appendChat(text, who="bot") {
    if (!chatWindow) return;
    // ensure flex column so margin-left:auto works for user messages
    chatWindow.style.display = "flex";
    chatWindow.style.flexDirection = "column";
    const d = document.createElement("div");
    d.className = "msg " + (who==="user"?"user":"bot");
    d.style.alignSelf = who === "user" ? "flex-end" : "flex-start";
    d.innerText = text;
    chatWindow.appendChild(d);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  if (chatForm) {
    chatForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const input = el("#chatInput");
      const q = input && input.value && input.value.trim();
      if (!q) return;
      const use_llm = !!el("#enableLLM") && el("#enableLLM").checked;
      appendChat(q, "user");
      input.value = "";
      setLoading("#chatBtn", true, "Thinking...");
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: CURRENT_SID || "", query: q, use_llm })
        });
        if (!res.ok) {
          const txt = await res.text().catch(()=>"");
          throw new Error(txt || "chat failed");
        }
        const data = await res.json();
        if (data.session_id) updateSessionId(data.session_id);
        appendChat(data.answer || "No answer", "bot");
      } catch (err) {
        console.error("Chat error", err);
        appendChat("Error: " + (err.message || "Chat failed"), "bot");
      } finally {
        setLoading("#chatBtn", false);
      }
    });
  }

  // End session button
  const endBtn = el("#endSessionBtn");
  if (endBtn) {
    endBtn.addEventListener("click", async (ev) => {
      if (!confirm("End session and clear all uploaded data?")) return;
      const payload = JSON.stringify({ session_id: CURRENT_SID || "" });
      try {
        if (navigator.sendBeacon) {
          navigator.sendBeacon("/clear_session", payload);
        } else {
          await fetch("/clear_session", { method: "POST", headers: { "Content-Type":"application/json" }, body: payload });
        }
        showToast("Session cleared — reloading");
        setTimeout(()=>location.reload(), 600);
      } catch (e) {
        console.error("clear session failed", e);
        showToast("Failed to clear session");
      }
    });
  }

  // Auto-clear on unload 
  window.addEventListener("unload", (ev) => {
    try {
      const payload = JSON.stringify({ session_id: CURRENT_SID || "" });
      if (navigator.sendBeacon) navigator.sendBeacon("/clear_session", payload);
    } catch (e) { /* swallow */ }
  });

  // Init
  (function init() {
    if (!CURRENT_SID) showToast("Session missing — a new one will be created on upload.");
    try {
      if (window.CHURN_EDA_FULL_JSON && window.CHI_charts && typeof window.CHI_charts.renderEdaFull === "function") {
        window.CHI_charts.renderEdaFull(window.CHURN_EDA_FULL_JSON);
      }
    } catch (e) { /* ignore */ }
  })();

})();
