// static/main.js
// Frontend wiring for Churn and Retention analysis single-page webapp.

(() => {
  "use strict";

  // --- Helpers & selectors ---
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
              <div style="font-weight:700;font-size:16px;margin-top:6px">${escapeHtml(String(value))}</div>
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
  
  function renderMetricList(metrics) {
    if (!metrics) return "";
    let html = "<div style='margin-top:10px;display:flex;gap:8px;flex-wrap:wrap'>";
    for (const k of Object.keys(metrics)) {
      const v = metrics[k];
      html += renderKpiCard(k, (typeof v === "number" ? (Math.round(v*10000)/100) + (k.toLowerCase().includes("rate") || k.toLowerCase().includes("recall") || k.toLowerCase().includes("precision") ? "%" : "") : String(v)), null);
    }
    html += "</div>";
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
        }

        showToast("Upload complete");
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
              } else if (data.status === "failed") {
                  stopPolling();
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
          // KPI cards
          html += "<div style='display:flex;gap:10px;flex-wrap:wrap'>";
          html += renderKpiCard("Rows scored", data.n_rows || 0, "");
          html += renderKpiCard("Predicted churn", ((data.churn_rate||0)*100).toFixed(2) + "%", "");
          if (data.revenue_summary) {
            html += renderKpiCard("Avg revenue", formatCurrency(data.revenue_summary.avg_revenue), "");
            html += renderKpiCard("Total revenue", formatCurrency(data.revenue_summary.sum_revenue), "");
            // quick profit/loss estimate - placeholder using churn count * avg revenue
            try {
              const rows = data.n_rows || 0;
              const churnCount = Math.round((data.churn_rate||0) * rows);
              const revenueLost = (data.revenue_summary.sum_revenue || ((data.revenue_summary.avg_revenue||0) * churnCount));
              html += renderKpiCard("Est. revenue at risk", formatCurrency(revenueLost), `${churnCount} customers`);
            } catch(e){}
          }
          html += "</div>";

          // Preview table
          if (data.preview_html) {
            html += `<div style="margin-top:10px">${data.preview_html}</div>`;
          } else if (data.rows_sample) {
            html += renderSmallTable(data.rows_sample);
          }

          setResult("#predictResult", html);
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
          setResult("#simulateResult", jsonPretty(data.simulate));
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

  // --- Chat 
  const chatForm = el("#chatForm");
  const chatWindow = el("#chatWindow");
  function appendChat(text, who="bot") {
    if (!chatWindow) return;
    const d = document.createElement("div");
    d.className = "msg " + (who==="user"?"user":"bot");
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
