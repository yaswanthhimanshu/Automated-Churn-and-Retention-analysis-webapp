// static/charts.js
// Lightweight EDA visualizer (no external chart libs).
// Expects eda_full JSON produced by core.generate_full_report().

(function () {
  "use strict";

  function el(sel) { return document.querySelector(sel); }

  function fmt(num, digits = 3) {
    if (num === null || num === undefined || Number.isNaN(num)) return "-";
    return Number(num).toFixed(digits);
  }

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  // Color scale for correlation values (-1..1)
  function corrColor(val) {
    // val in [-1,1]
    if (val === null || val === undefined || Number.isNaN(val)) return "#222";
    const v = clamp(Math.abs(val), 0, 1);
    // strong -> brighter
    const hue = val >= 0 ? 190 : 350; // teal for positive, pink for negative
    const lightness = Math.round(60 - (v * 35));
    const sat = Math.round(30 + (v * 60));
    return `hsl(${hue} ${sat}% ${lightness}%)`;
  }

  // Render numeric summary cards
  function renderNumericSummary(container, numeric_summary) {
    if (!numeric_summary) return;
    const keys = Object.keys(numeric_summary || {});
    if (!keys.length) return;

    const wrap = document.createElement("div");
    wrap.style.display = "grid";
    wrap.style.gridTemplateColumns = "repeat(auto-fit,minmax(160px,1fr))";
    wrap.style.gap = "10px";
    wrap.style.marginBottom = "12px";

    keys.forEach(col => {
      const stats = numeric_summary[col];
      const card = document.createElement("div");
      card.style.padding = "10px";
      card.style.borderRadius = "8px";
      card.style.background = "linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00))";
      card.style.border = "1px solid rgba(255,255,255,0.03)";
      card.innerHTML = `<div style="font-weight:700;color:#fff;margin-bottom:6px">${col}</div>
        <div style="font-size:13px;color:var(--muted)">count: ${stats.count || "-"}</div>
        <div style="font-size:13px;color:var(--muted)">mean: ${fmt(stats.mean)}</div>
        <div style="font-size:13px;color:var(--muted)">std: ${fmt(stats.std)}</div>
        <div style="font-size:13px;color:var(--muted)">min: ${fmt(stats.min)}</div>
        <div style="font-size:13px;color:var(--muted)">max: ${fmt(stats.max)}</div>`;
      wrap.appendChild(card);
    });

    container.appendChild(wrap);
  }

  // Render correlation table (colored cells)
  function renderCorrelation(container, correlation) {
    if (!correlation || Object.keys(correlation).length === 0) return;
    const cols = Object.keys(correlation);
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.style.marginBottom = "12px";
    table.style.fontSize = "12px";

    // header
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    hr.appendChild(document.createElement("th")); // corner blank
    cols.forEach(c => {
      const th = document.createElement("th");
      th.textContent = c;
      th.style.padding = "6px";
      th.style.textAlign = "left";
      th.style.color = "var(--muted)";
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    // body
    const tbody = document.createElement("tbody");
    cols.forEach(r => {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      th.textContent = r;
      th.style.padding = "6px";
      th.style.textAlign = "left";
      th.style.color = "var(--muted)";
      tr.appendChild(th);

      cols.forEach(c => {
        const td = document.createElement("td");
        td.style.padding = "6px";
        let val = correlation[r] && correlation[r][c];
        if (val === undefined || val === null) {
          td.textContent = "-";
        } else {
          const v = Number(val);
          td.textContent = fmt(v, 2);
          td.style.background = corrColor(v);
          td.style.color = (Math.abs(v) > 0.45) ? "#021124" : "#e6eef6";
          td.style.borderRadius = "6px";
          td.style.textAlign = "center";
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.appendChild(table);
  }

  // Render categorical top value bars (simple)
  function renderCategoricalTop(container, top_categories) {
    if (!top_categories) return;
    const keys = Object.keys(top_categories);
    if (!keys.length) return;

    // Limit number of columns to show for performance
    const showCols = keys.slice(0, 6);

    showCols.forEach(col => {
      const vals = top_categories[col];
      const wrap = document.createElement("div");
      wrap.style.marginBottom = "12px";
      const title = document.createElement("div");
      title.textContent = col;
      title.style.fontWeight = "700";
      title.style.marginBottom = "6px";
      title.style.color = "#fff";
      wrap.appendChild(title);

      const items = Object.entries(vals).slice(0, 10);
      const maxVal = Math.max(...items.map(i => i[1] || 0), 0.00001);

      items.forEach(([k, v]) => {
        const line = document.createElement("div");
        line.style.display = "flex";
        line.style.alignItems = "center";
        line.style.gap = "8px";
        line.style.marginBottom = "6px";

        const label = document.createElement("div");
        label.style.width = "120px";
        label.style.fontSize = "13px";
        label.style.color = "var(--muted)";
        label.textContent = k.length > 30 ? (k.slice(0, 27) + "...") : k;

        const barWrap = document.createElement("div");
        barWrap.style.flex = "1";
        barWrap.style.background = "rgba(255,255,255,0.02)";
        barWrap.style.borderRadius = "6px";
        barWrap.style.overflow = "hidden";
        barWrap.style.height = "14px";

        const bar = document.createElement("div");
        bar.style.height = "100%";
        const pct = (v / maxVal) * 100;
        bar.style.width = `${pct}%`;
        bar.style.borderRadius = "6px";
        bar.style.background = "linear-gradient(90deg,var(--accent),#7c3aed)";

        const valSpan = document.createElement("div");
        valSpan.style.width = "60px";
        valSpan.style.fontSize = "13px";
        valSpan.style.color = "var(--muted)";
        valSpan.textContent = (Math.round((v * 1000) / 10) / 100) + "%";

        barWrap.appendChild(bar);
        line.appendChild(label);
        line.appendChild(barWrap);
        line.appendChild(valSpan);
        wrap.appendChild(line);
      });

      container.appendChild(wrap);
    });
  }

  // Clear and render EDA result object into #edaFullResult
  function renderEdaFull(report) {
    const root = el("#edaFullResult");
    if (!root) return;
    root.innerHTML = ""; // clear
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.justifyContent = "space-between";
    header.style.alignItems = "center";
    header.style.marginBottom = "10px";

    const title = document.createElement("div");
    title.innerHTML = "<strong style='color:#fff'>Full EDA</strong>";
    header.appendChild(title);

    const meta = document.createElement("div");
    meta.style.color = "var(--muted)";
    meta.style.fontSize = "13px";
    meta.textContent = `rows: ${report.n_rows || "-"}, cols: ${report.n_cols || "-"}, missing total: ${report.missing_total || 0}`;
    header.appendChild(meta);

    root.appendChild(header);

    // numeric summary cards
    if (report.numeric_summary) {
      renderNumericSummary(root, report.numeric_summary);
    }

    // correlation
    if (report.correlation && Object.keys(report.correlation).length) {
      const sec = document.createElement("div");
      sec.style.marginTop = "8px";
      sec.innerHTML = "<div style='font-weight:700;margin-bottom:6px;color:#fff'>Correlation matrix</div>";
      root.appendChild(sec);
      renderCorrelation(root, report.correlation);
    }

    // top categories
    if (report.top_categories) {
      const sec2 = document.createElement("div");
      sec2.style.marginTop = "8px";
      sec2.innerHTML = "<div style='font-weight:700;margin-bottom:6px;color:#fff'>Top categories (sample)</div>";
      root.appendChild(sec2);
      renderCategoricalTop(root, report.top_categories);
    }

    // fallbacks raw JSON if nothing else
    if (!report.correlation && !report.top_categories && !report.numeric_summary) {
      const pre = document.createElement("pre");
      pre.style.whiteSpace = "pre-wrap";
      pre.style.color = "var(--muted)";
      pre.textContent = JSON.stringify(report, null, 2);
      root.appendChild(pre);
    }
  }

  // expose API
  window.CHI_charts = {
    renderEdaFull
  };

  // auto-runs if page has eda JSON embedded in #edaFullResult data attribute
  document.addEventListener("DOMContentLoaded", function () {
    try {
      const raw = window._CHURN_EDA_FULL || null;
      if (raw && typeof raw === "object") {
        renderEdaFull(raw);
      }
      // also if server placed a global var CHURN_EDA_FULL_JSON
      if (window.CHURN_EDA_FULL_JSON) {
        renderEdaFull(window.CHURN_EDA_FULL_JSON);
      }
    } catch (e) {
      // ignore
      console.warn("charts init error", e);
    }
  });

})();
