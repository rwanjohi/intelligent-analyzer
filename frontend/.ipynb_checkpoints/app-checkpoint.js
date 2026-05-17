/* ===========================================
   Claims Atlas — frontend logic
   =========================================== */

// ---------- DOM refs ---------- //
const dropzone     = document.getElementById("dropzone");
const fileInput    = document.getElementById("fileInput");
const browseBtn    = document.getElementById("browseBtn");
const filePill     = document.getElementById("filePill");
const fileName     = document.getElementById("fileName");
const fileClear    = document.getElementById("fileClear");
const analyzeBtn   = document.getElementById("analyzeBtn");

const heroSection      = document.getElementById("heroSection");
const progressSection  = document.getElementById("progressSection");
const resultsSection   = document.getElementById("resultsSection");
const errorSection     = document.getElementById("errorSection");
const errorText        = document.getElementById("errorText");
const errorReset       = document.getElementById("errorReset");
const resetBtn         = document.getElementById("resetBtn");
const downloadBtn      = document.getElementById("downloadBtn");
const resultsSubtitle  = document.getElementById("resultsSubtitle");

const progressBar      = document.getElementById("progressBarFill");
const todayDate        = document.getElementById("todayDate");

// Source tabs
const sourceTabs       = document.querySelectorAll(".source-tab");
const sourcePanels     = document.querySelectorAll(".source-panel");
const remoteTab        = document.getElementById("remoteTab");
const remoteHost       = document.getElementById("remoteHost");
const remoteHostInline = document.getElementById("remoteHostInline");

// Topics control
const topicsRange      = document.getElementById("topicsRange");
const topicsValue      = document.getElementById("topicsValue");

todayDate.textContent = new Date().toLocaleDateString("en-US", {
  month: "long", day: "numeric", year: "numeric",
});

let selectedFile = null;
let charts = {};
let lastResults = null;
let activeSource = "file";   // "file" | "remote"

// ---------- Init: check whether remote is configured ---------- //
(async function initRemoteCapability() {
  try {
    const r = await fetch("/api/remote-config");
    if (!r.ok) return;
    const cfg = await r.json();
    if (cfg.remote_configured) {
  remoteTab.hidden = false;
  const host = cfg.remote_url_hint || "configured";
  if (remoteHostInline) remoteHostInline.textContent = host;
}
  } catch (e) {
    // No remote configured — file-upload tab stays as only option
  }
})();

// ---------- Source-tab switching ---------- //
sourceTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    sourceTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    activeSource = tab.dataset.source;
    sourcePanels.forEach(p => {
      p.hidden = p.dataset.source !== activeSource;
    });
    updateAnalyzeEnabled();
  });
});

// ---------- Topics slider ---------- //
topicsRange.addEventListener("input", () => {
  topicsValue.textContent = topicsRange.value;
});

// ---------- File handling ---------- //
function setFile(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
    filePill.hidden = false;
  } else {
    filePill.hidden = true;
    fileInput.value = "";
  }
  updateAnalyzeEnabled();
}

function updateAnalyzeEnabled() {
  if (activeSource === "file") {
    analyzeBtn.disabled = !selectedFile;
  } else if (activeSource === "remote") {
    analyzeBtn.disabled = false;
  }
}

dropzone.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON" || e.target.tagName === "A") return;
  fileInput.click();
});
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});
["dragenter", "dragover"].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(ev =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); })
);
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
fileClear.addEventListener("click", (e) => { e.stopPropagation(); setFile(null); });

// ---------- Reset ---------- //
function resetAll() {
  setFile(null);
  resultsSection.hidden = true;
  errorSection.hidden = true;
  progressSection.hidden = true;
  heroSection.hidden = false;
  Object.values(charts).forEach(c => c && c.destroy && c.destroy());
  charts = {};
  lastResults = null;
  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active", "done");
    s.querySelector(".stage-status").textContent = "Queued";
  });
  progressBar.style.width = "0%";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
resetBtn.addEventListener("click", resetAll);
errorReset.addEventListener("click", resetAll);

// ---------- Download CSV ---------- //
downloadBtn.addEventListener("click", () => {
  if (!lastResults) return;
  const rows = lastResults.all_rows || lastResults.preview_rows || [];
  const cols = lastResults.columns || [];
  if (!rows.length || !cols.length) return;

  const surfaced = ["predicted_claim_type", "actual_claim_type",
                    "match_predicted_vs_actual", "topic",
                    "extracted_location", "extracted_state_code"];
  const ordered = [
    ...surfaced.filter(c => cols.includes(c)),
    ...cols.filter(c => !surfaced.includes(c)),
  ];
  const escape = v => {
    const s = String(v ?? "");
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [
    ordered.join(","),
    ...rows.map(r => ordered.map(c => escape(r[c])).join(","))
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `claims_analyzed_${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// ---------- Progress simulation ---------- //
const STAGE_LABELS = ["Fetching & cleaning", "Predicting claim types", "Topic modeling", "Extracting locations", "Generating word clouds"];

function setStage(idx, status) {
  document.querySelectorAll(".stage").forEach((el, i) => {
    el.classList.remove("active");
    if (i < idx) {
      el.classList.add("done");
      el.querySelector(".stage-status").textContent = "Done";
    }
    if (i === idx) {
      el.classList.add("active");
      el.querySelector(".stage-status").textContent = status || "Running";
    }
  });
  progressBar.style.width = `${((idx + 0.5) / STAGE_LABELS.length) * 100}%`;
}

async function runProgress() {
  const delays = [600, 700, 900, 600, 700];
  for (let i = 0; i < STAGE_LABELS.length; i++) {
    setStage(i);
    await new Promise(r => setTimeout(r, delays[i]));
  }
}

// ---------- Analyze ---------- //
analyzeBtn.addEventListener("click", async () => {
  if (activeSource === "file" && !selectedFile) return;

  heroSection.hidden = true;
  errorSection.hidden = true;
  progressSection.hidden = false;
  resultsSection.hidden = true;

  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active", "done");
    s.querySelector(".stage-status").textContent = "Queued";
  });
  progressBar.style.width = "0%";

  const n_topics = parseInt(topicsRange.value, 10) || 5;
  const uiPromise = runProgress();

  let response, body;
  try {
    if (activeSource === "file") {
      const formData = new FormData();
      formData.append("file", selectedFile);
      response = await fetch(`/api/analyze?n_topics=${n_topics}`, {
        method: "POST",
        body: formData,
      });
    } else {
      response = await fetch(`/api/analyze-remote?n_topics=${n_topics}`, {
        method: "GET",
      });
    }
    body = await response.json();
  } catch (err) {
    progressSection.hidden = true;
    showError("Network error — could not reach the analyzer. " + err.message);
    return;
  }

  await uiPromise;

  if (!response.ok) {
  // Mark the active stage as failed instead of letting the UI claim success
  const activeStage = document.querySelector(".stage.active");
  if (activeStage) {
    activeStage.classList.remove("active");
    const status = activeStage.querySelector(".stage-status");
    if (status) status.textContent = "Failed";
  }
  await new Promise(r => setTimeout(r, 300));
  progressSection.hidden = true;
  showError(body.detail || "Something went wrong during analysis.");
  return;
}

  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active");
    s.classList.add("done");
    s.querySelector(".stage-status").textContent = "Done";
  });
  progressBar.style.width = "100%";

  await new Promise(r => setTimeout(r, 350));
  progressSection.hidden = true;
  renderResults(body);
});

function showError(msg) {
  heroSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = false;
  errorText.textContent = msg;
}

// ---------- Render ---------- //
function renderResults(data) {
  resultsSection.hidden = false;
  lastResults = data;

  resultsSubtitle.textContent = data.source === "remote"
    ? "Analysis complete · live from remote source"
    : "Analysis complete";

  renderKpis(data.kpi);
  renderTopicChart(data.topic_counts);
  renderLocationChart(data.location_counts);
  renderHeatmap(data.state_code_counts || {});
  renderWordclouds(data.topic_wordclouds, data.topic_counts);
  renderTable(data.preview_rows, data.columns);

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderKpis(kpi) {
  const row = document.getElementById("kpiRow");
  const items = [
    { eyebrow: "Total claims",  value: kpi.total_cases,        label: "claims analyzed",   accent: false },
    { eyebrow: "Locations",     value: kpi.total_locations,    label: "distinct states",   accent: true  },
    { eyebrow: "Themes",        value: kpi.total_topics,       label: "topics surfaced",   accent: false },
    { eyebrow: "Categories",    value: kpi.total_claim_types,  label: "claim types",       accent: true  },
  ];
  if (kpi.has_ground_truth && kpi.match_accuracy !== null && kpi.match_accuracy !== undefined) {
    items[3] = {
      eyebrow: "Match accuracy",
      value: (kpi.match_accuracy * 100).toFixed(1) + "%",
      label: "predicted vs. actual",
      accent: true,
      raw: true,
    };
  }
  row.innerHTML = items.map(it => `
    <div class="kpi">
      <div class="kpi-eyebrow">${it.eyebrow}</div>
      <div class="kpi-value ${it.accent ? "accent" : ""}">${it.raw ? it.value : formatNum(it.value)}</div>
      <div class="kpi-label">${it.label}</div>
    </div>
  `).join("");
}

function formatNum(n) {
  return new Intl.NumberFormat("en-US").format(n || 0);
}

const CHART_PALETTE = [
  "#2d5a3d", "#2d7d9a", "#c47a1a", "#1a7a3a",
  "#7c5fa6", "#b5392a", "#6b6b64",
];

const CHART_DEFAULT = {
  responsive: true, maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "#1a1a1a",
      titleFont:  { family: "Segoe UI", size: 12, weight: 600 },
      bodyFont:   { family: "Segoe UI", size: 12 },
      padding: 10, cornerRadius: 6, displayColors: false,
    },
  },
  scales: {
    x: { grid: { display: false, drawBorder: false },
         ticks: { color: "#6b6b64", font: { family: "Segoe UI", size: 11 }, autoSkip: false, maxRotation: 35 } },
    y: { beginAtZero: true,
         grid: { color: "#e5e5de", drawBorder: false },
         ticks: { color: "#9c9c94", font: { family: "Segoe UI", size: 11 } } },
  },
};

function renderTopicChart(counts) {
  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const ctx = document.getElementById("topicChart").getContext("2d");
  if (charts.topic) charts.topic.destroy();
  charts.topic = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.map(l => l.length > 26 ? l.slice(0, 24) + "…" : l),
      datasets: [{
        data: values,
        backgroundColor: labels.map((_, i) => CHART_PALETTE[i % CHART_PALETTE.length]),
        borderRadius: 6, barThickness: "flex", maxBarThickness: 60,
      }],
    },
    options: CHART_DEFAULT,
  });
}

function renderLocationChart(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const filtered = entries.filter(([k]) => k !== "Unknown");
  const top = (filtered.length ? filtered : entries).slice(0, 12);
  const ctx = document.getElementById("locationChart").getContext("2d");
  if (charts.location) charts.location.destroy();
  charts.location = new Chart(ctx, {
    type: "bar",
    data: {
      labels: top.map(([k]) => k),
      datasets: [{
        data: top.map(([, v]) => v),
        backgroundColor: "#2d7d9a",
        hoverBackgroundColor: "#1a4d63",
        borderRadius: 6, barThickness: "flex", maxBarThickness: 30,
      }],
    },
    options: {
      ...CHART_DEFAULT,
      indexAxis: "y",
      scales: {
        x: { beginAtZero: true,
             grid: { color: "#e5e5de", drawBorder: false },
             ticks: { color: "#9c9c94", font: { family: "Segoe UI", size: 11 } } },
        y: { grid: { display: false, drawBorder: false },
             ticks: { color: "#1a1a1a", font: { family: "Segoe UI", size: 12, weight: 500 } } },
      },
    },
  });
}

// ---------- US heatmap via Plotly choropleth ---------- //
function renderHeatmap(stateCounts) {
  const heatmapDiv = document.getElementById("usHeatmap");
  if (!heatmapDiv) return;

  const codes = Object.keys(stateCounts);
  const values = Object.values(stateCounts);

  if (!codes.length) {
    heatmapDiv.innerHTML = `
      <div style="display:flex;height:100%;align-items:center;justify-content:center;color:#9c9c94;text-align:center;padding:20px;">
        <div>
          <div style="font-size:14px;font-weight:600;margin-bottom:6px;">No state-level locations extracted</div>
          <div style="font-size:12.5px;">The descriptions in this batch didn't contain identifiable US state references.</div>
        </div>
      </div>`;
    return;
  }

  const data = [{
    type: "choropleth",
    locationmode: "USA-states",
    locations: codes,
    z: values,
    colorscale: [
      [0,    "#f0f6f2"],
      [0.25, "#a8c8b6"],
      [0.5,  "#6da38a"],
      [0.75, "#3a7d5b"],
      [1,    "#2d5a3d"],
    ],
    autocolorscale: false,
    marker: { line: { color: "white", width: 1 } },
    colorbar: {
      title: { text: "Claims", side: "right", font: { family: "Segoe UI", size: 12, color: "#6b6b64" } },
      tickfont: { family: "Segoe UI", size: 11, color: "#6b6b64" },
      thickness: 14,
      len: 0.75,
      outlinewidth: 0,
    },
    hovertemplate: "<b>%{location}</b><br>%{z} claims<extra></extra>",
  }];

  const layout = {
    geo: {
      scope: "usa",
      projection: { type: "albers usa" },
      showlakes: true,
      lakecolor: "#f5f5f0",
      bgcolor: "#ffffff",
      showland: true,
      landcolor: "#fafaf7",
      showcoastlines: false,
      showsubunits: true,
      subunitcolor: "white",
    },
    margin: { l: 0, r: 0, t: 8, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { family: "Segoe UI", color: "#1a1a1a" },
  };

  const config = { displayModeBar: false, responsive: true };
  Plotly.newPlot(heatmapDiv, data, layout, config);
}

function renderWordclouds(topicWcs, topicCounts) {
  const grid = document.getElementById("wordcloudGrid");
  let html = "";
  for (const [topic, dataUrl] of Object.entries(topicWcs)) {
    const count = topicCounts[topic] || 0;
    html += `
      <div class="wc-tile">
        <div class="wc-tile-title">
          <span>${escapeHtml(topic)}</span>
          <span class="wc-tile-count">${count} ${count === 1 ? "claim" : "claims"}</span>
        </div>
        <img src="${dataUrl}" alt="${escapeHtml(topic)} word cloud" />
      </div>`;
  }
  grid.innerHTML = html;
}

function renderTable(rows, columns) {
  const head = document.getElementById("tableHead");
  const body = document.getElementById("tableBody");

  const surfaced = ["predicted_claim_type", "actual_claim_type",
                    "match_predicted_vs_actual", "topic", "extracted_location"];
  const hidden = new Set(["extracted_state_code"]);
  const visibleCols = columns.filter(c => !hidden.has(c));
  const ordered = [
    ...surfaced.filter(c => visibleCols.includes(c)),
    ...visibleCols.filter(c => !surfaced.includes(c)),
  ];

  const headerLabel = (c) => {
    switch (c) {
      case "predicted_claim_type":      return "Predicted";
      case "actual_claim_type":         return "Actual";
      case "match_predicted_vs_actual": return "Match";
      case "topic":                     return "Topic";
      case "extracted_location":        return "Location";
      default: return c;
    }
  };

  head.innerHTML = ordered.map(c => `<th>${escapeHtml(headerLabel(c))}</th>`).join("");
  body.innerHTML = rows.map(r => {
    return "<tr>" + ordered.map(c => {
      const v = r[c] ?? "";
      const safe = escapeHtml(String(v));
      if (c === "predicted_claim_type") return `<td class="col-pred" title="${safe}">${safe}</td>`;
      if (c === "actual_claim_type")    return `<td class="col-actual" title="${safe}">${safe}</td>`;
      if (c === "match_predicted_vs_actual") {
        const cls = v === "Y" ? "match-y" : (v === "N" ? "match-n" : "");
        return `<td class="col-match"><span class="${cls}">${safe || "—"}</span></td>`;
      }
      if (c === "extracted_location") return `<td class="col-loc" title="${safe}">${safe}</td>`;
      return `<td title="${safe}">${safe}</td>`;
    }).join("") + "</tr>";
  }).join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
