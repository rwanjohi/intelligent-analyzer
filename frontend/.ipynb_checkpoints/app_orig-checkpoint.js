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

const stagesEl       = document.getElementById("stages");
const progressBar    = document.getElementById("progressBarFill");

const todayDate      = document.getElementById("todayDate");

// ---------- Misc init ---------- //
todayDate.textContent = new Date().toLocaleDateString("en-US", {
  month: "long", day: "numeric", year: "numeric",
});

let selectedFile = null;
let charts = {};

// ---------- File handling ---------- //
function setFile(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
    filePill.hidden = false;
    analyzeBtn.disabled = false;
  } else {
    filePill.hidden = true;
    analyzeBtn.disabled = true;
    fileInput.value = "";
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
  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active", "done");
    s.querySelector(".stage-status").textContent = "queued";
  });
  progressBar.style.width = "0%";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
resetBtn.addEventListener("click", resetAll);
errorReset.addEventListener("click", resetAll);

// ---------- Progress simulation ---------- //
const STAGE_LABELS = ["Parsing & cleaning", "Predicting claim types", "Topic modeling", "Extracting locations", "Generating word clouds"];

function setStage(idx, status) {
  document.querySelectorAll(".stage").forEach((el, i) => {
    el.classList.remove("active");
    if (i < idx) {
      el.classList.add("done");
      el.querySelector(".stage-status").textContent = "done";
    }
    if (i === idx) {
      el.classList.add("active");
      el.querySelector(".stage-status").textContent = status || "running";
    }
  });
  progressBar.style.width = `${((idx + 0.5) / STAGE_LABELS.length) * 100}%`;
}

async function runProgress() {
  // Simulate visible stage progress while the backend works
  const delays = [600, 700, 900, 600, 700];
  for (let i = 0; i < STAGE_LABELS.length; i++) {
    setStage(i);
    await new Promise(r => setTimeout(r, delays[i]));
  }
}

// ---------- Analyze ---------- //
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  heroSection.hidden = true;
  errorSection.hidden = true;
  progressSection.hidden = false;
  resultsSection.hidden = true;

  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active", "done");
    s.querySelector(".stage-status").textContent = "queued";
  });
  progressBar.style.width = "0%";

  const formData = new FormData();
  formData.append("file", selectedFile);

  // Run progress UI in parallel with the actual upload
  const uiPromise = runProgress();
  let response, body;
  try {
    response = await fetch("/api/analyze", { method: "POST", body: formData });
    body = await response.json();
  } catch (err) {
    progressSection.hidden = true;
    showError("Network error — could not reach the analyzer. " + err.message);
    return;
  }

  await uiPromise;

  if (!response.ok) {
    progressSection.hidden = true;
    showError(body.detail || "Something went wrong during analysis.");
    return;
  }

  // Mark all stages done
  document.querySelectorAll(".stage").forEach(s => {
    s.classList.remove("active");
    s.classList.add("done");
    s.querySelector(".stage-status").textContent = "done";
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

  renderKpis(data.kpi);
  renderTopicChart(data.topic_counts);
  renderLocationChart(data.location_counts);
  renderWordclouds(data.topic_wordclouds, data.topic_counts, data.overall_wordcloud);
  renderTable(data.preview_rows, data.columns);

  // Smooth scroll into the results
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderKpis(kpi) {
  const row = document.getElementById("kpiRow");
  const items = [
    { eyebrow: "fig. i",   value: kpi.total_cases,        label: "claims analyzed",  italic: false },
    { eyebrow: "fig. ii",  value: kpi.total_locations,    label: "distinct locations", italic: true  },
    { eyebrow: "fig. iii", value: kpi.total_topics,       label: "themes surfaced",   italic: false },
    { eyebrow: "fig. iv",  value: kpi.total_claim_types,  label: "claim categories",  italic: true  },
  ];
  row.innerHTML = items.map(it => `
    <div class="kpi">
      <div class="kpi-eyebrow">${it.eyebrow}</div>
      <div class="kpi-value ${it.italic ? "italic" : ""}">${formatNum(it.value)}</div>
      <div class="kpi-label">${it.label}</div>
    </div>
  `).join("");
}

function formatNum(n) {
  return new Intl.NumberFormat("en-US").format(n || 0);
}

// Editorial palette for charts
const CHART_PALETTE = [
  "#7a1f1f",  // oxblood
  "#b8893f",  // gold
  "#5a5a2c",  // olive
  "#2d5959",  // teal
  "#c45a3a",  // terracotta
  "#561414",  // oxblood deep
  "#3d3830",  // ink soft
];

const CHART_DEFAULT = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "#1a1814",
      titleFont:  { family: "Inter Tight", size: 12, weight: 600 },
      bodyFont:   { family: "JetBrains Mono", size: 11 },
      padding: 10,
      cornerRadius: 2,
      displayColors: false,
    },
  },
  scales: {
    x: {
      grid: { display: false, drawBorder: false },
      ticks: {
        color: "#3d3830",
        font: { family: "JetBrains Mono", size: 10 },
        autoSkip: false,
        maxRotation: 35,
        minRotation: 0,
      },
    },
    y: {
      beginAtZero: true,
      grid: { color: "rgba(201,184,150,0.5)", drawBorder: false },
      ticks: {
        color: "#6b5f4f",
        font: { family: "JetBrains Mono", size: 10 },
      },
    },
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
        borderRadius: 2,
        barThickness: "flex",
        maxBarThickness: 60,
      }],
    },
    options: CHART_DEFAULT,
  });
}

function renderLocationChart(counts) {
  // Sort top 12, drop "Unknown" unless it's the only category
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
        backgroundColor: "#2d5959",
        hoverBackgroundColor: "#1a1814",
        borderRadius: 2,
        barThickness: "flex",
        maxBarThickness: 30,
      }],
    },
    options: {
      ...CHART_DEFAULT,
      indexAxis: "y",
      scales: {
        x: {
          beginAtZero: true,
          grid: { color: "rgba(201,184,150,0.5)", drawBorder: false },
          ticks: { color: "#6b5f4f", font: { family: "JetBrains Mono", size: 10 } },
        },
        y: {
          grid: { display: false, drawBorder: false },
          ticks: { color: "#3d3830", font: { family: "Fraunces", size: 12, style: "italic" } },
        },
      },
    },
  });
}

function renderWordclouds(topicWcs, topicCounts, overallWc) {
  const grid = document.getElementById("wordcloudGrid");
  let html = "";

  // Overall first
  // if (overallWc) {
  //   html += `
  //     <div class="wc-tile" style="grid-column: span 2;">
  //       <div class="wc-tile-title">
  //         <span>↳ overall</span>
  //         <span class="wc-tile-count">all topics</span>
  //       </div>
  //       <img src="${overallWc}" alt="Overall word cloud" />
  //     </div>`;
  // }

  for (const [topic, dataUrl] of Object.entries(topicWcs)) {
    const count = topicCounts[topic] || 0;
    html += `
      <div class="wc-tile">
        <div class="wc-tile-title">
          <span>${topic}</span>
          <span class="wc-tile-count">${count} ${count === 1 ? "claim" : "claims"}</span>
        </div>
        <img src="${dataUrl}" alt="${topic} word cloud" />
      </div>`;
  }

  grid.innerHTML = html;
}

function renderTable(rows, columns) {
  const head = document.getElementById("tableHead");
  const body = document.getElementById("tableBody");

  // Reorder columns: surface predicted_claim_type / topic / extracted_location near the front
  const surfaced = ["predicted_claim_type", "topic", "extracted_location"];
  const ordered = [
    ...surfaced.filter(c => columns.includes(c)),
    ...columns.filter(c => !surfaced.includes(c)),
  ];

  head.innerHTML = ordered.map(c => `<th>${c}</th>`).join("");
  body.innerHTML = rows.map(r => {
    return "<tr>" + ordered.map(c => {
      const v = r[c] ?? "";
      let cls = "";
      if (c === "predicted_claim_type") cls = "col-pred";
      if (c === "extracted_location") cls = "col-loc";
      return `<td class="${cls}" title="${escapeHtml(String(v))}">${escapeHtml(String(v))}</td>`;
    }).join("") + "</tr>";
  }).join("");
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
