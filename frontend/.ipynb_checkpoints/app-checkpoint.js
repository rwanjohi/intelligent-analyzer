/* ==========================================================================
   Claims Atlas — Frontend Controller and Data Orchestration Logic
   ========================================================================== */

// --- Register Chart-Geo Components for Chart.js v4 Compatibility ---
// In Chart.js v4+, third-party modules must be explicitly registered to instantiate controllers
if (typeof Chart !== 'undefined' && typeof ChartGeo !== 'undefined') {
  Chart.register(
    ChartGeo.ChoroplethController, 
    ChartGeo.GeoFeature, 
    ChartGeo.ColorScale, 
    ChartGeo.ProjectionScale
  );
}

// --- Component DOM Target Reference Pointers ---
const dropzone         = document.getElementById("dropzone");
const fileInput        = document.getElementById("fileInput");
const browseBtn        = document.getElementById("browseBtn");
const filePill         = document.getElementById("filePill");
const fileName         = document.getElementById("fileName");
const fileClear        = document.getElementById("fileClear");
const analyzeBtn       = document.getElementById("analyzeBtn");

const remoteTriggerBtn = document.getElementById("remoteTriggerBtn");
const numTopicsInput   = document.getElementById("numTopics");

const heroSection      = document.getElementById("heroSection");
const progressSection  = document.getElementById("progressSection");
const resultsSection   = document.getElementById("resultsSection");
const errorSection     = document.getElementById("errorSection");
const errorText        = document.getElementById("errorText");
const errorReset       = document.getElementById("errorReset");
const resetBtn         = document.getElementById("resetBtn");
const downloadBtn      = document.getElementById("downloadBtn");

const progressBar      = document.getElementById("progressBarFill");
const todayDate        = document.getElementById("todayDate");

// --- UI Framework Datetime Stamp Syncing ---
todayDate.textContent = new Date().toLocaleDateString("en-US", {
  month: "long", day: "numeric", year: "numeric",
});

let selectedFile = null;
let charts = {};
let lastResults = null;
let cachedUsFeaturesGeoJson = null;

/**
 * Prefetches and buffers continental topological vectors maps globally for geographical charting
 */
async function getUnitedStatesTopoJsonFeatures() {
  if (cachedUsFeaturesGeoJson) return cachedUsFeaturesGeoJson;
  try {
    const response = await fetch("https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json");
    const topoData = await response.json();
    cachedUsFeaturesGeoJson = ChartGeo.topojson.feature(topoData, topoData.objects.states).features;
    return cachedUsFeaturesGeoJson;
  } catch (err) {
    console.error("Critical Mapping Error: Unable to fetch layout coordinates profiles", err);
    return [];
  }
}
getUnitedStatesTopoJsonFeatures(); // Prime spatial maps model database on boot initialization

/**
 * Syncs staging state for local files provided to Option A input dropzone
 */
function applyFileStateSelection(file) {
  selectedFile = file;
  if (file) {
    fileName.textContent = `${file.name}  ·  (${(file.size / 1024).toFixed(1)} KB)`;
    filePill.hidden = false;
    analyzeBtn.disabled = false; // Enable local analysis runtime trigger
  } else {
    filePill.hidden = true;
    fileInput.value = "";
    analyzeBtn.disabled = true;
  }
}

// Drag, drop, and standard browsing attachment hooks
dropzone.addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON" || e.target.tagName === "A") return;
  fileInput.click();
});
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", (e) => { if (e.target.files[0]) applyFileStateSelection(e.target.files[0]); });

["dragenter", "dragover"].forEach(evName => {
  dropzone.addEventListener(evName, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
});
["dragleave", "drop"].forEach(evName => {
  dropzone.addEventListener(evName, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); });
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) applyFileStateSelection(e.dataTransfer.files[0]);
});

fileClear.addEventListener("click", (e) => { e.stopPropagation(); applyFileStateSelection(null); });

/**
 * Resets application workbench metrics back to system default baseline states
 */
function executeFullDashboardReset() {
  applyFileStateSelection(null);
  numTopicsInput.value = "5";
  resultsSection.hidden = true;
  errorSection.hidden = true;
  progressSection.hidden = true;
  heroSection.hidden = false;
  
  Object.keys(charts).forEach(chartKey => {
    if (charts[chartKey]) {
      charts[chartKey].destroy();
      charts[chartKey] = null;
    }
  });
  charts = {};
  lastResults = null;
  
  document.querySelectorAll(".stage").forEach(el => {
    el.classList.remove("active", "done");
    el.querySelector(".stage-status").textContent = "Queued";
  });
  progressBar.style.width = "0%";
  window.scrollTo({ top: 0, behavior: "smooth" });
}
resetBtn.addEventListener("click", executeFullDashboardReset);
errorReset.addEventListener("click", executeFullDashboardReset);

/**
 * Safe CSV compilation matrix handler for user results downloads
 */
downloadBtn.addEventListener("click", () => {
  if (!lastResults || !lastResults.all_rows) return;
  const entries = lastResults.all_rows;
  const headers = lastResults.columns;
  if (!entries.length || !headers.length) return;

  const priorityKeys = ["predicted_claim_type", "actual_claim_type", "match_predicted_vs_actual", "topic", "extracted_location"];
  const compiledOrder = [
    ...priorityKeys.filter(k => headers.includes(k)),
    ...headers.filter(k => !priorityKeys.includes(k))
  ];

  const captureSanitizedString = val => {
    const cleanStr = String(val ?? "");
    return /[",\n\r]/.test(cleanStr) ? `"${cleanStr.replace(/"/g, '""')}"` : cleanStr;
  };

  const csvPayload = [
    compiledOrder.join(","),
    ...entries.map(rowObj => compiledOrder.map(headKey => captureSanitizedString(rowObj[headKey])).join(","))
  ].join("\n");

  const blobTarget = new Blob([csvPayload], { type: "text/csv;charset=utf-8;" });
  const urlBlobString = URL.createObjectURL(blobTarget);
  const virtualAnchor = document.createElement("a");
  virtualAnchor.href = urlBlobString;
  virtualAnchor.download = `claims_atlas_processed_${new Date().toISOString().slice(0,10)}.csv`;
  document.body.appendChild(virtualAnchor);
  virtualAnchor.click();
  document.body.removeChild(virtualAnchor);
  URL.revokeObjectURL(urlBlobString);
});

// --- Dynamic Progress Metrics Feedback Loader UI ---
const STAGES_LIST = ["Parsing & cleaning", "Predicting claim types", "Dynamic Topic modeling", "Extracting locations", "Generating word clouds"];

function synchronizeLoadingStageView(index, indicatorText) {
  document.querySelectorAll(".stage").forEach((element, i) => {
    element.classList.remove("active");
    if (i < index) {
      element.classList.add("done");
      element.querySelector(".stage-status").textContent = "Done";
    } else if (i === index) {
      element.classList.add("active");
      element.querySelector(".stage-status").textContent = indicatorText || "Running";
    }
  });
  progressBar.style.width = `${((index + 0.5) / STAGES_LIST.length) * 100}%`;
}

async function triggerFrontendProgressSimulation() {
  const executionDelayIntervals = [500, 600, 850, 500, 600];
  for (let i = 0; i < STAGES_LIST.length; i++) {
    synchronizeLoadingStageView(i);
    await new Promise(resolve => setTimeout(resolve, executionDelayIntervals[i]));
  }
}

/**
 * Standard Processing Workflow Pipeline Launcher Data Dispatcher
 */
async function launchAnalyticalPipelineRun(isRemoteExecution) {
  if (!isRemoteExecution && !selectedFile) return;

  // Clear states and shift viewports immediately into loading view
  heroSection.hidden = true;
  errorSection.hidden = true;
  progressSection.hidden = false;
  resultsSection.hidden = true;

  const uploadFormPayload = new FormData();
  uploadFormPayload.append("num_topics", numTopicsInput.value || 5);
  
  if (isRemoteExecution) {
    uploadFormPayload.append("use_remote", "true");
  } else {
    uploadFormPayload.append("use_remote", "false");
    uploadFormPayload.append("file", selectedFile);
  }

  const networkUiEngineTracker = triggerFrontendProgressSimulation();
  let serverRawResult, payloadJson;
  
  try {
    serverRawResult = await fetch("/api/analyze", { method: "POST", body: uploadFormPayload });
    payloadJson = await serverRawResult.json();
  } catch (err) {
    progressSection.hidden = true;
    displayRuntimeErrorCard("Network Link Failure: Server cluster communication crashed. " + err.message);
    return;
  }

  await networkUiEngineTracker; // Ensure loader visual steps settle

  if (!serverRawResult.ok) {
    progressSection.hidden = true;
    displayRuntimeErrorCard(payloadJson.detail || "Processing Engine faulted during pipeline runtime execution.");
    return;
  }

  document.querySelectorAll(".stage").forEach(el => {
    el.classList.remove("active");
    el.classList.add("done");
    el.querySelector(".stage-status").textContent = "Done";
  });
  progressBar.style.width = "100%";

  await new Promise(r => setTimeout(r, 300));
  progressSection.hidden = true;
  renderCompiledDashboardOutput(payloadJson);
}

// Map Action execution button trigger hooks
analyzeBtn.addEventListener("click", () => launchAnalyticalPipelineRun(false));
remoteTriggerBtn.addEventListener("click", () => launchAnalyticalPipelineRun(true));

function displayRuntimeErrorCard(errorMessageText) {
  heroSection.hidden = true;
  resultsSection.hidden = true;
  errorSection.hidden = false;
  errorText.textContent = errorMessageText;
}

// --- Graphical Analytics View Layer Rendering Builders ---
function renderCompiledDashboardOutput(dataPackage) {
  resultsSection.hidden = false;
  lastResults = dataPackage;

  renderScorecardKpis(dataPackage.kpi);
  renderTopicDistributionsChart(dataPackage.topic_counts);
  renderRegionalLocationsChart(dataPackage.location_counts);
  renderUSGeographicHeatmap(dataPackage.location_counts);
  renderDynamicWordCloudsGrid(dataPackage.topic_wordclouds, dataPackage.topic_counts);
  renderPreviewSpreadsheetGrid(dataPackage.preview_rows, dataPackage.columns);

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderScorecardKpis(kpiObj) {
  const scorecardTargetRow = document.getElementById("kpiRow");
  const scoreCardConfigurationList = [
    { label: "Total Cases", value: kpiObj.total_cases, detail: "records analyzed", highlight: false },
    { label: "Geographies", value: kpiObj.total_locations, detail: "distinct US states", highlight: true },
    { label: "Surfaced Themes", value: kpiObj.total_topics, detail: "dynamic topics", highlight: false },
    { label: "Claim Families", value: kpiObj.total_claim_types, detail: "classified profiles", highlight: true }
  ];

  if (kpiObj.has_ground_truth && kpiObj.match_accuracy !== null) {
    scoreCardConfigurationList[3] = {
      label: "Prediction Match Accuracy",
      value: (kpiObj.match_accuracy * 100).toFixed(1) + "%",
      detail: "predicted vs ground truth",
      highlight: true,
      isLiteralString: true
    };
  }

  scorecardTargetRow.innerHTML = scoreCardConfigurationList.map(card => `
    <div class="kpi">
      <div class="kpi-eyebrow">${card.label}</div>
      <div class="kpi-value ${card.highlight ? "accent" : ""}">${card.isLiteralString ? card.value : new Intl.NumberFormat("en-US").format(card.value || 0)}</div>
      <div class="kpi-label">${card.detail}</div>
    </div>
  `).join("");
}

const PALETTE_VECTORS = ["#2d5a3d", "#2d7d9a", "#c47a1a", "#1a7a3a", "#7c5fa6", "#b5392a", "#6b6b64"];
const RENDER_BASE_BAR_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: "#1a1a1a",
      titleFont: { family: "Segoe UI", size: 12, weight: 600 },
      bodyFont: { family: "Segoe UI", size: 12 },
      padding: 10,
      cornerRadius: 6,
      displayColors: false
    }
  },
  scales: {
    x: { grid: { display: false }, ticks: { color: "#6b6b64", font: { family: "Segoe UI", size: 11 } } },
    y: { beginAtZero: true, grid: { color: "#e5e5de" }, ticks: { color: "#9c9c94", font: { family: "Segoe UI", size: 11 } } }
  }
};

function renderTopicDistributionsChart(countsObj) {
  const chartLabels = Object.keys(countsObj);
  const chartValues = Object.values(countsObj);
  const canvasCtx = document.getElementById("topicChart").getContext("2d");
  
  if (charts.topic) charts.topic.destroy();
  charts.topic = new Chart(canvasCtx, {
    type: "bar",
    data: {
      labels: chartLabels.map(str => str.length > 25 ? str.slice(0, 23) + "…" : str),
      datasets: [{
        data: chartValues,
        backgroundColor: chartLabels.map((_, idx) => PALETTE_VECTORS[idx % PALETTE_VECTORS.length]),
        borderRadius: 5,
        maxBarThickness: 50
      }]
    },
    options: RENDER_BASE_BAR_OPTIONS
  });
}

function renderRegionalLocationsChart(countsObj) {
  const locationSortedEntries = Object.entries(countsObj).sort((alpha, beta) => beta[1] - alpha[1]);
  const recordsMinusUnknowns = locationSortedEntries.filter(([stateKey]) => stateKey !== "Unknown");
  const topTwelveStates = (recordsMinusUnknowns.length ? recordsMinusUnknowns : locationSortedEntries).slice(0, 12);

  const canvasCtx = document.getElementById("locationChart").getContext("2d");
  if (charts.location) charts.location.destroy();
  charts.location = new Chart(canvasCtx, {
    type: "bar",
    data: {
      labels: topTwelveStates.map(([nameCode]) => nameCode),
      datasets: [{
        data: topTwelveStates.map(([, totalVal]) => totalVal),
        backgroundColor: "#2d7d9a",
        hoverBackgroundColor: "#1b4f63",
        borderRadius: 5,
        maxBarThickness: 30
      }]
    },
    options: {
      ...RENDER_BASE_BAR_OPTIONS,
      indexAxis: "y",
      scales: {
        x: { beginAtZero: true, grid: { color: "#e5e5de" }, ticks: { color: "#9c9c94", font: { family: "Segoe UI", size: 11 } } },
        y: { grid: { display: false }, ticks: { color: "#1a1a1a", font: { family: "Segoe UI", size: 12, weight: 500 } } }
      }
    }
  });
}

/**
 * Renders the interactive geographic US state choropleth map with data-defensive safety filters
 */
async function renderUSGeographicHeatmap(countsObj) {
  try {
    const stateVectorShapes = await getUnitedStatesTopoJsonFeatures();
    const canvasCtx = document.getElementById("usHeatmapCanvas");
    if (!canvasCtx) return;

    if (charts.heatmap) {
      charts.heatmap.destroy();
      charts.heatmap = null;
    }

    const labelToAbbrevMap = {
      "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
      "connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID",
      "illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
      "maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN",
      "mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV",
      "new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY",
      "north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK","oregon":"OR",
      "pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
      "tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA",
      "washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY"
    };

    // Diagnostic Check: Trace if any states were successfully evaluated from the ingestion pipeline
    const validStateEntries = Object.keys(countsObj).filter(k => k !== "Unknown" && countsObj[k] > 0);
    if (validStateEntries.length === 0) {
      console.warn("Geographic Plot Alert: No valid US state names or uppercase postal codes matched your dataset rows.");
    }

    charts.heatmap = new Chart(canvasCtx.getContext("2d"), {
      type: "choropleth",
      data: {
        labels: stateVectorShapes.map(shape => shape.properties.name),
        datasets: [{
          label: "Claims Density Distribution",
          outline: stateVectorShapes,
          data: stateVectorShapes.map(shape => {
            const matchedPostalCode = labelToAbbrevMap[shape.properties.name.toLowerCase()];
            return {
              feature: shape,
              value: countsObj[matchedPostalCode] || 0
            };
          })
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1a1a1a",
            padding: 12,
            cornerRadius: 6,
            titleFont: { family: "Segoe UI", size: 13, weight: 600 },
            bodyFont: { family: "Segoe UI", size: 12 },
            callbacks: {
              label: (toolTipContext) => ` Volume Density: ${toolTipContext.formattedValue} active claims`
            }
          }
        },
        scales: {
          projection: { axis: "x", projection: "albersUsa" },
          color: { axis: "x", interpolate: "blues", missing: "#e5e5de" }
        }
      }
    });
  } catch (err) {
    console.error("Critical Mapping Engine Crash: Map canvas could not be compiled.", err);
  }
}

function renderDynamicWordCloudsGrid(wordcloudsMap, topicVolCounts) {
  const displayGridBox = document.getElementById("wordcloudGrid");
  let templateAccumulatorHtml = "";
  
  for (const [topicNameKey, base64DataUrlString] of Object.entries(wordcloudsMap)) {
    const claimsVolumeCount = topicVolCounts[topicNameKey] || 0;
    const trackingPillLabelText = `${claimsVolumeCount} ${claimsVolumeCount === 1 ? "incident record" : "incident records"}`;
    
    templateAccumulatorHtml += `
      <div class="wc-tile" style="background: var(--surface); padding:14px; border:1px solid var(--border); border-radius:var(--radius);">
        <div class="wc-tile-title" style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:12px; font-weight:600;">
          <span style="color:var(--text-primary); text-overflow:ellipsis; overflow:hidden; white-space:nowrap; max-width:70%;">${escapeStringHtmlContent(topicNameKey)}</span>
          <span class="wc-tile-count" style="background:var(--accent-soft); color:var(--accent); padding:2px 8px; border-radius:12px; font-size:11px;">${trackingPillLabelText}</span>
        </div>
        ${base64DataUrlString ? `<img src="${base64DataUrlString}" alt="Topic cloud illustration" style="width:100%; height:auto; display:block; border-radius:4px;" />` : `<div style="height:120px; background:var(--surface-alt); display:flex; align-items:center; justify-content:center; font-size:12px; color:var(--text-tertiary);">Insufficient text variance for wordcloud rendering</div>`}
      </div>`;
  }
  displayGridBox.innerHTML = templateAccumulatorHtml;
}

function renderPreviewSpreadsheetGrid(rowRecords, schemaColumns) {
  const headerRowTarget = document.getElementById("tableHead");
  const bodyRowTarget = document.getElementById("tableBody");

  const surfaceColumnPriorityList = ["predicted_claim_type", "actual_claim_type", "match_predicted_vs_actual", "topic", "extracted_location"];
  const standardizedColumnSequence = [
    ...surfaceColumnPriorityList.filter(col => schemaColumns.includes(col)),
    ...schemaColumns.filter(col => !surfaceColumnPriorityList.includes(col))
  ];

  const captureUserFriendlyAliases = internalKey => {
    switch (internalKey) {
      case "predicted_claim_type":      return "Predicted Type";
      case "actual_claim_type":         return "Actual Ground Truth";
      case "match_predicted_vs_actual": return "Match Tracker";
      case "topic":                     return "Surfaced Theme Group";
      case "extracted_location":        return "Extracted State";
      default: return internalKey;
    }
  };

  headerRowTarget.innerHTML = standardizedColumnSequence.map(columnKey => `<th>${escapeStringHtmlContent(captureUserFriendlyAliases(columnKey))}</th>`).join("");

  bodyRowTarget.innerHTML = rowRecords.map(rowData => {
    return "<tr>" + standardizedColumnSequence.map(columnKey => {
      const cellValue = rowData[columnKey] ?? "";
      const escapedCleanText = escapeStringHtmlContent(String(cellValue));

      if (columnKey === "predicted_claim_type") return `<td class="col-pred" title="${escapedCleanText}">${escapedCleanText}</td>`;
      if (columnKey === "actual_claim_type") return `<td class="col-actual" title="${escapedCleanText}">${escapedCleanText || "—"}</td>`;
      if (columnKey === "match_predicted_vs_actual") {
        const structuralStatusStyleClass = cellValue === "Y" ? "match-y" : (cellValue === "N" ? "match-n" : "");
        return `<td class="col-match"><span class="${structuralStatusStyleClass}">${escapedCleanText || "—"}</span></td>`;
      }
      if (columnKey === "extracted_location") return `<td class="col-loc" style="font-weight:600; text-align:center;" title="${escapedCleanText}">${escapedCleanText}</td>`;
      return `<td title="${escapedCleanText}">${escapedCleanText}</td>`;
    }).join("") + "</tr>";
  }).join("");
}

function escapeStringHtmlContent(rawStringText) {
  return String(rawStringText)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}