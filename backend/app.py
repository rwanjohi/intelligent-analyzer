"""
FastAPI backend for Claims Atlas — claim analysis app.

Endpoints:
  POST /api/analyze            upload a file, run full pipeline
  GET  /api/analyze-remote     fetch from remote API, run full pipeline
  GET  /api/remote-config      tells the frontend whether the remote URL is configured
  GET  /api/sample             download the bundled sample CSV
  GET  /                       serve the frontend

Query params on both analyze endpoints:
  ?n_topics=N   integer 2..10 (default 5) — number of LDA topics

Pipeline:
  1. Robust read of CSV/Excel (handles BOM, encoding, whitespace headers)
  2. Predict claim type via the pickled model
  3. If a 'truth' column is present, add actual_claim_type and
     match_predicted_vs_actual ("Y"/"N") columns + match accuracy KPI
  4. LDA topic modeling with distinctive 2-keyword labels
  5. Location: extract US state from description text
  6. Per-topic word clouds (base64 PNG)

Remote data source (configurable, optional):
  Set REMOTE_DATA_URL and REMOTE_API_KEY in a .env file at the project root.
  When set, the frontend shows a "Pull from remote" button next to upload.
"""

import io
import os
import re
import json
import pickle
import base64
from collections import Counter
from typing import Dict, List, Any, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

# -----------------------------------------------------------------------------
# Paths and config
# -----------------------------------------------------------------------------
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "..", "models")
FRONTEND_DIR = os.path.join(BASE, "..", "frontend")
DATA_DIR = os.path.join(BASE, "..", "data")
ENV_PATH = os.path.join(BASE, "..", ".env")

# Try to load python-dotenv if installed; otherwise read .env manually.
def _load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        from dotenv import load_dotenv  # optional dependency
        load_dotenv(ENV_PATH)
    except ImportError:
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    for key in ("REMOTE_DATA_URL", "REMOTE_API_KEY", "REMOTE_AUTH_HEADER"):
        env[key] = os.environ.get(key, "").strip()
    return env

ENV = _load_env()
REMOTE_DATA_URL = ENV.get("REMOTE_DATA_URL", "")
REMOTE_API_KEY = ENV.get("REMOTE_API_KEY", "")
# Header name to send the API key in. Defaults to "Authorization: Bearer <key>".
# Override in .env with REMOTE_AUTH_HEADER=x-api-key (or any other name).
REMOTE_AUTH_HEADER = ENV.get("REMOTE_AUTH_HEADER", "")

# -----------------------------------------------------------------------------
# Load model artifacts at startup
# -----------------------------------------------------------------------------
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
    VECTORIZER = pickle.load(f)
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    LABEL_ENCODER = pickle.load(f)

# -----------------------------------------------------------------------------
# US state lookup tables
# -----------------------------------------------------------------------------
US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
STATE_NAME_TO_ABBR = {v.lower(): k for k, v in US_STATES.items()}


# =========================
# APP INIT
# =========================
app = FastAPI(title="Claims Atlas")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# =========================
# HELPERS
# =========================
def normalize_col(name: str) -> str:
    """Lowercase, strip BOM/whitespace, collapse internal spaces."""
    return re.sub(r"\s+", " ", str(name).replace("\ufeff", "").strip().lower())


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\xa0", " ").lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_text_list(series) -> List[str]:
    return [
        str(x).replace("\xa0", " ").strip() if pd.notnull(x) else ""
        for x in series
    ]


def safe_read_file(content: bytes, filename: str) -> pd.DataFrame:
    """Read CSV with encoding fallbacks, or Excel."""
    fn = (filename or "").lower()
    try:
        if fn.endswith(".csv"):
            for enc in ("utf-8-sig", "utf-8", "latin1", "cp1252"):
                try:
                    return pd.read_csv(io.BytesIO(content), encoding=enc)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(io.BytesIO(content), encoding="latin1",
                               sep=None, engine="python")
        if fn.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")


def parse_remote_payload(payload: Any, content_type: str = "") -> pd.DataFrame:
    """Accept the remote API's response and turn it into a DataFrame.
    Handles three shapes: JSON list-of-objects, JSON {data: [...]}, or CSV text."""
    # If response body is already bytes (CSV)
    if isinstance(payload, (bytes, bytearray)):
        # Try CSV first
        try:
            for enc in ("utf-8-sig", "utf-8", "latin1"):
                try:
                    return pd.read_csv(io.BytesIO(bytes(payload)), encoding=enc)
                except UnicodeDecodeError:
                    continue
        except Exception:
            pass
        # Fallback: try to decode as text and parse JSON
        try:
            payload = json.loads(bytes(payload).decode("utf-8", errors="replace"))
        except Exception:
            raise HTTPException(status_code=502,
                                detail="Remote response could not be parsed as CSV or JSON")
    if isinstance(payload, dict):
        # Common envelope keys: data, records, rows, results, items
        for key in ("data", "records", "rows", "results", "items"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=502, detail="Remote API returned empty list")
        return pd.DataFrame(payload)

    raise HTTPException(status_code=502,
                        detail=f"Unexpected remote payload shape: {type(payload).__name__}")


def extract_state(description: str) -> str:
    """Extract a US state from a description.
    Tries explicit ', XX' pattern first, then matches full state name."""
    if not isinstance(description, str):
        return "Unknown"

    m = re.search(r",\s*([A-Z]{2})\b", description)
    if m and m.group(1) in US_STATES:
        return US_STATES[m.group(1)]

    lower = description.lower()
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return US_STATES[abbr]

    return "Unknown"


def find_description_column(df: pd.DataFrame) -> str:
    candidates = [
        "description", "claim_description", "claim description",
        "details", "narrative", "claim_details", "text", "claim_text",
        "allegation synopsis", "allegation_synopsis", "allegation",
        "synopsis", "complaint", "complaint description",
        "incident description", "incident_description", "incident",
        "summary", "claim summary", "remarks", "comments", "notes",
        "issue description", "issue", "loss description",
    ]
    candidates_norm = {normalize_col(c) for c in candidates}
    col_map = {normalize_col(c): c for c in df.columns}

    for cand in candidates_norm:
        if cand in col_map:
            return col_map[cand]

    # Fallback: longest avg-length object column (>30 chars avg)
    best_col, best_score = None, 30
    for c in df.columns:
        if df[c].dtype == object:
            try:
                score = df[c].astype(str).str.len().mean()
                if score > best_score:
                    best_col, best_score = c, score
            except Exception:
                pass

    if not best_col:
        cols_preview = ", ".join(df.columns[:12])
        raise HTTPException(
            status_code=400,
            detail=(f"No description column found. Looked for 'description', "
                    f"'allegation synopsis', 'narrative', 'complaint', etc. "
                    f"Your file has columns: {cols_preview}"),
        )
    return best_col


def find_target_column(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    candidates = [
        "claim_type", "claim type",
        "allegation type", "allegation_type",
        "category", "true_label", "actual",
        "label", "type", "class",
    ]
    candidates_norm = {normalize_col(c) for c in candidates}
    col_map = {normalize_col(c): c for c in df.columns}
    for cand in candidates_norm:
        if cand in col_map and (exclude is None or col_map[cand] != exclude):
            return col_map[cand]
    return None


# =========================
# ML PREDICTION
# =========================
def predict_claim_types(descriptions: List[str]) -> List[str]:
    cleaned = [d if isinstance(d, str) and d.strip() else "unknown" for d in descriptions]
    X = VECTORIZER.transform(cleaned)
    preds = MODEL.predict(X)
    return LABEL_ENCODER.inverse_transform(preds).tolist()


# =========================
# TOPIC MODELING (n_topics is parameterized — user can request 2..10)
# =========================
def topic_model(descriptions: List[str], n_topics: int = 5) -> Dict[str, Any]:
    cleaned = [clean_text(d) for d in descriptions]
    cleaned = [c if c.strip() else "unknown" for c in cleaned]

    # Don't ask for more topics than we have documents / 2 (LDA needs samples)
    n_topics = max(2, min(n_topics, max(2, len(set(cleaned)) // 2 or 2)))

    cv = CountVectorizer(
        max_features=500,
        stop_words="english",
        min_df=2 if len(cleaned) > 20 else 1,
    )
    try:
        X = cv.fit_transform(cleaned)
    except ValueError:
        return _trivial_topic_result(descriptions)
    if X.shape[1] == 0:
        return _trivial_topic_result(descriptions)

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=20, learning_method="batch",
    )
    doc_topic = lda.fit_transform(X)

    feature_names = cv.get_feature_names_out()
    topic_keywords = {}
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-10:][::-1]
        topic_keywords[f"Topic {i + 1}"] = [feature_names[j] for j in top_idx]

    used = set()
    topic_labels = {}
    for i in range(n_topics):
        kws = topic_keywords[f"Topic {i + 1}"]
        distinctive = []
        for w in kws:
            if w not in used and len(distinctive) < 2:
                distinctive.append(w)
        if len(distinctive) < 2:
            distinctive = kws[:2]
        used.update(distinctive)
        topic_labels[f"Topic {i + 1}"] = f"Topic {i + 1}: {distinctive[0]}/{distinctive[1]}"

    doc_topics_idx = doc_topic.argmax(axis=1)
    doc_topics_labeled = [topic_labels[f"Topic {i + 1}"] for i in doc_topics_idx]
    topic_counts = Counter(doc_topics_labeled)
    topic_keywords_labeled = {topic_labels[k]: v for k, v in topic_keywords.items()}

    return {
        "doc_topics": doc_topics_labeled,
        "topic_keywords": topic_keywords_labeled,
        "topic_counts": dict(topic_counts),
        "n_topics_used": n_topics,
    }


def _trivial_topic_result(descriptions: List[str]) -> Dict[str, Any]:
    return {
        "doc_topics": ["Topic 1"] * len(descriptions),
        "topic_keywords": {"Topic 1": ["claim", "report", "incident"]},
        "topic_counts": {"Topic 1": len(descriptions)},
        "n_topics_used": 1,
    }


# =========================
# WORDCLOUD
# =========================
def make_wordcloud_b64(text: str) -> str:
    if not text.strip():
        text = "no data"
    stop = set(STOPWORDS) | {"claim", "claims", "incident"}
    try:
        wc = WordCloud(
            width=600, height=350, background_color="white",
            stopwords=stop, colormap="viridis", max_words=60,
            relative_scaling=0.5, prefer_horizontal=0.9,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Wordcloud error: {e}")
        return ""


# =========================
# SHARED PIPELINE
# =========================
def run_analysis(df: pd.DataFrame, n_topics: int = 5, source: str = "upload") -> Dict[str, Any]:
    """Shared analysis pipeline used by both /api/analyze and /api/analyze-remote."""
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Data is empty.")

    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    desc_col = find_description_column(df)
    target_col = find_target_column(df, exclude=desc_col)

    df[desc_col] = df[desc_col].fillna("").astype(str)
    df = df[df[desc_col].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400,
                            detail=f"No valid description rows in column '{desc_col}'.")

    descriptions = safe_text_list(df[desc_col])

    # Predict
    df["predicted_claim_type"] = predict_claim_types(descriptions)

    # Actual + match Y/N
    if target_col:
        df["actual_claim_type"] = df[target_col].fillna("").astype(str)
        df["match_predicted_vs_actual"] = [
            "Y" if str(p).strip().lower() == str(a).strip().lower() and a.strip() != "" else "N"
            for p, a in zip(df["predicted_claim_type"], df["actual_claim_type"])
        ]

    # Topics (with configurable count)
    topic_info = topic_model(descriptions, n_topics=n_topics)
    df["topic"] = topic_info["doc_topics"]

    # Location: extract state name AND state abbreviation
    df["extracted_location"] = [extract_state(d) for d in descriptions]
    # State abbreviation column for the heatmap (Plotly choropleth uses 2-letter codes)
    df["extracted_state_code"] = df["extracted_location"].map(
        lambda s: next((abbr for abbr, name in US_STATES.items() if name == s), "")
    )

    # Per-topic word clouds
    topic_wordclouds = {}
    for topic_label in topic_info["topic_counts"].keys():
        topic_descs = [d for d, t in zip(descriptions, topic_info["doc_topics"]) if t == topic_label]
        topic_text = " ".join(clean_text(d) for d in topic_descs)
        topic_wordclouds[topic_label] = make_wordcloud_b64(topic_text)

    # Aggregations
    location_counts = Counter(df["extracted_location"])
    type_counts = Counter(df["predicted_claim_type"])

    # State-code counts for heatmap (skip unknowns)
    state_code_counts = {
        code: int(count)
        for code, count in Counter(df["extracted_state_code"]).items()
        if code  # skip empty (Unknown)
    }

    match_accuracy = None
    if target_col and len(df) > 0:
        match_accuracy = round(
            (df["match_predicted_vs_actual"] == "Y").sum() / len(df), 4
        )

    preview = df.head(50).fillna("").astype(str).to_dict("records")
    all_rows = df.fillna("").astype(str).to_dict("records")

    return {
        "kpi": {
            "total_cases": int(len(df)),
            "total_locations": int(len([k for k in location_counts.keys() if k != "Unknown"])),
            "total_topics": int(len(topic_info["topic_counts"])),
            "total_claim_types": int(len(type_counts)),
            "match_accuracy": match_accuracy,
            "has_ground_truth": bool(target_col),
        },
        "claim_type_counts": dict(type_counts),
        "topic_counts": topic_info["topic_counts"],
        "topic_keywords": topic_info["topic_keywords"],
        "location_counts": dict(location_counts),
        "state_code_counts": state_code_counts,   # NEW: for choropleth
        "topic_wordclouds": topic_wordclouds,
        "preview_rows": preview,
        "all_rows": all_rows,
        "columns": list(df.columns),
        "description_column": desc_col,
        "target_column": target_col,
        "source": source,
        "n_topics_used": topic_info.get("n_topics_used", n_topics),
    }


# =========================
# ROUTES
# =========================
@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    n_topics: int = Query(5, ge=2, le=10),
):
    content = await file.read()
    df = safe_read_file(content, file.filename or "upload")
    return JSONResponse(run_analysis(df, n_topics=n_topics, source="upload"))


@app.get("/api/analyze-remote")
async def analyze_remote(
    n_topics: int = Query(5, ge=2, le=10),
):
    """Fetch data from the configured remote source, then run the same pipeline."""
    if not REMOTE_DATA_URL:
        raise HTTPException(
            status_code=503,
            detail=("Remote data source not configured. Set REMOTE_DATA_URL "
                    "in the .env file at the project root."),
        )

    headers = {"Accept": "application/json, text/csv"}
    if REMOTE_API_KEY:
        if REMOTE_AUTH_HEADER:
            headers[REMOTE_AUTH_HEADER] = REMOTE_API_KEY
        else:
            headers["Authorization"] = f"Bearer {REMOTE_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.get(REMOTE_DATA_URL, headers=headers)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Remote request failed: {e}")

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Remote API returned HTTP {resp.status_code}: {resp.text[:300]}",
        )

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        df = parse_remote_payload(resp.json(), content_type)
    elif "text/csv" in content_type or REMOTE_DATA_URL.lower().endswith(".csv"):
        df = parse_remote_payload(resp.content, content_type)
    else:
        # Try JSON first, fall back to CSV
        try:
            df = parse_remote_payload(resp.json(), content_type)
        except Exception:
            df = parse_remote_payload(resp.content, content_type)

    return JSONResponse(run_analysis(df, n_topics=n_topics, source="remote"))


@app.get("/api/remote-config")
async def remote_config():
    """Tell the frontend whether the remote pull button should be shown."""
    return JSONResponse({
        "remote_configured": bool(REMOTE_DATA_URL),
        # Useful for showing in the UI; don't expose the API key value
        "remote_url_hint": _redact_url(REMOTE_DATA_URL) if REMOTE_DATA_URL else None,
    })


def _redact_url(url: str) -> str:
    """Return the host portion of the URL for display, hiding query params/paths."""
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else "configured"


@app.get("/api/sample")
async def sample():
    path = os.path.join(DATA_DIR, "sample_claims.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, filename="sample_claims.csv", media_type="text/csv")


# Static frontend (must be mounted last so API routes take precedence)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
