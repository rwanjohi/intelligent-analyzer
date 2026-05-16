"""
FastAPI backend for Claims Atlas — claim analysis app.

Endpoints:
  POST /api/analyze   upload file, run full pipeline
  GET  /api/sample    download a sample CSV
  GET  /              serve the frontend

Pipeline:
  1. Robust file read (CSV utf-8 / utf-8-sig / latin1 / cp1252 fallback; Excel)
  2. Column normalization (strip BOM, whitespace; case-insensitive matching)
  3. Predict claim type
  4. If a target column (e.g. claim_type) is present in the upload, also
     surface actual_claim_type and match_predicted_vs_actual ("Y"/"N")
  5. Topic modeling (LDA, max 5 topics; distinctive labels)
  6. Location: extract US state from description text
  7. Word clouds per topic
"""

import io
import os
import re
import pickle
import base64
from collections import Counter
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "..", "models")
FRONTEND_DIR = os.path.join(BASE, "..", "frontend")
DATA_DIR = os.path.join(BASE, "..", "data")

# Load model artifacts at startup
with open(os.path.join(MODELS_DIR, "model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
    VECTORIZER = pickle.load(f)
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    LABEL_ENCODER = pickle.load(f)

# US state lookup table for location extraction
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


app = FastAPI(title="Claims Atlas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HELPERS
# =========================
def normalize_col(name: str) -> str:
    """Strip BOM, lowercase, collapse whitespace, strip edges — for matching."""
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

        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {filename}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")


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
    """Find the description / synopsis / narrative column."""
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
            detail=(
                f"No description column found. Looked for 'description', "
                f"'allegation synopsis', 'narrative', 'complaint', etc. "
                f"Your file has columns: {cols_preview}"
            ),
        )
    return best_col


def find_target_column(df: pd.DataFrame, exclude: str | None = None) -> str | None:
    """Find a 'truth' column if one is present (e.g. claim_type, allegation type).
    Excludes the description column to avoid accidental match."""
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
# TOPIC MODELING
# =========================
def topic_model(descriptions: List[str], n_topics: int = 5) -> Dict[str, Any]:
    """LDA topic modeling with distinctive 2-keyword labels per topic."""
    cleaned = [clean_text(d) for d in descriptions]
    cleaned = [c if c.strip() else "unknown" for c in cleaned]

    n_topics = min(n_topics, max(2, len(set(cleaned)) // 2 or 2))
    n_topics = min(n_topics, 5)

    cv = CountVectorizer(
        max_features=500,
        stop_words="english",
        min_df=2 if len(cleaned) > 20 else 1,
    )
    try:
        X = cv.fit_transform(cleaned)
    except ValueError:
        return {
            "doc_topics": ["Topic 1"] * len(descriptions),
            "topic_keywords": {"Topic 1": ["claim", "report", "incident"]},
            "topic_counts": {"Topic 1": len(descriptions)},
        }

    if X.shape[1] == 0:
        return {
            "doc_topics": ["Topic 1"] * len(descriptions),
            "topic_keywords": {"Topic 1": ["claim", "report", "incident"]},
            "topic_counts": {"Topic 1": len(descriptions)},
        }

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

    # Build distinctive labels — don't reuse top words across topics
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
            width=600, height=350,
            background_color="white",
            stopwords=stop,
            colormap="viridis",
            max_words=60,
            relative_scaling=0.5,
            prefer_horizontal=0.9,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Wordcloud error: {e}")
        return ""


# =========================
# ROUTES
# =========================
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    df = safe_read_file(content, file.filename)

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Strip BOM / whitespace from headers up front
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    desc_col = find_description_column(df)
    target_col = find_target_column(df, exclude=desc_col)

    # Drop rows with missing description
    df[desc_col] = df[desc_col].fillna("").astype(str)
    df = df[df[desc_col].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=400,
                            detail=f"No valid description rows found in column '{desc_col}'.")

    descriptions = safe_text_list(df[desc_col])

    # ---- Pipeline ----
    # Stage: predict
    predicted = predict_claim_types(descriptions)
    df["predicted_claim_type"] = predicted

    # Stage: actual + match Y/N (only when target column exists)
    if target_col:
        df["actual_claim_type"] = df[target_col].fillna("").astype(str)
        df["match_predicted_vs_actual"] = [
            "Y" if str(p).strip().lower() == str(a).strip().lower() and a.strip() != "" else "N"
            for p, a in zip(df["predicted_claim_type"], df["actual_claim_type"])
        ]

    # Stage: topic modeling
    topic_info = topic_model(descriptions, n_topics=5)
    df["topic"] = topic_info["doc_topics"]

    # Stage: location
    df["extracted_location"] = [extract_state(d) for d in descriptions]

    # Stage: per-topic word clouds
    topic_wordclouds = {}
    for topic_label in topic_info["topic_counts"].keys():
        topic_descs = [d for d, t in zip(descriptions, topic_info["doc_topics"]) if t == topic_label]
        topic_text = " ".join(clean_text(d) for d in topic_descs)
        topic_wordclouds[topic_label] = make_wordcloud_b64(topic_text)

    # Aggregations
    location_counts = Counter(df["extracted_location"])
    type_counts = Counter(df["predicted_claim_type"])

    # Match accuracy (when we have ground truth)
    match_accuracy = None
    if target_col and len(df) > 0:
        match_accuracy = round(
            (df["match_predicted_vs_actual"] == "Y").sum() / len(df), 4
        )

    # Preview + full rows for download
    preview = df.head(50).fillna("").astype(str).to_dict("records")
    all_rows = df.fillna("").astype(str).to_dict("records")

    return JSONResponse({
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
        "topic_wordclouds": topic_wordclouds,
        "preview_rows": preview,
        "all_rows": all_rows,
        "columns": list(df.columns),
        "description_column": desc_col,
        "target_column": target_col,
    })


@app.get("/api/sample")
async def sample():
    path = os.path.join(DATA_DIR, "sample_claims.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, filename="sample_claims.csv", media_type="text/csv")


# Static frontend
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
