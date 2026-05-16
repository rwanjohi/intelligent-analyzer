"""
FastAPI backend for claim analysis app.

Endpoints:
- POST /api/analyze         -> upload file, run full analysis
- GET  /api/sample          -> download a sample CSV
- GET  /                    -> serve UI

Pipeline:
  Stage 1: Data cleaning
  Stage 2: Claim type prediction
  Stage 3: Topic modeling (LDA, max 5 topics)
  Stage 4: Location extraction
  Stage 5: Word cloud generation per topic
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

app = FastAPI(title="Claim Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pipeline helpers ---------- #

def clean_text(text: str) -> str:
    """Lowercase, strip non-alpha, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_state(description: str) -> str:
    """Extract a US state from a free-text description.
    Order: explicit state abbr (', NY') -> full state name."""
    if not isinstance(description, str):
        return "Unknown"

    # Try ", XX" or ", XX." pattern (case sensitive abbreviation)
    m = re.search(r",\s*([A-Z]{2})\b", description)
    if m and m.group(1) in US_STATES:
        return US_STATES[m.group(1)]

    # Try full state name (case-insensitive)
    lower = description.lower()
    for name, abbr in STATE_NAME_TO_ABBR.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            return US_STATES[abbr]

    return "Unknown"


def predict_claim_types(descriptions: List[str]) -> List[str]:
    cleaned = [d if isinstance(d, str) and d.strip() else "unknown" for d in descriptions]
    X = VECTORIZER.transform(cleaned)
    preds = MODEL.predict(X)
    return LABEL_ENCODER.inverse_transform(preds).tolist()


def topic_model(descriptions: List[str], n_topics: int = 5) -> Dict[str, Any]:
    """Run LDA, return per-doc topic assignments + topic keywords."""
    cleaned = [clean_text(d) for d in descriptions]
    cleaned = [c if c.strip() else "unknown" for c in cleaned]

    # Adjust n_topics if very few docs
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
        # All-empty edge case
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
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(X)

    feature_names = cv.get_feature_names_out()
    topic_keywords = {}
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-10:][::-1]
        keywords = [feature_names[j] for j in top_idx]
        topic_keywords[f"Topic {i + 1}"] = keywords

    # Build distinctive labels by picking words that aren't already used
    used = set()
    topic_labels = {}
    for i in range(n_topics):
        kws = topic_keywords[f"Topic {i + 1}"]
        # Pick first 2 words not yet used by another topic; fall back to top-2
        distinctive = []
        for w in kws:
            if w not in used and len(distinctive) < 2:
                distinctive.append(w)
        if len(distinctive) < 2:
            distinctive = kws[:2]
        used.update(distinctive)
        topic_labels[f"Topic {i + 1}"] = f"Topic {i + 1}: {distinctive[0]}/{distinctive[1]}"

    doc_topics_idx = doc_topic.argmax(axis=1)
    doc_topics = [f"Topic {i + 1}" for i in doc_topics_idx]

    topic_counts = Counter(doc_topics)
    topic_counts_dict = {topic_labels[k]: v for k, v in topic_counts.items()}
    topic_keywords_labeled = {topic_labels[k]: v for k, v in topic_keywords.items()}
    doc_topics_labeled = [topic_labels[t] for t in doc_topics]

    return {
        "doc_topics": doc_topics_labeled,
        "topic_keywords": topic_keywords_labeled,
        "topic_counts": topic_counts_dict,
    }


def make_wordcloud_b64(text: str, color: str = "#0ea5e9") -> str:
    """Generate a wordcloud PNG, return base64 data URL."""
    if not text.strip():
        text = "no data"
    stop = set(STOPWORDS) | {"claim", "claims", "incident"}
    try:
        wc = WordCloud(
            width=600,
            height=350,
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
        # Fallback empty image
        print(f"Wordcloud error: {e}")
        return ""


def find_description_column(df: pd.DataFrame) -> str:
    """Find the column most likely to hold claim descriptions."""
    # Direct match on common names
    candidates = ["description", "claim_description", "claim description",
                  "details", "narrative", "claim_details", "text", "claim_text"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]

    # Otherwise pick the string column with the longest avg length
    best_col, best_len = None, 0
    for c in df.columns:
        if df[c].dtype == object:
            try:
                avg = df[c].astype(str).str.len().mean()
                if avg > best_len:
                    best_len, best_col = avg, c
            except Exception:
                pass

    if best_col is None:
        raise ValueError("No text column found in uploaded file")
    return best_col


# ---------- Routes ---------- #

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    content = await file.read()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Upload a .csv, .xlsx, or .xls file")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        desc_col = find_description_column(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Stage 1: clean
    df = df.dropna(subset=[desc_col]).reset_index(drop=True)
    df[desc_col] = df[desc_col].astype(str)
    descriptions = df[desc_col].tolist()

    if not descriptions:
        raise HTTPException(status_code=400, detail="No valid descriptions found")

    # Stage 2: predict claim types
    predicted_types = predict_claim_types(descriptions)
    df["predicted_claim_type"] = predicted_types

    # Stage 3: topic modeling
    topic_info = topic_model(descriptions, n_topics=5)
    df["topic"] = topic_info["doc_topics"]

    # Stage 4: location
    df["extracted_location"] = [extract_state(d) for d in descriptions]

    # Stage 5: wordclouds — overall + per topic
    overall_text = " ".join(clean_text(d) for d in descriptions)
    overall_wc = make_wordcloud_b64(overall_text)

    topic_wordclouds = {}
    for topic_label in topic_info["topic_counts"].keys():
        topic_descs = [d for d, t in zip(descriptions, topic_info["doc_topics"]) if t == topic_label]
        topic_text = " ".join(clean_text(d) for d in topic_descs)
        topic_wordclouds[topic_label] = make_wordcloud_b64(topic_text)

    # KPI / aggregations
    location_counts = Counter(df["extracted_location"])
    type_counts = Counter(predicted_types)

    # Top-10 preview
    preview_cols = [c for c in df.columns]
    preview = df.head(50).fillna("").astype(str).to_dict(orient="records")

    response = {
        "kpi": {
            "total_cases": int(len(df)),
            "total_locations": int(len([k for k in location_counts.keys() if k != "Unknown"])),
            "total_topics": int(len(topic_info["topic_counts"])),
            "total_claim_types": int(len(type_counts)),
        },
        "claim_type_counts": dict(type_counts),
        "topic_counts": topic_info["topic_counts"],
        "topic_keywords": topic_info["topic_keywords"],
        "location_counts": dict(location_counts),
        "overall_wordcloud": overall_wc,
        "topic_wordclouds": topic_wordclouds,
        "preview_rows": preview,
        "columns": preview_cols,
        "description_column": desc_col,
    }

    return JSONResponse(response)


@app.get("/api/sample")
async def sample():
    path = os.path.join(DATA_DIR, "sample_claims.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path, filename="sample_claims.csv", media_type="text/csv")


# Static frontend
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
