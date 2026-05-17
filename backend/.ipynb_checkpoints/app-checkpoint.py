"""
FastAPI backend for Claims Atlas — claim analysis app.

Endpoints:
  POST /api/analyze   Process a local file OR pull from the pre-configured remote repository URL
  GET  /api/sample    Download a sample CSV dataset
  GET  /              Serve the frontend UI
"""

import io
import os
import re
import pickle
import base64
from collections import Counter
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE, "..", "models"))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE, "..", "frontend"))
DATA_DIR = os.path.abspath(os.path.join(BASE, "..", "data"))

# --- Secure Backend Configuration Layer ---
# Both the destination URL and authentication keys are locked strictly on the server-side
import os
from dotenv import load_dotenv

REMOTE_DATA_URL = os.getenv("REMOTE_DATA_URL", "https://raw.githubusercontent.com/user/repo/main/claims_stream.csv")
REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")
THIRD_PARTY_BEARER_TOKEN = os.getenv("THIRD_PARTY_BEARER_TOKEN", "")

app = FastAPI(title="Claims Atlas Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

US_STATES_MAP = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD', 'massachusetts': 'MA',
    'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS', 'missouri': 'MO', 'montana': 'MT',
    'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM',
    'new york': 'NY', 'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT', 'vermont': 'VT',
    'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY'
}

def extract_state_from_text(text: str) -> str:
    """Extracts US State code from text bodies using boundary matching rules."""
    if not isinstance(text, str) or pd.isna(text):
        return "Unknown"
    text_lower = text.lower()
    for state_name, code in US_STATES_MAP.items():
        if re.search(r'\b' + re.escape(state_name) + r'\b', text_lower):
            return code
    postal_matches = re.findall(r'\b[A-Z]{2}\b', text)
    valid_codes = set(US_STATES_MAP.values())
    for match in postal_matches:
        if match in valid_codes:
            return match
    return "Unknown"

@app.post("/api/analyze")
async def analyze_claims(
    file: Optional[UploadFile] = File(None),
    use_remote: bool = Form(False),
    num_topics: int = Form(5)
):
    # --- 1. Data Intake Routing Architecture ---
    if use_remote:
        # Pathway B: User triggered remote data pull via UI button click
        if not REMOTE_DATA_URL:
            raise HTTPException(status_code=500, detail="Server Configuration Error: Remote target database URL is missing.")
        try:
            headers = {}
            if REMOTE_API_KEY:
                headers["X-API-Key"] = REMOTE_API_KEY
            if THIRD_PARTY_BEARER_TOKEN:
                headers["Authorization"] = f"Bearer {THIRD_PARTY_BEARER_TOKEN}"

            async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
                response = await client.get(REMOTE_DATA_URL.strip(), headers=headers)
                response.raise_for_status()
                data_buffer = io.BytesIO(response.content)
                filename = REMOTE_DATA_URL.split("?")[0].split("/")[-1] or "remote_dataset.csv"
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Network Intake Failure: Failed pulling data stream from internal repository feed: {str(e)}"
            )
    elif file is not None and file.filename != "":
        # Pathway A: User dropped or browsed a file from their local machine
        contents = await file.read()
        data_buffer = io.BytesIO(contents)
        filename = file.filename
    else:
        raise HTTPException(
            status_code=400,
            detail="Data Action Required: Please supply a local upload file or trigger the remote repository stream pull."
        )

    # --- 2. File Parsing Engine ---
    try:
        if filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(data_buffer)
        else:
            try:
                data_buffer.seek(0)
                df = pd.read_csv(data_buffer, encoding="utf-8")
            except Exception:
                try:
                    data_buffer.seek(0)
                    df = pd.read_csv(data_buffer, encoding="utf-8-sig")
                except Exception:
                    data_buffer.seek(0)
                    df = pd.read_csv(data_buffer, encoding="cp1252")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed parsing spreadsheet data array structured: {str(e)}")

    df.columns = [str(c).strip() for c in df.columns]
    desc_col = next((c for c in df.columns if "description" in c.lower()), None)
    if not desc_col:
        raise HTTPException(status_code=400, detail="Incompatible Matrix: Target sheet must contain an explicit 'description' column text field.")

    target_col = next((c for c in df.columns if c.lower() in ["claim_type", "category", "actual_type"]), None)
    df[desc_col] = df[desc_col].fillna("").astype(str)

    # --- 3. Dynamic Vectorization & Topic Modeling Layer ---
    valid_topics_n = max(2, min(int(num_topics), 15))
    descriptions_list = df[desc_col].tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=1)
    
    try:
        if len(descriptions_list) > 0 and any(d.strip() for d in descriptions_list):
            dtm = vectorizer.fit_transform(descriptions_list)
            lda = LatentDirichletAllocation(n_components=valid_topics_n, random_state=42, max_iter=10)
            lda.fit(dtm)
            
            topic_distributions = lda.transform(dtm)
            df["topic_idx"] = topic_distributions.argmax(axis=1)
            
            feature_names = vectorizer.get_feature_names_out()
            topic_keywords_map = {}
            topic_counts_map = {}
            topic_wordclouds_encoded = {}

            for t_idx in range(valid_topics_n):
                top_word_indices = lda.components_[t_idx].argsort()[:-6:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                topic_label = f"Topic {t_idx + 1}: " + ", ".join(top_words)
                
                matching_rows = df[df["topic_idx"] == t_idx]
                topic_counts_map[topic_label] = int(len(matching_rows))
                
                combined_words_text = " ".join(matching_rows[desc_col].tolist())
                if combined_words_text.strip():
                    wc = WordCloud(background_color="white", width=400, height=200, max_words=30, stopwords=STOPWORDS).generate(combined_words_text)
                    img_buf = io.BytesIO()
                    wc.to_image().save(img_buf, format="PNG")
                    img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
                    topic_wordclouds_encoded[topic_label] = f"data:image/png;base64,{img_base64}"
                else:
                    topic_wordclouds_encoded[topic_label] = ""
                    
            df["topic"] = df["topic_idx"].apply(lambda idx: list(topic_counts_map.keys())[idx] if idx < len(topic_counts_map) else "Unclassified")
        else:
            raise ValueError("Empty strings matrix array")
    except Exception:
        topic_counts_map = {f"Default Topic Group {i+1}": 0 for i in range(valid_topics_n)}
        topic_wordclouds_encoded = {f"Default Topic Group {i+1}": "" for i in range(valid_topics_n)}
        df["topic"] = "Unclassified"

    # --- 4. Location Processing ---
    df["extracted_location"] = df[desc_col].apply(extract_state_from_text)
    location_counts = Counter(df["extracted_location"].tolist())

    # --- 5. Downstream Analytics Classification Simulations ---
    mock_classes = ["Property Damage", "Bodily Injury", "Workers Comp", "General Liability"]
    np.random.seed(42)
    df["predicted_claim_type"] = np.random.choice(mock_classes, size=len(df))
    type_counts = df["predicted_claim_type"].value_counts().to_dict()

    match_accuracy = None
    if target_col:
        df["actual_claim_type"] = df[target_col].fillna("").astype(str)
        df["match_predicted_vs_actual"] = np.where(
            df["predicted_claim_type"].str.lower() == df["actual_claim_type"].str.lower(), "Y", "N"
        )
        match_accuracy = round(float((df["match_predicted_vs_actual"] == "Y").sum() / len(df)), 4)
    else:
        df["match_predicted_vs_actual"] = ""

    preview_data = df.head(50).fillna("").astype(str).to_dict("records")
    full_download_data = df.fillna("").astype(str).to_dict("records")

    return JSONResponse({
        "kpi": {
            "total_cases": int(len(df)),
            "total_locations": int(len([k for k in location_counts.keys() if k != "Unknown"])),
            "total_topics": int(valid_topics_n),
            "total_claim_types": int(len(type_counts)),
            "match_accuracy": match_accuracy,
            "has_ground_truth": bool(target_col),
        },
        "claim_type_counts": type_counts,
        "topic_counts": topic_counts_map,
        "location_counts": dict(location_counts),
        "topic_wordclouds": topic_wordclouds_encoded,
        "preview_rows": preview_data,
        "all_rows": full_download_data,
        "columns": list(df.columns),
    })

@app.get("/api/sample")
async def sample():
    path = os.path.join(DATA_DIR, "sample_claims.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        pd.DataFrame({
            "Incident Description": [
                "Slip and fall accident in grocery storefront aisle in St. Louis Missouri, claims severe back injuries.",
                "Water utility leak from broken line caused deep structural property damage inside Austin TX warehouse office.",
                "Worker sustained minor arm fractures while operating warehouse sorting machinery in Los Angeles California.",
                "Delivery vehicle collided into storefront exterior loading dock access gate in Chicago IL."
            ],
            "Claim_Type": ["Bodily Injury", "Property Damage", "Workers Comp", "General Liability"]
        }).to_csv(path, index=False)
    return FileResponse(path, filename="sample_claims.csv", media_type="text/csv")

if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")