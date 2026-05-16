# Claims Atlas — Claim Analyzer

A FastAPI + vanilla-JS app that takes a CSV/Excel of claim records and produces:

- **Claim type prediction** from the description (TF-IDF + Logistic Regression)
- **Topic modeling** — up to 5 themes via LDA
- **Location extraction** — US state from the description text
- **Word clouds** — overall + one per topic
- **Charts & KPIs** — Chart.js bar graphs, scrollable preview table

## Project layout

```
claim_analyzer/
├── backend/
│   ├── app.py                      # FastAPI app
│   ├── generate_training_data.py   # Builds synthetic training CSV
│   └── train_model.py              # Trains classifier, saves pickles
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── models/
│   ├── model.pkl                   # LogisticRegression
│   ├── vectorizer.pkl              # TfidfVectorizer
│   └── label_encoder.pkl           # LabelEncoder
├── data/
│   ├── training_data.csv           # 560 synthetic rows, 7 classes
│   └── sample_claims.csv           # 56-row sample for testing the UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## (Re)train the model

```bash
cd backend
python generate_training_data.py   # creates data/training_data.csv + sample_claims.csv
python train_model.py              # creates models/{model,vectorizer,label_encoder}.pkl
```

## Run the app

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. Drop in a CSV/Excel with a column called
`description` (or any text-heavy column — the app picks the best candidate).
Click **Analyze**.

A test file is available at <http://localhost:8000/api/sample>.

## How it works

| Stage | Module                                      |
|-------|---------------------------------------------|
| Parse & clean   | pandas + light regex                |
| Predict type    | TfidfVectorizer → LogisticRegression |
| Topic model     | CountVectorizer → LatentDirichletAllocation |
| Location        | regex on `, XX` patterns + state-name match |
| Word clouds     | `wordcloud.WordCloud`, base64 embedded in JSON |

The backend returns one JSON payload; the frontend renders KPIs, two bar
charts (topics + top-12 locations), per-topic word clouds, and a scrollable
preview table. Progress stages animate while the request is in flight.

## Claim categories the model knows

Auto Accident · Property Damage · Medical · Theft · Natural Disaster ·
Liability · Workers Compensation
