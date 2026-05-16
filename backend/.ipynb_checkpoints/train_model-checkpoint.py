"""
Train and compare multiple claim-type classification models.

Pipeline
--------
1. Load training data
2. Fit a single shared TF-IDF vectorizer + label encoder
3. For each model in the registry:
     - cross-validate on the training fold (CV accuracy)
     - fit on the full training fold
     - score on the held-out test set (test accuracy + macro-F1)
     - save the trained estimator as <name>.pkl
4. Pick the best model by test macro-F1 and copy it to model.pkl
   (so backend/app.py can keep loading model.pkl unchanged)
5. Save vectorizer.pkl, label_encoder.pkl, and a metrics.json report

Add a new model
---------------
Append a tuple to MODEL_REGISTRY below:

    ("my_model", MyEstimator(...))

Anything sklearn-compatible works (must implement .fit / .predict).

Usage
-----
    python train_model.py
    python train_model.py --models logreg random_forest
    python train_model.py --test-size 0.25 --cv 3
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "..", "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Model registry — add new models here
# ----------------------------------------------------------------------------
MODEL_REGISTRY: list[tuple[str, Any]] = [
    (
        "logreg",
        LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"),
    ),
    (
        "linear_svm",
        LinearSVC(C=1.0, class_weight="balanced", max_iter=3000),
    ),
    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    ),
    (
        "gradient_boost",
        GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        ),
    ),
    (
        "naive_bayes",
        MultinomialNB(alpha=0.5),
    ),
    (
        "knn",
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    ),
]


# ----------------------------------------------------------------------------
# Vectorizer (shared across all models)
# ----------------------------------------------------------------------------
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        sublinear_tf=True,
    )


# ----------------------------------------------------------------------------
# Per-model train/eval
# ----------------------------------------------------------------------------
def train_and_evaluate(
    name: str,
    estimator: Any,
    X_train,
    X_test,
    y_train,
    y_test,
    target_names: list[str],
    cv: int,
) -> dict:
    """Train one model, score it, save it, return metrics."""
    print(f"\n┌─ {name}")
    t0 = time.time()

    # Cross-validation on the training fold (accuracy)
    try:
        cv_scores = cross_val_score(
            estimator, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())
    except Exception as e:
        print(f"│  CV failed: {e}")
        cv_mean = cv_std = float("nan")

    # Fit on full training fold
    estimator.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    # Score on held-out test set
    y_pred = estimator.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred, average="macro"))

    print(f"│  CV accuracy : {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"│  Test acc    : {test_acc:.4f}")
    print(f"│  Test macroF1: {test_f1:.4f}")
    print(f"│  Fit time    : {fit_seconds:.2f}s")

    # Save the trained estimator
    out_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(estimator, f)
    print(f"└─ saved -> {os.path.basename(out_path)}")

    # Per-class report (compact)
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "name": name,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "fit_seconds": round(fit_seconds, 3),
        "per_class_f1": {
            cls: round(report[cls]["f1-score"], 4)
            for cls in target_names if cls in report
        },
        "model_file": f"{name}.pkl",
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_names = [n for n, _ in MODEL_REGISTRY]
    parser.add_argument(
        "--models", nargs="+", choices=all_names, default=all_names,
        help="Subset of models to train (default: all)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} rows from {os.path.basename(DATA)}")
    print(f"Class distribution:\n{df['claim_type'].value_counts().to_string()}\n")

    X = df["description"].astype(str).values
    y_raw = df["claim_type"].astype(str).values

    # Fit shared vectorizer + label encoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    vectorizer = build_vectorizer()
    X_vec = vectorizer.fit_transform(X)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix : {X_vec.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print(f"Models to train: {', '.join(args.models)}")

    # Save shared artifacts up front (vectorizer + encoder don't depend on the model)
    with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"Saved shared artifacts: vectorizer.pkl, label_encoder.pkl")

    # Train every requested model
    target_names = list(le.classes_)
    selected = [(n, est) for n, est in MODEL_REGISTRY if n in args.models]
    results = []
    for name, estimator in selected:
        try:
            metrics = train_and_evaluate(
                name, estimator,
                X_train, X_test, y_train, y_test,
                target_names, args.cv,
            )
            results.append(metrics)
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results.append({"name": name, "error": str(e)})

    # Pick winner by test_macro_f1 (tie-break on test_accuracy, then -fit_seconds)
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo model finished successfully.")
        return

    valid.sort(
        key=lambda r: (r["test_macro_f1"], r["test_accuracy"], -r["fit_seconds"]),
        reverse=True,
    )
    winner = valid[0]

    # Pretty leaderboard
    print("\n" + "=" * 78)
    print("LEADERBOARD (sorted by test macro-F1)")
    print("=" * 78)
    print(f"{'rank':<5}{'model':<18}{'cv_acc':>10}{'test_acc':>11}{'macroF1':>10}{'fit (s)':>10}")
    print("-" * 78)
    for i, r in enumerate(valid, 1):
        marker = "★" if r["name"] == winner["name"] else " "
        print(
            f"{i:<5}{r['name']:<18}"
            f"{r['cv_accuracy_mean']:>9.4f} "
            f"{r['test_accuracy']:>10.4f} "
            f"{r['test_macro_f1']:>9.4f} "
            f"{r['fit_seconds']:>9.2f}  {marker}"
        )
    print("=" * 78)
    print(f"\nWinner: {winner['name']}  →  copying to model.pkl (used by backend)")

    # Copy winner to the canonical model.pkl that the backend loads
    src = os.path.join(MODELS_DIR, winner["model_file"])
    dst = os.path.join(MODELS_DIR, "model.pkl")
    shutil.copyfile(src, dst)

    # Persist a metrics report for traceability
    report_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "winner": winner["name"],
                "test_size": args.test_size,
                "cv_folds": args.cv,
                "seed": args.seed,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "classes": target_names,
                "results": results,
            },
            f, indent=2,
        )
    print(f"Saved metrics report -> {os.path.basename(report_path)}")

    # Per-class F1 of the winner
    print(f"\nPer-class F1 ({winner['name']}):")
    for cls, score in winner["per_class_f1"].items():
        bar = "█" * int(score * 30)
        print(f"  {cls:<22} {score:.4f}  {bar}")


if __name__ == "__main__":
    main()