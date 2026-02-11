"""
Train classifiers and a regressor to infer bowling parameters from commentary.

Design choices and rationale (in comments):
- Use sentence-transformers embeddings as primary text representation.
- Append simple contextual features (`over` and `innings phase`) to help
  disambiguate cases where text is sparse.
- Use lightweight sklearn models: LogisticRegression for multi-class classification
  (fast and interpretable) and RandomForestRegressor for speed with per-tree
  uncertainty estimate.

This is a prototype meant to be CPU-friendly and interpretable.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import extract_speed_kph, innings_phase_from_over
from utils.model_utils import load_embedding_model, build_features


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def prepare_dataset(df, embedder):
    texts = df['commentary'].fillna('').astype(str).tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)
    df = df.copy()
    df['speed_extracted'] = df['commentary'].apply(lambda t: extract_speed_kph(str(t)))
    df['speed'] = df.apply(lambda r: float(r['speed_kph']) if pd.notnull(r.get('speed_kph')) else (r['speed_extracted'] if r['speed_extracted'] is not None else np.nan), axis=1)
    df['phase'] = df['over'].apply(lambda o: innings_phase_from_over(o) if pd.notnull(o) else 'unknown')
    return df, np.array(embeddings)


def train(args):
    data = load_jsonl(args.data)
    embedder = load_embedding_model()
    df, embeddings = prepare_dataset(data, embedder)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Train classifiers: line, length, swing
    classifiers = {}
    for label in ['line', 'length', 'swing']:
        mask = df[label].notnull()
        if mask.sum() == 0:
            print(f"No labels for {label}, skipping")
            continue
        X_raw = embeddings[mask.values]
        y = df.loc[mask, label].astype(str).values
        X, scaler = build_features(X_raw, df.loc[mask, 'over'].fillna(0).astype(float).values[0] if False else 0, 'unknown')
        # Note: we re-build features per-sample in inference; here we use a simple scaler fit placeholder.
        # For robustness we'll instead build per-row features properly below.
        # Proper build: include correct per-row over and phase
        X_rows = []
        for i, emb in enumerate(X_raw):
            over = float(df.loc[mask].iloc[i]['over']) if pd.notnull(df.loc[mask].iloc[i].get('over')) else 0
            phase = df.loc[mask].iloc[i]['phase']
            xi, _ = build_features(emb, over, phase)
            X_rows.append(xi[0])
        X_rows = np.vstack(X_rows)
        # Handle small datasets: check if stratification is safe
        try:
            X_train, X_val, y_train, y_val = train_test_split(X_rows, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            # fall back to non-stratified split if classes are too sparse
            X_train, X_val, y_train, y_val = train_test_split(X_rows, y, test_size=0.2, random_state=42, stratify=None)
        clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print(f"=== {label} classifier report ===")
        print(classification_report(y_val, y_pred))
        classifiers[label] = clf

    # Train speed regressor on rows where speed is known
    mask_speed = df['speed'].notnull()
    speed_model = None
    speed_uncertainty = None
    if mask_speed.sum() >= 2:
        X_raw = embeddings[mask_speed.values]
        y_speed = df.loc[mask_speed, 'speed'].astype(float).values
        X_rows = []
        for i, emb in enumerate(X_raw):
            row = df.loc[mask_speed].iloc[i]
            over = float(row['over']) if pd.notnull(row.get('over')) else 0
            phase = row['phase']
            xi, scaler = build_features(emb, over, phase)
            X_rows.append(xi[0])
        X_rows = np.vstack(X_rows)
        # For small datasets, use all data with less test split
        test_size = 0.2 if len(X_rows) > 10 else 0.0
        if test_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(X_rows, y_speed, test_size=test_size, random_state=42)
        else:
            X_train, X_val, y_train, y_val = X_rows, np.array([]), y_speed, np.array([])
        speed_model = RandomForestRegressor(n_estimators=10, random_state=42)
        speed_model.fit(X_train, y_train)
        # use tree-by-tree std as uncertainty estimate
        all_preds = np.stack([t.predict(X_train) for t in speed_model.estimators_], axis=1)
        pred_mean = all_preds.mean(axis=1)
        pred_std = all_preds.std(axis=1)
        print("Speed MAE:", mean_absolute_error(y_train, pred_mean))
        speed_uncertainty = pred_std.mean()

    # Save models and needed metadata
    if classifiers:
        joblib.dump(classifiers, Path(args.out) / 'classifiers.joblib')
    if speed_model is not None:
        joblib.dump({'model': speed_model, 'uncertainty': float(speed_uncertainty)}, Path(args.out) / 'speed_model.joblib')
    print(f"Models saved to {args.out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/example_train.jsonl')
    parser.add_argument('--out', type=str, default='models')
    args = parser.parse_args()
    train(args)
