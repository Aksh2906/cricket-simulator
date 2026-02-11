"""
Load trained models and infer bowling parameters from commentary text.

Outputs a JSON structure consumable by Unity, including confidence scores per prediction.

Confidence semantics:
- For classifiers: use `predict_proba` per class and return the top class + its probability
- For speed regression: return predicted speed and an uncertainty (std) estimated from RF trees
"""
import argparse
import json
from pathlib import Path
import numpy as np
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import extract_speed_kph, innings_phase_from_over
from utils.model_utils import load_embedding_model, build_features


def load_models(models_dir: str):
    models_dir = Path(models_dir)
    classifiers = joblib.load(models_dir / 'classifiers.joblib')
    speed_info = None
    try:
        speed_info = joblib.load(models_dir / 'speed_model.joblib')
    except Exception:
        speed_info = None
    return classifiers, speed_info


def infer(text: str, over: float, models_dir: str):
    embedder = load_embedding_model()
    classifiers, speed_info = load_models(models_dir)
    emb = embedder.encode([text])[0]
    phase = innings_phase_from_over(over)
    X, _ = build_features(emb, over, phase)
    X_row = X[0]

    out = {
        'commentary': text,
        'over': over,
        'phase': phase,
        'predictions': {}
    }

    # classifiers
    for name, clf in classifiers.items():
        probs = clf.predict_proba(X_row.reshape(1, -1))[0]
        classes = clf.classes_
        top_idx = int(np.argmax(probs))
        out['predictions'][name] = {
            'label': str(classes[top_idx]),
            'confidence': float(probs[top_idx]),
            'all_probs': {str(c): float(p) for c, p in zip(classes, probs)}
        }

    # speed: prefer explicit extraction if present, otherwise predict and give uncertainty
    speed_extracted = extract_speed_kph(text)
    if speed_extracted is not None:
        out['predictions']['speed'] = {
            'speed_kph': float(speed_extracted),
            'confidence': 0.99,
            'method': 'extracted'
        }
    elif speed_info is not None:
        model = speed_info['model']
        # predict per-tree to get std
        tree_preds = np.stack([t.predict(X_row.reshape(1, -1)).ravel() for t in model.estimators_], axis=0)
        mean_pred = float(tree_preds.mean())
        std_pred = float(tree_preds.std())
        # transform std into a heuristic confidence: higher std -> lower confidence
        conf = max(0.0, min(0.99, 1.0 - (std_pred / max(1.0, mean_pred))))
        out['predictions']['speed'] = {
            'speed_kph': mean_pred,
            'uncertainty_std': std_pred,
            'confidence': conf,
            'method': 'regressed'
        }
    else:
        out['predictions']['speed'] = {
            'speed_kph': None,
            'confidence': 0.0,
            'method': 'none'
        }

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='models')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--over', type=float, default=0.0)
    args = parser.parse_args()
    res = infer(args.text, args.over, args.models)
    print(json.dumps(res, indent=2))
