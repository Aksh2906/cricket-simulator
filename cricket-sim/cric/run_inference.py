# """
# Load trained models and infer bowling parameters from commentary text.

# Outputs a JSON structure consumable by Unity, including confidence scores per prediction.

# Confidence semantics:
# - For classifiers: use `predict_proba` per class and return the top class + its probability
# - For speed regression: return predicted speed and an uncertainty (std) estimated from RF trees
# """
# import argparse
# import json
# from pathlib import Path
# import numpy as np
# import joblib

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.text_utils import extract_speed_kph, innings_phase_from_over
# from utils.model_utils import load_embedding_model, build_features


# def load_models(models_dir: str):
#     models_dir = Path(models_dir)
#     classifiers = joblib.load(models_dir / 'classifiers.joblib')
#     speed_info = None
#     try:
#         speed_info = joblib.load(models_dir / 'speed_model.joblib')
#     except Exception:
#         speed_info = None
#     return classifiers, speed_info


# def infer(text: str, over: float, models_dir: str):
#     embedder = load_embedding_model()
#     classifiers, speed_info = load_models(models_dir)
#     emb = embedder.encode([text])[0]
#     phase = innings_phase_from_over(over)
#     X, _ = build_features(emb, over, phase)
#     X_row = X[0]

#     out = {
#         'commentary': text,
#         'over': over,
#         'phase': phase,
#         'predictions': {}
#     }

#     # classifiers
#     for name, clf in classifiers.items():
#         probs = clf.predict_proba(X_row.reshape(1, -1))[0]
#         classes = clf.classes_
#         top_idx = int(np.argmax(probs))
#         out['predictions'][name] = {
#             'label': str(classes[top_idx]),
#             # 'confidence': float(probs[top_idx]),
#             # 'all_probs': {str(c): float(p) for c, p in zip(classes, probs)}
#         }

#     # speed: prefer explicit extraction if present, otherwise predict and give uncertainty
#     speed_extracted = extract_speed_kph(text)
#     if speed_extracted is not None:
#         out['predictions']['speed'] = {
#             'speed_kph': float(speed_extracted),
#             # 'confidence': 0.99,
#             # 'method': 'extracted'
#         }
#     elif speed_info is not None:
#         model = speed_info['model']
#         # predict per-tree to get std
#         tree_preds = np.stack([t.predict(X_row.reshape(1, -1)).ravel() for t in model.estimators_], axis=0)
#         mean_pred = float(tree_preds.mean())
#         std_pred = float(tree_preds.std())
#         # transform std into a heuristic confidence: higher std -> lower confidence
#         conf = max(0.0, min(0.99, 1.0 - (std_pred / max(1.0, mean_pred))))
#         out['predictions']['speed'] = {
#             'speed_kph': mean_pred,
#             # 'uncertainty_std': std_pred,
#             # 'confidence': conf,
#             # 'method': 'regressed'
#         }
#     else:
#         out['predictions']['speed'] = {
#             'speed_kph': None,
#             # 'confidence': 0.0,
#             # 'method': 'none'
#         }

#     return out


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--models', type=str, default='models')
#     parser.add_argument('--text', type=str, required=True)
#     parser.add_argument('--over', type=float, default=0.0)
#     args = parser.parse_args()
#     res = infer(args.text, args.over, args.models)
#     print(json.dumps(res, indent=2))

# """
# Train classifiers and a regressor to infer bowling parameters from commentary.
# Also supports inference (prediction) on new data.

# Usage:
#     Train:   python script.py --mode train --data train.jsonl --model_dir models
#     Predict: python script.py --mode predict --input new_data.jsonl --output predicted.jsonl --model_dir models
# """
# import argparse
# import json
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import joblib
# import warnings

# # Suppress benign warnings
# warnings.filterwarnings("ignore", category=UserWarning) 
# warnings.filterwarnings("ignore", category=FutureWarning)

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, mean_absolute_error

# # Add parent directory to path to find utils
# sys.path.insert(0, str(Path(__file__).parent.parent))

# try:
#     from utils.text_utils import extract_speed_kph, innings_phase_from_over
#     from utils.model_utils import load_embedding_model, build_features
# except ImportError:
#     # Fallback for standalone testing if utils are missing
#     print("Warning: utils module not found. Ensure you are in the correct directory.")
#     sys.exit(1)


# def load_jsonl(path):
#     """Robust JSONL loader that handles BOM and errors."""
#     rows = []
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"Input file not found: {path}")

#     with open(path, 'r', encoding='utf-8-sig') as f:
#         for i, line in enumerate(f):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rows.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 print(f"Skipping malformed JSON on line {i+1}: {e}")
#                 continue
#     return pd.DataFrame(rows)


# def prepare_dataset(df, embedder):
#     """Prepares text embeddings and extracts ground truth labels for training."""
#     texts = df['commentary'].fillna('').astype(str).tolist()
#     print("Generating embeddings...")
#     embeddings = embedder.encode(texts, show_progress_bar=True)
    
#     df = df.copy()
#     # Extract speed from text if explicit label is missing
#     df['speed_extracted'] = df['commentary'].apply(lambda t: extract_speed_kph(str(t)))
    
#     # Prioritize 'speed_kph' column, fall back to regex extraction
#     def get_speed(r):
#         if 'speed_kph' in r and pd.notnull(r['speed_kph']):
#             return float(r['speed_kph'])
#         if r['speed_extracted'] is not None:
#             return float(r['speed_extracted'])
#         return np.nan

#     df['speed'] = df.apply(get_speed, axis=1)
#     df['phase'] = df['over'].apply(lambda o: innings_phase_from_over(o) if pd.notnull(o) else 'unknown')
    
#     return df, np.array(embeddings)


# def train(args):
#     print(f"--- Training Mode: Loading {args.data} ---")
#     data = load_jsonl(args.data)
#     embedder = load_embedding_model()
#     df, embeddings = prepare_dataset(data, embedder)

#     output_dir = Path(args.model_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 1. Train Classifiers (Line, Length, Swing)
#     classifiers = {}
#     for label in ['line', 'length']:
#         if label not in df.columns:
#             continue

#         mask = df[label].notnull()
#         if mask.sum() < 2:
#             print(f"Not enough data for {label}, skipping.")
#             continue
        
#         print(f"\nTraining {label} classifier...")
#         X_raw = embeddings[mask.values]
#         y = df.loc[mask, label].astype(str).values
        
#         # Build features row-by-row to ensure correct context
#         X_rows = []
#         subset = df.loc[mask]
#         for i in range(len(subset)):
#             row = subset.iloc[i]
#             over = float(row['over']) if pd.notnull(row.get('over')) else 0.0
#             phase = row['phase']
#             xi, _ = build_features(X_raw[i], over, phase)
#             X_rows.append(xi[0])
        
#         X_rows = np.vstack(X_rows)

#         # Stratified split
#         try:
#             X_train, X_val, y_train, y_val = train_test_split(X_rows, y, test_size=0.2, random_state=42, stratify=y)
#         except ValueError:
#             X_train, X_val, y_train, y_val = train_test_split(X_rows, y, test_size=0.2, random_state=42)

#         clf = LogisticRegression(max_iter=1000)
#         clf.fit(X_train, y_train)
        
#         if len(y_val) > 0:
#             y_pred = clf.predict(X_val)
#             print(classification_report(y_val, y_pred, zero_division=0))
        
#         classifiers[label] = clf

#     # 2. Train Speed Regressor
#     mask_speed = df['speed'].notnull()
#     speed_model = None
#     if mask_speed.sum() >= 2:
#         print("\nTraining Speed Regressor...")
#         X_raw = embeddings[mask_speed.values]
#         y_speed = df.loc[mask_speed, 'speed'].astype(float).values
        
#         X_rows = []
#         subset = df.loc[mask_speed]
#         for i in range(len(subset)):
#             row = subset.iloc[i]
#             over = float(row['over']) if pd.notnull(row.get('over')) else 0.0
#             phase = row['phase']
#             xi, _ = build_features(X_raw[i], over, phase)
#             X_rows.append(xi[0])
            
#         X_rows = np.vstack(X_rows)
        
#         speed_model = RandomForestRegressor(n_estimators=20, random_state=42)
#         speed_model.fit(X_rows, y_speed) # Training on all available valid data for final model
#         print("Speed model trained.")

#     # Save
#     if classifiers:
#         joblib.dump(classifiers, output_dir / 'classifiers.joblib')
#     if speed_model:
#         joblib.dump({'model': speed_model}, output_dir / 'speed_model.joblib')
    
#     print(f"\nModels saved to {output_dir}")


# def predict(args):
#     print(f"--- Inference Mode: Reading {args.input} ---")
    
#     # 1. Check Paths
#     model_dir = Path(args.model_dir)
#     clf_path = model_dir / 'classifiers.joblib'
#     speed_path = model_dir / 'speed_model.joblib'
    
#     if not clf_path.exists() or not speed_path.exists():
#         print(f"Error: Models not found in {model_dir}. Run --mode train first.")
#         return

#     # 2. Load Models
#     print("Loading models...")
#     classifiers = joblib.load(clf_path)
#     speed_data = joblib.load(speed_path)
#     speed_model = speed_data['model']
#     embedder = load_embedding_model()

#     # 3. Load Data
#     df = load_jsonl(args.input)
#     if df.empty:
#         print("Input file is empty.")
#         return

#     # 4. Generate Embeddings
#     print(f"Encoding {len(df)} commentaries...")
#     texts = df['commentary'].fillna('').astype(str).tolist()
#     embeddings = embedder.encode(texts, show_progress_bar=True)

#     # 5. Inference Loop
#     output_rows = []
    
#     print("Running predictions...")
#     # Convert dataframe to list of dicts to preserve original structure
#     records = df.to_dict('records')
    
#     for i, record in enumerate(tqdm(records)):
#         # Prepare context
#         over = float(record.get('over', 0)) if pd.notnull(record.get('over')) else 0.0
#         phase = innings_phase_from_over(over)
        
#         # Build features (Using single embedding vector)
#         emb = embeddings[i]
#         xi, _ = build_features(emb, over, phase)
#         # build_features returns (1, n_features), we need (1, n_features) for prediction
        
#         # Predict Classifiers
#         for label, clf in classifiers.items():
#             pred_label = clf.predict(xi)[0]
#             record[label] = pred_label # e.g. record['line'] = 'wide'

#         # Predict Speed
#         pred_speed = speed_model.predict(xi)[0]
#         record['speed'] = round(pred_speed, 1)

#         output_rows.append(record)

#     # 6. Save Output
#     out_path = Path(args.output)
#     print(f"Saving results to {out_path}...")
#     with open(out_path, 'w', encoding='utf-8') as f:
#         for row in output_rows:
#             f.write(json.dumps(row) + '\n')
#     print("Done.")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help="Operation mode")
    
#     # Training args
#     parser.add_argument('--data', type=str, default='example_train.jsonl', help="Input file for training")
    
#     # Prediction args
#     parser.add_argument('--input', type=str, help="Input file for prediction")
#     parser.add_argument('--output', type=str, default='predictions.jsonl', help="Output file for prediction")
    
#     # Shared args
#     parser.add_argument('--model_dir', type=str, default='models', help="Directory to save/load models")
    
#     args = parser.parse_args()

#     if args.mode == 'train':
#         train(args)
#     elif args.mode == 'predict':
#         if not args.input:
#             print("Error: --input is required for prediction mode.")
#         else:
#             predict(args)

# import argparse
# import json
# from pathlib import Path
# import numpy as np
# import joblib
# import sys

# # Ensure utils are discoverable
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from utils.text_utils import extract_speed_kph, innings_phase_from_over
# from utils.model_utils import load_embedding_model, build_features


# def load_resources(models_dir: str):
#     """
#     Loads the embedding model, classifiers, and speed regression model once.
#     """
#     print(f"Loading resources from {models_dir}...", file=sys.stderr)
    
#     # Load Embedder
#     embedder = load_embedding_model()
    
#     # Load Models
#     models_path = Path(models_dir)
#     classifiers = joblib.load(models_path / 'classifiers.joblib')
    
#     speed_info = None
#     try:
#         speed_info = joblib.load(models_path / 'speed_model.joblib')
#     except Exception:
#         speed_info = None
        
#     return embedder, classifiers, speed_info


# def infer_row(text: str, over: float, embedder, classifiers, speed_info):
#     """
#     Performs inference for a single row of data.
#     Returns a dict with only speed, line, and length.
#     """
#     # 1. Encode text and build features
#     emb = embedder.encode([text])[0]
#     phase = innings_phase_from_over(over)
#     X, _ = build_features(emb, over, phase)
#     X_row = X[0]

#     output = {
#         'speed': None,
#         'line': None,
#         'length': None
#     }

#     # 2. Predict Classifiers (Line & Length)
#     # We iterate through classifiers and only store 'line' and 'length' predictions
#     for name, clf in classifiers.items():
#         if name in ['line', 'length']:
#             probs = clf.predict_proba(X_row.reshape(1, -1))[0]
#             classes = clf.classes_
#             top_idx = int(np.argmax(probs))
#             output[name] = str(classes[top_idx])

#     # 3. Predict Speed
#     # Priority: Explicit extraction -> Regression Model -> None
#     speed_extracted = extract_speed_kph(text)
    
#     if speed_extracted is not None:
#         output['speed'] = float(speed_extracted)
#     elif speed_info is not None:
#         model = speed_info['model']
#         # Average prediction from trees (RandomForest logic)
#         tree_preds = np.stack([t.predict(X_row.reshape(1, -1)).ravel() for t in model.estimators_], axis=0)
#         output['speed'] = float(tree_preds.mean())
    
#     return output


# def process_file(input_path: str, output_path: str, models_dir: str):
#     # Load models once
#     embedder, classifiers, speed_info = load_resources(models_dir)

#     processed_count = 0
#     with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
#         for line in f_in:
#             line = line.strip()
#             if not line:
#                 continue

#             try:
#                 data = json.loads(line)
                
#                 # Extract inputs (expecting 'commentary' and 'over')
#                 text = data.get('commentary', '')
#                 over = float(data.get('over', 0.0))
                
#                 # Infer
#                 result = infer_row(text, over, embedder, classifiers, speed_info)
                
#                 # Write to output as JSONL
#                 f_out.write(json.dumps(result) + '\n')
#                 processed_count += 1
                
#             except json.JSONDecodeError:
#                 print(f"Skipping invalid JSON line: {line[:50]}...", file=sys.stderr)
#             except Exception as e:
#                 print(f"Error processing line: {e}", file=sys.stderr)

#     print(f"Processing complete. {processed_count} lines written to {output_path}", file=sys.stderr)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--models', type=str, default='models', help="Directory containing model files")
#     parser.add_argument('--input', type=str, required=True, help="Path to input JSONL file")
#     parser.add_argument('--output', type=str, required=True, help="Path to output JSONL file")
    
#     args = parser.parse_args()
    
#     process_file(args.input, args.output, args.models)

import argparse
import json
import time
from pathlib import Path
import numpy as np
import joblib
import sys

# Ensure utils are discoverable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import extract_speed_kph, innings_phase_from_over
from utils.model_utils import load_embedding_model, build_features


def load_resources(models_dir: str):
    """
    Loads the embedding model, classifiers, and speed regression model once.
    """
    print(f"Loading resources from {models_dir}...", file=sys.stderr)
    
    # Load Embedder
    embedder = load_embedding_model()
    
    # Load Models
    models_path = Path(models_dir)
    classifiers = joblib.load(models_path / 'classifiers.joblib')
    
    speed_info = None
    try:
        speed_info = joblib.load(models_path / 'speed_model.joblib')
    except Exception:
        speed_info = None
        
    return embedder, classifiers, speed_info


def infer_row(text: str, over: float, embedder, classifiers, speed_info):
    """
    Performs inference for a single row of data.
    Returns a dict with only speed, line, and length.
    """
    # 1. Encode text and build features
    emb = embedder.encode([text])[0]
    phase = innings_phase_from_over(over)
    X, _ = build_features(emb, over, phase)
    X_row = X[0]

    output = {
        'speed': None,
        'line': None,
        'length': None
    }

    # 2. Predict Classifiers (Line & Length)
    for name, clf in classifiers.items():
        if name in ['line', 'length']:
            probs = clf.predict_proba(X_row.reshape(1, -1))[0]
            classes = clf.classes_
            top_idx = int(np.argmax(probs))
            output[name] = str(classes[top_idx])

    # 3. Predict Speed
    speed_extracted = extract_speed_kph(text)
    
    if speed_extracted is not None:
        output['speed'] = float(speed_extracted)
    elif speed_info is not None:
        model = speed_info['model']
        tree_preds = np.stack([t.predict(X_row.reshape(1, -1)).ravel() for t in model.estimators_], axis=0)
        output['speed'] = float(tree_preds.mean())
    
    return output


def process_file_continuous(input_path: str, output_path: str, models_dir: str):
    # Load models once
    embedder, classifiers, speed_info = load_resources(models_dir)

    print(f"Starting continuous processing. Reading from {input_path}...", file=sys.stderr)

    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                
                # Extract inputs
                text = data.get('commentary', '')
                over = float(data.get('over', 0.0))
                
                # Infer
                result = infer_row(text, over, embedder, classifiers, speed_info)
                
                # Overwrite the output file with the current ball's data
                # We open and close the file inside the loop to ensure Unity sees the update
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    json.dump(result, f_out, indent=2)
                
                print(f"Updated {output_path} for over {over}. Waiting 5 seconds...", file=sys.stderr)
                
                # Wait for 5 seconds before processing the next ball
                time.sleep(5)
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...", file=sys.stderr)
            except Exception as e:
                print(f"Error processing line: {e}", file=sys.stderr)

    print("End of input file reached.", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='models', help="Directory containing model files")
    parser.add_argument('--input', type=str, required=True, help="Path to input JSONL file")
    parser.add_argument('--output', type=str, required=True, help="Path to output JSON file (will be overwritten repeatedly)")
    
    args = parser.parse_args()
    
    process_file_continuous(args.input, args.output, args.models)