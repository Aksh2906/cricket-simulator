# Cricket commentary → bowling parameters

**Research-backed prototype** that converts ball-by-ball cricket commentary into structured bowling parameters for a Unity simulation engine.

## Overview

This system uses **sentence embeddings + ML classifiers + probabilistic inference** to parse natural language cricket commentary and output:
- **speed** (km/h) — extracted or regression-predicted
- **line** (off / middle / leg / wide)
- **length** (yorker / full / good / short)
- **swing** (inswing / outswing / none)
- **confidence scores** per prediction

## Design Philosophy

**Not keyword-based.** The system:
1. Uses `SentenceTransformers` to convert commentary text into dense embeddings
2. Appends lightweight contextual features (over number, innings phase)
3. Trains sklearn classifiers (`LogisticRegression` for line/length/swing, `RandomForestRegressor` for speed)
4. Returns full probability distributions and uncertainty estimates

This approach generalizes better than regex/keyword rules and allows the model to learn semantic patterns.

## Architecture

```
data/
  └─ example_train.jsonl       # 16 labeled examples (line, length, swing, speed)

models/
  ├─ classifiers.joblib         # Line, Length, Swing (LogisticRegression)
  └─ speed_model.joblib         # Speed regressor (RandomForest) + uncertainty

training/
  └─ train_models.py            # Train all classifiers and regressor

inference/
  └─ run_inference.py           # Load models + predict from commentary

utils/
  ├─ text_utils.py              # Speed extraction + phase inference
  └─ model_utils.py             # Embeddings + feature engineering
```

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train models

```bash
python training/train_models.py --data data/example_train.jsonl --out models/
```

Expects JSONL with fields:
```json
{
  "commentary": "Beautiful full delivery, 132 kph",
  "over": 3.0,
  "innings": 1,
  "speed_kph": 132,
  "line": "middle",
  "length": "full",
  "swing": "none"
}
```

### 3. Run inference

```bash
python inference/run_inference.py --models models/ --text "Short and wide, batsman scoops, 137 kph" --over 19
```

**Output JSON** (ready for Unity):
```json
{
  "commentary": "Short and wide, batsman scoops, 137 kph",
  "over": 19.0,
  "phase": "death",
  "predictions": {
    "line": {
      "label": "middle",
      "confidence": 0.362,
      "all_probs": {"leg": 0.162, "middle": 0.362, "off": 0.226, "wide": 0.249}
    },
    "length": {
      "label": "good",
      "confidence": 0.294,
      "all_probs": {"full": 0.234, "good": 0.294, "short": 0.260, "yorker": 0.212}
    },
    "swing": {
      "label": "none",
      "confidence": 0.774,
      "all_probs": {"inswing": 0.133, "none": 0.774, "outswing": 0.093}
    },
    "speed": {
      "speed_kph": 137.0,
      "confidence": 0.99,
      "method": "extracted"
    }
  }
}
```

## ML Pipeline

### Feature Engineering

1. **Text embedding**: Convert commentary via `SentenceTransformers` (`all-MiniLM-L6-v2`, ~90MB)
2. **Context features**:
   - Over number (normalized 0–50)
   - Innings phase (one-hot: powerplay, middle, death)
3. **Combined**: Embedding + context → 389-dim feature vector

### Speed handling

- **If explicit speed in text** (regex `\d{2,3}\s*kph`): Extract directly, confidence 0.99
- **If missing**: Use RandomForest regressor trained on labeled examples
  - Output: predicted speed + per-tree uncertainty (via `std` of tree predictions)
  - Confidence = 1 − (std / max_speed), clamped [0, 1]

### Line / Length / Swing prediction

- **LogisticRegression** multiclass (fast, interpretable, ~10 parameters per class)
- Output: class label + softmax probabilities for all classes

## Example Dataset

See `data/example_train.jsonl` (16 samples) with diverse:
- Over phases (powerplay → middle → death)
- Ball types (yorker, full, good, short)
- Lines (leg, middle, off, wide)
- Speeds (115–142 kph)
- Swing modes

To train on real data, collect commentary with crowdsourced annotations or weak labels.

## Performance Notes

- **Dataset**: 16 examples → ~50% accuracy on validation (expected; needs 100–500 examples)
- **Speed MAE**: ~2.3 kph (regressor uncertainty ~7–8 kph)
- **Model size**: ~100 KB (classifiers + metadata)
- **Inference latency**: ~200 ms/ball (embedding download on first run, then cached)
- **CPU-friendly**: Runs on laptop; embedding model fits in memory

## Integration with Unity

1. **Input**: Commentary text (string) + over number (float)
2. **Process**: Call `python inference/run_inference.py --models models/ --text "..." --over X`
3. **Output**: JSON with predictions and confidences
4. **Unity workflow**:
   - Deserialize JSON
   - Use `predictions.line.label`, `predictions.speed.speed_kph`, etc.
   - Threshold on confidence scores (e.g., only accept line if confidence > 0.5)
   - For missing predictions (confidence too low), use defaults

Example JSON format in `example_outputs.json`.

## Extensions

- **Expand training data**: Collect / annotate more commentary
- **Add more attributes**: Seam/spin info, batsman response, outcome
- **Fine-tune embeddings**: Train a task-specific embedding model
- **Ensemble**: Combine multiple models or feature sets
- **Uncertainty calibration**: Tune confidence thresholds per use case

## Requirements

- Python 3.9+
- `sentence-transformers>=2.2.2`
- `scikit-learn>=1.2.0`
- `joblib>=1.2.0`
- `numpy>=1.24.0`, `pandas>=1.5.0`, `tqdm>=4.64.0`

## License

Research prototype. Use freely for non-commercial purposes.
