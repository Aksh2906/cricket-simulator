# Cricket NLP System: Architecture & ML Design

## Problem Statement

Convert free-text ball-by-ball cricket commentary into structured bowling parameters:
- **Speed** (km/h)
- **Line** (off / middle / leg / wide)
- **Length** (yorker / full / good / short)
- **Swing** (inswing / outswing / none)

Each prediction must include a **confidence score** for downstream filtering/thresholding.

## Design Principles

1. **No hand-crafted rules**: Avoid regex-only or keyword-matching approaches
2. **Semantic understanding**: Use neural embeddings to capture meaning beyond keywords
3. **Probabilistic outputs**: Return full distributions, not just hard labels
4. **Lightweight inference**: CPU-friendly, fast enough for real-time simulation
5. **Uncertainty quantification**: Honest confidence scores that reflect model doubt

## Technical Architecture

### 1. Text Representation: Sentence Embeddings

**Model**: `SentenceTransformers` (`all-MiniLM-L6-v2`)

**Why this choice?**
- Small (~90 MB), fast (~200 ms/embedding), trained on diverse downstream tasks
- 384-dim vectors: dense, semantic representation
- Pre-trained on millions of sentence pairs → captures general cricket language patterns
- No task-specific fine-tuning needed for prototype

**Example embeddings similarity** (cosine):
- "Short and wide" vs "Yorker" → 0.35 (dissimilar)
- "Full delivery" vs "Full ball" → 0.92 (similar)
- "Outswing" vs "Inswinger" → 0.72 (moderately similar)

**Key insight**: Embeddings alone can't fully distinguish cricket subtleties (e.g., "full" vs "good"), so we combine with explicit features.

### 2. Contextual Features

**Phase of Innings** (from over number):
```
Powerplay:  [1,1,0]   (overs 0-6)
Middle:     [0,1,0]   (overs 6-15)
Death:      [0,0,1]   (overs 15+)
```

**Why?** Cricket is phase-dependent:
- Powerplay: aggressive bowling, higher speeds, more short balls
- Middle: building pressure, varied tactics
- Death: yorkers/slower balls dominant

**Over number** (normalized 0–1):
```
over_normalized = min(over, 50) / 50
```

**Rationale**: Captures continuous trend. Most T20 matches finish by over 20, so capping at 50 avoids extreme outliers.

### 3. Combined Feature Vector

```
features = [embedding (384 dims), over_normalized, phase_one_hot (3 dims)]
         = 384 + 1 + 3 = 388 dimensions
```

**Processing**:
- Embedding scaled via StandardScaler
- Context features one-hot/normalized
- Stack horizontally

### 4. Speed Prediction: Regression

**Target**: Speed in km/h (numeric)

**Model**: `RandomForestRegressor` (10 trees)

**Why Random Forest?**
- Handles nonlinear speed patterns (e.g., speeds vary by over phase non-monotonically)
- Multi-tree ensemble allows uncertainty estimation (std of tree predictions)
- Fast inference on CPU
- Interpretable feature importance

**Uncertainty Quantification**:
```
predictions_per_tree = [tree_1_pred, tree_2_pred, ..., tree_10_pred]
mean_pred = mean(predictions_per_tree)
std_pred = std(predictions_per_tree)
```

**Confidence mapping** (heuristic):
```
confidence = max(0.0, min(0.99, 1.0 - (std_pred / max_speed)))
```

Example:
- If all trees predict 135±2 kph → low std → confidence ≈ 0.98
- If trees predict 120±15 kph → high std → confidence ≈ 0.70

**Speed Extraction Override**:
If commentary contains explicit speed (e.g., "137 kph"), extract via regex:
```
regex: r"(\d{2,3})(?:\s*(?:kph|km/h|kmh|kph\.))"
confidence: 0.99 (direct observation)
method: "extracted"
```

### 5. Line / Length / Swing Classification

**Model**: `LogisticRegression` (multiclass, multinomial)

**Why Logistic Regression?**
- Outputs well-calibrated probabilities (softmax)
- Inherently multiclass (avoids one-vs-rest inefficiency)
- Interpretable: linear + softmax = transparent decision boundary
- Fast training/inference

**Per-class training**:
```
Line:   classes = [leg, middle, off, wide]
Length: classes = [yorker, full, good, short]
Swing:  classes = [inswing, outswing, none]
```

**Output**: 
```json
{
  "label": "off",
  "confidence": 0.43,
  "all_probs": {"leg": 0.16, "middle": 0.36, "off": 0.43, "wide": 0.25}
}
```

**Softmax interpretation**: All probabilities sum to 1.0; top class is most likely.

## Training Pipeline

### Data Format (JSONL)

```json
{
  "commentary": "Short and wide, 137 kph",
  "over": 19.0,
  "innings": 1,
  "speed_kph": 137,
  "line": "wide",
  "length": "short",
  "swing": "none"
}
```

### Missing Labels

If a ball lacks a field (e.g., no `speed_kph`), that sample is excluded from speed training but included in classifier training.

**Example**: 
```json
{"commentary": "Slower ball, full", "over": 16, "line": "off", "length": "full", "swing": "none"}
// No speed_kph → skip speed training, but train line/length/swing
```

### Handling Small Datasets

With 16 examples:
- Some classes have ≤2 samples
- Stratified k-fold split fails

**Solution**: Graceful fallback
```python
try:
    train_test_split(..., stratify=y)
except ValueError:
    train_test_split(..., stratify=None)  # random split fails, use non-stratified
```

Also reduce RF trees (10 instead of 50) and test set (0% if <10 samples) to avoid sparse validation.

## Inference Pipeline

```
Input Commentary + Over
         ↓
   Encode via SentenceTransformers (384 dims)
         ↓
   Build context features (phase, over)
         ↓
   Stack into 388-dim feature vector
         ↓
   ┌─────────────────────────────────┐
   │  Try regex speed extraction?    │
   │  Yes → return confidence=0.99   │
   │  No → RF regressor → (mean, std)│
   └─────────────────────────────────┘
         ↓
   Line classifier → softmax probs
   Length classifier → softmax probs
   Swing classifier → softmax probs
         ↓
   JSON output with all probabilities + confidence
```

## Example: "Short and wide, 137 kph" at over 19

1. **Embedding**: Text → 384-dim vector via SentenceTransformers
   - Keywords detected: "short", "wide", "kph"
   - Semantic meaning: attacking delivery, batsman in danger
   
2. **Context**: over=19 → phase="death" → [0,0,1]

3. **Speed**: Regex finds "137 kph" → extracted, confidence=0.99

4. **Line classifier**:
   - Input: 388-dim vector
   - Logit scores: leg=-1.2, middle=+0.5, off=-0.3, wide=+0.1
   - After softmax: leg=0.16, middle=0.36, off=0.23, wide=0.25
   - Top: "middle" (0.36)
   
5. **Length classifier**:
   - Logit scores favor "short" = 0.26, good=0.24, full=0.23, yorker=0.27
   - Top: "short" (but close!)
   
6. **Swing classifier**:
   - Clear separation: none=0.77, inswing=0.13, outswing=0.10
   - Top: "none" (0.77)

7. **Output**:
```json
{
  "commentary": "Short and wide, 137 kph",
  "over": 19.0,
  "phase": "death",
  "predictions": {
    "speed": {"speed_kph": 137, "confidence": 0.99, "method": "extracted"},
    "line": {"label": "middle", "confidence": 0.36, "all_probs": {...}},
    "length": {"label": "short", "confidence": 0.26, "all_probs": {...}},
    "swing": {"label": "none", "confidence": 0.77, "all_probs": {...}}
  }
}
```

## Limitations & Future Work

### Current prototype limitations:
1. **Tiny dataset** (16 examples) → classifier accuracy ~50%
2. **No fine-tuning** on cricket-specific embeddings
3. **Limited attributes** (could add seam vs spin, outcome, batsman response)
4. **No calibration** of confidence scores on held-out test set

### Improvements for production:
1. **Data**: Collect/annotate 500–5000 examples from commentary corpora
2. **Fine-tuning**: Train custom embedding model on cricket commentary
3. **Ensemble**: Combine multiple models (transformer-based + traditional ML)
4. **Calibration**: Use Platt scaling or isotonic regression on validation set
5. **Active learning**: Prioritize labeling hard/uncertain examples
6. **External features**: Batsman handedness, bowler type, match situation

## References

- **SentenceTransformers**: https://www.sbert.net/
- **Embedding models**: https://huggingface.co/sentence-transformers
- **sklearn**: https://scikit-learn.org/

## Author Notes

This is a **research prototype** demonstrating how modern NLP techniques (embeddings + probabilistic ML) can be applied to domain-specific text understanding, without hard-coded rules. The architecture is modular and extensible—swapping in a BERT-based classifier or fine-tuned embeddings is straightforward.
