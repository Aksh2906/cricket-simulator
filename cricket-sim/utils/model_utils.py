from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def build_features(embeddings: np.ndarray, over: float, innings_phase: str, scaler: StandardScaler=None) -> Tuple[np.ndarray, StandardScaler]:
    """Combine embedding vector with simple numeric/context features.

    We append normalized `over` and a small one-hot-ish encoding for `innings_phase`.
    This keeps models lightweight while allowing context to influence predictions.
    """
    phase_map = {"powerplay": 0, "middle": 1, "death": 2, "unknown": -1}
    phase_val = phase_map.get(innings_phase, -1)     
    # shape handling
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    n = embeddings.shape[0]
    over_arr = np.zeros((n, 1))
    over_arr[:, 0] = np.clip(np.array([over]*n, dtype=float), 0, 50) / 50.0
    phase_arr = np.zeros((n, 3))
    if phase_val >= 0:
        phase_arr[:, phase_val] = 1.0
    X = np.hstack([embeddings, over_arr, phase_arr])
    if scaler is None:
        scaler = StandardScaler()
        X[:, -4:-1] = scaler.fit_transform(X[:, -4:-1])
    else:
        X[:, -4:-1] = scaler.transform(X[:, -4:-1])
    return X, scaler

def load_embedding_model(name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    # choose a small, fast model suitable for CPU inference
    return SentenceTransformer(name)
