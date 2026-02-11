from typing import Dict
import re

# Text helper utilities: minimal rules for numeric extraction (speed) and phase inference

def extract_speed_kph(text: str):
    """Try to extract numeric speed mentioned in commentary.

    This is not the primary prediction method; it's only used when a numeric speed
    is explicitly present in the commentary. We avoid keyword-only approaches by
    relying on embeddings+models for main predictions.
    """
    if not text:
        return None
    # look for patterns like '137 kph', '137 km/h', or '137kmh' or just '137'
    m = re.search(r"(\d{2,3})(?:\s*(?:kph|km/h|kmh|kph\.|kph))", text.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # fallback: find isolated 2-3 digit number with 'k' or 'km' next to it
    m2 = re.search(r"(\d{2,3})\s*(kph|km|km/h|kmh)?", text.lower())
    if m2:
        # be conservative: only accept if 'k' token appeared nearby
        if m2.group(2):
            return float(m2.group(1))
    return None

def innings_phase_from_over(over: float) -> str:
    """Simple heuristic to derive phase of innings from over number.

    - powerplay: 0-6
    - middle: 6-15
    - death: 15+

    These are crude cricket-logic features used as additional inputs to ML models
    when commentary lacks explicit contextual clues.
    """
    try:
        o = float(over)
    except Exception:
        return "unknown"
    if o <= 6:
        return "powerplay"
    if o <= 15:
        return "middle"
    return "death"
