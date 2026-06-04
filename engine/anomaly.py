"""
Anomaly detector.
Flags "out-of-distribution" queries: when the best hybrid match score
falls below the 5th-percentile threshold computed across the catalog.

If a query's best match score < threshold → the query is unusual / no good match.
"""

import numpy as np
import pandas as pd

# Precomputed or set at startup
_threshold: float = None


def calibrate(best_scores_array: np.ndarray, percentile: float = 5.0) -> float:
    """
    best_scores_array: 1-D array of best-match hybrid scores for all catalog
                       items (computed during startup by evaluate.py or app startup).
    Returns the threshold below which a query is "anomalous".
    """
    global _threshold
    _threshold = float(np.percentile(best_scores_array, percentile))
    return _threshold


def is_anomalous(best_match_score: float, threshold: float = None) -> bool:
    t = threshold if threshold is not None else _threshold
    if t is None:
        return False  # not calibrated → assume normal
    return best_match_score < t


def get_threshold() -> float:
    return _threshold
