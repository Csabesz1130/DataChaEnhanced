import os
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.io_utils.io_utils import ATFHandler
from src.utils.logger import app_logger


def _safe_downsample(arr: np.ndarray, max_points: int = 5000) -> np.ndarray:
    if arr is None:
        return np.array([])
    if len(arr) <= max_points:
        return arr
    step = max(1, len(arr) // max_points)
    return arr[::step]


@lru_cache(maxsize=256)
def load_signal(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load time and first trace (#1) from ATF and return numpy arrays.
    Downsampled to a manageable size for transport to frontend.
    """
    handler = ATFHandler(filepath)
    handler.load_atf()
    time_s = handler.get_column("Time")
    current_pA = handler.get_column("#1")

    time_s = np.asarray(time_s)
    current_pA = np.asarray(current_pA)

    # Downsample for transport
    ds_time = _safe_downsample(time_s)
    ds_current = _safe_downsample(current_pA)

    if len(ds_time) != len(ds_current):
        min_len = min(len(ds_time), len(ds_current))
        ds_time = ds_time[:min_len]
        ds_current = ds_current[:min_len]

    return ds_time, ds_current


def extract_basic_features(time_s: np.ndarray, current_pA: np.ndarray) -> Dict[str, float]:
    """Compute lightweight, generic features from a single trace."""
    if time_s is None or current_pA is None or len(current_pA) == 0:
        return {}

    x = np.asarray(current_pA, dtype=float)
    # basic stats
    mean = float(np.mean(x))
    std = float(np.std(x))
    ptp = float(np.ptp(x))
    max_val = float(np.max(x))
    min_val = float(np.min(x))
    # energy and roughness
    energy = float(np.mean(x ** 2))
    diff_energy = float(np.mean(np.diff(x) ** 2)) if len(x) > 1 else 0.0

    # percentiles
    p10 = float(np.percentile(x, 10))
    p50 = float(np.percentile(x, 50))
    p90 = float(np.percentile(x, 90))

    # simple integral approx (pA*ms ~ pC), assume uniform step
    if len(time_s) > 1:
        dt_ms = float(np.mean(np.diff(time_s)) * 1000.0)
    else:
        dt_ms = 0.1
    integral_pC = float(np.sum(x) * dt_ms / 1000.0)

    return {
        "mean": mean,
        "std": std,
        "ptp": ptp,
        "max": max_val,
        "min": min_val,
        "energy": energy,
        "diff_energy": diff_energy,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "integral_pC": integral_pC,
    }


def extract_features_for_file(filepath: str) -> Dict[str, float]:
    try:
        time_s, current_pA = load_signal(filepath)
        features = extract_basic_features(time_s, current_pA)
        features["num_points"] = int(len(current_pA))
        return features
    except Exception as e:
        app_logger.error(f"Feature extraction failed for {filepath}: {e}")
        return {}


def batch_extract_features(filepaths: List[str]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    index: List[str] = []
    for fp in filepaths:
        feats = extract_features_for_file(fp)
        records.append(feats)
        index.append(fp)
    df = pd.DataFrame.from_records(records, index=index)
    df.index.name = "path"
    return df


def list_atf_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".atf"):
                files.append(os.path.join(dirpath, name))
    return sorted(files)