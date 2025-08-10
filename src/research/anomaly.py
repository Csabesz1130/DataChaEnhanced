# src/research/anomaly.py
from typing import Dict, Any
import numpy as np


def detect_anomalies(signal: Dict[str, Any], window: int = 101, k: float = 6.0) -> Dict[str, Any]:
    """
    Egyszerű anomália detektor patch-clamp jelre.
    - gördülő medián + MAD alapú küszöbölés
    - visszaad indexeket és metrikákat
    """
    t = signal.get('time')
    y = signal.get('current')
    if t is None or y is None or len(y) < max(window, 20):
        return {"indices": [], "count": 0}

    y = np.asarray(y)
    t = np.asarray(t)

    # padding a gördülő mediánhoz
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode='edge')
    med = np.array([
        np.median(y_pad[i:i+window]) for i in range(len(y))
    ])
    mad = np.array([
        np.median(np.abs(y_pad[i:i+window] - np.median(y_pad[i:i+window])))
        for i in range(len(y))
    ]) * 1.4826

    thr = k * (mad + 1e-9)
    resid = np.abs(y - med)
    idx = np.where(resid > thr)[0]

    return {
        "indices": idx.tolist(),
        "count": int(len(idx)),
        "threshold_mean": float(np.mean(thr)),
        "residual_mean": float(np.mean(resid)),
    }