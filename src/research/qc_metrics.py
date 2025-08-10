# src/research/qc_metrics.py
from typing import Dict, Any
import numpy as np


def compute_qc_metrics(signal: Dict[str, Any]) -> Dict[str, float]:
    """
    Alap QC metrikák: SNR, drift, outlier arány, mintavételezési stabilitás.
    Várja: {
      'time': np.ndarray (s),
      'current': np.ndarray (pA)
    }
    """
    t = signal.get('time')
    y = signal.get('current')
    if t is None or y is None or len(y) < 10:
        return {"error": 1.0}

    y = np.asarray(y)
    t = np.asarray(t)

    # SNR: jel RMS / zaj RMS (nagyobb jobb)
    # durva becslés: zaj = felső kvantilis- detrend
    y_detr = y - np.polyval(np.polyfit(t, y, deg=1), t)
    noise_std = np.median(np.abs(y_detr - np.median(y_detr))) * 1.4826
    signal_rms = np.sqrt(np.mean(np.square(y)))
    snr = (signal_rms / (noise_std + 1e-9))

    # Drift: lineáris trend meredeksége pA/s abszolút értékben
    slope = np.polyfit(t, y, deg=1)[0]
    drift = abs(slope)

    # Outlier arány: 5*noise_std fölötti pontok
    outlier_ratio = float(np.mean(np.abs(y_detr) > 5 * (noise_std + 1e-9)))

    # Mintavételi stabilitás: medián dt szórása
    dt = np.diff(t)
    sampling_jitter = float(np.std(dt) / (np.median(dt) + 1e-12))

    return {
        "snr": float(snr),
        "drift_pa_per_s": float(drift),
        "outlier_ratio": float(outlier_ratio),
        "sampling_jitter": sampling_jitter,
    }