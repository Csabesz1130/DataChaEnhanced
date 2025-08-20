from __future__ import annotations

import math
from typing import List, Optional

from src.utils.logger import app_logger


class DriftMonitor:
    """Egyszerű KS‑szerű drift becslés listás vektorokra, külső csomagok nélkül."""

    @staticmethod
    def _cdf(values: List[float]) -> List[float]:
        if not values:
            return []
        xs = sorted(values)
        n = len(xs)
        cdf = []
        for i, _ in enumerate(xs, start=1):
            cdf.append(i / n)
        return cdf

    @staticmethod
    def ks_distance(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = sorted(a), sorted(b)
        ia = ib = 0
        na, nb = len(sa), len(sb)
        da = db = 0
        dmax = 0.0
        while ia < na and ib < nb:
            if sa[ia] <= sb[ib]:
                ia += 1
            else:
                ib += 1
            da = ia / na
            db = ib / nb
            dmax = max(dmax, abs(da - db))
        return dmax

    def check_drift(self, baseline: List[List[float]], recent: List[List[float]], threshold: float = 0.25) -> float:
        # projekció: az első dimenziót hasonlítjuk (egyszerűsítés)
        base_1d = [v[0] for v in baseline if v and not math.isnan(v[0])]
        recent_1d = [v[0] for v in recent if v and not math.isnan(v[0])]
        d = self.ks_distance(base_1d, recent_1d)
        app_logger.info(f"Drift KS distance: {d:.3f} (threshold {threshold})")
        return d


