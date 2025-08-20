from __future__ import annotations

from typing import List, Tuple

from src.utils.logger import app_logger


class AnomalyDetector:
    """
    Unsupervised anomaly detector.
    - If scikit-learn IsolationForest is available, use it.
    - Otherwise, fall back to simple z-score on vector norms.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self._use_iforest = False
        self._iforest = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore

            self._iforest = IsolationForest(
                contamination=self.contamination, random_state=self.random_state
            )
            self._use_iforest = True
            app_logger.info("AnomalyDetector: using IsolationForest")
        except Exception as exc:
            app_logger.warning(
                f"AnomalyDetector: sklearn not available ({exc}); using z-score fallback"
            )

    def fit(self, embeddings: List[List[float]]) -> None:
        if not embeddings:
            return
        if self._use_iforest:
            self._iforest.fit(embeddings)

    def predict(self, embeddings: List[List[float]]) -> Tuple[List[int], List[float]]:
        if not embeddings:
            return ([], [])
        if self._use_iforest:
            labels = self._iforest.predict(embeddings)  # 1 normal, -1 anomaly
            try:
                scores = self._iforest.score_samples(embeddings)
            except Exception:
                scores = [0.0 for _ in embeddings]
            return (list(labels), list(scores))

        # Fallback: z-score on vector norms
        norms = [sum(v * v for v in vec) ** 0.5 for vec in embeddings]
        if not norms:
            return ([], [])
        mean = sum(norms) / len(norms)
        var = sum((n - mean) ** 2 for n in norms) / max(1, len(norms) - 1)
        std = var ** 0.5 if var > 0 else 1.0
        zscores = [(n - mean) / std for n in norms]
        # mark anomalies with |z| > 3
        labels = [(-1 if abs(z) > 3.0 else 1) for z in zscores]
        return (labels, zscores)


