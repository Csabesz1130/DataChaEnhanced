from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from src.utils.logger import app_logger
from src.processors import ParserNexus
from src.ai.cognitive_core import CognitiveCore
from src.ai.anomaly_detector import AnomalyDetector
from src.nexus.domain import ConfidenceLevel


class CognitionServices:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.core = CognitiveCore()
        self.detector = AnomalyDetector()

    def embed_texts(self, texts: List[str]):
        if not texts:
            return []
        return self.core.fit_corpus(texts)

    def anomaly_scores(self, embeddings):
        self.detector.fit(embeddings)
        labels, scores = self.detector.predict(embeddings)
        # Convert to 0..1 anomaly score (rough heuristic)
        if not scores:
            return 0.0
        norms = [abs(s) for s in scores]
        mx = max(norms) or 1.0
        return float(np.mean([n / mx for n in norms]))

    def confidence_from_scores(self, scores: Dict[str, float]) -> ConfidenceLevel:
        if not scores:
            return ConfidenceLevel.UNCERTAIN
        avg = float(np.mean(list(scores.values())))
        if avg > 0.8:
            return ConfidenceLevel.HIGH
        if avg > 0.5:
            return ConfidenceLevel.MEDIUM
        if avg > 0.3:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNCERTAIN


