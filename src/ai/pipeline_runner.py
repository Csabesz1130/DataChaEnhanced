from __future__ import annotations

from typing import List, Tuple

from src.processors import ParserNexus
from src.ai.cognitive_core import CognitiveCore
from src.ai.anomaly_detector import AnomalyDetector
from src.utils.logger import app_logger


def run_embedding_and_anomaly(filepaths: List[str]) -> Tuple[List[List[float]], List[int]]:
    irs = [ParserNexus.parse_file(p) for p in filepaths]
    texts = ParserNexus.flatten_texts(irs)
    if not texts:
        app_logger.warning("No texts for embeddings")
        return ([], [])
    core = CognitiveCore()
    embeddings = core.fit_corpus(texts)
    detector = AnomalyDetector()
    detector.fit(embeddings)
    labels, _ = detector.predict(embeddings)
    anomalies = sum(1 for l in labels if l == -1)
    app_logger.info(f"Embeddings: {len(embeddings)} | anomalies: {anomalies}")
    return (embeddings, labels)


