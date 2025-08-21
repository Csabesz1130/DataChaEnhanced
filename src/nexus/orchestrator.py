from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime

from src.utils.logger import app_logger
from src.nexus.domain import ProcessingContext
from src.nexus.infra import CeleryInfra
from src.nexus.services import CognitionServices


class NexusCognitionEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_store: Dict[str, ProcessingContext] = {}
        self.infra = CeleryInfra(config)
        self.services = CognitionServices(config)
        app_logger.info("NexusCognitionEngine initialized")

    def process_document_sync(self, file_path: str, user_id: Optional[str] = None) -> ProcessingContext:
        ctx = ProcessingContext(user_id=user_id, file_path=file_path)
        self.session_store[ctx.session_id] = ctx
        ctx.processing_steps.append("ingest")
        # Parse + embed
        from src.processors import ParserNexus
        ir = ParserNexus.parse_file(file_path)
        texts = [c.text for c in ir.chunks]
        ctx.processing_steps.append("parsed")
        if not texts:
            ctx.error_log.append("No content extracted")
            return ctx
        emb = self.services.embed_texts(texts)
        ctx.processing_steps.append("embedded")
        # Simple confidence proxy: average length ratio
        avg_len = sum(len(t) for t in texts) / max(1, len(texts))
        ctx.confidence_scores["content_richness"] = min(1.0, avg_len / 200.0)
        ctx.anomaly_score = self.services.anomaly_scores(emb)
        ctx.processing_steps.append("anomaly")
        return ctx


_engine: Optional[NexusCognitionEngine] = None


def get_nexus_engine() -> NexusCognitionEngine:
    global _engine
    if _engine is None:
        _engine = NexusCognitionEngine(
            {
                "redis_url": "redis://localhost:6379/0",
                "embedding_model": "all-MiniLM-L6-v2",
                "uncertainty_threshold": 0.7,
                "drift_threshold": 0.1,
                "retrain_threshold": 100,
            }
        )
    return _engine


