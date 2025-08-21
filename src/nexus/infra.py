from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.logger import app_logger


class CeleryInfra:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.celery_app = None
        self.redis_client = None
        self._init()

    def _init(self) -> None:
        try:
            from celery import Celery  # type: ignore
            from redis import Redis  # type: ignore

            self.celery_app = Celery(
                "nexus_cognition",
                broker=self.config.get("redis_url", "redis://localhost:6379/0"),
                backend=self.config.get("redis_url", "redis://localhost:6379/0"),
            )
            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                worker_prefetch_multiplier=1,
                task_acks_late=True,
            )
            self.redis_client = Redis.from_url(self.config.get("redis_url"))
            app_logger.info("Celery/Redis initialized")
        except Exception as exc:
            app_logger.warning(f"Celery/Redis unavailable: {exc}")


