from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import app_logger


@dataclass
class AppConfig:
    # Security
    forbidden_types: tuple = (
        b"\x7fELF",  # Linux ELF
        b"MZ",       # Windows PE
    )
    signature_db_path: str = "config/signatures.json"

    # AI thresholds
    drift_ks_threshold: float = 0.25
    drift_baseline_path: str = "storage/drift_baseline.json"

    # Server / infra
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storage
    raw_storage_dir: str = "storage/raw"
    jobs_db_path: str = "storage/jobs.json"


def load_config(env: Optional[str] = None) -> AppConfig:
    """Load config from optional JSON files and env overrides."""
    cfg = AppConfig()
    # ensure dirs
    os.makedirs(cfg.raw_storage_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.jobs_db_path), exist_ok=True)

    # Optional JSON override
    json_path = os.getenv("APP_CONFIG_JSON", "")
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
        except Exception as exc:
            app_logger.warning(f"Config JSON load failed: {exc}")

    # ENV overrides (simple)
    api_port = os.getenv("APP_API_PORT")
    if api_port:
        try:
            cfg.api_port = int(api_port)
        except ValueError:
            pass
    return cfg


