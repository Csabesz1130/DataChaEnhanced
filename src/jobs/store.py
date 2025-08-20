from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Optional

from src.utils.logger import app_logger


class JobStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    QUARANTINED_SECURITY_RISK = "QUARANTINED_SECURITY_RISK"
    QUARANTINED_EXTRACTION_FAILED = "QUARANTINED_EXTRACTION_FAILED"
    FAILED = "FAILED"


@dataclass
class Job:
    id: str
    filepath: str
    status: JobStatus
    message: str = ""


class JobStore:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _read(self) -> Dict[str, dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write(self, data: Dict[str, dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def upsert(self, job: Job) -> None:
        data = self._read()
        data[job.id] = asdict(job)
        self._write(data)
        app_logger.info(f"Job {job.id} -> {job.status}")

    def get(self, job_id: str) -> Optional[Job]:
        data = self._read()
        raw = data.get(job_id)
        if not raw:
            return None
        return Job(
            id=raw["id"], filepath=raw["filepath"], status=JobStatus(raw["status"]), message=raw.get("message", "")
        )


