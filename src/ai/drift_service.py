from __future__ import annotations

import json
import os
from typing import List, Tuple

from src.ai.cognitive_core import CognitiveCore
from src.monitoring.drift import DriftMonitor
from src.processors import ParserNexus
from src.config import load_config
from src.utils.logger import app_logger


class DriftService:
    def __init__(self):
        self.cfg = load_config()
        self.monitor = DriftMonitor()

    def _save_baseline(self, embeddings: List[List[float]]) -> None:
        os.makedirs(os.path.dirname(self.cfg.drift_baseline_path), exist_ok=True)
        with open(self.cfg.drift_baseline_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)
        app_logger.info("Drift baseline saved")

    def _load_baseline(self) -> List[List[float]]:
        if not os.path.exists(self.cfg.drift_baseline_path):
            return []
        try:
            with open(self.cfg.drift_baseline_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def build_embeddings(self, paths: List[str]) -> List[List[float]]:
        irs = [ParserNexus.parse_file(p) for p in paths]
        texts = ParserNexus.flatten_texts(irs)
        if not texts:
            return []
        core = CognitiveCore()
        return core.fit_corpus(texts)

    def update_baseline_from_folder(self, folder: str, exts=(".xlsx", ".xls", ".csv", ".pdf", ".docx", ".atf", ".txt")) -> int:
        import glob, os as _os
        files: List[str] = []
        for ext in exts:
            files.extend(glob.glob(_os.path.join(folder, f"*{ext}")))
        emb = self.build_embeddings(files)
        if emb:
            self._save_baseline(emb)
        return len(emb)

    def check_folder_against_baseline(self, folder: str) -> float:
        baseline = self._load_baseline()
        if not baseline:
            app_logger.warning("No baseline available; create one first")
            return 0.0
        import glob, os as _os
        files: List[str] = []
        for ext in (".xlsx", ".xls", ".csv", ".pdf", ".docx", ".atf", ".txt"):
            files.extend(glob.glob(_os.path.join(folder, f"*{ext}")))
        recent = self.build_embeddings(files)
        if not recent:
            app_logger.warning("No recent embeddings to compare")
            return 0.0
        d = self.monitor.check_drift(baseline, recent, threshold=self.cfg.drift_ks_threshold)
        if d > self.cfg.drift_ks_threshold:
            app_logger.warning("Drift threshold exceeded — retraining suggested")
        return d


