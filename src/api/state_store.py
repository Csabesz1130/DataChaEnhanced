import json
from pathlib import Path
from typing import Dict, List, Optional


class LabelStore:
    def __init__(self, path: str = "analysis_history.json") -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")
        self._data: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        try:
            raw = self.path.read_text(encoding="utf-8").strip() or "{}"
            data = json.loads(raw)
            # data: { filepath: label }
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def set_label(self, filepath: str, label: str) -> None:
        self._data[str(filepath)] = str(label)
        self.save()

    def set_many(self, labels: Dict[str, str]) -> None:
        self._data.update({str(k): str(v) for k, v in labels.items()})
        self.save()

    def get_label(self, filepath: str) -> Optional[str]:
        return self._data.get(str(filepath))

    def all(self) -> Dict[str, str]:
        return dict(self._data)


class Cache:
    def __init__(self) -> None:
        self.features: Dict[str, Dict] = {}