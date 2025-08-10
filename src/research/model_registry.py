# src/research/model_registry.py
from typing import Callable, Dict, Any

class ModelRegistry:
    """
    Egyszerű modell-regiszter. A modellek callable-ek, amelyek a következő
    aláírást követik: fn(signal: Dict[str, Any]) -> Dict[str, Any]
    """
    def __init__(self):
        self._models: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register(self, name: str, model_fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        if not callable(model_fn):
            raise TypeError("Model function must be callable")
        self._models[name] = model_fn

    def available(self):
        return sorted(self._models.keys())

    def run(self, name: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        return self._models[name](signal)

    def run_all(self, signal: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {name: fn(signal) for name, fn in self._models.items()}