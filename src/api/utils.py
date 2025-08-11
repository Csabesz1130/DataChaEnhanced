import os
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def resolve_path(p: str) -> str:
    pth = Path(p)
    if not pth.is_absolute():
        pth = (PROJECT_ROOT / p).resolve()
    return str(pth)


def list_data_files(exts: List[str] = [".atf"]) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(DATA_DIR):
        for name in filenames:
            if any(name.lower().endswith(e) for e in exts):
                files.append(str(Path(dirpath) / name))
    return sorted(files)