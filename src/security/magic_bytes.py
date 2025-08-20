from __future__ import annotations

from typing import Optional, Tuple

from src.utils.logger import app_logger


MAGIC_SIGNATURES: Tuple[Tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip"),  # xlsx, docx, pptx are zip
    (b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", "ole"),  # old office
    (b"\x50\x4B\x03\x04", "zip"),
    (b"MZ", "pe"),
    (b"\x7fELF", "elf"),
)


def detect_true_file_type(head: bytes) -> Optional[str]:
    for sig, name in MAGIC_SIGNATURES:
        if head.startswith(sig):
            return name
    return None


def is_forbidden_magic(head: bytes) -> bool:
    t = detect_true_file_type(head)
    if t in {"elf", "pe"}:
        app_logger.warning(f"Forbidden magic detected: {t}")
        return True
    return False


