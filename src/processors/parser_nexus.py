from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from src.utils.logger import app_logger


@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentIR:
    doc_id: str
    source_path: str
    doc_type: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseParser:
    def parse(self, filepath: str) -> DocumentIR:
        raise NotImplementedError


class ExcelParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"ExcelParser: parsing {filepath}")
        if pd is None:
            app_logger.warning("Pandas nem elérhető, Excel feldolgozás kihagyva.")
            return DocumentIR(
                doc_id=os.path.basename(filepath),
                source_path=filepath,
                doc_type="excel",
                chunks=[DocumentChunk(text="Excel fájl tartalma nem érhető el pandas nélkül.")],
            )
        xls = pd.ExcelFile(filepath)
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="excel",
            metadata={"sheets": xls.sheet_names},
        )
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                df = df.convert_dtypes()
                # Create contextual row sentences
                header_list = list(df.columns)
                for row_idx, row in df.iterrows():
                    parts: List[str] = []
                    for col in header_list:
                        val = row[col]
                        try:
                            is_na = pd.isna(val)
                        except Exception:
                            is_na = False
                        if is_na:
                            continue
                        parts.append(f"{col}={val}")
                    if not parts:
                        continue
                    text = f"Sheet {sheet_name} | Row {row_idx+1}: " + "; ".join(parts)
                    chunk = DocumentChunk(
                        text=text,
                        metadata={
                            "sheet": sheet_name,
                            "row_index": int(row_idx),
                            "columns": header_list,
                            "source": filepath,
                        },
                    )
                    ir.chunks.append(chunk)
            except Exception as exc:
                app_logger.warning(f"ExcelParser: sheet '{sheet_name}' parse failed: {exc}")
        return ir


class CSVParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"CSVParser: parsing {filepath}")
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="csv",
        )
        if pd is not None:
            try:
                df = pd.read_csv(filepath)
                df = df.convert_dtypes()
                header_list = list(df.columns)
                for row_idx, row in df.iterrows():
                    parts: List[str] = []
                    for col in header_list:
                        val = row[col]
                        try:
                            is_na = pd.isna(val)
                        except Exception:
                            is_na = False
                        if is_na:
                            continue
                        parts.append(f"{col}={val}")
                    if not parts:
                        continue
                    text = f"CSV | Row {row_idx+1}: " + "; ".join(parts)
                    chunk = DocumentChunk(
                        text=text,
                        metadata={
                            "row_index": int(row_idx),
                            "columns": header_list,
                            "source": filepath,
                        },
                    )
                    ir.chunks.append(chunk)
                return ir
            except Exception as exc:
                app_logger.warning(f"CSVParser: pandas parse failed ({exc}), fallback a csv modulra")
        # Fallback: beépített csv modul
        import csv
        try:
            with open(filepath, newline='', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return ir
            header_list = rows[0]
            for row_idx, row in enumerate(rows[1:]):
                parts: List[str] = []
                for col_idx, val in enumerate(row):
                    if val is None or str(val).strip() == "":
                        continue
                    col = header_list[col_idx] if col_idx < len(header_list) else f"col{col_idx+1}"
                    parts.append(f"{col}={val}")
                if not parts:
                    continue
                text = f"CSV | Row {row_idx+1}: " + "; ".join(parts)
                chunk = DocumentChunk(
                    text=text,
                    metadata={
                        "row_index": int(row_idx),
                        "columns": header_list,
                        "source": filepath,
                    },
                )
                ir.chunks.append(chunk)
        except Exception as exc:
            app_logger.warning(f"CSVParser fallback sikertelen: {exc}")
        return ir


class ATFParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"ATFParser: parsing {filepath}")
        try:
            # Lazán importáljuk, hogy numpy hiánya esetén ne törjön más formátum
            from src.io_utils.io_utils import ATFHandler  # type: ignore
        except Exception as exc:
            app_logger.warning(f"ATF parser nem elérhető (függőség hiányzik): {exc}")
            return DocumentIR(
                doc_id=os.path.basename(filepath),
                source_path=filepath,
                doc_type="atf",
                chunks=[DocumentChunk(text="ATF feldolgozás kikapcsolva: numpy/ATFHandler hiányzik.")],
            )
        handler = ATFHandler(filepath)
        handler.load_atf()
        headers = getattr(handler, "headers", [])
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="atf",
            metadata={"headers": headers},
        )
        # Describe each trace as a semantic chunk
        for idx, header in enumerate(headers):
            # Skip time columns; describe traces
            if "time" in header.lower():
                continue
            text = f"ATF Trace '{header}' measured over time. Typical electrophysiology current trace (pA) vs time (s)."
            chunk = DocumentChunk(
                text=text,
                metadata={
                    "trace_index": idx,
                    "trace_name": header,
                    "source": filepath,
                },
            )
            ir.chunks.append(chunk)
        # Add a global chunk with file-level metadata
        meta_text = f"ATF file with {len(headers)} columns. Headers: {', '.join(headers[:8])}{'...' if len(headers) > 8 else ''}"
        ir.chunks.append(DocumentChunk(text=meta_text, metadata={"source": filepath}))
        return ir


class PDFParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"PDFParser: parsing {filepath}")
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="pdf",
        )
        try:
            import pdfplumber  # type: ignore
        except Exception as exc:
            app_logger.warning(f"pdfplumber not available ({exc}). Skipping PDF text extraction.")
            return ir
        try:
            with pdfplumber.open(filepath) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        continue
                    ir.chunks.append(
                        DocumentChunk(
                            text=text,
                            metadata={"page": page_idx + 1, "source": filepath},
                        )
                    )
        except Exception as exc:
            app_logger.warning(f"PDF parse failed: {exc}")
        return ir


class DOCXParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"DOCXParser: parsing {filepath}")
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="docx",
        )
        try:
            import docx  # type: ignore
        except Exception as exc:
            app_logger.warning(f"python-docx not available ({exc}). Skipping DOCX text extraction.")
            return ir
        try:
            document = docx.Document(filepath)
            for para_idx, para in enumerate(document.paragraphs):
                text = (para.text or "").strip()
                if not text:
                    continue
                ir.chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata={"paragraph": para_idx + 1, "source": filepath},
                    )
                )
        except Exception as exc:
            app_logger.warning(f"DOCX parse failed: {exc}")
        return ir


class TextParser(BaseParser):
    def parse(self, filepath: str) -> DocumentIR:
        app_logger.info(f"TextParser: parsing {filepath}")
        ir = DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type="text",
        )
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if content.strip():
                ir.chunks.append(DocumentChunk(text=content.strip(), metadata={"source": filepath}))
        except Exception as exc:
            app_logger.warning(f"Text parse failed: {exc}")
        return ir


class ParserNexus:
    """
    Unified Ingestion Layer with fallback chain:
    1) Native structured parser by extension
    2) Robust text extraction (generic)
    3) OCR for image-like docs (optional, if deps exist)
    """

    _parsers: Dict[str, BaseParser] = {
        ".xlsx": ExcelParser(),
        ".xls": ExcelParser(),
        ".csv": CSVParser(),
        ".atf": ATFParser(),
        ".pdf": PDFParser(),
        ".docx": DOCXParser(),
        ".txt": TextParser(),
    }

    @classmethod
    def parse_file(cls, filepath: str) -> DocumentIR:
        ext = os.path.splitext(filepath)[1].lower()
        # Attempt 1: native structured
        try:
            parser = cls._parsers.get(ext, TextParser())
            ir = parser.parse(filepath)
            if ir and ir.chunks:
                return ir
        except Exception as exc:
            app_logger.warning(f"Native parser failed: {exc}")

        # Attempt 2: robust plain text
        try:
            ir_text = TextParser().parse(filepath)
            if ir_text and ir_text.chunks and len(ir_text.chunks[0].text) > 20:
                return ir_text
        except Exception as exc:
            app_logger.warning(f"Text fallback failed: {exc}")

        # Attempt 3: OCR for image-like/pdf
        if ext in {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}:
            try:
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore
                import tempfile
                # Basic OCR path: try open as image; for PDF this would need pdf->image (omitted if deps missing)
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img)
                text = (text or "").strip()
                if text:
                    return DocumentIR(
                        doc_id=os.path.basename(filepath),
                        source_path=filepath,
                        doc_type="ocr",
                        chunks=[DocumentChunk(text=text, metadata={"source": filepath})],
                    )
            except Exception as exc:
                app_logger.warning(f"OCR attempt failed or unsupported: {exc}")

        # All failed: empty IR
        return DocumentIR(
            doc_id=os.path.basename(filepath),
            source_path=filepath,
            doc_type=ext.lstrip("."),
            chunks=[],
        )

    @staticmethod
    def flatten_texts(ir_list: List[DocumentIR]) -> List[str]:
        texts: List[str] = []
        for ir in ir_list:
            for chunk in ir.chunks:
                texts.append(chunk.text)
        return texts

    @staticmethod
    def flatten_chunks(ir_list: List[DocumentIR]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for ir in ir_list:
            chunks.extend(ir.chunks)
        return chunks


