from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.models.schemas import DocumentChunk
from app.utils.config import get_settings
from app.utils.logging import get_logger
from ingestion.ocr_processor import OCRProcessor

logger = get_logger(__name__)


class PDFProcessor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ocr = OCRProcessor()

    def process(self, file_path: Path) -> list[DocumentChunk]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._process_pdf(file_path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._process_image(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _process_pdf(self, file_path: Path) -> list[DocumentChunk]:
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        chunks: list[DocumentChunk] = []
        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(file_path) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    raw_text = page.extract_text() or ""
                    confidence = 0.98 if raw_text.strip() else 0.0

                    if not raw_text.strip():
                        logger.info("No embedded text on page %s, OCR fallback would be required.", page_index)

                    cleaned_text = self._clean_text(raw_text)
                    chunks.extend(
                        self._chunk_page(
                            document_id=document_id,
                            page_number=page_index,
                            text=cleaned_text,
                            confidence=confidence,
                            metadata={
                                "source_path": str(file_path),
                                "file_type": "pdf",
                                "page_width": page.width,
                                "page_height": page.height,
                            },
                        )
                    )
        except Exception as exc:
            logger.exception("Failed to process PDF %s", file_path)
            raise RuntimeError(f"PDF processing failed: {exc}") from exc
        return chunks

    def _process_image(self, file_path: Path) -> list[DocumentChunk]:
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        text, confidence = self.ocr.extract_text(file_path)
        cleaned_text = self._clean_text(text)
        return self._chunk_page(
            document_id=document_id,
            page_number=1,
            text=cleaned_text,
            confidence=confidence,
            metadata={"source_path": str(file_path), "file_type": file_path.suffix.lower().lstrip(".")},
        )

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_page(
        self,
        document_id: str,
        page_number: int,
        text: str,
        confidence: float,
        metadata: dict[str, object],
    ) -> list[DocumentChunk]:
        if not text:
            return [
                DocumentChunk(
                    chunk_id=f"{document_id}_p{page_number}_c1",
                    document_id=document_id,
                    page_number=page_number,
                    text="",
                    confidence=confidence,
                    metadata=metadata,
                )
            ]

        chunks: list[DocumentChunk] = []
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        start = 0
        chunk_number = 1
        while start < len(text):
            end = min(len(text), start + size)
            chunk_text = text[start:end].strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}_p{page_number}_c{chunk_number}",
                    document_id=document_id,
                    page_number=page_number,
                    text=chunk_text,
                    confidence=confidence,
                    metadata=metadata | {"char_start": start, "char_end": end},
                )
            )
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
            chunk_number += 1
        return chunks
