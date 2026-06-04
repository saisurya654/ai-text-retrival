from __future__ import annotations

from pathlib import Path
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


class OCRProcessor:
    """OCR wrapper with optional OpenCV preprocessing and EasyOCR inference."""

    def __init__(self) -> None:
        self._reader = None

    def _ensure_reader(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore

            self._reader = easyocr.Reader(["en"], gpu=False)
        except Exception as exc:  # pragma: no cover - depends on native install
            logger.warning("EasyOCR unavailable, OCR fallback disabled: %s", exc)
            self._reader = False
        return self._reader

    def preprocess(self, image_path: Path) -> Path:
        try:
            import cv2  # type: ignore

            image = cv2.imread(str(image_path))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            filtered = cv2.fastNlMeansDenoising(gray)
            _, thresholded = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            output_path = image_path.with_name(f"{image_path.stem}_preprocessed{image_path.suffix}")
            cv2.imwrite(str(output_path), thresholded)
            return output_path
        except Exception as exc:  # pragma: no cover - depends on native install
            logger.warning("OpenCV preprocessing skipped for %s: %s", image_path, exc)
            return image_path

    def extract_text(self, image_path: Path) -> tuple[str, float]:
        reader = self._ensure_reader()
        if reader is False:
            return ("", 0.0)

        processed_path = self.preprocess(image_path)
        results = reader.readtext(str(processed_path), detail=1, paragraph=True)
        if not results:
            return ("", 0.0)

        text_parts = [item[1] for item in results]
        confidence = sum(float(item[2]) for item in results) / len(results)
        return ("\n".join(text_parts), round(confidence, 4))
