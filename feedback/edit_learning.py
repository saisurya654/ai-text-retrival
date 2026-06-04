from __future__ import annotations

import difflib
import json
from pathlib import Path

from app.models.schemas import FeedbackRecord


class FeedbackMemory:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.records: list[FeedbackRecord] = []
        self._load()

    def analyze_and_store(self, original: str, edited: str, reason: str = "") -> FeedbackRecord:
        change_type = self._classify_change(original, edited)
        learned_pattern = self._extract_pattern(original, edited)
        record = FeedbackRecord(
            original=original,
            edited=edited,
            change_type=change_type,
            reason=reason,
            learned_pattern=learned_pattern,
        )
        self.records.append(record)
        self._persist()
        return record

    def apply_preferences(self, text: str) -> str:
        updated = text
        for record in self.records[-20:]:
            if "->" in record.learned_pattern:
                source, target = [part.strip() for part in record.learned_pattern.split("->", maxsplit=1)]
                if source and target:
                    updated = updated.replace(source, target)
        return updated

    def history(self) -> list[FeedbackRecord]:
        return self.records

    def _classify_change(self, original: str, edited: str) -> str:
        if original.lower() == edited.lower():
            return "formatting"
        if len(edited) > len(original) * 1.2:
            return "expansion"
        if len(edited) < len(original) * 0.8:
            return "condensation"
        return "terminology"

    def _extract_pattern(self, original: str, edited: str) -> str:
        if original.strip() and original.strip() in edited and original.strip() != edited.strip():
            return f"{original.strip()} -> {edited.strip()}"
        matcher = difflib.SequenceMatcher(a=original.split(), b=edited.split())
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "replace":
                before = " ".join(original.split()[a0:a1]).strip()
                after = " ".join(edited.split()[b0:b1]).strip()
                if before or after:
                    return f"{before} -> {after}"
        return "style reinforcement"

    def _persist(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(
            json.dumps([record.model_dump(mode="json") for record in self.records], indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.records = [FeedbackRecord(**item) for item in payload]
