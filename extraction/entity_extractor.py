from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from app.models.schemas import (
    DiscoveredField,
    DocumentChunk,
    DocumentUnderstanding,
    DynamicSchema,
    ExtractedEntity,
    Relationship,
    SectionNode,
    TableCell,
    TableData,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Schema-free document analyzer that infers structure from content."""

    def __init__(self, llm_fallback: Callable[[str], dict] | None = None) -> None:
        self.llm_fallback = llm_fallback
        self._nlp = None

    def _ensure_nlp(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
        except Exception as exc:  # pragma: no cover
            logger.warning("spaCy model unavailable, using heuristic semantic analysis: %s", exc)
            self._nlp = False
        return self._nlp

    def extract(self, chunks: list[DocumentChunk]) -> DocumentUnderstanding:
        sections = self._build_sections(chunks)
        entities = self._collect_entities(sections, chunks)
        relationships = self._discover_relationships(entities)
        document_text = "\n".join(chunk.text for chunk in chunks if chunk.text.strip())
        dynamic_schema = self._generate_dynamic_schema(sections, entities)
        document_type = self._infer_document_type(document_text, sections)
        summary = self._summarize_document(sections, document_type)
        confidence = self._mean_confidence(chunks)
        metadata = self._build_metadata(chunks, sections, entities)

        result = DocumentUnderstanding(
            document_id=chunks[0].document_id,
            document_type=document_type,
            document_summary=summary,
            confidence=confidence,
            dynamic_schema=dynamic_schema,
            sections=sections,
            entities=entities,
            relationships=relationships,
            metadata=metadata,
        )

        if self.llm_fallback:
            try:
                llm_payload = self.llm_fallback(document_text)
                result = result.model_copy(update=llm_payload)
            except Exception as exc:
                logger.warning("LLM fallback enrichment failed: %s", exc)
        return result

    def _build_sections(self, chunks: list[DocumentChunk]) -> list[SectionNode]:
        sections: list[SectionNode] = []
        for chunk in chunks:
            lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
            if not lines:
                continue

            current_title = self._infer_heading(lines[0], index=0)
            current_lines: list[str] = []
            subsection_counter = 1

            def flush(title: str, body_lines: list[str]) -> None:
                if not body_lines and not title:
                    return
                body_text = "\n".join(body_lines).strip()
                section_id = f"{chunk.chunk_id}_s{subsection_counter + len(sections)}"
                sections.append(
                    SectionNode(
                        section_id=section_id,
                        title=title or "Untitled Section",
                        level=self._heading_level(title),
                        page_start=chunk.page_number,
                        page_end=chunk.page_number,
                        confidence=self._section_confidence(title, body_text, chunk.confidence),
                        text=body_text,
                        summary=self._summarize_text(body_text),
                        key_values=self._extract_key_values(body_text, chunk.chunk_id),
                        entities=[],
                        tables=self._extract_tables(body_lines, section_id),
                        paragraphs=self._extract_paragraphs(body_text),
                        subsections=[],
                        metadata={"source_chunk_id": chunk.chunk_id},
                    )
                )

            for idx, line in enumerate(lines[1:], start=1):
                maybe_heading = self._infer_heading(line, index=idx)
                if maybe_heading and current_lines:
                    flush(current_title, current_lines)
                    current_title = maybe_heading
                    current_lines = []
                    subsection_counter += 1
                else:
                    current_lines.append(line)
            flush(current_title, current_lines or lines[1:])

        return self._attach_hierarchy(sections)

    def _collect_entities(self, sections: list[SectionNode], chunks: list[DocumentChunk]) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        for section in sections:
            section_entities = self._extract_entities(section.text, section.section_id, section.page_start, section.metadata.get("source_chunk_id", ""))
            section.entities = section_entities
            entities.extend(section_entities)
            for subsection in section.subsections:
                subsection_entities = self._extract_entities(
                    subsection.text,
                    subsection.section_id,
                    subsection.page_start,
                    subsection.metadata.get("source_chunk_id", ""),
                )
                subsection.entities = subsection_entities
                entities.extend(subsection_entities)
        if not entities:
            entities.extend(self._extract_entities("\n".join(chunk.text for chunk in chunks), None, None, ""))
        return entities

    def _extract_entities(
        self,
        text: str,
        section_id: str | None,
        page_number: int | None,
        chunk_id: str,
    ) -> list[ExtractedEntity]:
        results: list[ExtractedEntity] = []
        nlp = self._ensure_nlp()
        seen: set[tuple[str, str]] = set()
        if nlp is not False and text.strip():
            doc = nlp(text)
            for ent in doc.ents:
                key = (ent.text.strip(), ent.label_)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    ExtractedEntity(
                        name=ent.text.strip(),
                        entity_type=ent.label_,
                        value=ent.text.strip(),
                        confidence=0.82,
                        page_number=page_number,
                        section_id=section_id,
                        evidence_chunk_ids=[chunk_id] if chunk_id else [],
                    )
                )

        patterns = {
            "key_value_candidate": r"([A-Za-z][A-Za-z0-9 \/_-]{1,40})\s*:\s*([^\n]+)",
            "reference": r"\b[A-Z]{2,}-\d{2,}\b",
            "date": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
            "amount": r"(?:Rs\.?|INR|\$)\s?[\d,]+(?:\.\d+)?",
            "identifier": r"\b[A-Z0-9]{4,}(?:[-\/][A-Z0-9]+)+\b",
        }
        for match in re.finditer(patterns["key_value_candidate"], text):
            key = match.group(1).strip()
            value = match.group(2).strip()
            pair_key = (f"{key}: {value}", "KEY_VALUE")
            if pair_key in seen:
                continue
            seen.add(pair_key)
            results.append(
                ExtractedEntity(
                    name=key,
                    entity_type="KEY_VALUE",
                    value=value,
                    confidence=0.74,
                    page_number=page_number,
                    section_id=section_id,
                    evidence_chunk_ids=[chunk_id] if chunk_id else [],
                )
            )

        for label, pattern in patterns.items():
            if label == "key_value_candidate":
                continue
            for match in re.finditer(pattern, text):
                raw = match.group(0).strip()
                key = (raw, label.upper())
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    ExtractedEntity(
                        name=raw,
                        entity_type=label.upper(),
                        value=raw,
                        confidence=0.68,
                        page_number=page_number,
                        section_id=section_id,
                        evidence_chunk_ids=[chunk_id] if chunk_id else [],
                    )
                )
        return results

    def _discover_relationships(self, entities: list[ExtractedEntity]) -> list[Relationship]:
        relationships: list[Relationship] = []
        by_section: dict[str, list[ExtractedEntity]] = {}
        for entity in entities:
            if not entity.section_id:
                continue
            by_section.setdefault(entity.section_id, []).append(entity)

        for section_entities in by_section.values():
            kv_entities = [entity for entity in section_entities if entity.entity_type == "KEY_VALUE"]
            context_entities = [entity for entity in section_entities if entity.entity_type != "KEY_VALUE"]
            for kv in kv_entities:
                for context in context_entities[:3]:
                    relationships.append(
                        Relationship(
                            source=context.name,
                            relation=f"described_by_{kv.name.lower().replace(' ', '_')}",
                            target=kv.value,
                            confidence=min(kv.confidence, context.confidence),
                            evidence_chunk_ids=list(set(kv.evidence_chunk_ids + context.evidence_chunk_ids)),
                        )
                    )
        return relationships

    def _generate_dynamic_schema(self, sections: list[SectionNode], entities: list[ExtractedEntity]) -> DynamicSchema:
        field_names = []
        descriptions: dict[str, str] = {}
        for section in sections:
            field_names.append(self._normalize_schema_name(section.title))
            descriptions[self._normalize_schema_name(section.title)] = f"Content discovered under section '{section.title}'."
            for item in section.key_values:
                normalized = self._normalize_schema_name(item.key)
                field_names.append(normalized)
                descriptions[normalized] = f"Dynamically discovered field derived from document content: {item.key}"
        for entity in entities:
            normalized = self._normalize_schema_name(entity.entity_type)
            field_names.append(normalized)
            descriptions.setdefault(normalized, f"Observed semantic entity of type {entity.entity_type}.")

        deduped = [name for name, _ in Counter(field_names).most_common()]
        return DynamicSchema(
            schema_name="dynamic_document_schema",
            inferred_fields=deduped[:50],
            field_descriptions=descriptions,
            confidence=0.78 if deduped else 0.35,
        )

    def _infer_document_type(self, text: str, sections: list[SectionNode]) -> str:
        joined_titles = " | ".join(section.title.lower() for section in sections if section.title)
        first_lines = " ".join(section.summary for section in sections[:3]).lower()
        lexical_profile = f"{joined_titles} {first_lines} {text[:400].lower()}"

        candidate_terms = []
        for pattern in [
            r"\b[a-z]+(?:\s+[a-z]+){0,2}\s+(?:report|notice|invoice|agreement|statement|application|letter|memorandum|form|summary)\b",
            r"\b(?:minutes of meeting|purchase order|bank statement|medical report|inspection note)\b",
        ]:
            candidate_terms.extend(match.group(0).title() for match in re.finditer(pattern, lexical_profile))

        if candidate_terms:
            return Counter(candidate_terms).most_common(1)[0][0]
        if sections and sections[0].title and sections[0].title != "Untitled Section":
            return f"Inferred from heading: {sections[0].title}"
        return "Unseen Document Pattern"

    def _summarize_document(self, sections: list[SectionNode], document_type: str) -> str:
        if not sections:
            return "Information not found in source documents."
        section_titles = ", ".join(section.title for section in sections[:4])
        return f"This document appears to be '{document_type}' and is organized around sections such as {section_titles}."

    def _build_metadata(
        self,
        chunks: list[DocumentChunk],
        sections: list[SectionNode],
        entities: list[ExtractedEntity],
    ) -> dict[str, object]:
        return {
            "page_count": len({chunk.page_number for chunk in chunks}),
            "section_count": len(sections),
            "entity_count": len(entities),
            "source_files": sorted({chunk.metadata.get("source_path", "") for chunk in chunks}),
            "ocr_confidence_average": self._mean_confidence(chunks),
        }

    def _extract_key_values(self, text: str, chunk_id: str) -> list[DiscoveredField]:
        fields: list[DiscoveredField] = []
        for match in re.finditer(r"([A-Za-z][A-Za-z0-9 \/_-]{1,40})\s*:\s*([^\n]+)", text):
            fields.append(
                DiscoveredField(
                    key=match.group(1).strip(),
                    value=match.group(2).strip(),
                    confidence=0.76,
                    evidence_chunk_ids=[chunk_id],
                )
            )
        return fields

    def _extract_tables(self, lines: list[str], section_id: str) -> list[TableData]:
        tables: list[TableData] = []
        table_lines = [line for line in lines if "|" in line or "\t" in line]
        if not table_lines:
            return tables

        rows = [re.split(r"\s*\|\s*|\t+", line.strip("| ")) for line in table_lines]
        headers = rows[0] if rows else []
        cells: list[TableCell] = []
        for row_idx, row in enumerate(rows):
            for col_idx, value in enumerate(row):
                cells.append(TableCell(row=row_idx, column=col_idx, value=value, confidence=0.72))

        tables.append(
            TableData(
                table_id=f"{section_id}_t1",
                title="Discovered Table",
                headers=headers,
                rows=rows[1:] if len(rows) > 1 else [],
                cells=cells,
                confidence=0.72,
            )
        )
        return tables

    def _extract_paragraphs(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]

    def _attach_hierarchy(self, sections: list[SectionNode]) -> list[SectionNode]:
        if not sections:
            return []
        top_level: list[SectionNode] = []
        stack: list[SectionNode] = []
        for section in sections:
            section.subsections = []
            while stack and stack[-1].level >= section.level:
                stack.pop()
            if stack:
                stack[-1].subsections.append(section)
            else:
                top_level.append(section)
            stack.append(section)
        return top_level

    def _infer_heading(self, line: str, index: int) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        if index == 0:
            return stripped
        if stripped.isupper() and len(stripped.split()) <= 8:
            return stripped.title()
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
            return stripped
        if stripped.endswith(":") and len(stripped.split()) <= 8:
            return stripped.rstrip(":")
        return ""

    def _heading_level(self, title: str) -> int:
        if not title:
            return 1
        if re.match(r"^\d+\.\d+", title):
            return 2
        if re.match(r"^\d+\.\d+\.\d+", title):
            return 3
        if title.isupper():
            return 1
        return 1

    def _section_confidence(self, title: str, body_text: str, chunk_confidence: float) -> float:
        structure_bonus = 0.1 if title and title != "Untitled Section" else 0.0
        length_bonus = 0.08 if len(body_text.split()) > 20 else 0.03
        return round(min(0.99, chunk_confidence * 0.75 + structure_bonus + length_bonus), 4)

    def _summarize_text(self, text: str, limit: int = 180) -> str:
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else f"{clean[:limit].rstrip()}..."

    def _normalize_schema_name(self, value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return normalized or "unnamed_field"

    def _mean_confidence(self, chunks: list[DocumentChunk]) -> float:
        if not chunks:
            return 0.0
        return round(sum(chunk.confidence for chunk in chunks) / len(chunks), 4)
