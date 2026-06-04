from __future__ import annotations

from app.models.schemas import DocumentUnderstanding, DraftResponse, EvidenceTrace, GeneratedSection, RetrievalHit
from feedback.edit_learning import FeedbackMemory


class DraftGenerator:
    def __init__(self, feedback_memory: FeedbackMemory) -> None:
        self.feedback_memory = feedback_memory

    def generate(
        self,
        draft_type: str,
        query: str,
        hits: list[RetrievalHit],
        understanding: DocumentUnderstanding | None = None,
    ) -> DraftResponse:
        inferred_type = understanding.document_type if understanding else "Unknown"
        summary = self._summary_section(query, hits, inferred_type, understanding)
        key_facts = self._facts_section(hits, understanding)
        evidence_used = self._evidence_section(hits)
        risks = self._risk_section(hits, understanding)
        missing = self._missing_information_section(hits, understanding)
        recommendations = self._recommendations_section(hits, inferred_type)

        return DraftResponse(
            draft_type=draft_type,
            inferred_document_type=inferred_type,
            summary=summary,
            key_facts=key_facts,
            evidence_used=evidence_used,
            risks=risks,
            missing_information=missing,
            recommendations=recommendations,
        )

    def _summary_section(
        self,
        query: str,
        hits: list[RetrievalHit],
        inferred_type: str,
        understanding: DocumentUnderstanding | None,
    ) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Summary")
        lead = hits[0]
        context = understanding.document_summary if understanding else self._clip(lead.text)
        content = (
            f"For the request '{query}', the uploaded material is best understood as '{inferred_type}'. "
            f"Grounded evidence suggests: {context}"
        )
        content = self.feedback_memory.apply_preferences(content)
        return self._section("Summary", content, hits[:2])

    def _facts_section(self, hits: list[RetrievalHit], understanding: DocumentUnderstanding | None) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Key Facts")
        lines = []
        if understanding:
            for section in understanding.sections[:4]:
                lines.append(f"- Section '{section.title}': {section.summary}")
        if not lines:
            lines = [f"- {self._clip(hit.text)}" for hit in hits[:4]]
        return self._section("Key Facts", "\n".join(lines), hits[:4])

    def _evidence_section(self, hits: list[RetrievalHit]) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Evidence Used")
        lines = []
        for hit in hits:
            section_title = hit.metadata.get("section_title", "Unknown Section")
            lines.append(f"- {hit.chunk_id} | {section_title} | page {hit.page_number} | score={hit.score:.3f}")
        return self._section("Evidence Used", "\n".join(lines), hits)

    def _risk_section(self, hits: list[RetrievalHit], understanding: DocumentUnderstanding | None) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Risks")
        risks = []
        if any(hit.score < 0.45 for hit in hits):
            risks.append("Some retrieved sections have low semantic similarity, so draft conclusions may need reviewer confirmation.")
        if understanding and understanding.confidence < 0.7:
            risks.append("Document understanding confidence is moderate, which may indicate an unfamiliar layout or OCR noise.")
        if not risks:
            risks.append("No major grounding risk detected in the retrieved sections.")
        return self._section("Risks", " ".join(risks), hits[:2])

    def _missing_information_section(
        self,
        hits: list[RetrievalHit],
        understanding: DocumentUnderstanding | None,
    ) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Missing Information")
        notes = []
        if understanding:
            if not understanding.entities:
                notes.append("- Entities: Information not found in source documents.")
            if not understanding.relationships:
                notes.append("- Relationships: Information not found in source documents.")
            if not any(section.tables for section in understanding.sections):
                notes.append("- Tables: Information not found in source documents.")
        if not notes:
            notes.append("No obvious structural gaps detected from the retrieved sections.")
        return self._section("Missing Information", "\n".join(notes), hits[:2])

    def _recommendations_section(self, hits: list[RetrievalHit], inferred_type: str) -> GeneratedSection:
        if not hits:
            return self._not_found_section("Recommendations")
        content = (
            f"Adapt review steps to the inferred document type '{inferred_type}', validate the cited sections directly, "
            "and capture any operator corrections to improve future extraction and drafting."
        )
        content = self.feedback_memory.apply_preferences(content)
        return self._section("Recommendations", content, hits[:2])

    def _section(self, title: str, content: str, hits: list[RetrievalHit]) -> GeneratedSection:
        traces = [
            EvidenceTrace(claim=self._clip(hit.text), source_chunk=hit.chunk_id, page=hit.page_number)
            for hit in hits
        ]
        return GeneratedSection(
            title=title,
            content=content,
            evidence_chunk_ids=[hit.chunk_id for hit in hits],
            traces=traces,
        )

    def _not_found_section(self, title: str) -> GeneratedSection:
        return GeneratedSection(
            title=title,
            content="Information not found in source documents.",
            evidence_chunk_ids=[],
            traces=[],
        )

    @staticmethod
    def _clip(text: str, limit: int = 220) -> str:
        clean = " ".join(text.split())
        return clean if len(clean) <= limit else f"{clean[:limit].rstrip()}..."
