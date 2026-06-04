from app.models.schemas import DocumentUnderstanding, DynamicSchema, RetrievalHit
from feedback.edit_learning import FeedbackMemory
from generation.draft_generator import DraftGenerator


def test_draft_generator_adapts_to_inferred_document_type(tmp_path):
    generator = DraftGenerator(FeedbackMemory(tmp_path / "feedback.json"))
    hits = [
        RetrievalHit(
            chunk_id="section_12",
            document_id="doc_1",
            page_number=3,
            score=0.91,
            text="The report states the deployment phase is blocked by a vendor approval delay.",
            metadata={"section_title": "Risks"},
        )
    ]
    understanding = DocumentUnderstanding(
        document_id="doc_1",
        document_type="Project Status Report",
        document_summary="This document summarizes project progress, risks, and next actions.",
        confidence=0.88,
        dynamic_schema=DynamicSchema(schema_name="dynamic", inferred_fields=["risks"], field_descriptions={}, confidence=0.8),
        sections=[],
        entities=[],
        relationships=[],
        metadata={},
    )
    draft = generator.generate("Adaptive Grounded Draft", "summarize risks", hits, understanding)
    assert draft.inferred_document_type == "Project Status Report"
    assert draft.summary.evidence_chunk_ids == ["section_12"]
