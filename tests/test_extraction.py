from pathlib import Path

from app.models.schemas import DocumentChunk
from extraction.entity_extractor import EntityExtractor
from feedback.edit_learning import FeedbackMemory


def test_document_understanding_is_schema_free_and_structured():
    extractor = EntityExtractor()
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            page_number=1,
            text=(
                "PROJECT STATUS REPORT\n"
                "Prepared By: Meera Shah\n"
                "Date: 12/05/2025\n\n"
                "1. Executive Summary\n"
                "The migration program is 80 percent complete.\n\n"
                "2. Risks\n"
                "Vendor dependency remains unresolved."
            ),
            confidence=0.97,
            metadata={},
        )
    ]
    result = extractor.extract(chunks)
    assert result.document_type
    assert result.sections
    assert result.dynamic_schema.inferred_fields
    assert result.metadata["section_count"] >= 1


def test_extraction_adapts_to_feedback(tmp_path: Path):
    feedback_file = tmp_path / "feedback.json"
    memory = FeedbackMemory(feedback_file)
    
    # Store feedback to correct "Meera Shah" to "Meera Shah Patel"
    memory.analyze_and_store(
        original="Meera Shah",
        edited="Meera Shah Patel",
        reason="prefer full legal name",
    )
    
    # Store feedback to correct the inferred document type
    # The default inferred type is "Project Status Report" (since it matches report/notice pattern)
    memory.analyze_and_store(
        original="Project Status Report",
        edited="Cloud Migration Progress Report",
        reason="correct document classification",
    )

    extractor = EntityExtractor(feedback_memory=memory)
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            page_number=1,
            text=(
                "PROJECT STATUS REPORT\n"
                "Prepared By: Meera Shah\n"
                "Date: 12/05/2025\n\n"
                "1. Executive Summary\n"
                "The migration program is 80 percent complete.\n\n"
                "2. Risks\n"
                "Vendor dependency remains unresolved."
            ),
            confidence=0.97,
            metadata={},
        )
    ]
    
    result = extractor.extract(chunks)
    
    # Assert that document type is dynamically corrected from feedback
    assert result.document_type == "Cloud Migration Progress Report"
    
    # Assert that key-values and entities are dynamically corrected
    prepared_by_field = None
    for section in result.sections:
        for kv in section.key_values:
            if kv.key == "Prepared By":
                prepared_by_field = kv.value

    assert prepared_by_field == "Meera Shah Patel"
    
    # Verify that the entity list has been updated
    entity_values = [ent.value for ent in result.entities]
    assert "Meera Shah Patel" in entity_values
    assert "Meera Shah" not in entity_values

