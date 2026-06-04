from app.models.schemas import DocumentChunk
from extraction.entity_extractor import EntityExtractor


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
