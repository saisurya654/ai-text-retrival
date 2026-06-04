from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SupportedDocumentType(str, Enum):
    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    entities: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveredField(BaseModel):
    key: str
    value: Any
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str
    entity_type: str
    value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page_number: int | None = None
    section_id: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class TableCell(BaseModel):
    row: int
    column: int
    value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TableData(BaseModel):
    table_id: str
    title: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SectionNode(BaseModel):
    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    text: str = ""
    summary: str = ""
    key_values: list[DiscoveredField] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    tables: list[TableData] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)
    subsections: list["SectionNode"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicSchema(BaseModel):
    schema_name: str
    inferred_fields: list[str] = Field(default_factory=list)
    field_descriptions: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentUnderstanding(BaseModel):
    document_id: str
    document_type: str
    document_summary: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dynamic_schema: DynamicSchema
    sections: list[SectionNode] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceTrace(BaseModel):
    claim: str
    source_chunk: str
    page: int


class GeneratedSection(BaseModel):
    title: str
    content: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    traces: list[EvidenceTrace] = Field(default_factory=list)


class DraftResponse(BaseModel):
    draft_type: str
    inferred_document_type: str = ""
    summary: GeneratedSection
    key_facts: GeneratedSection
    evidence_used: GeneratedSection
    risks: GeneratedSection
    missing_information: GeneratedSection
    recommendations: GeneratedSection


class FeedbackRecord(BaseModel):
    original: str
    edited: str
    change_type: str
    reason: str = ""
    learned_pattern: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationReport(BaseModel):
    ocr_accuracy: float
    extraction_accuracy: float
    recall_at_k: float
    precision_at_k: float
    groundedness_score: float
    citation_coverage: float
    improvement_rate: float
    repeated_error_reduction: float
    details: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_processed: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    query: str
    draft_type: str = "Adaptive Grounded Draft"
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class FeedbackRequest(BaseModel):
    original: str
    edited: str
    reason: str = ""


class ExtractRequest(BaseModel):
    document_id: str


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
