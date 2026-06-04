# Architecture

## System Architecture Diagram

```mermaid
flowchart LR
    A["Operator"] --> B["Streamlit Console"]
    B --> C["FastAPI Platform"]
    C --> D["Ingestion Layer"]
    D --> E["OCR + Text Extraction"]
    E --> F["Layout Analysis"]
    F --> G["Semantic Understanding Engine"]
    G --> H["Dynamic Schema Generator"]
    H --> I["Universal JSON Store"]
    G --> J["Section Vector Index"]
    J --> K["Grounded Retrieval"]
    K --> L["Adaptive Draft Generator"]
    B --> M["Feedback Review"]
    M --> N["Learning Memory"]
    N --> G
    N --> K
    N --> L
```

## Data Flow Diagram

```mermaid
flowchart TD
    A["Upload PDF/Image"] --> B["OCR or native text extraction"]
    B --> C["Clean text and collect page metadata"]
    C --> D["Detect headings, boundaries, key-values, tables, paragraphs"]
    D --> E["Infer document type and semantic structure"]
    E --> F["Generate dynamic schema"]
    E --> G["Create universal document JSON"]
    G --> H["Persist understanding"]
    E --> I["Split into semantic sections"]
    I --> J["Embed and index sections"]
    J --> K["Retrieve grounded evidence"]
    K --> L["Generate adaptive draft with citations"]
    L --> M["Operator edits output"]
    M --> N["Extract learning patterns"]
```

## Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit
    participant A as FastAPI
    participant P as Pipeline
    participant X as Understanding Engine
    participant V as Vector Store
    participant F as Feedback Memory

    U->>S: Upload document
    S->>A: POST /upload
    A->>P: persist file and extract raw text
    P-->>A: document_id
    A-->>S: upload response

    U->>S: Run document understanding
    S->>A: POST /extract
    A->>P: load document chunks
    P->>X: infer sections, schema, entities, relationships
    X-->>P: universal understanding object
    P->>V: index semantic sections
    P-->>A: structured result
    A-->>S: understanding JSON

    U->>S: Generate draft
    S->>A: POST /generate
    A->>P: retrieve sections by query
    P->>V: semantic section search
    V-->>P: grounded hits
    P->>F: apply learned preferences
    P-->>A: adaptive grounded draft
    A-->>S: draft with evidence traces
```

## Component Interaction Diagram

```mermaid
classDiagram
    class LegalMindService {
      +upload_document(path)
      +extract(document_id)
      +retrieve(query, top_k, document_id)
      +generate(draft_type, query, top_k, document_id)
      +record_feedback(original, edited, reason)
      +evaluation()
    }

    class PDFProcessor
    class OCRProcessor
    class EntityExtractor
    class VectorStore
    class EmbeddingService
    class DraftGenerator
    class FeedbackMemory
    class EvaluationEngine

    LegalMindService --> PDFProcessor
    PDFProcessor --> OCRProcessor
    LegalMindService --> EntityExtractor
    LegalMindService --> VectorStore
    VectorStore --> EmbeddingService
    LegalMindService --> DraftGenerator
    DraftGenerator --> FeedbackMemory
    LegalMindService --> EvaluationEngine
```

## Design notes

- The understanding engine is schema-free and content-driven.
- Sections are the primary retrieval units.
- Universal JSON separates document understanding from downstream task assumptions.
- Feedback memory is positioned as a reusable adaptation layer across extraction, retrieval, and drafting.
