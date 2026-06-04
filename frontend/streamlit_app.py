from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(page_title="Agentic Document Intelligence", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

st.title("Agentic Document Intelligence Platform")
st.caption("Schema-free document understanding, section-aware retrieval, grounded drafting, and feedback learning.")

pages = [
    "Upload Document",
    "Document Understanding",
    "Evidence Explorer",
    "Draft Generator",
    "Feedback Review",
    "Analytics Dashboard",
]
page = st.sidebar.radio("Pages", pages)

if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = ""


def request_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
    except Exception:
        pass
    return response.text or f"HTTP {response.status_code}"


def safe_post_json(path: str, payload: dict):
    response = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
    if not response.ok:
        st.error(request_error_message(response))
        return None
    return response.json()


def safe_get_json(path: str):
    response = requests.get(f"{API_BASE}{path}", timeout=60)
    if not response.ok:
        st.error(request_error_message(response))
        return None
    return response.json()


def render_section_tree(sections: list[dict], indent: int = 0):
    for section in sections:
        prefix = "  " * indent
        st.markdown(
            f"{prefix}- **{section['title']}** | level `{section['level']}` | confidence `{section['confidence']}` | page `{section['page_start']}`"
        )
        if section.get("summary"):
            st.caption(f"{prefix}{section['summary']}")
        if section.get("subsections"):
            render_section_tree(section["subsections"], indent + 1)


def render_understanding_view(data: dict):
    st.subheader("Document Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Document Type", data.get("document_type", ""))
    col2.metric("Understanding Confidence", data.get("confidence", 0.0))
    col3.metric("Sections", len(data.get("sections", [])))
    st.write(data.get("document_summary", ""))

    st.subheader("Dynamic Schema")
    schema = data.get("dynamic_schema", {})
    st.write(f"Schema Name: `{schema.get('schema_name', '')}`")
    st.dataframe(
        [{"field": field, "description": schema.get("field_descriptions", {}).get(field, "")} for field in schema.get("inferred_fields", [])],
        use_container_width=True,
    )

    st.subheader("Section Hierarchy")
    render_section_tree(data.get("sections", []))

    st.subheader("Entities")
    st.dataframe(data.get("entities", []), use_container_width=True)

    st.subheader("Relationships")
    st.dataframe(data.get("relationships", []), use_container_width=True)

    st.subheader("Metadata")
    st.json(data.get("metadata", {}))

    download_json = json.dumps(data, indent=2).encode("utf-8")
    st.download_button(
        "Download Understanding JSON",
        data=download_json,
        file_name="document_understanding.json",
        mime="application/json",
    )


if page == "Upload Document":
    uploaded_file = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file and st.button("Process Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
        if response.ok:
            payload = response.json()
            st.session_state.current_document_id = payload["document_id"]
            st.success("Document uploaded successfully.")
            st.write(f"Document ID: `{payload['document_id']}`")
            st.json(payload)
        else:
            st.error(request_error_message(response))

    st.subheader("Available Documents")
    documents = safe_get_json("/documents")
    if documents:
        st.dataframe(documents, use_container_width=True)

if page == "Document Understanding":
    document_id = st.text_input("Document ID", value=st.session_state.current_document_id)
    c1, c2 = st.columns(2)
    if c1.button("Run Understanding"):
        payload = safe_post_json("/extract", {"document_id": document_id})
        if payload:
            st.session_state.current_document_id = document_id
            render_understanding_view(payload)
    if c2.button("Load Saved Understanding"):
        record = safe_get_json(f"/documents/{document_id}")
        if record and record.get("extraction"):
            render_understanding_view(record["extraction"])
        elif record:
            st.info("No saved understanding yet. Run understanding first.")

if page == "Evidence Explorer":
    document_id = st.text_input("Filter by Document ID", value=st.session_state.current_document_id)
    query = st.text_input("Semantic Section Query", value="core purpose, obligations, entities, relationships")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    if st.button("Retrieve Evidence"):
        hits = safe_post_json("/retrieve", {"query": query, "top_k": top_k, "document_id": document_id or None})
        if hits:
            st.dataframe(hits, use_container_width=True)

if page == "Draft Generator":
    document_id = st.text_input("Document ID for Drafting", value=st.session_state.current_document_id)
    draft_type = st.text_input("Draft Type", value="Adaptive Grounded Draft")
    query = st.text_area("Draft Request", value="Generate a grounded brief adapted to the document's purpose, structure, and key relationships.")
    top_k = st.slider("Section Retrieval Depth", min_value=1, max_value=10, value=5)
    if st.button("Generate Draft"):
        draft = safe_post_json(
            "/generate",
            {"query": query, "draft_type": draft_type, "top_k": top_k, "document_id": document_id or None},
        )
        if draft:
            st.subheader("Inferred Document Type")
            st.write(draft.get("inferred_document_type", ""))
            st.subheader("Summary")
            st.write(draft["summary"]["content"])
            st.subheader("Key Facts")
            st.write(draft["key_facts"]["content"])
            st.subheader("Evidence Used")
            st.dataframe(draft["evidence_used"]["traces"], use_container_width=True)
            st.subheader("Risks")
            st.write(draft["risks"]["content"])
            st.subheader("Missing Information")
            st.write(draft["missing_information"]["content"])
            st.subheader("Recommendations")
            st.write(draft["recommendations"]["content"])
            st.download_button(
                "Download Draft JSON",
                data=json.dumps(draft, indent=2).encode("utf-8"),
                file_name="grounded_adaptive_draft.json",
                mime="application/json",
            )

if page == "Feedback Review":
    original = st.text_area("Original Draft")
    edited = st.text_area("Edited Draft")
    reason = st.text_input("Reason")
    if st.button("Store Feedback"):
        payload = safe_post_json("/feedback", {"original": original, "edited": edited, "reason": reason})
        if payload:
            st.json(payload)
    if st.button("Load Feedback History"):
        payload = safe_get_json("/history")
        if payload is not None:
            st.dataframe(payload, use_container_width=True)

if page == "Analytics Dashboard":
    if st.button("Refresh Evaluation"):
        payload = safe_get_json("/evaluation")
        if payload is not None:
            st.json(payload)
    report_path = Path("samples/outputs/evaluation_report.json")
    if report_path.exists():
        st.subheader("Latest Saved Report")
        st.json(json.loads(report_path.read_text(encoding="utf-8")))
