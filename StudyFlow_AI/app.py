from __future__ import annotations

import json
from typing import Any

import streamlit as st

from agents import StudyFlowError, run_studyflow
from pdf_processing import MAX_PAGES, extract_pdf_text
from utils import make_export_record


st.set_page_config(
    page_title="StudyFlow AI",
    page_icon="📚",
    layout="wide",
)


def render_metadata(metadata: dict[str, Any]) -> None:
    columns = st.columns(5)

    columns[0].metric(
        "Pages analyzed",
        f"{metadata.get('processed_pages', 0)} / "
        f"{metadata.get('total_pages', 0)}",
    )

    columns[1].metric(
        "Study sections",
        metadata.get("analysis_chunk_count", 0),
    )

    columns[2].metric(
        "Characters read",
        f"{metadata.get('extracted_characters', 0):,}",
    )

    columns[3].metric(
        "OCR pages",
        len(metadata.get("ocr_pages", [])),
    )

    masked = sum(
        metadata.get("redaction_counts", {}).values()
    )

    columns[4].metric(
        "Identifiers masked",
        masked,
    )

    if metadata.get("fully_processed"):
        st.success(
            metadata.get(
                "coverage_status",
                "Full PDF analyzed",
            )
        )
    else:
        st.warning(
            metadata.get(
                "coverage_status",
                "Partial PDF coverage",
            )
        )


def render_notes(result: dict[str, Any]) -> None:
    notes = result["notes"]

    st.subheader("Quick summary")

    st.write(
        notes.get(
            "quick_summary",
            "No quick summary was returned.",
        )
    )

    st.subheader("Structured notes")

    for topic in notes.get("structured_notes", []):
        title = topic.get(
            "topic",
            "Untitled topic",
        )

        pages = topic.get(
            "page_range",
            "",
        )

        expander_title = title

        if pages:
            expander_title = f"{title} — {pages}"

        with st.expander(
            expander_title,
            expanded=False,
        ):
            st.write(topic.get("summary", ""))

            sections = [
                ("Key points", "key_points"),
                ("Formulas or rules", "formulas_or_rules"),
                ("Examples from the source", "examples_from_source"),
                ("Revision focus", "exam_focus"),
            ]

            for label, key in sections:
                if topic.get(key):
                    st.markdown(f"**{label}**")

                    for item in topic[key]:
                        st.markdown(f"- {item}")

            if topic.get("definitions"):
                st.markdown("**Definitions**")

                for definition in topic["definitions"]:
                    term = definition.get(
                        "term",
                        "Term",
                    )

                    explanation = definition.get(
                        "definition",
                        "",
                    )

                    st.markdown(
                        f"- **{term}:** {explanation}"
                    )


def render_explanations(result: dict[str, Any]) -> None:
    explanations = result["explanations"].get(
        "simple_explanations",
        [],
    )

    for item in explanations:
        heading = item.get(
            "concept",
            "Concept",
        )

        if item.get("page_range"):
            heading += f" — {item['page_range']}"

        st.markdown(f"### {heading}")

        st.write(
            item.get(
                "plain_explanation",
                "",
            )
        )

        if item.get("common_confusion"):
            st.info(
                f"**Common confusion:** "
                f"{item['common_confusion']}"
            )

        if item.get("memory_tip"):
            st.success(
                f"**Memory tip:** "
                f"{item['memory_tip']}"
            )


def render_practice(result: dict[str, Any]) -> None:
    practice = result["practice"]

    st.subheader("Multiple-choice questions")

    for number, question in enumerate(
        practice.get("mcqs", []),
        start=1,
    ):
        title = (
            f"MCQ {number}: "
            f"{question.get('question', '')}"
        )

        if question.get("page_range"):
            title += f" ({question['page_range']})"

        with st.expander(title):
            options = question.get("options", {})

            for letter in ("A", "B", "C", "D"):
                st.write(
                    f"{letter}. "
                    f"{options.get(letter, '')}"
                )

            st.markdown(
                f"**Answer:** "
                f"{question.get('answer', '')}"
            )

            st.write(
                question.get(
                    "explanation",
                    "",
                )
            )

    st.subheader("Short-answer questions")

    for number, question in enumerate(
        practice.get("short_answer_questions", []),
        start=1,
    ):
        title = (
            f"Question {number}: "
            f"{question.get('question', '')}"
        )

        if question.get("page_range"):
            title += f" ({question['page_range']})"

        with st.expander(title):
            st.markdown(
                f"**Model answer:** "
                f"{question.get('model_answer', '')}"
            )

    st.subheader("Flashcards")

    for number, card in enumerate(
        practice.get("flashcards", []),
        start=1,
    ):
        title = (
            f"Flashcard {number}: "
            f"{card.get('front', '')}"
        )

        if card.get("page_range"):
            title += f" ({card['page_range']})"

        with st.expander(title):
            st.write(card.get("back", ""))


def render_quality(result: dict[str, Any]) -> None:
    topic_map = result["topic_map"]
    quality = result["quality_check"]

    st.subheader("Detected topics")

    for item in topic_map.get("main_topics", []):
        title = item.get("title", "")
        pages = item.get("page_range", "")
        focus = item.get("focus", "")

        st.markdown(
            f"- **{title}** ({pages}): {focus}"
        )

    st.subheader("Coverage check")

    for item in quality.get("covered_topics", []):
        st.markdown(
            f"- {item.get('topic', '')} — "
            f"{item.get('page_range', '')}"
        )

    st.subheader("Revision checklist")

    for index, item in enumerate(
        quality.get("revision_checklist", [])
    ):
        st.checkbox(
            item,
            key=f"revision_{index}_{item}",
        )

    unclear = quality.get(
        "possibly_unclear_or_missing",
        [],
    )

    if unclear:
        st.subheader(
            "Sections to double-check in the original PDF"
        )

        for item in unclear:
            st.warning(item)

    st.caption(
        quality.get(
            "confidence_note",
            "",
        )
    )


def main() -> None:
    with st.sidebar:
        st.title("📚 StudyFlow AI")

        st.caption(
            "Multi-agent academic PDF study assistant"
        )

        st.divider()

        st.subheader("What it creates")

        st.markdown(
            "- Full-document topic map\n"
            "- Structured notes\n"
            "- Simple explanations\n"
            "- MCQs and short-answer practice\n"
            "- Flashcards\n"
            "- Revision checklist"
        )

        st.divider()

        st.subheader("PDF rules")

        st.markdown(
            f"- One PDF at a time\n"
            f"- Maximum 10 MB\n"
            f"- Up to {MAX_PAGES} pages analyzed\n"
            f"- Longer PDFs are clearly marked as partial\n"
            f"- Scanned PDFs use OCR when Tesseract is available"
        )

        st.divider()

        st.info(
            "StudyFlow AI is a learning-support prototype. "
            "It does not replace the original course material, "
            "your instructor, or academic integrity rules."
        )

    st.title("StudyFlow AI")

    st.caption(
        "Turn one academic PDF into structured notes, "
        "explanations, practice questions, flashcards, "
        "and a revision plan."
    )

    with st.form("studyflow_form"):
        uploaded_pdf = st.file_uploader(
            "Upload one academic PDF",
            type=["pdf"],
            help=(
                "Use course material you are permitted to process. "
                "Avoid confidential student or university records."
            ),
        )

        left, middle = st.columns(2)

        with left:
            language = st.selectbox(
                "Output language",
                [
                    "English",
                    "Arabic",
                    "Same language as the PDF",
                ],
            )

        with middle:
            study_goal = st.selectbox(
                "Study goal",
                [
                    "Understand the concepts",
                    "Prepare for revision",
                    "Practice for an exam",
                ],
            )

        acknowledged = st.checkbox(
            "I confirm that I am allowed to use this academic PDF "
            "and understand that the system creates study support, "
            "not guaranteed exam content."
        )

        submitted = st.form_submit_button(
            "Create my study pack",
            use_container_width=True,
        )

    if submitted:
        if uploaded_pdf is None:
            st.error("Please upload one PDF first.")
            return

        if not acknowledged:
            st.error(
                "Please tick the confirmation box before continuing."
            )
            return

        if uploaded_pdf.size > 10 * 1024 * 1024:
            st.error(
                "This PDF is larger than 10 MB. "
                "Please upload a smaller file."
            )
            return

        try:
            with st.status(
                "Reading the PDF and analyzing all permitted sections...",
                expanded=True,
            ) as status:
                status.write(
                    "Reading the PDF in memory, splitting it into "
                    "page-based study sections, and masking common identifiers."
                )

                processed = extract_pdf_text(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name,
                )

                if not processed["text_extracted"]:
                    status.update(
                        label="The PDF could not be read",
                        state="error",
                    )

                    st.error(
                        "No readable text was found. Try a text-based PDF, "
                        "or configure Tesseract for a scanned PDF."
                    )

                    for warning in processed["warnings"]:
                        st.warning(warning)

                    return

                status.write(
                    f"Prepared {processed['analysis_chunk_count']} sections "
                    f"covering {processed['processed_pages']} of "
                    f"{processed['total_pages']} pages."
                )

                result = run_studyflow(
                    source_chunks=processed["analysis_chunks"],
                    language=language,
                    study_goal=study_goal,
                    document_metadata=processed,
                    progress=status.write,
                )

                st.session_state["studyflow_result"] = result

                status.update(
                    label="Your full study pack is ready.",
                    state="complete",
                )

        except StudyFlowError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"Something unexpected happened: {error}"
            )

    result = st.session_state.get("studyflow_result")

    if not result:
        return

    st.divider()

    metadata = result["document_metadata"]

    render_metadata(metadata)

    for warning in metadata.get("warnings", []):
        st.warning(warning)

    tabs = st.tabs(
        [
            "Overview",
            "Structured Notes",
            "Simple Explanations",
            "Practice & Flashcards",
            "Quality Check",
            "Export",
        ]
    )

    with tabs[0]:
        topic_map = result["topic_map"]

        st.subheader(
            topic_map.get(
                "course_title_guess",
                "Academic PDF",
            )
        )

        st.write(
            topic_map.get(
                "document_scope_note",
                "",
            )
        )

        st.markdown("**Learning objectives**")

        for item in topic_map.get(
            "learning_objectives",
            [],
        ):
            st.markdown(f"- {item}")

        st.markdown("**Agents used**")

        st.write(
            " → ".join(
                result["workflow"]
            )
        )

    with tabs[1]:
        render_notes(result)

    with tabs[2]:
        render_explanations(result)

    with tabs[3]:
        render_practice(result)

    with tabs[4]:
        render_quality(result)

    with tabs[5]:
        export = make_export_record(result)

        st.write(
            "The JSON export contains generated study material "
            "and safe file metadata. It does not contain the uploaded PDF "
            "or raw extracted text."
        )

        st.download_button(
            "Download study pack JSON",
            data=json.dumps(
                export,
                ensure_ascii=False,
                indent=2,
            ),
            file_name="studyflow_study_pack.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()