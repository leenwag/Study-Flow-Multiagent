"""StudyFlow AI: chunked multi-agent workflow for academic PDFs."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI


ProgressCallback = Callable[[str], None]


class StudyFlowError(RuntimeError):
    """A user-friendly error shown in the Streamlit app."""


@lru_cache(maxsize=1)
def get_settings() -> dict[str, str]:
    """Load DeepSeek settings from the local .env file."""
    load_dotenv(override=False)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://api.deepseek.com",
    ).strip()

    model = os.getenv(
        "LLM_MODEL",
        "deepseek-v4-flash",
    ).strip()

    if not api_key:
        raise StudyFlowError(
            "Your DEEPSEEK_API_KEY is empty. "
            "Open the .env file, add your key, save it, "
            "then restart Streamlit."
        )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create one reusable DeepSeek client."""
    settings = get_settings()

    return OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"],
    )


def _to_json(data: Any) -> str:
    """Turn Python data into readable JSON for the next agent."""
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
    )


def _page_label(start_page: int, end_page: int) -> str:
    """Create a readable page-range label."""
    if start_page == end_page:
        return f"Page {start_page}"

    return f"Pages {start_page}-{end_page}"


def _safe_document_metadata(
    document_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove temporary raw text before returning results.

    Raw extracted PDF text and chunks must not appear in final output
    or in the downloadable JSON record.
    """
    blocked_keys = {
        "analysis_chunks",
        "source_text",
        "redacted_text",
        "file_bytes",
    }

    return {
        key: value
        for key, value in document_metadata.items()
        if key not in blocked_keys
    }


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Read JSON even if the model adds extra text or Markdown fences.

    Examples this can handle:
    - ```json { ... } ```
    - Here is the JSON: { ... }
    - Normal JSON: { ... }
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()

    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    decoder = json.JSONDecoder()

    for position, character in enumerate(cleaned):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(cleaned[position:])

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "No complete JSON object was found in the model response."
    )


def _request_model(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Send one request to DeepSeek and return content plus finish status."""
    settings = get_settings()

    response = get_client().chat.completions.create(
        model=settings["model"],
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.1,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]

    content = choice.message.content or ""
    finish_reason = choice.finish_reason or "unknown"

    return content, finish_reason


def _ask_agent(
    *,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """
    Run one agent.

    It retries once if DeepSeek returns incomplete JSON, Markdown,
    or a response that cannot be parsed.
    """
    last_error = ""

    for attempt in range(2):
        retry_instruction = ""

        if attempt == 1:
            retry_instruction = """

IMPORTANT RETRY:
Return one short, complete JSON object only.
Start directly with { and finish directly with }.
Do not use Markdown, explanations, headings, or code fences.
Keep lists concise so your response is not cut off.
"""

        current_max_tokens = max_tokens

        if attempt == 1:
            current_max_tokens = min(
                max_tokens + 1000,
                8000,
            )

        try:
            content, finish_reason = _request_model(
                system_prompt=system_prompt + retry_instruction,
                user_prompt=user_prompt,
                max_tokens=current_max_tokens,
            )

            if not content.strip():
                last_error = "The model returned an empty response."
                continue

            try:
                return _extract_json_object(content)

            except ValueError as error:
                if finish_reason == "length":
                    last_error = (
                        "The model response was cut off before "
                        "the JSON was complete."
                    )
                else:
                    last_error = str(error)

        except StudyFlowError:
            raise

        except Exception as error:
            last_error = str(error)

    raise StudyFlowError(
        f"{agent_name} could not finish. "
        f"{last_error or 'The model did not return usable JSON.'}"
    )


def run_studyflow(
    *,
    source_chunks: list[dict[str, Any]],
    language: str,
    study_goal: str,
    document_metadata: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Run StudyFlow's multi-agent workflow.

    1. Chunk Reading Agent
    2. Topic Mapping Agent
    3. Structured Notes Agent
    4. Concept Explanation Agent
    5. Practice Agent
    6. Quality Check Agent
    """
    if not source_chunks:
        raise StudyFlowError(
            "No readable PDF text was available for analysis."
        )

    def update(message: str) -> None:
        if progress is not None:
            progress(message)

    base_rules = f"""
You are part of StudyFlow AI, an academic learning-support prototype.

Write in: {language}
Student study goal: {study_goal}

Use only the supplied academic source material.

Do not invent:
- facts
- page numbers
- formulas
- sources
- guaranteed exam questions

Return valid JSON only.
Start with {{ and end with }}.
Do not use Markdown code fences.
"""

    # ---------------------------------------------------------
    # Agent 1: Read every chunk separately.
    # ---------------------------------------------------------
    chunk_reports: list[dict[str, Any]] = []
    total_chunks = len(source_chunks)

    for chunk in source_chunks:
        chunk_number = int(chunk["chunk_number"])
        start_page = int(chunk["start_page"])
        end_page = int(chunk["end_page"])

        page_range = _page_label(
            start_page,
            end_page,
        )

        chunk_text = str(
            chunk.get("text", "")
        ).strip()

        if not chunk_text:
            continue

        update(
            f"Chunk Reading Agent: section "
            f"{chunk_number}/{total_chunks}, {page_range}."
        )

        report = _ask_agent(
            agent_name=(
                f"Chunk Reading Agent ({page_range})"
            ),
            system_prompt=base_rules
            + """
Your role is Chunk Reading Agent.

Read one academic PDF section and create compact factual notes
for the later agents.

Rules:
- Use the supplied page range only.
- Identify a maximum of 4 important topics.
- Do not create practice questions.
- Keep the lists short and useful.

Return exactly this JSON shape:

{
  "chunk_summary": "string",
  "topics": [
    {
      "title": "string",
      "focus": "string",
      "key_points": ["string"],
      "definitions": [
        {
          "term": "string",
          "definition": "string"
        }
      ],
      "formulas_or_rules": ["string"],
      "examples_or_diagrams": ["string"]
    }
  ],
  "possible_unclear_content": ["string"]
}
""",
            user_prompt=(
                f"CHUNK NUMBER: {chunk_number}\n"
                f"REAL SOURCE PAGE RANGE: {page_range}\n\n"
                f"SOURCE TEXT:\n{chunk_text}"
            ),
            max_tokens=2500,
        )

        # These are added locally, so the AI cannot invent page ranges.
        report["chunk_number"] = chunk_number
        report["page_range"] = page_range
        report["start_page"] = start_page
        report["end_page"] = end_page

        chunk_reports.append(report)

    if not chunk_reports:
        raise StudyFlowError(
            "The PDF was read, but no usable study sections "
            "were prepared for the AI agents."
        )

    # The coverage statement is created locally for accuracy.
    if document_metadata.get("fully_processed"):
        coverage_note = (
            f"Full coverage: all "
            f"{document_metadata.get('total_pages', 0)} PDF pages "
            f"were analyzed in {len(chunk_reports)} sections."
        )
    else:
        coverage_note = (
            f"Partial coverage: "
            f"{document_metadata.get('processed_pages', 0)} of "
            f"{document_metadata.get('total_pages', 0)} pages "
            f"were analyzed. Review the original PDF for the rest."
        )

    # ---------------------------------------------------------
    # Agent 2: Combine chunks into one topic map.
    # ---------------------------------------------------------
    update(
        "Topic Mapping Agent is combining all sections "
        "into one course outline."
    )

    topic_map = _ask_agent(
        agent_name="Topic Mapping Agent",
        system_prompt=base_rules
        + """
Your role is Topic Mapping Agent.

Combine the supplied chunk reports into one logical outline.

Rules:
- Use only page ranges already supplied in the reports.
- Do not invent new source pages.
- Include a maximum of 10 main topics.

Return exactly this JSON shape:

{
  "course_title_guess": "string",
  "subject_area": "string",
  "main_topics": [
    {
      "title": "string",
      "page_range": "Page 1 or Pages 1-4",
      "focus": "string",
      "key_concepts": ["string"]
    }
  ],
  "learning_objectives": ["string"]
}
""",
        user_prompt=(
            f"CHUNK REPORTS:\n"
            f"{_to_json(chunk_reports)}"
        ),
        max_tokens=3000,
    )

    topic_map["document_scope_note"] = coverage_note

    # ---------------------------------------------------------
    # Agent 3: Create structured notes.
    # ---------------------------------------------------------
    update(
        "Structured Notes Agent is creating full notes."
    )

    notes = _ask_agent(
        agent_name="Structured Notes Agent",
        system_prompt=base_rules
        + """
Your role is Structured Notes Agent.

Create organized, student-friendly notes from the topic map
and all chunk reports.

Rules:
- Cover every major topic in the topic map.
- Keep page ranges based only on supplied reports.
- Create a maximum of 10 note sections.
- Keep each section focused and concise.

Return exactly this JSON shape:

{
  "quick_summary": "string",
  "structured_notes": [
    {
      "topic": "string",
      "page_range": "Page 1 or Pages 1-4",
      "summary": "string",
      "key_points": ["string"],
      "definitions": [
        {
          "term": "string",
          "definition": "string"
        }
      ],
      "formulas_or_rules": ["string"],
      "examples_from_source": ["string"],
      "exam_focus": ["string"]
    }
  ]
}
""",
        user_prompt=(
            f"TOPIC MAP:\n{_to_json(topic_map)}\n\n"
            f"ALL CHUNK REPORTS:\n{_to_json(chunk_reports)}"
        ),
        max_tokens=6000,
    )

    # ---------------------------------------------------------
    # Agent 4: Explain difficult concepts simply.
    # ---------------------------------------------------------
    update(
        "Concept Explanation Agent is simplifying difficult concepts."
    )

    explanations = _ask_agent(
        agent_name="Concept Explanation Agent",
        system_prompt=base_rules
        + """
Your role is Concept Explanation Agent.

Choose up to 8 important or difficult concepts from the notes.
Explain them in simple student-friendly language.

Return exactly this JSON shape:

{
  "simple_explanations": [
    {
      "concept": "string",
      "page_range": "string",
      "plain_explanation": "string",
      "common_confusion": "string",
      "memory_tip": "string"
    }
  ]
}
""",
        user_prompt=(
            f"STRUCTURED NOTES:\n{_to_json(notes)}"
        ),
        max_tokens=3200,
    )

    # ---------------------------------------------------------
    # Agent 5: Create practice material.
    # ---------------------------------------------------------
    update(
        "Practice Agent is creating questions and flashcards."
    )

    practice = _ask_agent(
        agent_name="Practice Agent",
        system_prompt=base_rules
        + """
Your role is Practice Agent.

Create revision material only from the structured notes.

Create:
- 8 multiple-choice questions
- 5 short-answer questions
- 12 flashcards

Rules:
- Vary correct MCQ answers between A, B, C, and D.
- These are practice questions, not guaranteed exam questions.
- Keep page ranges based on the supplied notes.

Return exactly this JSON shape:

{
  "mcqs": [
    {
      "question": "string",
      "page_range": "string",
      "options": {
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      },
      "answer": "A/B/C/D",
      "explanation": "string"
    }
  ],
  "short_answer_questions": [
    {
      "question": "string",
      "page_range": "string",
      "model_answer": "string"
    }
  ],
  "flashcards": [
    {
      "front": "string",
      "back": "string",
      "page_range": "string"
    }
  ]
}
""",
        user_prompt=(
            f"STRUCTURED NOTES:\n{_to_json(notes)}\n\n"
            f"TOPIC MAP:\n{_to_json(topic_map)}"
        ),
        max_tokens=5200,
    )

    # ---------------------------------------------------------
    # Agent 6: Check coverage and revision priorities.
    # ---------------------------------------------------------
    update(
        "Quality Check Agent is checking coverage and revision priorities."
    )

    quality_check = _ask_agent(
        agent_name="Quality Check Agent",
        system_prompt=base_rules
        + """
Your role is Quality Check Agent.

Compare the topic map, notes, and chunk reports.

Identify:
- topics covered
- sections that may need review in the original PDF
- a clear revision checklist

Do not claim perfect accuracy.

Return exactly this JSON shape:

{
  "covered_topics": [
    {
      "topic": "string",
      "page_range": "string"
    }
  ],
  "possibly_unclear_or_missing": ["string"],
  "revision_checklist": ["string"],
  "confidence_note": "string"
}
""",
        user_prompt=(
            f"TOPIC MAP:\n{_to_json(topic_map)}\n\n"
            f"STRUCTURED NOTES:\n{_to_json(notes)}\n\n"
            f"CHUNK REPORTS:\n{_to_json(chunk_reports)}"
        ),
        max_tokens=2400,
    )

    return {
        "project": "StudyFlow AI",
        "workflow": [
            "Chunk Reading Agent",
            "Topic Mapping Agent",
            "Structured Notes Agent",
            "Concept Explanation Agent",
            "Practice Agent",
            "Quality Check Agent",
        ],
        "document_metadata": _safe_document_metadata(
            document_metadata
        ),
        "topic_map": topic_map,
        "notes": notes,
        "explanations": explanations,
        "practice": practice,
        "quality_check": quality_check,
    }