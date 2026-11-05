"""In-memory PDF extraction and chunking for long academic PDFs."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

try:
    import pytesseract
    OCR_IMPORT_ERROR: str | None = None
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    OCR_IMPORT_ERROR = "pytesseract is not installed."

from utils import redact_sensitive_text



MAX_PAGES = 60

# Safety limits
MAX_EXTRACTED_CHARACTERS = 120_000
MAX_CHARACTERS_PER_PAGE = 6_000
MAX_CHUNK_CHARACTERS = 12_000
MIN_SELECTABLE_TEXT = 120


def _configure_tesseract() -> None:
    """Use a path from .env or the normal Windows Tesseract path."""
    if pytesseract is None:
        return

    configured_path = os.getenv("TESSERACT_CMD", "").strip()
    windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    if configured_path and Path(configured_path).exists():
        pytesseract.pytesseract.tesseract_cmd = configured_path
    elif windows_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_default)


def _ocr_page(page: Any) -> str:
    """Try OCR only when normal selectable PDF text is weak."""
    if pytesseract is None:
        raise RuntimeError(OCR_IMPORT_ERROR or "OCR is unavailable.")

    _configure_tesseract()

    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    )

    image = Image.open(
        io.BytesIO(pixmap.tobytes("png"))
    ).convert("RGB")

    return pytesseract.image_to_string(image)


def _add_counts(
    total: dict[str, int],
    new_counts: dict[str, int],
) -> None:
    """Add privacy-redaction counts together."""
    for key, value in new_counts.items():
        total[key] = total.get(key, 0) + value


def _build_chunks(
    redacted_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Group pages into manageable chunks before sending them to the AI.

    Each chunk keeps page labels such as:
    --- Page 1 ---
    --- Page 2 ---
    """

    chunks: list[dict[str, Any]] = []

    current_blocks: list[str] = []
    current_pages: list[int] = []
    current_length = 0

    for page in redacted_pages:
        page_number = page["page_number"]

        block = (
            f"--- Page {page_number} ---\n"
            f"{page['text']}"
        )

        # Start a new chunk if adding this page would make the
        # current chunk too large.
        if (
            current_blocks
            and current_length + len(block) > MAX_CHUNK_CHARACTERS
        ):
            chunks.append(
                {
                    "chunk_number": len(chunks) + 1,
                    "start_page": current_pages[0],
                    "end_page": current_pages[-1],
                    "page_numbers": current_pages.copy(),
                    "text": "\n\n".join(current_blocks),
                }
            )

            current_blocks = []
            current_pages = []
            current_length = 0

        current_blocks.append(block)
        current_pages.append(page_number)
        current_length += len(block)

    # Save the final chunk.
    if current_blocks:
        chunks.append(
            {
                "chunk_number": len(chunks) + 1,
                "start_page": current_pages[0],
                "end_page": current_pages[-1],
                "page_numbers": current_pages.copy(),
                "text": "\n\n".join(current_blocks),
            }
        )

    return chunks


def extract_pdf_text(
    file_bytes: bytes,
    file_name: str,
    *,
    use_ocr_fallback: bool = True,
    max_pages: int = MAX_PAGES,
) -> dict[str, Any]:
    """
    Extract PDF text in memory.

    The returned analysis_chunks are only used during the current Streamlit run.
    They are not included in the downloaded JSON export.
    """

    warnings: list[str] = []

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as error:
        raise ValueError(
            "This file could not be opened as a PDF."
        ) from error

    try:
        total_pages = len(document)
        processed_pages = min(total_pages, max_pages)
        fully_processed = processed_pages == total_pages

        if not fully_processed:
            warnings.append(
                f"This PDF has {total_pages} pages. "
                f"The prototype analyzed the first {processed_pages} pages only."
            )

        redacted_pages: list[dict[str, Any]] = []
        ocr_pages: list[int] = []
        redaction_counts: dict[str, int] = {}

        remaining_characters = MAX_EXTRACTED_CHARACTERS

        for page_index in range(processed_pages):
            page_number = page_index + 1
            page = document[page_index]

            text = page.get_text(
                "text",
                sort=True,
            ).strip()

            # Use OCR only if regular PDF text is nearly empty.
            if use_ocr_fallback and len(text) < MIN_SELECTABLE_TEXT:
                try:
                    ocr_text = _ocr_page(page).strip()

                    if len(ocr_text) > len(text):
                        text = ocr_text
                        ocr_pages.append(page_number)

                except Exception:
                    warnings.append(
                        f"Page {page_number} has little selectable text "
                        f"and OCR could not read it."
                    )

            if not text:
                continue

            # Prevent one very large page from consuming too much space.
            if len(text) > MAX_CHARACTERS_PER_PAGE:
                text = text[:MAX_CHARACTERS_PER_PAGE]

                warnings.append(
                    f"Page {page_number} was shortened to "
                    f"{MAX_CHARACTERS_PER_PAGE:,} characters before analysis."
                )

            if remaining_characters <= 0:
                warnings.append(
                    f"The extracted-text safety limit of "
                    f"{MAX_EXTRACTED_CHARACTERS:,} characters was reached."
                )
                break

            if len(text) > remaining_characters:
                text = text[:remaining_characters]

                warnings.append(
                    f"The extracted-text safety limit of "
                    f"{MAX_EXTRACTED_CHARACTERS:,} characters was reached."
                )

            # Redact before text is sent to DeepSeek.
            redacted_text, page_counts = redact_sensitive_text(text)

            _add_counts(
                redaction_counts,
                page_counts,
            )

            redacted_pages.append(
                {
                    "page_number": page_number,
                    "text": redacted_text,
                }
            )

            remaining_characters -= len(text)

            if remaining_characters <= 0:
                break

        extracted_characters = sum(
            len(page["text"])
            for page in redacted_pages
        )

        analysis_chunks = _build_chunks(redacted_pages)

        text_extracted = bool(analysis_chunks)

        if not text_extracted:
            warnings.append(
                "No readable text was found. This may be a scanned, "
                "protected, or image-only PDF."
            )

        if fully_processed and remaining_characters > 0:
            coverage_status = "Full PDF analyzed"

        elif remaining_characters <= 0:
            coverage_status = "Partial: text safety limit reached"

        else:
            coverage_status = (
                f"Partial: first {processed_pages} pages analyzed"
            )

        return {
            "file_name": file_name,
            "file_type": "PDF",
            "size_kb": round(len(file_bytes) / 1024, 1),
            "total_pages": total_pages,
            "processed_pages": processed_pages,
            "fully_processed": (
                fully_processed
                and remaining_characters > 0
            ),
            "coverage_status": coverage_status,
            "text_extracted": text_extracted,
            "extracted_characters": extracted_characters,
            "characters_sent_to_ai": extracted_characters,
            "analysis_chunk_count": len(analysis_chunks),
            "ocr_pages": ocr_pages,
            "warnings": warnings,
            "redaction_counts": redaction_counts,

            # Used only while the app is running.
            # It will not be exported to JSON.
            "analysis_chunks": analysis_chunks,
        }

    finally:
        document.close()