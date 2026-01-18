"""
PDF Ingestion for Medical Coding Exams

Extracts multiple-choice questions from PDF exam files.
"""

import re
import fitz  # PyMuPDF

from src.models import Question


def extract_questions_from_pdf(pdf_path: str) -> list[Question]:
    """
    Extract questions and options from a medical coding exam PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of Question objects with id, text, and options
    """
    doc = fitz.open(pdf_path)
    full_text = "\n".join(_extract_page_text(page) for page in doc)

    # Remove common headers/footers
    for noise in ["Medical Coding Ace", "TIMER START", "4 HOURS"]:
        full_text = full_text.replace(noise, "")

    return _parse_questions(full_text)


def _extract_page_text(page: fitz.Page) -> str:
    """Extract page text with basic multi-column handling."""
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return page.get_text("text")

    text_blocks = [b for b in blocks if str(b[4]).strip()]
    if not text_blocks:
        return page.get_text("text")

    page_width = page.rect.width
    mid_x = page_width / 2

    left_blocks = [b for b in text_blocks if b[0] < mid_x]
    right_blocks = [b for b in text_blocks if b[0] >= mid_x]

    def block_key(block: tuple) -> tuple:
        return (block[1], block[0])

    if left_blocks and right_blocks:
        left_max = max(b[2] for b in left_blocks)
        right_min = min(b[0] for b in right_blocks)
        gap = right_min - left_max
        if gap > page_width * 0.05:
            ordered = sorted(left_blocks, key=block_key) + sorted(right_blocks, key=block_key)
        else:
            ordered = sorted(text_blocks, key=block_key)
    else:
        ordered = sorted(text_blocks, key=block_key)

    lines = []
    for block in ordered:
        block_text = str(block[4]).strip()
        if not block_text:
            continue
        block_text = re.sub(r"[ \t]+", " ", block_text)
        lines.append(block_text)

    return "\n".join(lines)


def _parse_questions(text: str) -> list[Question]:
    """Parse question text into Question objects."""
    questions = []
    current_id = None
    current_text = []
    current_options = {}

    q_pattern = re.compile(r"^(\d+)\.\s+(.*)")
    opt_pattern = re.compile(r"^([A-D])\.\s+(.*)")

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        q_match = q_pattern.match(line)
        opt_match = opt_pattern.match(line)

        if q_match:
            # Save previous question
            if current_id is not None:
                questions.append(Question(
                    id=current_id,
                    text=" ".join(current_text).strip(),
                    options=current_options
                ))
            # Start new question
            current_id = int(q_match.group(1))
            current_text = [q_match.group(2)]
            current_options = {}

        elif opt_match:
            current_options[opt_match.group(1)] = opt_match.group(2)

        elif current_id is not None:
            # Continuation text
            if current_options:
                last_key = sorted(current_options.keys())[-1]
                current_options[last_key] += " " + line
            else:
                current_text.append(line)

    # Don't forget the last question
    if current_id is not None:
        questions.append(Question(
            id=current_id,
            text=" ".join(current_text).strip(),
            options=current_options
        ))

    return questions
