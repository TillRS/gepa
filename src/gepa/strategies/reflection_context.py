"""Compact repeated reflection evidence without dropping distinct task content."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

MIN_REFERENCE_CHARS = 256
REFLECTION_CONTEXT_CONTRACT = {
    "version": 1,
    "duplicates": "exact_text_and_paragraph_references_within_each_prompt",
    "minimum_reference_chars": MIN_REFERENCE_CHARS,
    "repeated_lines": "retain_first_and_count_when_at_least_three_lines_save_256_chars",
    "unique_text": "preserved_without_character_truncation",
}


def _compact_repeated_lines(text: str) -> str:
    """Replace long runs of identical lines with one line and an explicit count."""
    lines = text.splitlines(keepends=True)
    result = []
    index = 0
    while index < len(lines):
        end = index + 1
        while end < len(lines) and lines[end] == lines[index]:
            end += 1
        result.append(lines[index])
        omitted = end - index - 1
        if omitted >= 2 and len(lines[index]) * omitted >= MIN_REFERENCE_CHARS:
            result.append(f"[Previous line repeated {omitted} additional times.]\n")
        else:
            result.extend(lines[index + 1 : end])
        index = end
    return "".join(result)


def compact_reflection_records(
    records: Sequence[Mapping[str, Any]], *, current_instruction: str | None = None
) -> list[dict[str, Any]]:
    """Reference repeated long text while keeping example boundaries and original data.

    Args:
        records: Evidence included together in one model request.
        current_instruction: Instruction already displayed before the evidence.

    Returns:
        New records with exact duplicate strings/paragraphs replaced by readable
        references and long identical-line runs represented with counts. Short
        values, distinct text, and non-text objects remain intact.
    """
    seen: dict[str, str] = {}
    if current_instruction:
        seen[current_instruction] = "Current instruction above"
        for index, paragraph in enumerate(re.split(r"\n[ \t]*\n", current_instruction), 1):
            seen.setdefault(paragraph, f"Current instruction above / paragraph {index}")

    def reference(text: str, location: str) -> str:
        """Keep the first long text occurrence and point subsequent copies to it."""
        if len(text) < MIN_REFERENCE_CHARS:
            return text
        if text in seen:
            return f"[Repeated text; see {seen[text]}.]"
        seen[text] = location
        return text

    def visit(value: Any, location: str) -> Any:
        """Copy nested evidence while compacting only repeated text."""
        if isinstance(value, Mapping):
            return {key: visit(item, f"{location} / {key}") for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [visit(item, f"{location} / Item {index}") for index, item in enumerate(value, 1)]
        if not isinstance(value, str):
            return value
        whole = reference(value, location)
        if whole != value:
            return whole
        parts = re.split(r"(\n[ \t]*\n)", value)
        if len(parts) > 1:
            for index in range(0, len(parts), 2):
                parts[index] = reference(parts[index], f"{location} / paragraph {index // 2 + 1}")
        return _compact_repeated_lines("".join(parts))

    return [visit(record, f"Example {index}") for index, record in enumerate(records, 1)]
