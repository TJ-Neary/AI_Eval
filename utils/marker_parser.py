"""
Marker-based content replacement for Markdown files.

Finds paired markers in a file and replaces the content between them.
Designed for updating sections of tracked files (README.md, catalog files)
without overwriting manually-written content.

Markers must appear on their own line (not embedded in other text):
    <!-- AI_EVAL:BEGIN -->
    (generated content here)
    <!-- AI_EVAL:END -->
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BEGIN_MARKER = "<!-- AI_EVAL:BEGIN -->"
DEFAULT_END_MARKER = "<!-- AI_EVAL:END -->"


def _find_standalone_marker(content: str, marker: str, start: int = 0) -> int:
    """Find a marker that appears as the sole content on its line.

    Returns the index of the marker in the content, or -1 if not found.
    """
    pos = start
    while True:
        idx = content.find(marker, pos)
        if idx == -1:
            return -1

        # Check that marker is the only non-whitespace content on its line
        line_start = content.rfind("\n", 0, idx) + 1  # 0 if no newline found
        line_end = content.find("\n", idx + len(marker))
        if line_end == -1:
            line_end = len(content)

        line = content[line_start:line_end]
        if line.strip() == marker:
            return idx

        # Not standalone, keep searching after this occurrence
        pos = idx + len(marker)


def has_markers(
    file_path: Path,
    begin_marker: str = DEFAULT_BEGIN_MARKER,
    end_marker: str = DEFAULT_END_MARKER,
) -> bool:
    """Check whether a file contains the marker pair on standalone lines."""
    if not file_path.exists():
        return False
    content = file_path.read_text()
    begin_idx = _find_standalone_marker(content, begin_marker)
    if begin_idx == -1:
        return False
    end_idx = _find_standalone_marker(content, end_marker, begin_idx + len(begin_marker))
    return end_idx != -1


def read_marker_content(
    file_path: Path,
    begin_marker: str = DEFAULT_BEGIN_MARKER,
    end_marker: str = DEFAULT_END_MARKER,
) -> Optional[str]:
    """
    Read the content between markers in a file.

    Returns:
        Content between markers (excluding marker lines), or None if not found.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    content = file_path.read_text()
    begin_idx = _find_standalone_marker(content, begin_marker)
    if begin_idx == -1:
        return None
    end_idx = _find_standalone_marker(content, end_marker, begin_idx + len(begin_marker))
    if end_idx == -1:
        return None

    # Content starts after the begin marker line
    start = content.index("\n", begin_idx) + 1
    return content[start:end_idx]


def replace_marker_content(
    file_path: Path,
    new_content: str,
    begin_marker: str = DEFAULT_BEGIN_MARKER,
    end_marker: str = DEFAULT_END_MARKER,
) -> bool:
    """
    Replace content between markers in a file.

    Preserves the marker lines and all content outside them.

    Returns:
        True if content was replaced, False if markers not found.
    """
    content = file_path.read_text()
    begin_idx = _find_standalone_marker(content, begin_marker)
    if begin_idx == -1:
        return False
    end_idx = _find_standalone_marker(content, end_marker, begin_idx + len(begin_marker))
    if end_idx == -1:
        return False

    # Find end of begin marker line
    start = content.index("\n", begin_idx) + 1

    # Build new content: before + begin marker line + new content + end marker + after
    updated = content[:start] + new_content + content[end_idx:]
    file_path.write_text(updated)
    return True
