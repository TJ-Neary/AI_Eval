"""Tests for the marker parser utility."""

import pytest
from pathlib import Path
from utils.marker_parser import has_markers, read_marker_content, replace_marker_content


@pytest.fixture
def file_with_markers(tmp_path: Path) -> Path:
    """Create a file with marker section."""
    f = tmp_path / "test.md"
    f.write_text(
        "# Title\n\nSome content.\n\n"
        "<!-- AI_EVAL:BEGIN -->\n"
        "old results here\n"
        "<!-- AI_EVAL:END -->\n\n"
        "## Footer\n"
    )
    return f


@pytest.fixture
def file_without_markers(tmp_path: Path) -> Path:
    """Create a file without markers."""
    f = tmp_path / "plain.md"
    f.write_text("# Title\n\nJust a normal file.\n")
    return f


class TestHasMarkers:
    def test_returns_true_when_both_markers_present(self, file_with_markers: Path) -> None:
        assert has_markers(file_with_markers) is True

    def test_returns_false_when_no_markers(self, file_without_markers: Path) -> None:
        assert has_markers(file_without_markers) is False

    def test_returns_false_when_only_begin_marker(self, tmp_path: Path) -> None:
        f = tmp_path / "partial.md"
        f.write_text("<!-- AI_EVAL:BEGIN -->\nstuff\n")
        assert has_markers(f) is False

    def test_returns_false_for_nonexistent_file(self, tmp_path: Path) -> None:
        assert has_markers(tmp_path / "nope.md") is False


class TestReadMarkerContent:
    def test_reads_content_between_markers(self, file_with_markers: Path) -> None:
        content = read_marker_content(file_with_markers)
        assert content == "old results here\n"

    def test_returns_none_when_no_markers(self, file_without_markers: Path) -> None:
        assert read_marker_content(file_without_markers) is None

    def test_reads_empty_content_between_markers(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.md"
        f.write_text("<!-- AI_EVAL:BEGIN -->\n<!-- AI_EVAL:END -->\n")
        assert read_marker_content(f) == ""

    def test_reads_multiline_content(self, tmp_path: Path) -> None:
        f = tmp_path / "multi.md"
        f.write_text(
            "<!-- AI_EVAL:BEGIN -->\nline1\nline2\nline3\n<!-- AI_EVAL:END -->\n"
        )
        assert read_marker_content(f) == "line1\nline2\nline3\n"

    def test_raises_on_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_marker_content(tmp_path / "nope.md")


class TestReplaceMarkerContent:
    def test_replaces_content_between_markers(self, file_with_markers: Path) -> None:
        result = replace_marker_content(file_with_markers, "new results\n")
        assert result is True
        assert "new results" in file_with_markers.read_text()
        assert "old results here" not in file_with_markers.read_text()

    def test_preserves_content_outside_markers(self, file_with_markers: Path) -> None:
        replace_marker_content(file_with_markers, "replaced\n")
        content = file_with_markers.read_text()
        assert "# Title" in content
        assert "## Footer" in content

    def test_returns_false_when_no_markers(self, file_without_markers: Path) -> None:
        assert replace_marker_content(file_without_markers, "stuff") is False

    def test_preserves_markers(self, file_with_markers: Path) -> None:
        replace_marker_content(file_with_markers, "new\n")
        content = file_with_markers.read_text()
        assert "<!-- AI_EVAL:BEGIN -->" in content
        assert "<!-- AI_EVAL:END -->" in content

    def test_handles_multiline_replacement(self, file_with_markers: Path) -> None:
        replace_marker_content(file_with_markers, "line1\nline2\nline3\n")
        content = read_marker_content(file_with_markers)
        assert content == "line1\nline2\nline3\n"


class TestInlineMarkerIgnored:
    """Markers embedded in other text (e.g. documentation) must be ignored."""

    def test_ignores_inline_markers_in_backticks(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text(
            "# Docs\n\n"
            "Uses `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers.\n\n"
            "<!-- AI_EVAL:BEGIN -->\n"
            "real content\n"
            "<!-- AI_EVAL:END -->\n"
        )
        content = read_marker_content(f)
        assert content == "real content\n"

    def test_replace_skips_inline_markers(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text(
            "Markers: `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->`\n\n"
            "<!-- AI_EVAL:BEGIN -->\n"
            "old\n"
            "<!-- AI_EVAL:END -->\n"
        )
        replace_marker_content(f, "new\n")
        text = f.read_text()
        # Inline reference should be untouched
        assert "`<!-- AI_EVAL:BEGIN -->`" in text
        # Real section should be replaced
        assert read_marker_content(f) == "new\n"

    def test_has_markers_false_when_only_inline(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text(
            "Uses `<!-- AI_EVAL:BEGIN -->` / `<!-- AI_EVAL:END -->` markers.\n"
        )
        assert has_markers(f) is False
