"""Unit tests for digest markdown assembly (pure function, no I/O)."""

from pathlib import Path

from video_digest.digest import (
    SUMMARY_TODO,
    assemble_digest,
    build_visual_notes_section,
)


class TestBuildVisualNotesSection:
    def test_empty_frames_returns_placeholder_message(self):
        assert build_visual_notes_section([]) == "(no keyframes extracted)"

    def test_one_line_per_frame_with_relative_path(self):
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg"), Path("scratch/frames/002.jpg")]
        )
        lines = section.splitlines()
        assert len(lines) == 2
        assert "frames/001.jpg" in lines[0]
        assert "frames/002.jpg" in lines[1]

    def test_includes_timestamp_label_when_provided(self):
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")], frame_timestamps=[12.0]
        )
        assert "[00:12]" in section

    def test_omits_timestamp_label_when_not_provided(self):
        section = build_visual_notes_section([Path("scratch/frames/001.jpg")])
        assert "[" not in section.split("`")[0]


class TestAssembleDigest:
    def test_contains_all_required_sections(self):
        md = assemble_digest(
            title="my-video",
            source="https://example.com/v/123",
            frame_paths=[Path("scratch/frames/001.jpg")],
            transcript_text="[00:00 - 00:05] hello world",
            frame_timestamps=[0.0],
        )
        for heading in (
            "## Summary",
            "## Visual notes",
            "## Ideas to steal",
            "## Action items",
            "## Full transcript",
        ):
            assert heading in md

    def test_transcript_is_folded_in_details_block(self):
        md = assemble_digest(
            title="my-video",
            source="local.mp4",
            frame_paths=[],
            transcript_text="hello world transcript body",
            frame_timestamps=None,
        )
        assert "<details>" in md
        assert "</details>" in md
        details_block = md.split("<details>")[1].split("</details>")[0]
        assert "hello world transcript body" in details_block

    def test_summary_placeholder_used_when_no_summary_given(self):
        md = assemble_digest(
            title="t", source="s", frame_paths=[], transcript_text="x", summary_text=None
        )
        assert SUMMARY_TODO in md

    def test_provided_summary_text_is_used_instead_of_placeholder(self):
        md = assemble_digest(
            title="t",
            source="s",
            frame_paths=[],
            transcript_text="x",
            summary_text="A concise mechanical summary.",
        )
        assert "A concise mechanical summary." in md
        assert SUMMARY_TODO not in md

    def test_empty_transcript_shows_no_transcript_placeholder(self):
        md = assemble_digest(title="t", source="s", frame_paths=[], transcript_text="")
        assert "(no transcript available)" in md

    def test_frontmatter_has_title_date_source(self):
        md = assemble_digest(
            title="World v4 redesign",
            source="https://tiktok.com/@x/video/1",
            frame_paths=[],
            transcript_text="x",
            digest_date="2026-07-23",
        )
        assert md.startswith("---\n")
        assert 'title: "video-digest — World v4 redesign"' in md
        assert "date: 2026-07-23" in md
        assert "source: https://tiktok.com/@x/video/1" in md

    def test_does_not_perform_visual_synthesis_itself(self):
        # The assembled digest should only ever contain the TODO placeholder text
        # for visual notes -- it must never fabricate a description of a frame.
        md = assemble_digest(
            title="t",
            source="s",
            frame_paths=[Path("scratch/frames/001.jpg")],
            transcript_text="x",
            frame_timestamps=[0.0],
        )
        assert "TODO" in md
