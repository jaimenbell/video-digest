"""Unit tests for digest markdown assembly (pure function, no I/O)."""

from pathlib import Path

from video_digest.digest import (
    SUMMARY_TODO,
    UNREADABLE_FRAME_PREFIX,
    VISUAL_NOTE_TODO,
    VISUAL_ROLLUP_NOT_RUN,
    assemble_digest,
    build_visual_notes_section,
    build_visual_rollup_section,
)
from video_digest.visual import FrameOutlier, FrameVisual, Swatch, VideoVisual


def _ok_visual(index=1, path="scratch/frames/001.jpg", **overrides):
    defaults = dict(
        index=index,
        path=path,
        ok=True,
        width=576,
        height=1024,
        sampled_pixels=589824,
        palette=(
            Swatch("#101018", (16, 16, 24), 0.7),
            Swatch("#2f6fb0", (47, 111, 176), 0.2),
            Swatch("#ffcc33", (255, 204, 51), 0.1),
        ),
        substrate=Swatch("#101018", (16, 16, 24), 0.7),
        substrate_luma=16.57,
        mean_luma=52.4,
        median_luma=16.57,
        dark_fraction=0.7,
        mean_saturation=0.46,
        vivid_fraction=0.3,
        bright_mark_density=0.1,
        glow_fraction=0.12,
    )
    defaults.update(overrides)
    return FrameVisual(**defaults)


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


class TestVisualNotesWithMeasuredData:
    def test_renders_hex_swatches_with_coverage(self):
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")],
            frame_timestamps=[15.0],
            frame_visuals=[_ok_visual()],
        )
        assert "[00:15]" in section
        assert "`#101018` 70.0%" in section
        assert "`#2f6fb0` 20.0%" in section
        assert "`#ffcc33` 10.0%" in section

    def test_renders_substrate_verdict_luminance_saturation_and_bright_marks(self):
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")], frame_visuals=[_ok_visual()]
        )
        assert "near-black" in section
        assert "70.0% of pixels below luma 32" in section
        assert "mean 0.46" in section
        assert "10.00% of pixels at/above luma 200" in section

    def test_todo_is_dropped_for_an_analysed_frame(self):
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")], frame_visuals=[_ok_visual()]
        )
        assert VISUAL_NOTE_TODO not in section

    def test_unreadable_frame_is_reported_and_keeps_its_todo(self):
        broken = FrameVisual(
            index=1, path="scratch/frames/001.jpg", ok=False, error="file not found"
        )
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")], frame_visuals=[broken]
        )
        assert UNREADABLE_FRAME_PREFIX in section
        assert "file not found" in section
        assert VISUAL_NOTE_TODO in section

    def test_unreadable_frame_is_never_silently_skipped(self):
        broken = FrameVisual(index=2, path="scratch/frames/002.jpg", ok=False, error="boom")
        section = build_visual_notes_section(
            [Path("scratch/frames/001.jpg"), Path("scratch/frames/002.jpg")],
            frame_visuals=[_ok_visual(), broken],
        )
        assert "frames/002.jpg" in section

    def test_falls_back_to_todo_when_no_visuals_supplied(self):
        section = build_visual_notes_section([Path("scratch/frames/001.jpg")])
        assert VISUAL_NOTE_TODO in section

    def test_optional_description_is_rendered_only_when_present(self):
        without = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")], frame_visuals=[_ok_visual()]
        )
        assert "description:" not in without
        with_desc = build_visual_notes_section(
            [Path("scratch/frames/001.jpg")],
            frame_visuals=[_ok_visual(description="a future VLM caption")],
        )
        assert "description: a future VLM caption" in with_desc


class TestVisualRollupSection:
    def _rollup(self, **overrides):
        defaults = dict(
            analyzed_count=30,
            failed_count=0,
            palette=(
                Swatch("#232c28", (35, 44, 40), 0.3697),
                Swatch("#7a0b24", (122, 11, 36), 0.157),
            ),
            substrate=Swatch("#232c28", (35, 44, 40), 0.3697),
            substrate_verdict="very dark gray",
            mean_luma=77.41,
            mean_dark_fraction=0.1921,
            mean_saturation=0.574,
            mean_vivid_fraction=0.5736,
            mean_bright_mark_density=0.0293,
        )
        defaults.update(overrides)
        return VideoVisual(**defaults)

    def test_not_run_placeholder_when_rollup_is_none(self):
        assert build_visual_rollup_section(None) == VISUAL_ROLLUP_NOT_RUN

    def test_renders_palette_table_with_roles(self):
        section = build_visual_rollup_section(self._rollup())
        assert "| Role | Hex | Coverage |" in section
        assert "| substrate | `#232c28` | 36.97% |" in section
        assert "| accent 1 | `#7a0b24` | 15.70% |" in section

    def test_reports_substrate_verdict_and_dark_fraction(self):
        section = build_visual_rollup_section(self._rollup())
        assert "very dark gray" in section
        assert "19.2% of pixels below luma 32" in section

    def test_reports_outlier_frames(self):
        rollup = self._rollup(
            outliers=(
                FrameOutlier(
                    index=6,
                    path="scratch/frames/006.jpg",
                    z_score=4.8,
                    reason="high bright-mark density (z=+4.80)",
                ),
            )
        )
        section = build_visual_rollup_section(rollup)
        assert "frames/006.jpg" in section
        assert "high bright-mark density" in section

    def test_states_uniformity_when_no_outliers(self):
        assert "none" in build_visual_rollup_section(self._rollup())

    def test_reports_all_frames_unreadable_honestly(self):
        section = build_visual_rollup_section(
            VideoVisual(analyzed_count=0, failed_count=3, failed_indices=(1, 2, 3))
        )
        assert "3 unreadable" in section


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

    def test_measured_visuals_replace_the_todo_placeholder(self):
        md = assemble_digest(
            title="t",
            source="s",
            frame_paths=[Path("scratch/frames/001.jpg")],
            transcript_text="x",
            frame_timestamps=[0.0],
            frame_visuals=[_ok_visual()],
        )
        visual_block = md.split("## Visual notes")[1].split("## Visual roll-up")[0]
        assert VISUAL_NOTE_TODO not in visual_block
        assert "`#101018`" in visual_block

    def test_rollup_section_is_rendered_when_supplied(self):
        rollup = VideoVisual(
            analyzed_count=2,
            failed_count=0,
            palette=(Swatch("#232c28", (35, 44, 40), 0.37),),
            substrate=Swatch("#232c28", (35, 44, 40), 0.37),
            substrate_verdict="very dark gray",
            mean_luma=77.4,
            mean_dark_fraction=0.192,
            mean_saturation=0.574,
            mean_vivid_fraction=0.573,
            mean_bright_mark_density=0.029,
        )
        md = assemble_digest(
            title="t",
            source="s",
            frame_paths=[Path("scratch/frames/001.jpg")],
            transcript_text="x",
            frame_visuals=[_ok_visual()],
            visual_rollup=rollup,
        )
        assert "## Visual roll-up" in md
        assert "very dark gray" in md
        assert "`#232c28`" in md

    def test_rollup_section_says_not_run_when_absent(self):
        md = assemble_digest(title="t", source="s", frame_paths=[], transcript_text="x")
        assert VISUAL_ROLLUP_NOT_RUN in md

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
