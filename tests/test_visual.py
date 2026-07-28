"""Unit tests for deterministic, model-free visual analysis of keyframes.

Every image used here is synthesised in-test as a lossless PNG so pixel values
are exact -- no JPEG rounding, no real video, no network, no model.
"""

import numpy as np
import pytest
from PIL import Image

from video_digest.visual import (
    BRIGHT_LUMA_THRESHOLD,
    DARK_LUMA_THRESHOLD,
    VIVID_SATURATION_THRESHOLD,
    FrameVisual,
    analyze_frame,
    analyze_frames,
    classify_substrate,
    kmeans_palette,
    luminance,
    rgb_to_hex,
    saturation,
    summarize_visuals,
)

# Three exact colours with known coverage: 70% near-black, 20% mid blue, 10% warm gold.
SUBSTRATE = (16, 16, 24)  # #101018
ACCENT_BLUE = (47, 111, 176)  # #2f6fb0
ACCENT_GOLD = (255, 204, 51)  # #ffcc33


def _write_three_band_png(path):
    """100x100 PNG: rows 0-69 substrate, 70-89 blue, 90-99 gold."""
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[0:70, :] = SUBSTRATE
    arr[70:90, :] = ACCENT_BLUE
    arr[90:100, :] = ACCENT_GOLD
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _write_flat_png(path, rgb, size=40):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = rgb
    Image.fromarray(arr, mode="RGB").save(path)
    return path


class TestLuminanceAndSaturation:
    def test_pure_black_is_zero_luminance(self):
        assert luminance(np.array([[0, 0, 0]], dtype=np.float64))[0] == pytest.approx(0.0)

    def test_pure_white_is_max_luminance(self):
        assert luminance(np.array([[255, 255, 255]], dtype=np.float64))[0] == pytest.approx(255.0)

    def test_green_weighs_more_than_blue_rec709(self):
        green = luminance(np.array([[0, 255, 0]], dtype=np.float64))[0]
        blue = luminance(np.array([[0, 0, 255]], dtype=np.float64))[0]
        assert green > blue
        assert green == pytest.approx(0.7152 * 255)
        assert blue == pytest.approx(0.0722 * 255)

    def test_neutral_gray_has_zero_saturation(self):
        assert saturation(np.array([[128, 128, 128]], dtype=np.float64))[0] == pytest.approx(0.0)

    def test_pure_red_is_fully_saturated(self):
        assert saturation(np.array([[255, 0, 0]], dtype=np.float64))[0] == pytest.approx(1.0)

    def test_black_saturation_is_zero_not_nan(self):
        value = saturation(np.array([[0, 0, 0]], dtype=np.float64))[0]
        assert value == pytest.approx(0.0)
        assert not np.isnan(value)


class TestRgbToHex:
    def test_lowercase_six_digit_hex(self):
        assert rgb_to_hex((255, 204, 51)) == "#ffcc33"

    def test_clamps_and_rounds(self):
        assert rgb_to_hex((-3.4, 15.6, 300.0)) == "#0010ff"


class TestClassifySubstrate:
    def test_bands_are_monotonic_labels(self):
        assert classify_substrate(2.0) == "pure black"
        assert classify_substrate(16.0) == "near-black"
        assert classify_substrate(40.0) == "very dark gray"
        assert classify_substrate(80.0) == "dark gray"
        assert classify_substrate(140.0) == "mid gray"
        assert classify_substrate(220.0) == "light"


class TestKmeansPalette:
    def _samples(self):
        rows = [SUBSTRATE] * 70 + [ACCENT_BLUE] * 20 + [ACCENT_GOLD] * 10
        return np.array(rows, dtype=np.float64)

    def test_recovers_exact_colours_and_coverage(self):
        palette = kmeans_palette(self._samples(), k=3, seed=0)
        assert [s.hex_color for s in palette] == ["#101018", "#2f6fb0", "#ffcc33"]
        assert [round(s.coverage, 4) for s in palette] == [0.7, 0.2, 0.1]

    def test_sorted_by_coverage_descending(self):
        palette = kmeans_palette(self._samples(), k=3, seed=0)
        coverages = [s.coverage for s in palette]
        assert coverages == sorted(coverages, reverse=True)

    def test_deterministic_across_repeated_runs(self):
        first = kmeans_palette(self._samples(), k=3, seed=0)
        second = kmeans_palette(self._samples(), k=3, seed=0)
        assert first == second

    def test_seed_is_honoured_and_output_still_pinned(self):
        # Different seed, same well-separated clusters -> same recovered palette.
        assert kmeans_palette(self._samples(), k=3, seed=1234) == kmeans_palette(
            self._samples(), k=3, seed=0
        )

    def test_k_capped_by_distinct_colour_count(self):
        palette = kmeans_palette(self._samples(), k=8, seed=0)
        assert len(palette) == 3

    def test_empty_sample_set_returns_empty_palette(self):
        assert kmeans_palette(np.zeros((0, 3), dtype=np.float64), k=3, seed=0) == []

    def test_weighted_samples_drive_coverage(self):
        samples = np.array([SUBSTRATE, ACCENT_GOLD], dtype=np.float64)
        weights = np.array([0.9, 0.1], dtype=np.float64)
        palette = kmeans_palette(samples, k=2, seed=0, weights=weights)
        assert palette[0].hex_color == "#101018"
        assert palette[0].coverage == pytest.approx(0.9)
        assert palette[1].coverage == pytest.approx(0.1)


class TestAnalyzeFrame:
    def test_measures_palette_luminance_saturation_and_bright_marks(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        fv = analyze_frame(frame, index=1, palette_size=3)

        assert fv.ok is True
        assert fv.error is None
        assert [s.hex_color for s in fv.palette] == ["#101018", "#2f6fb0", "#ffcc33"]
        assert [round(s.coverage, 4) for s in fv.palette] == [0.7, 0.2, 0.1]

        assert fv.mean_luma == pytest.approx(52.40, abs=0.05)
        assert fv.median_luma == pytest.approx(16.57, abs=0.05)
        assert fv.dark_fraction == pytest.approx(0.7)
        assert fv.mean_saturation == pytest.approx(0.4599, abs=0.001)
        assert fv.vivid_fraction == pytest.approx(0.3)
        assert fv.bright_mark_density == pytest.approx(0.1)

    def test_substrate_is_largest_region_and_classified(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        fv = analyze_frame(frame, index=1, palette_size=3)
        assert fv.substrate is not None
        assert fv.substrate.hex_color == "#101018"
        assert fv.substrate_luma == pytest.approx(16.57, abs=0.05)
        assert fv.substrate_verdict == "near-black"

    def test_dark_fraction_uses_documented_threshold(self, tmp_path):
        # A flat frame just above the dark threshold is not counted as dark.
        just_above = int(DARK_LUMA_THRESHOLD) + 4
        frame = _write_flat_png(tmp_path / "a.png", (just_above, just_above, just_above))
        assert analyze_frame(frame, index=1).dark_fraction == pytest.approx(0.0)

    def test_bright_mark_density_uses_documented_threshold(self, tmp_path):
        just_above = int(BRIGHT_LUMA_THRESHOLD) + 4
        frame = _write_flat_png(tmp_path / "b.png", (just_above, just_above, just_above))
        assert analyze_frame(frame, index=1).bright_mark_density == pytest.approx(1.0)

    def test_vivid_fraction_uses_documented_threshold(self, tmp_path):
        frame = _write_flat_png(tmp_path / "c.png", (255, 0, 0))
        fv = analyze_frame(frame, index=1)
        assert VIVID_SATURATION_THRESHOLD <= 1.0
        assert fv.vivid_fraction == pytest.approx(1.0)

    def test_same_file_twice_gives_identical_result(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        assert analyze_frame(frame, index=1) == analyze_frame(frame, index=1)

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        fv = analyze_frame(tmp_path / "nope.png", index=7)
        assert isinstance(fv, FrameVisual)
        assert fv.ok is False
        assert fv.error
        assert fv.palette == ()
        assert fv.substrate is None
        assert fv.index == 7

    def test_corrupt_file_is_reported_not_raised(self, tmp_path):
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"this is not an image")
        fv = analyze_frame(bad, index=2)
        assert fv.ok is False
        assert fv.error

    def test_never_invents_a_description(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        assert analyze_frame(frame, index=1).description is None


class TestAnalyzeFrames:
    def test_returns_one_entry_per_input_including_failures(self, tmp_path):
        good = _write_three_band_png(tmp_path / "001.png")
        missing = tmp_path / "002.png"
        results = analyze_frames([good, missing])
        assert len(results) == 2
        assert results[0].ok is True
        assert results[1].ok is False
        assert results[0].index == 1 and results[1].index == 2

    def test_empty_input_returns_empty_list(self):
        assert analyze_frames([]) == []


class TestOptionalVlmSeam:
    """The seam exists and is exercised, but nothing in the package wires a model to it."""

    def test_no_describer_by_default(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        assert analyze_frames([frame])[0].description is None

    def test_describer_output_is_attached_when_supplied(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")
        calls = []

        def fake_describer(path, fv):
            calls.append((path, fv.index))
            return "a caption from some future vision model"

        results = analyze_frames([frame], describe_fn=fake_describer)
        assert results[0].description == "a caption from some future vision model"
        assert len(calls) == 1

    def test_describer_not_called_for_unreadable_frame(self, tmp_path):
        calls = []
        analyze_frames(
            [tmp_path / "missing.png"], describe_fn=lambda p, f: calls.append(p) or "x"
        )
        assert calls == []

    def test_describer_failure_is_reported_not_raised(self, tmp_path):
        frame = _write_three_band_png(tmp_path / "001.png")

        def boom(path, fv):
            raise RuntimeError("vision backend down")

        fv = analyze_frames([frame], describe_fn=boom)[0]
        assert fv.description is None
        assert fv.describe_error and "vision backend down" in fv.describe_error


class TestSummarizeVisuals:
    def _three_frames(self, tmp_path):
        a = _write_three_band_png(tmp_path / "001.png")
        b = _write_three_band_png(tmp_path / "002.png")
        c = _write_flat_png(tmp_path / "003.png", (250, 250, 250))
        return analyze_frames([a, b, c])

    def test_rollup_counts_analyzed_and_failed(self, tmp_path):
        frames = analyze_frames(
            [_write_three_band_png(tmp_path / "001.png"), tmp_path / "missing.png"]
        )
        rollup = summarize_visuals(frames)
        assert rollup.analyzed_count == 1
        assert rollup.failed_count == 1
        assert rollup.failed_indices == (2,)

    def test_overall_palette_is_built_from_frame_palettes(self, tmp_path):
        frames = self._three_frames(tmp_path)
        rollup = summarize_visuals(frames, palette_size=4)
        hexes = [s.hex_color for s in rollup.palette]
        assert "#101018" in hexes
        assert sum(s.coverage for s in rollup.palette) == pytest.approx(1.0, abs=1e-6)

    def test_rollup_is_deterministic(self, tmp_path):
        frames = self._three_frames(tmp_path)
        assert summarize_visuals(frames) == summarize_visuals(frames)

    def test_flags_the_odd_frame_out(self, tmp_path):
        frames = self._three_frames(tmp_path)
        rollup = summarize_visuals(frames, outlier_z=1.0)
        outlier_indices = [o.index for o in rollup.outliers]
        assert 3 in outlier_indices
        assert 1 not in outlier_indices
        assert rollup.outliers[0].reason

    def test_no_outliers_when_all_frames_identical(self, tmp_path):
        frames = analyze_frames(
            [
                _write_three_band_png(tmp_path / "001.png"),
                _write_three_band_png(tmp_path / "002.png"),
            ]
        )
        assert summarize_visuals(frames).outliers == ()

    def test_empty_input_is_honest_not_fabricated(self):
        rollup = summarize_visuals([])
        assert rollup.analyzed_count == 0
        assert rollup.palette == ()
        assert rollup.substrate_verdict is None

    def test_all_failed_input_reports_no_palette(self, tmp_path):
        frames = analyze_frames([tmp_path / "a.png", tmp_path / "b.png"])
        rollup = summarize_visuals(frames)
        assert rollup.analyzed_count == 0
        assert rollup.failed_count == 2
        assert rollup.palette == ()
        assert rollup.substrate_verdict is None
