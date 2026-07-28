"""Stage 4: deterministic, model-free visual analysis of the extracted keyframes.

This is the module that actually *looks at the pixels*. It deliberately does NOT
use a vision model:

* the local vLLM this pipeline talks to (qwen3-14b) is text-only and cannot
  describe an image at all, and
* for an art-direction reference, **measured colour data beats prose**: it costs
  nothing, it runs offline (the "$0, fully local" constraint), it is reproducible
  bit-for-bit, and it cannot hallucinate a colour that isn't in the frame.

So every number below is a straight measurement over the frame's pixels:

    palette              k-means over sampled pixels -> top-N hex swatches + coverage
    luminance            Rec.709 weighted luma over sRGB bytes (0-255), plus the
                         fraction of pixels below `DARK_LUMA_THRESHOLD` -- i.e. a
                         defensible answer to "how black is the substrate?"
    saturation           HSV S = (max-min)/max per pixel; mean + "vivid" fraction
    bright-mark density  fraction of pixels at/above `BRIGHT_LUMA_THRESHOLD` --
                         the emissive-density measure ("how many small glowing
                         marks per unit area")

Determinism: k-means is seeded (`DEFAULT_SEED`), initialised with a seeded
k-means++ draw, run for a fixed iteration cap, empty clusters are re-seeded by a
deterministic rule (first farthest point wins), and swatches are sorted by
(-coverage, hex). Pixel subsampling is a fixed stride, never a random draw. Same
frames in -> same hex values out. `tests/test_visual.py` pins this.

Honest degradation: a missing or corrupt frame yields a `FrameVisual` with
`ok=False` and the error text. It is never silently skipped, and its numbers are
never invented.

Optional-VLM seam: `analyze_frames(..., describe_fn=...)` accepts a callable
`(path, FrameVisual) -> str` whose output lands in `FrameVisual.description`.
**Nothing in this package wires a model to that seam.** Choosing and hosting a
VLM is an architecture decision, not a side effect of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

# --- Measurement constants (documented, not magic) ---------------------------

#: Rec.709 luma weights, applied to gamma-encoded sRGB bytes (0-255). This is the
#: "how bright does it look on screen" number, not linear-light luminance.
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

#: A pixel below this luma counts as "substrate dark" (0-255 scale).
DARK_LUMA_THRESHOLD = 32.0
#: A pixel at/above this luma counts as a bright/emissive mark.
BRIGHT_LUMA_THRESHOLD = 200.0
#: A softer tier below the bright threshold -- ambient glow rather than a hot mark.
GLOW_LUMA_THRESHOLD = 160.0
#: HSV saturation at/above which a pixel counts as "vivid".
VIVID_SATURATION_THRESHOLD = 0.5

DEFAULT_PALETTE_SIZE = 5
DEFAULT_SEED = 0
DEFAULT_KMEANS_ITERATIONS = 25
#: Cap on pixels used for the histogram/threshold statistics (stride-subsampled).
DEFAULT_MAX_STAT_PIXELS = 1_000_000
#: Cap on pixels fed to k-means (stride-subsampled from the stat sample).
DEFAULT_MAX_PALETTE_PIXELS = 20_000
#: |z| at/above which a frame is called a visual outlier in the roll-up.
DEFAULT_OUTLIER_Z = 2.0

_SUBSTRATE_BANDS = (
    (8.0, "pure black"),
    (24.0, "near-black"),
    (48.0, "very dark gray"),
    (96.0, "dark gray"),
    (160.0, "mid gray"),
)


# --- Data types --------------------------------------------------------------


@dataclass(frozen=True)
class Swatch:
    """One palette entry: a colour and the share of pixels it accounts for."""

    hex_color: str
    rgb: tuple[int, int, int]
    coverage: float  # 0.0 - 1.0

    @property
    def coverage_pct(self) -> float:
        return self.coverage * 100.0


@dataclass(frozen=True)
class FrameVisual:
    """Measured visual facts about one keyframe. No prose, no inference."""

    index: int
    path: str
    ok: bool
    error: str | None = None
    width: int = 0
    height: int = 0
    sampled_pixels: int = 0
    palette: tuple[Swatch, ...] = ()
    substrate: Swatch | None = None
    substrate_luma: float = 0.0
    mean_luma: float = 0.0
    median_luma: float = 0.0
    dark_fraction: float = 0.0
    mean_saturation: float = 0.0
    vivid_fraction: float = 0.0
    bright_mark_density: float = 0.0
    glow_fraction: float = 0.0
    #: Optional-VLM seam. Always None unless a caller supplies `describe_fn`.
    description: str | None = None
    #: Set when a supplied `describe_fn` raised -- reported, never swallowed.
    describe_error: str | None = None

    @property
    def substrate_verdict(self) -> str | None:
        if not self.ok or self.substrate is None:
            return None
        return classify_substrate(self.substrate_luma)


@dataclass(frozen=True)
class FrameOutlier:
    index: int
    path: str
    z_score: float
    reason: str


@dataclass(frozen=True)
class VideoVisual:
    """Roll-up across all analysed keyframes."""

    analyzed_count: int
    failed_count: int
    failed_indices: tuple[int, ...] = ()
    palette: tuple[Swatch, ...] = ()
    substrate: Swatch | None = None
    substrate_verdict: str | None = None
    mean_luma: float = 0.0
    mean_dark_fraction: float = 0.0
    mean_saturation: float = 0.0
    mean_vivid_fraction: float = 0.0
    mean_bright_mark_density: float = 0.0
    outliers: tuple[FrameOutlier, ...] = ()


DescribeFn = Callable[[Path, FrameVisual], str]


# --- Pixel math (pure) -------------------------------------------------------


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 weighted luma of gamma-encoded sRGB values, on the same 0-255 scale.

    `rgb` is an (N, 3) float array. Returns an (N,) float array.
    """
    weights = np.array(LUMA_WEIGHTS, dtype=np.float64)
    return rgb.astype(np.float64) @ weights


def saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation, (max-min)/max per pixel, 0.0 for pure black (no NaN)."""
    values = rgb.astype(np.float64)
    high = values.max(axis=1)
    low = values.min(axis=1)
    out = np.zeros_like(high)
    nonzero = high > 0
    out[nonzero] = (high[nonzero] - low[nonzero]) / high[nonzero]
    return out


def rgb_to_hex(rgb: Sequence[float]) -> str:
    """Round + clamp an RGB triple to a lowercase `#rrggbb` string."""
    r, g, b = (int(min(255, max(0, round(float(c))))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def classify_substrate(luma: float) -> str:
    """Turn a substrate luma (0-255) into a defensible label."""
    for ceiling, label in _SUBSTRATE_BANDS:
        if luma <= ceiling:
            return label
    return "light"


def _stride_subsample(arr: np.ndarray, max_rows: int) -> np.ndarray:
    """Deterministic fixed-stride subsample -- never a random draw, and it keeps
    exact pixel values (unlike resampling, which would blur small bright marks)."""
    count = arr.shape[0]
    if max_rows <= 0 or count <= max_rows:
        return arr
    stride = int(math.ceil(count / max_rows))
    return arr[::stride]


# --- k-means (seeded, deterministic) -----------------------------------------


def _kmeans_plus_plus_init(
    samples: np.ndarray, weights: np.ndarray, k: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    probabilities = weights / weights.sum()
    first = int(rng.choice(n, p=probabilities))
    centroids = [samples[first]]
    distances = np.sum((samples - centroids[0]) ** 2, axis=1)
    while len(centroids) < k:
        scores = distances * weights
        total = scores.sum()
        if total <= 0:
            # Every remaining point coincides with a chosen centroid: fall back to
            # the first unused distinct row (deterministic).
            for row in samples:
                if not any(np.array_equal(row, c) for c in centroids):
                    centroids.append(row)
                    break
            else:
                break
            distances = np.minimum(
                distances, np.sum((samples - centroids[-1]) ** 2, axis=1)
            )
            continue
        pick = int(rng.choice(n, p=scores / total))
        centroids.append(samples[pick])
        distances = np.minimum(distances, np.sum((samples - samples[pick]) ** 2, axis=1))
    return np.array(centroids, dtype=np.float64)


def kmeans_palette(
    samples: np.ndarray,
    k: int = DEFAULT_PALETTE_SIZE,
    seed: int = DEFAULT_SEED,
    iterations: int = DEFAULT_KMEANS_ITERATIONS,
    weights: np.ndarray | None = None,
) -> list[Swatch]:
    """Seeded k-means over (N, 3) RGB samples -> swatches sorted by coverage.

    `weights` (optional, shape (N,)) lets a caller cluster already-summarised
    colours -- e.g. the roll-up clusters every frame's swatches weighted by that
    swatch's coverage, so it needs no second pass over the pixels.

    Deterministic by construction: seeded k-means++ init, fixed iteration cap,
    first-farthest-point re-seed for empty clusters, and a total order on output.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return []
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("samples must have shape (N, 3)")
    if k < 1:
        raise ValueError("k must be at least 1")

    if weights is None:
        weights = np.ones(samples.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (samples.shape[0],):
            raise ValueError("weights must have shape (N,)")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if weights.sum() <= 0:
            return []

    distinct = np.unique(samples, axis=0)
    k = min(k, distinct.shape[0])

    centroids = _kmeans_plus_plus_init(samples, weights, k, seed)
    k = centroids.shape[0]
    assignments = np.zeros(samples.shape[0], dtype=np.int64)

    for _ in range(iterations):
        # (N, k) squared distances; argmin ties resolve to the lowest cluster index.
        deltas = samples[:, None, :] - centroids[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(new_assignments, assignments) and _ > 0:
            assignments = new_assignments
            break
        assignments = new_assignments
        for cluster in range(k):
            mask = assignments == cluster
            cluster_weight = weights[mask].sum()
            if cluster_weight > 0:
                centroids[cluster] = (
                    samples[mask] * weights[mask][:, None]
                ).sum(axis=0) / cluster_weight
            else:
                # Deterministic empty-cluster re-seed: the point farthest from its
                # own centroid, first index wins on a tie.
                own = distances[np.arange(samples.shape[0]), assignments]
                centroids[cluster] = samples[int(np.argmax(own))]

    total_weight = weights.sum()
    swatches: list[Swatch] = []
    for cluster in range(k):
        mask = assignments == cluster
        cluster_weight = float(weights[mask].sum())
        if cluster_weight <= 0:
            continue
        centre = (samples[mask] * weights[mask][:, None]).sum(axis=0) / cluster_weight
        rgb = tuple(int(min(255, max(0, round(float(c))))) for c in centre)
        swatches.append(
            Swatch(
                hex_color=rgb_to_hex(centre),
                rgb=rgb,  # type: ignore[arg-type]
                coverage=cluster_weight / float(total_weight),
            )
        )

    swatches.sort(key=lambda s: (-s.coverage, s.hex_color))
    return swatches


# --- Per-frame analysis ------------------------------------------------------


def analyze_pixels(
    rgb: np.ndarray,
    index: int,
    path: str,
    width: int,
    height: int,
    palette_size: int = DEFAULT_PALETTE_SIZE,
    seed: int = DEFAULT_SEED,
    max_stat_pixels: int = DEFAULT_MAX_STAT_PIXELS,
    max_palette_pixels: int = DEFAULT_MAX_PALETTE_PIXELS,
) -> FrameVisual:
    """Pure measurement over an (H, W, 3) or (N, 3) uint8/float RGB array."""
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    flat = _stride_subsample(flat, max_stat_pixels)
    if flat.shape[0] == 0:
        return FrameVisual(
            index=index, path=path, ok=False, error="frame contains no pixels"
        )

    lumas = luminance(flat)
    sats = saturation(flat)
    palette = kmeans_palette(
        _stride_subsample(flat, max_palette_pixels), k=palette_size, seed=seed
    )
    substrate = palette[0] if palette else None
    substrate_luma = (
        float(luminance(np.array([substrate.rgb], dtype=np.float64))[0])
        if substrate
        else 0.0
    )

    return FrameVisual(
        index=index,
        path=path,
        ok=True,
        width=width,
        height=height,
        sampled_pixels=int(flat.shape[0]),
        palette=tuple(palette),
        substrate=substrate,
        substrate_luma=substrate_luma,
        mean_luma=float(lumas.mean()),
        median_luma=float(np.median(lumas)),
        dark_fraction=float((lumas < DARK_LUMA_THRESHOLD).mean()),
        mean_saturation=float(sats.mean()),
        vivid_fraction=float((sats >= VIVID_SATURATION_THRESHOLD).mean()),
        bright_mark_density=float((lumas >= BRIGHT_LUMA_THRESHOLD).mean()),
        glow_fraction=float((lumas >= GLOW_LUMA_THRESHOLD).mean()),
    )


def analyze_frame(
    frame_path: Path | str,
    index: int = 1,
    palette_size: int = DEFAULT_PALETTE_SIZE,
    seed: int = DEFAULT_SEED,
    max_stat_pixels: int = DEFAULT_MAX_STAT_PIXELS,
    max_palette_pixels: int = DEFAULT_MAX_PALETTE_PIXELS,
) -> FrameVisual:
    """Measure one keyframe. A missing/corrupt frame is REPORTED, never raised
    and never silently skipped."""
    path = Path(frame_path)
    try:
        with Image.open(path) as image:
            image.load()
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            arr = np.asarray(rgb_image, dtype=np.uint8)
    except FileNotFoundError:
        return FrameVisual(index=index, path=str(path), ok=False, error="file not found")
    except UnidentifiedImageError:
        return FrameVisual(
            index=index, path=str(path), ok=False, error="not a readable image file"
        )
    except OSError as exc:  # truncated / unreadable image data
        return FrameVisual(
            index=index, path=str(path), ok=False, error=f"unreadable image: {exc}"
        )

    return analyze_pixels(
        arr,
        index=index,
        path=str(path),
        width=width,
        height=height,
        palette_size=palette_size,
        seed=seed,
        max_stat_pixels=max_stat_pixels,
        max_palette_pixels=max_palette_pixels,
    )


def analyze_frames(
    frame_paths: Iterable[Path | str],
    palette_size: int = DEFAULT_PALETTE_SIZE,
    seed: int = DEFAULT_SEED,
    describe_fn: DescribeFn | None = None,
) -> list[FrameVisual]:
    """Measure every keyframe, in order, one `FrameVisual` per input path.

    `describe_fn` is the optional-VLM seam described in the module docstring: it
    is None everywhere in this package, is only ever called for a frame that read
    successfully, and a failure inside it is recorded on `describe_error` rather
    than taking down the run.
    """
    results: list[FrameVisual] = []
    for i, frame_path in enumerate(frame_paths, start=1):
        visual = analyze_frame(frame_path, index=i, palette_size=palette_size, seed=seed)
        if describe_fn is not None and visual.ok:
            try:
                visual = replace(
                    visual, description=describe_fn(Path(frame_path), visual)
                )
            except Exception as exc:  # a broken seam must not fabricate or crash
                visual = replace(visual, description=None, describe_error=str(exc))
        results.append(visual)
    return results


# --- Roll-up across frames ---------------------------------------------------

_OUTLIER_FEATURES = (
    ("mean_luma", "overall brightness"),
    ("mean_saturation", "saturation"),
    ("bright_mark_density", "bright-mark density"),
    ("dark_fraction", "dark-substrate share"),
)


def summarize_visuals(
    frames: Sequence[FrameVisual],
    palette_size: int = DEFAULT_PALETTE_SIZE,
    seed: int = DEFAULT_SEED,
    outlier_z: float = DEFAULT_OUTLIER_Z,
) -> VideoVisual:
    """Roll the per-frame measurements up into a whole-video palette + outliers.

    The overall palette is a weighted k-means over every frame's swatches (weight
    = that swatch's coverage / frame count), so it needs no second pixel pass and
    stays deterministic. Outliers are frames whose feature z-score reaches
    `outlier_z` on any measured feature -- distinct scenes/districts cluster apart
    and show up here.
    """
    ok_frames = [f for f in frames if f.ok]
    failed = [f for f in frames if not f.ok]
    failed_indices = tuple(f.index for f in failed)

    if not ok_frames:
        return VideoVisual(
            analyzed_count=0,
            failed_count=len(failed),
            failed_indices=failed_indices,
        )

    swatch_rows: list[list[float]] = []
    swatch_weights: list[float] = []
    for frame in ok_frames:
        for swatch in frame.palette:
            swatch_rows.append([float(c) for c in swatch.rgb])
            swatch_weights.append(swatch.coverage / len(ok_frames))

    palette = (
        kmeans_palette(
            np.array(swatch_rows, dtype=np.float64),
            k=palette_size,
            seed=seed,
            weights=np.array(swatch_weights, dtype=np.float64),
        )
        if swatch_rows
        else []
    )
    substrate = palette[0] if palette else None
    substrate_verdict = (
        classify_substrate(float(luminance(np.array([substrate.rgb], dtype=np.float64))[0]))
        if substrate
        else None
    )

    def _mean(attr: str) -> float:
        return float(np.mean([getattr(f, attr) for f in ok_frames]))

    outliers: list[FrameOutlier] = []
    if len(ok_frames) >= 3:
        best: dict[int, tuple[float, str]] = {}
        for attr, label in _OUTLIER_FEATURES:
            values = np.array([getattr(f, attr) for f in ok_frames], dtype=np.float64)
            std = float(values.std())
            if std <= 1e-12:
                continue
            zs = (values - float(values.mean())) / std
            for frame, z in zip(ok_frames, zs):
                if abs(float(z)) >= outlier_z:
                    current = best.get(frame.index)
                    if current is None or abs(float(z)) > abs(current[0]):
                        direction = "high" if z > 0 else "low"
                        best[frame.index] = (
                            float(z),
                            f"{direction} {label} (z={float(z):+.2f})",
                        )
        for frame in ok_frames:
            if frame.index in best:
                z, reason = best[frame.index]
                outliers.append(
                    FrameOutlier(
                        index=frame.index, path=frame.path, z_score=z, reason=reason
                    )
                )

    return VideoVisual(
        analyzed_count=len(ok_frames),
        failed_count=len(failed),
        failed_indices=failed_indices,
        palette=tuple(palette),
        substrate=substrate,
        substrate_verdict=substrate_verdict,
        mean_luma=_mean("mean_luma"),
        mean_dark_fraction=_mean("dark_fraction"),
        mean_saturation=_mean("mean_saturation"),
        mean_vivid_fraction=_mean("vivid_fraction"),
        mean_bright_mark_density=_mean("bright_mark_density"),
        outliers=tuple(outliers),
    )
