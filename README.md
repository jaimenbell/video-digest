# video-digest

Turn a video (a URL, or a local mp4 path) into keyframes + a transcript + a
structured markdown digest — because Claude can read images and text, but not
video or audio directly. Drop a video in, get working material out: reference
designs, competitor teardowns, idea capture.

**Fully local, $0.** No cloud API calls. The only network activity is
`yt-dlp` fetching a URL you gave it — the video is never uploaded anywhere.

## Pipeline

```
input: a video URL (yt-dlp) OR a local mp4 path
 1. acquire   -> yt-dlp download to gitignored scratch/ (skipped if a local path is given)
 2. keyframes -> ffmpeg scene-change extraction (select='gt(scene,0.3)')
                 + a floor of 1 frame / N seconds, capped at ~20-40 frames total
                 -> scratch/frames/NNN.jpg
 3. audio     -> ffmpeg extract mono 16kHz wav -> scratch/audio.wav
 4. transcript-> faster-whisper (local, $0) -> scratch/transcript.txt (timestamped)
 5. visual    -> deterministic, model-free colour analysis of every keyframe:
                 k-means palette (hex + coverage), luminance histogram /
                 substrate darkness, saturation distribution, bright-mark
                 (emissive) density, plus a whole-video roll-up + outlier frames
 6. digest    -> assemble markdown: Summary / Visual notes per keyframe /
                 Visual roll-up / Ideas to steal / Action items /
                 Full transcript (folded)
                 optional first-pass mechanical Summary from a local vLLM
                 server (text-only — vLLM never sees the keyframe images)
output: digests/video-digest-YYYY-MM-DD-<slug>.md
```

### What the visual stage does (and deliberately does not) do

Stage 5 **measures** the frames; it never describes them. For an art-direction
reference that is the useful half: `#232c28` at 37% coverage is a fact you can
paste into a palette, reproducible bit-for-bit, produced offline at $0 — whereas
prose like "dark, moody, neon accents" is unverifiable and, from a model that
cannot see, would be fabricated. Concretely, per frame:

| Measure | What it answers |
|---|---|
| k-means palette (top-N hex + coverage %) | what colours are actually on screen, and how much of the frame each one owns |
| substrate hex + luma + % below luma 32 | "is the background truly black, or dark gray?" — as a number, not a vibe |
| mean/median luma | overall exposure of the frame |
| mean saturation + vivid fraction (S>=0.50) | how "vibrant" the frame really is |
| bright-mark density (% of pixels >= luma 200) | emissive density — how many small glowing marks per unit area |

The roll-up clusters every frame's swatches into one whole-video palette and
flags outlier frames by z-score, so distinct scenes/districts surface on their
own.

Everything is deterministic: k-means is seeded, pixel subsampling is a fixed
stride (never a random draw), and swatches are totally ordered. The same frames
always produce the same hex values — `tests/test_visual.py` pins this.

A missing or corrupt frame is **reported** in the digest (`FRAME UNREADABLE`,
with the error, keeping its TODO line). It is never silently skipped and its
numbers are never invented.

`video_digest/visual.py` also exposes an **optional-VLM seam**
(`analyze_frames(..., describe_fn=...)`) for a future vision model, but nothing
in this repo wires one up: the local vLLM this pipeline talks to (`qwen3-14b`)
is text-only and cannot describe an image, and adding a VLM would be an
architecture decision, not a default. Writing the *interpretation* — visual
notes prose, ideas to steal, action items — is still a human or
Claude-in-the-loop step after the pipeline runs (Claude is the part of this
workflow that can actually see the images).

## Requirements

- Python 3.10+
- `ffmpeg` + `ffprobe` on `PATH` (this repo was built/tested against ffmpeg 8.1.1)
  - Windows: `winget install Gyan.FFmpeg` (or grab a build from gyan.dev) and add
    its `bin/` to `PATH`
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg` / your distro's package manager
- The Python packages in `requirements.txt` (`yt-dlp`, `faster-whisper`,
  `requests`, `pillow`, `numpy` — the last two are used only by the local,
  model-free keyframe colour analysis)

## Quickstart

```bash
pip install -r requirements.txt   # yt-dlp, faster-whisper, requests, pillow, numpy

# From a URL:
python -m video_digest "https://www.tiktok.com/@user/video/123"

# From a local file:
python -m video_digest "C:/Users/you/Downloads/clip.mp4"

# Options:
python -m video_digest <url-or-path> \
  --output digests/ \
  --scratch scratch/ \
  --model base \
  --vault-inbox "C:/path/to/your/obsidian/vault/inbox" \
  --no-vllm-summary \
  --no-visual
```

- `--output` — where the digest markdown lands (default `digests/`, gitignored).
- `--model` — faster-whisper model size: `tiny`/`base`/`small`/`medium`/`large-v3`
  (default `base` — good balance of speed and accuracy for short clips).
- `--vault-inbox` — optional: also copy the finished digest into this directory
  (e.g. an Obsidian vault inbox). Wiring a *default* vault path is deliberately
  left out of this repo since the vault location is operator-specific; pass it
  explicitly, or copy the file yourself after a run.
- `--no-vllm-summary` — skip the local vLLM first-pass summary and leave a
  `TODO` placeholder in the Summary section instead. By default, if a local
  vLLM server is reachable at `http://localhost:8000`, its `qwen3-14b` model is
  asked for a mechanical, text-only first-pass summary of the transcript
  (vLLM never sees the keyframe images — that part is Claude/human-only). If
  the server isn't reachable, the pipeline falls back to the placeholder
  automatically — this is never a hard dependency.
- `--no-visual` — skip the deterministic keyframe colour analysis and leave the
  per-frame `TODO` placeholders instead. The analysis runs by default: it is
  local, model-free, and costs nothing but a second or two of CPU.

## After a run

The digest markdown at `digests/video-digest-<date>-<slug>.md` references
`frames/NNN.jpg` inside `scratch/`. Open the digest, review the keyframes, and
fill in:

- **Summary** (if not already filled by the optional vLLM first pass)
- **Visual notes** — the measured colour data is already filled in per keyframe
  (palette, substrate, saturation, bright-mark density); add what's *on screen*
- **Ideas to steal**
- **Action items**

The full timestamped transcript is folded into a `<details>` block at the
bottom for reference.

## Development

```bash
pip install -r requirements.txt
pip install pytest
pytest
```

126 tests, 126 passing (re-verified fresh). Every subprocess call
(`yt-dlp`, `ffmpeg`, `ffprobe`) and every model/HTTP call (`faster-whisper`'s
`WhisperModel`, the local vLLM request) is dependency-injected behind a
`runner=subprocess.run` / `model_factory=` / `http_post=` default argument.
The test suite mocks all of these — it never invokes a real binary, loads a
real Whisper model, or makes a real network request. The keyframe colour tests
synthesise their own lossless PNGs in `tmp_path` (exact pixel values, no JPEG
rounding), so they need no sample video and pin the k-means output to literal
hex values.

## Project layout

```
video_digest/
  acquire.py       # yt-dlp download or local-path passthrough
  keyframes.py     # scene-change + floor-interval timestamp math, ffmpeg frame extraction
  audio.py         # ffmpeg mono 16kHz wav extraction
  transcript.py    # faster-whisper transcription + timestamp formatting
  vllm_summary.py  # optional local-vLLM text-only first-pass summary
  visual.py        # deterministic, model-free keyframe colour analysis (palette/luma/sat/density)
  digest.py        # pure markdown template assembly (renders measurements, never invents prose)
  cli.py           # argparse entry point + pipeline orchestration
tests/             # unit tests, one file per module, subprocess/model/HTTP all mocked
scratch/           # gitignored working directory (video, frames, audio, transcript)
digests/           # gitignored default output directory
```

## Why this exists

Claude can read text and images, but can't natively watch a video or listen to
audio. This tool does the mechanical extraction (frames + transcript) so that
a video becomes something Claude — or a human — can actually work with: pull
design references out of a UI walkthrough, pull ideas out of a talk, teardown
a competitor's demo. Standalone repo, no cloud dependency, no video leaves
your machine except to be fetched from wherever you pointed `yt-dlp` at.

## Limitations

- Developed and tested on Windows; `ffmpeg`/`ffprobe` invocation uses
  subprocess args that should be POSIX-compatible but have not been verified
  on macOS/Linux.
- The optional vLLM summary step requires you to already have a local vLLM
  (or OpenAI-compatible) server running at the configured endpoint — this
  repo doesn't set one up for you, and the pipeline works fine without it
  (falls back to a `TODO` placeholder).
- `faster-whisper` model download/inference speed and accuracy depend on the
  `--model` size you pick (`tiny`..`large-v3`) and your CPU/GPU.
- Scene-change keyframe detection is a heuristic (ffmpeg's `scene` filter +
  a floor interval); it won't perfectly match every video's actual cut
  points, especially on long or slow-paced source material.
- The visual stage measures colour; it does not recognise objects, read on-screen
  text, or describe composition. That is deliberate (see above) — no vision model
  is wired up, only a seam for one.
- Palette k-means samples up to 20k pixels per frame and statistics up to 1M
  pixels (fixed stride, deterministic). Very fine detail below that sampling
  density can be missed.
- No packaging/publish to PyPI yet — install from source.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Jaime Bell.
