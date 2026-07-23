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
 5. digest    -> assemble markdown: Summary / Visual notes per keyframe /
                 Ideas to steal / Action items / Full transcript (folded)
                 optional first-pass mechanical Summary from a local vLLM
                 server (text-only — vLLM never sees the keyframe images)
output: digests/video-digest-YYYY-MM-DD-<slug>.md
```

The digest-assembly step does **not** perform visual synthesis itself — it
assembles a template with a TODO placeholder per keyframe. Actually looking at
the frames and writing the visual notes / ideas / action items is a human or
Claude-in-the-loop step, done after the pipeline runs (Claude is the part of
this workflow that can actually see the images).

## Quickstart

```bash
pip install -r requirements.txt   # yt-dlp, faster-whisper, requests
# ffmpeg must already be on PATH (this repo was built against ffmpeg 8.1.1)

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
  --no-vllm-summary
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

## After a run

The digest markdown at `digests/video-digest-<date>-<slug>.md` references
`frames/NNN.jpg` inside `scratch/`. Open the digest, review the keyframes, and
fill in:

- **Summary** (if not already filled by the optional vLLM first pass)
- **Visual notes** — one line per keyframe range describing what's on screen
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

Every subprocess call (`yt-dlp`, `ffmpeg`, `ffprobe`) and every model/HTTP call
(`faster-whisper`'s `WhisperModel`, the local vLLM request) is dependency-
injected behind a `runner=subprocess.run` / `model_factory=` / `http_post=`
default argument. The test suite mocks all of these — it never invokes a real
binary, loads a real Whisper model, or makes a real network request.

## Project layout

```
video_digest/
  acquire.py       # yt-dlp download or local-path passthrough
  keyframes.py     # scene-change + floor-interval timestamp math, ffmpeg frame extraction
  audio.py         # ffmpeg mono 16kHz wav extraction
  transcript.py    # faster-whisper transcription + timestamp formatting
  vllm_summary.py  # optional local-vLLM text-only first-pass summary
  digest.py        # pure markdown template assembly (no visual synthesis)
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
