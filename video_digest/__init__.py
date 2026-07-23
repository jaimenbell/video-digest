"""video-digest: turn a video (URL or local file) into keyframes + transcript + markdown digest.

Fully local, $0 pipeline: yt-dlp (acquire) -> ffmpeg (keyframes + audio) ->
faster-whisper (transcript) -> optional local vLLM first-pass summary -> markdown digest.
"""

__version__ = "0.1.0"
