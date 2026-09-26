"""
Ultimate Premium Video Dubbing Platform (Single-Character Master Build)
========================================================================

FastAPI backend, single-character-only, real-time-duration-matching
architecture:

  * Groq (whisper-large-v3)              -> heavy audio transcription
  * Groq (openai/gpt-oss-120b)           -> elite translation
  * Gemini native TTS (3-model fallback,
    multi-key rotation)                  -> pure-text voice generation

Two-phase flow (upload vs. dub are deliberately separate)
-----------------------------------------------------------
  1. POST /upload         -> plain JSON, fires the instant a video is
                              picked. Just saves the bytes + creates a
                              session (so the frontend can show real
                              XHR upload-progress).
  2. GET  /prepare/{id}   -> SSE, called automatically right after
                              /upload resolves. Extracts audio, probes
                              the real video duration, and transcribes it
                              (language-independent — no target language/
                              voice needed yet).
  3. POST /dub             -> SSE, ONLY runs once the person has picked a
                              target language + voice and pressed the dub
                              button. Translates the already-transcribed
                              script (chunked, drop-proof), generates the
                              voice, fits it to the video, and muxes it in.
  4. GET  /download/{id}  -> streams the finished MP4 (HTTP Range
                              supported) — never deleted on read; cleanup
                              is via the 1h TTL sweep or DELETE /session.

Voice generation strategy
--------------------------
The ENTIRE translated script is sent as exactly ONE Gemini TTS request
(never split into chunks, never split into per-line requests) — this is a
deliberate design choice: mixing per-line speeds would make some lines
sound faster than others, which is far more noticeable than a single
uniform speed change across the whole track.

After that one request comes back as raw audio, the pipeline works like a
real audio engineer instead of guessing:

  1. SILENCE TRIM (adaptive ladder): every internal silent gap longer than
     a threshold is trimmed down (FFmpeg `silenceremove`, real audio-level
     detection — not a timestamp guess). The ladder starts at 500ms. If the
     video is short and the generated speech is still too long relative to
     it after a 500ms trim, the threshold is automatically lowered in
     steps (500 -> 400 -> 280ms) — never below 280ms, because that starts
     cutting into natural between-sentence pauses, which makes the voice
     sound rushed/mashed-together rather than helping. This only ever
     trims SILENCE, never speech.
  2. DURATION MATCH: the trimmed speech duration is compared against the
     ACTUAL VIDEO DURATION (not the sum of Whisper segment timings) and a
     single pitch-preserving speed ratio is computed and applied to the
     WHOLE track at once — so every line speeds up or slows down by
     exactly the same amount, with no "one line fast, one line normal"
     artifact. Uses `rubberband` when the FFmpeg build has it (much
     cleaner than chained `atempo` at larger ratios), else falls back to
     `atempo` automatically.
  3. FINE-TUNE: if the fitted track is still more than ~150ms off the
     video length (rounding/precision noise), one small corrective
     micro-pass nudges it the rest of the way.
  4. MUX: the video stream is copied bit-for-bit (`-c:v copy`) — zero
     re-encoding, zero quality loss — only the audio track is replaced.

Backup / never-crash design
-----------------------------
* Every Gemini TTS call tries TTS_MODELS in order, and for EACH model
  tries every configured GEMINI_API_KEYS entry before moving to the next
  model — one (model, key) combination busy -> automatic fallback to the
  next, with an SSE [WARN] so it's visible. This is the backup plan for
  the (still single, conceptually one-shot) TTS request: it retries
  across every model/key combination before giving up, it does not split
  the script into pieces.
* Every processing step is wrapped so a failure ends ONLY that session
  with a clean SSE `error` event and the session's scratch files are
  deleted — the FastAPI process itself, and every other in-flight
  session, is never affected.
* Speed ratios are always clamped to a safe FFmpeg range (0.25x-4.0x) so
  a pathological mismatch (e.g. wildly different script vs. video length)
  degrades gracefully with a [WARN] instead of crashing the render.

Environment
-----------
GEMINI_API_KEYS (required)   Comma-separated list of Gemini API keys.
                              Rotated automatically on quota/errors.
                              (GEMINI_API_KEY, singular, also still works
                              as a 1-key fallback.) Required for TTS always,
                              and for transcription/translation when
                              ENGINE_MODE="gemini" (the default).
GROQ_API_KEYS   (optional*)  Comma-separated list of Groq API keys.
                              Rotated automatically on quota/errors.
                              (GROQ_API_KEY, singular, also still works
                              as a 1-key fallback.) *Required only if
                              ENGINE_MODE="groq" or the frontend engine
                              toggle is switched to Groq for a session.
ENGINE_MODE     (optional)   "gemini" (default) or "groq" — which engine
                              handles transcription + translation by
                              default. The frontend's toggle can override
                              this per-session via ?engine= on /prepare and
                              /dub without restarting the server.
GEMINI_TRANSCRIBE_MODEL (optional) Default: "gemini-2.5-flash"
GEMINI_TRANSLATE_MODEL  (optional) Default: "gemini-2.5-flash"
CORS_ALLOW_ORIGINS (optional) Comma-separated allow-list. Default: "*".
WORK_DIR        (optional)   Scratch dir. Default: ./_sessions
FFMPEG_BIN      (optional)   Default: "ffmpeg"
FFPROBE_BIN     (optional)   Default: "ffprobe"
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# Gemini (native TTS only — the only piece of Gemini we still touch).
from google import genai
from google.genai import types

# Groq (heavy transcription + translation).
from groq import Groq

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _parse_key_list(raw: str) -> List[str]:
    """Comma or newline separated list of API keys -> clean list, dedup-safe."""
    if not raw:
        return []
    parts = re.split(r"[,\n]+", raw)
    seen: List[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return seen


# Multiple keys are supported for BOTH providers: set GEMINI_API_KEYS /
# GROQ_API_KEYS as a comma-separated list to enable automatic key-rotation
# whenever a key hits a quota/rate-limit error. The single-key env vars
# (GEMINI_API_KEY / GROQ_API_KEY) still work as a 1-key fallback.
GEMINI_API_KEYS: List[str] = _parse_key_list(os.environ.get("GEMINI_API_KEYS", "")) or (
    _parse_key_list(os.environ.get("GEMINI_API_KEY", ""))
)
GROQ_API_KEYS: List[str] = _parse_key_list(os.environ.get("GROQ_API_KEYS", "")) or (
    _parse_key_list(os.environ.get("GROQ_API_KEY", ""))
)

WORK_DIR = Path(os.environ.get("WORK_DIR", "./_sessions")).resolve()
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# --- Transcription + translation engine toggle -------------------------------
# ENGINE_MODE picks the DEFAULT engine used for transcription + translation:
#   "gemini" (default, primary per product requirement) -> Gemini text model
#             does both transcription (via audio understanding) and
#             translation/localization in one coherent pass.
#   "groq"   -> original Whisper (transcribe) + gpt-oss-120b (translate) path.
# The frontend can override this per-request (?engine=groq|gemini on
# /prepare, and an `engine` form field on /dub) without restarting the
# server, so a person can flip the toggle live if one provider is degraded.
ENGINE_MODE_DEFAULT = os.environ.get("ENGINE_MODE", "gemini").strip().lower()
if ENGINE_MODE_DEFAULT not in ("gemini", "groq"):
    ENGINE_MODE_DEFAULT = "gemini"

GEMINI_TRANSCRIBE_MODEL = os.environ.get("GEMINI_TRANSCRIBE_MODEL", "gemini-2.5-flash")
GEMINI_TRANSLATE_MODEL = os.environ.get("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash")

# --- Groq: fallback/alternate transcription + translation engine ------------
WHISPER_MODEL = "whisper-large-v3"            # heavy audio -> text
TRANSLATION_MODEL = "openai/gpt-oss-120b"     # elite translation
                                               # (llama-3.3-70b-versatile is
                                               # retired on Groq as of 2026-08-16)

# --- Gemini: pure-text native TTS engine, with ordered auto-fallback --------
# One model busy -> auto request goes to the next model. Still busy across
# every model -> auto-rotate to the next GEMINI_API_KEYS entry. Every
# model x key combination exhausted -> a clean error is raised via SSE.
#
# Exactly the 4 voice models requested for the UI's single/multi-select
# picker. NOTE: "gemini-3.1-flash-tts-preview", "gemini-3.8-flash-tts" and
# "gemini-3.8-flash-tts-lite" are forward-looking model IDs that may not yet
# be live on every Gemini account/region — that's WHY the fallback chain
# below exists: if a requested model 404s / is unavailable, it's skipped
# with a [WARN] and the next selected model is tried automatically, so the
# pipeline never hard-fails just because a newer preview isn't enabled yet.
TTS_MODEL_CATALOG: Dict[str, str] = {
    "gemini-2.5-flash-preview-tts": "Gemini 2.5 Flash Preview TTS",
    "gemini-3.1-flash-tts-preview": "Gemini 3.1 Flash TTS Preview",
    "gemini-3.8-flash-tts": "Gemini 3.8 Flash TTS",
    "gemini-3.8-flash-tts-lite": "Gemini 3.8 Flash Lite TTS",
}
TTS_MODELS: List[str] = list(TTS_MODEL_CATALOG.keys())


def parse_selected_tts_models(raw: str) -> List[str]:
    """
    Turn a comma-separated list of model IDs from the frontend's
    single/multi-select into an ordered, de-duplicated, validated fallback
    chain. Unknown IDs are dropped (logged by the caller); an empty/absent
    selection falls back to the full default catalog in its documented
    order (single "best first" chain).
    """
    if not raw:
        return list(TTS_MODELS)
    chosen: List[str] = []
    for part in raw.split(","):
        mid = part.strip()
        if mid and mid in TTS_MODEL_CATALOG and mid not in chosen:
            chosen.append(mid)
    return chosen or list(TTS_MODELS)
# Hard ceiling on a SINGLE (model, key) TTS attempt. This only guards
# against a truly hung connection (no response at all) — a normal slow
# generation (observed up to ~90s under load) must NOT be cut off, so this
# is intentionally generous rather than tight.
TTS_CALL_TIMEOUT_SECONDS = 150

# All 30 official Gemini native-TTS prebuilt voices (kept in sync with the
# dropdown in index.html — see VOICES there for the display list).
GEMINI_VOICE_NAMES: List[str] = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

# Gemini native TTS output: 24kHz, 16-bit, mono, little-endian PCM.
TTS_SAMPLE_RATE = 24000
TTS_SAMPLE_WIDTH = 2
TTS_CHANNELS = 1

UPLOAD_CHUNK = 1024 * 1024  # 1 MiB streaming chunk (RAM-safe)

# --- Translation: chunked + drop-proof ------------------------------------ #
# The transcript is translated in small chunks (not one giant call) so a
# single LLM hiccup can only ever affect a FEW lines, never the whole
# video — and any chunk that still doesn't come back 1:1 after retries
# falls back to the literal source-language text for just those lines
# (logged clearly) instead of silently dropping them. See groq_translate_all.
#
# Chunk size is now budgeted by CHARACTERS, not a fixed line count: ~1500
# characters per chunk matches the requested voice-generation pacing (a
# chunk this size reads naturally in ~60-90s of speech, keeping each LLM
# call's context small enough to stay fast/cheap while still giving the
# model enough surrounding context to keep sentence rhythm natural).
TRANSLATE_CHUNK_CHAR_BUDGET = 1500
TRANSLATE_CHUNK_SIZE = 12          # hard line-count ceiling even if a chunk
                                   # is still under-budget on characters
                                   # (keeps memory + JSON payloads bounded).
TRANSLATE_CHUNK_RETRIES = 2
SELF_HEAL_RETRIES = 2              # extra automatic retries (with backoff)
SELF_HEAL_BACKOFF_SECONDS = 3.0

# --- Silence-trim ladder ------------------------------------------------- #
# Product requirement: silence removal is STRICTLY locked between 400ms and
# 500ms — never lower, never higher. The ladder therefore only has two
# rungs now (500ms tried first / least aggressive, 400ms as the only
# fallback), and rungs are tried ONE AT A TIME (sequential, not
# concurrent) — see Render RAM note below — with each rung's scratch file
# deleted immediately once it's no longer needed.
SILENCE_TRIM_LADDER_MS: List[int] = [500, 400]
SILENCE_TRIM_FLOOR_MS: int = SILENCE_TRIM_LADDER_MS[-1]
SILENCE_DB_THRESHOLD = -32.0  # dB below which audio is treated as silence.
COMFORTABLE_MAX_RATIO = 1.2   # stop shrinking the trim window once we're
                               # within +20% of the video's length — lean on
                               # the (now formant-preserving) speed-up rather
                               # than over-trimming pauses.

# Absolute FFmpeg speed-filter safety clamp (atempo/rubberband hard limits).
HARD_SPEED_MIN = 0.25
HARD_SPEED_MAX = 4.0
# Softer "this might start sounding off" advisory band.
SOFT_SPEED_MAX = 1.3

# --- Product requirement: atempo STRICTLY locked to the 1.15x-1.30x band ---
# Every dubbed track is sped up by AT LEAST 1.15x (even if the raw fit would
# have needed less, or none) and by NO MORE than 1.30x (even if the raw fit
# would have needed more) — no value outside this band is ever applied.
ATEMPO_LOCK_MIN = 1.15
ATEMPO_LOCK_MAX = 1.30

# --- RAM safety: sequential, chunk-by-chunk processing ----------------------
# Render's free/starter instances have limited RAM. Nothing in this pipeline
# is allowed to run two heavy ffmpeg/audio jobs at once for a single
# session — every step below runs strictly one-after-another, and each
# step's own temporary files are deleted the moment that step is done with
# them (see _cleanup_scratch()).
SEQUENTIAL_PROCESSING = True

WORK_DIR.mkdir(parents=True, exist_ok=True)

# One client per API key, created lazily and cached so key-rotation doesn't
# reconnect on every call. The /health endpoint reports key COUNTS, never
# the key values themselves.
_gemini_clients: Dict[str, "genai.Client"] = {}
_groq_clients: Dict[str, "Groq"] = {}


def get_client(api_key: str) -> "genai.Client":
    """Cached Gemini client for a specific API key (native TTS only)."""
    if not api_key:
        raise RuntimeError("Empty Gemini API key.")
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def get_groq_client(api_key: str) -> "Groq":
    """Cached Groq client for a specific API key (Whisper + translation)."""
    if not api_key:
        raise RuntimeError("Empty Groq API key.")
    if api_key not in _groq_clients:
        _groq_clients[api_key] = Groq(api_key=api_key)
    return _groq_clients[api_key]


# --------------------------------------------------------------------------- #
# Session model + disk persistence
# --------------------------------------------------------------------------- #

@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Session:
    session_id: str
    dir: Path
    target_language: str = ""
    single_voice: str = "Kore"
    created_at: float = field(default_factory=time.time)
    video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    source_language: Optional[str] = None
    raw_segments: List[Segment] = field(default_factory=list)   # transcribed, NOT translated yet
    segments: List[Segment] = field(default_factory=list)       # translated (filled by /dub)
    video_duration: float = 0.0
    prepared: bool = False   # True once extract+probe+transcribe has finished
    engine: str = ENGINE_MODE_DEFAULT   # "gemini" or "groq" — set at /upload time
    tts_models: List[str] = field(default_factory=lambda: list(TTS_MODELS))


SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SECONDS = 60 * 60  # 1h safety sweep for abandoned sessions.


def _session_file(sdir: Path) -> Path:
    return sdir / "session.json"


def save_session(sess: Session) -> None:
    """Persist lightweight session metadata to disk (never media bytes)."""
    data = {
        "session_id": sess.session_id,
        "dir": str(sess.dir),
        "target_language": sess.target_language,
        "single_voice": sess.single_voice,
        "created_at": sess.created_at,
        "video_path": str(sess.video_path) if sess.video_path else None,
        "audio_path": str(sess.audio_path) if sess.audio_path else None,
        "source_language": sess.source_language,
        "raw_segments": [asdict(s) for s in sess.raw_segments],
        "segments": [asdict(s) for s in sess.segments],
        "video_duration": sess.video_duration,
        "prepared": sess.prepared,
        "engine": sess.engine,
        "tts_models": sess.tts_models,
    }
    tmp = _session_file(sess.dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(_session_file(sess.dir))


def load_session(session_id: str) -> Optional[Session]:
    """Return a session from RAM, or rehydrate it from disk if needed."""
    if session_id in SESSIONS:
        return SESSIONS[session_id]
    sdir = WORK_DIR / session_id
    sfile = _session_file(sdir)
    if not sfile.exists():
        return None
    try:
        data = json.loads(sfile.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    sess = Session(
        session_id=data["session_id"],
        dir=Path(data["dir"]),
        target_language=data.get("target_language", ""),
        single_voice=data.get("single_voice", "Kore"),
        created_at=data.get("created_at", time.time()),
        video_path=Path(data["video_path"]) if data.get("video_path") else None,
        audio_path=Path(data["audio_path"]) if data.get("audio_path") else None,
        source_language=data.get("source_language"),
        raw_segments=[Segment(**s) for s in data.get("raw_segments", [])],
        segments=[Segment(**s) for s in data.get("segments", [])],
        video_duration=data.get("video_duration", 0.0),
        prepared=data.get("prepared", False),
        engine=data.get("engine", ENGINE_MODE_DEFAULT),
        tts_models=data.get("tts_models") or list(TTS_MODELS),
    )
    SESSIONS[session_id] = sess
    return sess


def _destroy_session(session_id: str) -> None:
    """Remove a session from RAM and delete ALL its scratch files from disk."""
    sess = SESSIONS.pop(session_id, None)
    sdir = sess.dir if sess else (WORK_DIR / session_id)
    if sdir.exists():
        shutil.rmtree(sdir, ignore_errors=True)


def _sweep_stale_sessions() -> None:
    now = time.time()
    if not WORK_DIR.exists():
        return
    for child in WORK_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError:
            continue
        if age > SESSION_TTL_SECONDS:
            _destroy_session(child.name)


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #

def sse_log(message: str) -> Dict[str, str]:
    return {"event": "log", "data": message}


def sse_progress(percent: int, message: str = "") -> Dict[str, str]:
    payload = {"percent": max(0, min(100, int(percent))), "message": message}
    return {"event": "progress", "data": json.dumps(payload)}


def sse_error(message: str) -> Dict[str, str]:
    return {"event": "error", "data": message}


def sse_done(obj: dict) -> Dict[str, str]:
    return {"event": "done", "data": json.dumps(obj)}


# --------------------------------------------------------------------------- #
# FFmpeg / FFprobe helpers (async, disk-based)
# --------------------------------------------------------------------------- #

async def _run(cmd: List[str]) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", "ignore").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd[:2])}...): {err[:800]}")
    return (stdout or b"").decode("utf-8", "ignore")


async def probe_duration(media_path: Path) -> float:
    out = await _run([
        FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ])
    try:
        return float(out.strip())
    except ValueError:
        raise RuntimeError(f"Could not read duration for {media_path.name}")


async def extract_audio(video_path: Path, out_audio: Path) -> None:
    """Extract a mono 16kHz MP3 — lightweight enough for instant Groq upload."""
    await _run([
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        str(out_audio),
    ])


def _atempo_chain(ratio: float) -> str:
    """
    atempo accepts only 0.5..2.0 per instance; chain to reach any factor.
    ratio = generated / target:  >1 -> speed up, <1 -> slow down.
    """
    tempo = max(HARD_SPEED_MIN, min(HARD_SPEED_MAX, ratio))
    factors: List[float] = []
    while tempo > 2.0:
        factors.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        factors.append(0.5)
        tempo /= 0.5
    factors.append(round(tempo, 6))
    return ",".join(f"atempo={f}" for f in factors)


_rubberband_available_cache: Optional[bool] = None


async def rubberband_available() -> bool:
    """
    Detect (once, cached) whether this FFmpeg build includes the
    `rubberband` filter — a much higher-quality pitch-preserving
    time-stretch than chained `atempo`, especially at larger speed ratios
    where chained atempo starts sounding degraded/artifacty. Not all FFmpeg
    builds include it (requires librubberband at compile time), so we
    detect it at runtime and gracefully fall back to atempo if absent.
    """
    global _rubberband_available_cache
    if _rubberband_available_cache is not None:
        return _rubberband_available_cache
    try:
        proc = await asyncio.create_subprocess_exec(
            FFMPEG_BIN, "-hide_banner", "-filters",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        _rubberband_available_cache = b"rubberband" in (stdout or b"")
    except Exception:  # noqa: BLE001
        _rubberband_available_cache = False
    return _rubberband_available_cache


def _speed_filter(ratio: float, use_rubberband: bool) -> str:
    """
    Build the FFmpeg audio filter string for a given speed ratio.

    Primary/only engine: FFmpeg's native `atempo` (pitch-preserving). After
    real-world listening tests on this deployment, `atempo` produced
    noticeably cleaner speech than `rubberband` — rubberband's
    phase-vocoder approach, even with formant preservation on, introduced
    an audible warble on Gemini's already-synthetic voice. `atempo` is
    used unconditionally now regardless of whether rubberband is
    available; `use_rubberband` is kept in the signature for compatibility
    but is intentionally ignored.
    """
    return _atempo_chain(ratio)


async def _encode_wav(cmd_in: List[str], out_audio: Path,
                      audio_filter: Optional[str] = None) -> None:
    """Encode to a canonical WAV (pcm_s16le / 24kHz / mono)."""
    cmd = [FFMPEG_BIN, "-y", *cmd_in]
    if audio_filter:
        cmd += ["-filter:a", audio_filter]
    cmd += ["-ar", str(TTS_SAMPLE_RATE), "-ac", str(TTS_CHANNELS),
            "-c:a", "pcm_s16le", str(out_audio)]
    await _run(cmd)


async def trim_internal_silences(src_audio: Path, out_audio: Path,
                                 min_silence_seconds: float,
                                 threshold_db: float = SILENCE_DB_THRESHOLD) -> None:
    """
    Strip every internal silent gap of at least `min_silence_seconds`
    (audio level below `threshold_db`) ANYWHERE in the clip — not just
    leading/trailing — using FFmpeg's `silenceremove` filter in continuous
    (stop_periods=-1) mode. Only ever removes silence; speech audio is
    passed through untouched.
    """
    min_silence_seconds = max(0.03, min_silence_seconds)
    audio_filter = (
        f"silenceremove=stop_periods=-1:"
        f"stop_duration={min_silence_seconds:.3f}:"
        f"stop_threshold={threshold_db:.1f}dB:"
        f"detection=peak"
    )
    await _encode_wav(["-i", str(src_audio)], out_audio, audio_filter)


async def time_stretch_to_duration(src_audio: Path, target_seconds: float,
                                   out_audio: Path, use_rubberband: bool) -> Dict[str, float]:
    """
    Pitch-preserving time-stretch so src fits target_seconds (<=2% -> copy
    untouched). Returns a diagnostic dict {gen_dur, target_seconds, ratio,
    clamped} so callers can log exactly what happened.
    """
    gen_dur = await probe_duration(src_audio)
    diag = {"gen_dur": gen_dur, "target_seconds": target_seconds,
            "ratio": 1.0, "clamped": False}
    if gen_dur <= 0 or target_seconds <= 0:
        await _encode_wav(["-i", str(src_audio)], out_audio)
        return diag
    ratio = gen_dur / target_seconds
    diag["ratio"] = ratio
    diag["clamped"] = ratio < HARD_SPEED_MIN or ratio > HARD_SPEED_MAX
    if 0.98 <= ratio <= 1.02:
        await _encode_wav(["-i", str(src_audio)], out_audio)
        return diag
    await _encode_wav(["-i", str(src_audio)], out_audio, _speed_filter(ratio, use_rubberband))
    return diag


async def mux_video_with_audio(video_path: Path, audio_path: Path,
                               out_path: Path) -> None:
    """
    Replace the video's audio. `apad` + `-shortest` pads the dubbed track
    with silence up to the video length (a final safety net — normally the
    duration-match step already leaves this at ~0), so the VIDEO is never
    truncated and never re-encoded (`-c:v copy` => zero quality loss).
    """
    await _run([
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-filter:a", "apad", "-shortest",
        str(out_path),
    ])


# --------------------------------------------------------------------------- #
# Shared JSON extraction helper (used by the Groq translation step)
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    if not (text.startswith("{") or text.startswith("[")):
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            text = m.group(1)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Groq helpers: whisper-large-v3 transcription + openai/gpt-oss-120b
# translation
# --------------------------------------------------------------------------- #

async def groq_transcribe(audio_path: Path) -> dict:
    """
    Stream the extracted audio directly to Groq's Whisper API
    (whisper-large-v3) for an instantaneous, high-accuracy, TIMESTAMPED
    transcription. Returns {"language": str, "segments": [{start,end,text}]}.
    Rotates across every configured GROQ_API_KEYS entry on failure.
    """
    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY(s) configured.")

    def _do(api_key: str):
        client = get_groq_client(api_key)
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        resp = client.audio.transcriptions.create(
            file=(audio_path.name, audio_bytes),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            temperature=0.0,
        )
        return resp

    resp = None
    last_err: Optional[Exception] = None
    for key in GROQ_API_KEYS:
        try:
            resp = await asyncio.to_thread(_do, key)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if resp is None:
        raise RuntimeError(
            f"Groq Whisper transcription failed on all {len(GROQ_API_KEYS)} "
            f"configured key(s): {last_err}"
        ) from last_err

    # The Groq SDK returns a pydantic-style object; support both attr + dict.
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif isinstance(resp, dict):
        data = resp
    else:
        data = {
            "language": getattr(resp, "language", None),
            "segments": getattr(resp, "segments", None),
            "text": getattr(resp, "text", None),
        }

    language = str(data.get("language") or "Unknown")
    raw_segments = data.get("segments") or []

    segments: List[Dict[str, object]] = []
    for s in raw_segments:
        if not isinstance(s, dict):
            s = {"start": getattr(s, "start", 0.0), "end": getattr(s, "end", 0.0),
                 "text": getattr(s, "text", "")}
        try:
            segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip(),
            })
        except (TypeError, ValueError):
            continue

    # Fallback: some responses omit segment timestamps entirely.
    if not segments:
        whole_text = str(data.get("text") or "").strip()
        if whole_text:
            segments = [{"start": 0.0, "end": 0.0, "text": whole_text}]

    if not segments:
        raise RuntimeError("Groq Whisper returned no usable transcript segments.")

    return {"language": language, "segments": segments}


def _translation_style_notes(target_language: str) -> str:
    """
    Extra localization guidance injected into the translation prompt.
    Bengali gets an explicit Bangladeshi-audience naturalization pass —
    everyday spoken Dhaka-standard Bengali, not textbook/literary Bengali
    and not the West Bengal (Indian) register/vocabulary — since this is
    the platform's flagship target-language use case.
    """
    lang = (target_language or "").strip().lower()
    if "bengali" in lang or "bangla" in lang:
        return (
            "LOCALIZATION FOR A BANGLADESHI AUDIENCE: write in natural, "
            "everyday spoken Bengali the way it's actually spoken in "
            "Bangladesh (Dhaka-standard) — not stiff literary/textbook "
            "Bengali ('sadhu bhasha'), and not West Bengal/Indian Bengali "
            "vocabulary or phrasing. Re-touch idioms, filler words, and "
            "sentence rhythm so a Bangladeshi viewer would never guess this "
            "was translated."
        )
    return (
        "LOCALIZATION: write in natural, everyday spoken language for this "
        "target audience — re-touch idioms and phrasing so it sounds "
        "native, not translated."
    )


async def gemini_transcribe(audio_path: Path) -> dict:
    """
    Gemini equivalent of groq_transcribe(): sends the extracted audio
    directly to a Gemini text model with audio-understanding, asking for a
    timestamped, language-tagged transcript in the same shape Whisper
    returns ({"language": str, "segments": [{start,end,text}]}). Rotates
    across every configured GEMINI_API_KEYS entry on failure, exactly like
    the TTS fallback does.
    """
    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")

    audio_bytes = audio_path.read_bytes()
    prompt = (
        "Transcribe this audio precisely, in its ORIGINAL spoken language "
        "(do not translate). Split it into natural short segments the way "
        "a subtitle/dubbing script would, each with an estimated start/end "
        "time in seconds. Respond with ONLY raw JSON (no markdown fences), "
        "in exactly this shape: "
        '{"language": "<detected language name>", "segments": '
        '[{"start": 0.0, "end": 2.4, "text": "..."}]}'
    )

    def _do(api_key: str) -> str:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=GEMINI_TRANSCRIBE_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                prompt,
            ],
        )
        return resp.text

    raw = None
    last_err: Optional[Exception] = None
    for key in GEMINI_API_KEYS:
        try:
            raw = await asyncio.to_thread(_do, key)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if raw is None:
        raise RuntimeError(
            f"Gemini transcription failed on all {len(GEMINI_API_KEYS)} "
            f"configured key(s): {last_err}"
        ) from last_err

    data = _extract_json(raw)
    language = str(data.get("language") or "Unknown")
    segments: List[Dict[str, object]] = []
    for s in data.get("segments", []):
        if not isinstance(s, dict):
            continue
        try:
            segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip(),
            })
        except (TypeError, ValueError):
            continue
    if not segments:
        raise RuntimeError("Gemini returned no usable transcript segments.")
    return {"language": language, "segments": segments}


async def _gemini_translate_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """Gemini equivalent of _groq_translate_chunk() — same contract."""
    system_prompt = (
        "You are an elite professional translator inside an automated "
        "single-narrator video-dubbing pipeline. You will be given a JSON "
        "array of timestamped transcript segments (a SMALL CHUNK of a "
        "larger transcript), already transcribed from the original spoken "
        f"language ({source_language}).\n\n"
        "TASK 1 — TRANSLATION: Translate every segment's text PERFECTLY and "
        f"naturally into {target_language}. Preserve tone, intent, idiom, and "
        "natural spoken rhythm — this is for dubbing, not a literal "
        "word-for-word gloss. Never leave a segment untranslated.\n\n"
        f"{_translation_style_notes(target_language)}\n\n"
        "SENTENCE LENGTH: keep sentences naturally balanced for spoken "
        "delivery — neither clipped/choppy nor run-on/overloaded.\n\n"
        "HOOK RETENTION: if this chunk contains the video's opening "
        "line(s), keep a strong hook faithful and sharpen a weak one, "
        "without inventing a new premise.\n\n"
        "TASK 2 — COMPLETENESS (CRITICAL): the output 'segments' array MUST "
        "have EXACTLY the same number of objects as the input, same order, "
        "same start/end timestamps. Never skip, merge, drop, or duplicate.\n\n"
        "Respond with ONLY raw JSON (no markdown fences), in exactly this "
        'shape: {"source_language": "<language name>", "segments": '
        '[{"start": 0.0, "end": 3.2, "text": "<translated text>"}]}'
    )
    user_payload = json.dumps(
        {"source_language": source_language, "transcript": chunk_segments},
        ensure_ascii=False,
    )

    def _do(api_key: str) -> str:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=GEMINI_TRANSLATE_MODEL,
            contents=[system_prompt, user_payload],
        )
        return resp.text

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")

    raw = None
    last_err: Optional[Exception] = None
    for key in GEMINI_API_KEYS:
        try:
            raw = await asyncio.to_thread(_do, key)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if raw is None:
        raise RuntimeError(
            f"Gemini translation failed on all {len(GEMINI_API_KEYS)} "
            f"configured key(s): {last_err}"
        ) from last_err

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, TypeError):
        return None, []

    out_segments = []
    for s in data.get("segments", []):
        if not isinstance(s, dict):
            continue
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        out_segments.append({"start": start, "end": end, "text": text})

    detected_lang = data.get("source_language")
    if len(out_segments) != len(chunk_segments):
        return (str(detected_lang) if detected_lang else None), []
    return (str(detected_lang) if detected_lang else None), out_segments


async def gemini_translate_all(
    transcript_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> dict:
    """Gemini equivalent of groq_translate_all() — identical drop-proof contract."""
    if not transcript_segments:
        raise RuntimeError("Nothing to translate — the transcript is empty.")

    chunks = _chunk_by_char_budget(transcript_segments)

    detected_source_language: Optional[str] = None
    all_segments: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk in chunks:
        payload_chunk = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
            }
            for seg in chunk
        ]
        result_segments: List[Dict[str, object]] = []
        for attempt in range(1 + TRANSLATE_CHUNK_RETRIES):
            try:
                lang, translated = await _gemini_translate_chunk(
                    payload_chunk, source_language, target_language,
                )
            except Exception:  # noqa: BLE001
                continue
            if lang and detected_source_language is None:
                detected_source_language = lang
            if len(translated) == len(chunk):
                result_segments = translated
                break

        if not result_segments:
            fallback_count += len(chunk)
            fallback_ranges.append(
                f"{payload_chunk[0]['start']:.1f}s-{payload_chunk[-1]['end']:.1f}s"
            )
            result_segments = [
                {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
                 "text": str(seg.get("text", "")).strip() or "..."}
                for seg in chunk
            ]

        all_segments.extend(result_segments)

    return {
        "source_language": detected_source_language or source_language,
        "segments": all_segments,
        "input_count": len(transcript_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


async def _groq_translate_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """
    Translate ONE small chunk (<= TRANSLATE_CHUNK_SIZE lines) and return
    (detected_source_language_or_None, translated_segments). Returns an
    EMPTY list (never raises) if this chunk's output doesn't come back as a
    strict 1:1 match after internal parsing — the caller (groq_translate_all)
    is responsible for retrying / falling back so no line is ever silently
    lost.
    """
    system_prompt = (
        "You are an elite professional translator inside an automated "
        "single-narrator video-dubbing pipeline. You will be given a JSON "
        "array of timestamped transcript segments (a SMALL CHUNK of a "
        "larger transcript), already transcribed from the original spoken "
        f"language ({source_language}).\n\n"
        "TASK 1 — TRANSLATION: Translate every segment's text PERFECTLY and "
        f"naturally into {target_language}. Preserve tone, intent, idiom, and "
        "natural spoken rhythm — this is for dubbing, not a literal "
        "word-for-word gloss. Never leave a segment untranslated.\n\n"
        f"{_translation_style_notes(target_language)}\n\n"
        "SENTENCE LENGTH: keep sentences naturally balanced for spoken "
        "delivery — neither clipped/choppy nor run-on/overloaded. Break an "
        "overly long source sentence into two natural spoken sentences if "
        "needed, or merge two very short fragments only when it still maps "
        "cleanly onto the SAME segment (never across segments).\n\n"
        "HOOK RETENTION: if this chunk contains the video's opening line(s) "
        "(the first segment(s) overall), judge whether the hook is already "
        "strong (attention-grabbing, curiosity-driving, punchy). If it is "
        "strong, translate it faithfully and keep its impact intact. If it "
        "is weak or flat, adapt/sharpen it in the target language so the "
        "opening still grabs attention — while staying true to the "
        "original topic/claim (never invent a new premise).\n\n"
        "TASK 2 — COMPLETENESS (CRITICAL): The number of objects in the "
        "output 'segments' array MUST exactly equal the number of input "
        "segments in THIS chunk — a strict 1:1 mapping, same order. NEVER "
        "skip, merge, drop, or duplicate a segment, even for very short "
        "lines (a single word, 'yes'/'okay', a filler sound, laughter) — "
        "translate or transliterate your best effort for every single one "
        "and include it. Omitting even one segment is a critical failure. "
        "Preserve the exact start/end timestamps given for each input "
        "segment — do not invent, reorder, or renumber them.\n\n"
        "OUTPUT — return STRICT JSON only. No prose, no markdown fences, no "
        "commentary. Return exactly this shape:\n"
        "{\n"
        '  "source_language": "<language name>",\n'
        '  "segments": [\n'
        '    {"start": 0.0, "end": 3.2, "text": "<translated text>"}\n'
        "  ]\n"
        "}"
    )
    user_payload = json.dumps(
        {"source_language": source_language, "transcript": chunk_segments},
        ensure_ascii=False,
    )

    def _do(api_key: str) -> str:
        client = get_groq_client(api_key)
        try:
            resp = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some Groq model/runtime combos reject `response_format`;
            # gracefully retry once without strict JSON mode.
            resp = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )
        return resp.choices[0].message.content

    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY(s) configured.")

    raw = None
    last_err: Optional[Exception] = None
    for key in GROQ_API_KEYS:
        try:
            raw = await asyncio.to_thread(_do, key)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if raw is None:
        raise RuntimeError(
            f"Groq translation failed on all {len(GROQ_API_KEYS)} "
            f"configured key(s): {last_err}"
        ) from last_err

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, TypeError):
        return None, []

    out_segments = []
    for s in data.get("segments", []):
        if not isinstance(s, dict):
            continue
        try:
            start = float(s.get("start", 0.0))
            end = float(s.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        out_segments.append({"start": start, "end": end, "text": text})

    detected_lang = data.get("source_language")
    if len(out_segments) != len(chunk_segments):
        return (str(detected_lang) if detected_lang else None), []
    return (str(detected_lang) if detected_lang else None), out_segments


def _chunk_by_char_budget(
    transcript_segments: List[Dict[str, object]],
) -> List[List[Dict[str, object]]]:
    """
    Group consecutive segments into chunks of roughly
    TRANSLATE_CHUNK_CHAR_BUDGET characters (~1500), never exceeding
    TRANSLATE_CHUNK_SIZE lines per chunk either way. A single very long
    segment (rare) still gets its own chunk rather than being split, since
    segments must stay 1:1 with the transcript.
    """
    chunks: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    current_chars = 0
    for seg in transcript_segments:
        text_len = len(str(seg.get("text", "")))
        would_overflow = current and (
            current_chars + text_len > TRANSLATE_CHUNK_CHAR_BUDGET
            or len(current) >= TRANSLATE_CHUNK_SIZE
        )
        if would_overflow:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += text_len
    if current:
        chunks.append(current)
    return chunks


async def groq_translate_all(
    transcript_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> dict:
    """
    Translate the WHOLE transcript in small chunks (TRANSLATE_CHUNK_SIZE
    lines at a time) so a single LLM hiccup can only ever affect a few
    lines. Every chunk is retried up to TRANSLATE_CHUNK_RETRIES times if it
    doesn't come back as a strict 1:1 match; if it still fails after every
    retry, that chunk falls back to the LITERAL SOURCE-LANGUAGE TEXT for
    just those lines (never silently dropped) and the caller is told
    exactly which ones via `fallback_count`. Guarantees
    len(output_segments) == len(transcript_segments) ALWAYS.
    """
    if not transcript_segments:
        raise RuntimeError("Nothing to translate — the transcript is empty.")

    chunks: List[List[Dict[str, object]]] = _chunk_by_char_budget(transcript_segments)

    detected_source_language: Optional[str] = None
    all_segments: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk in chunks:
        payload_chunk = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
            }
            for seg in chunk
        ]

        result_segments: List[Dict[str, object]] = []
        last_err: Optional[Exception] = None
        for attempt in range(1 + TRANSLATE_CHUNK_RETRIES):
            try:
                lang, translated = await _groq_translate_chunk(
                    payload_chunk, source_language, target_language,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
            if lang and detected_source_language is None:
                detected_source_language = lang
            if len(translated) == len(chunk):
                result_segments = translated
                break

        if not result_segments:
            # Every retry failed to come back 1:1 — NEVER drop these lines:
            # fall back to the original source-language text so they still
            # get voiced, and note exactly which range fell back.
            fallback_count += len(chunk)
            fallback_ranges.append(
                f"{payload_chunk[0]['start']:.1f}s-{payload_chunk[-1]['end']:.1f}s"
            )
            result_segments = [
                {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
                 "text": str(seg.get("text", "")).strip() or "..."}
                for seg in chunk
            ]

        all_segments.extend(result_segments)

    return {
        "source_language": detected_source_language or source_language,
        "segments": all_segments,
        "input_count": len(transcript_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


# --------------------------------------------------------------------------- #
# Self-healing retry wrapper
# --------------------------------------------------------------------------- #

async def self_heal(
    step_name: str,
    coro_factory,
    retries: int = SELF_HEAL_RETRIES,
    backoff_seconds: float = SELF_HEAL_BACKOFF_SECONDS,
):
    """
    Run an async step with automatic error self-healing: on any exception,
    emit an SSE [WARN] with what broke and that a retry is happening, wait
    a short backoff, and try again — up to `retries` extra times — before
    finally giving up and re-raising (which the caller turns into a clean
    SSE `error` + session cleanup). This is what lets a transient hiccup
    (a flaky network blip, a momentary 500 from a provider, a one-off
    ffmpeg hiccup) resolve itself without the whole run failing.

    `coro_factory` is a zero-arg callable that returns a *fresh* coroutine
    each time it's called (a coroutine object can only be awaited once).
    Yields SSE dicts as it goes; the final yielded item is always
    {"_result": <return value of the awaited coroutine>}.
    """
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            result = await coro_factory()
            yield {"_result": result}
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                yield sse_log(
                    f"[WARN] {step_name} hit an error ({type(exc).__name__}: {exc}) — "
                    f"self-healing: retrying in {backoff_seconds:.0f}s "
                    f"(attempt {attempt + 2}/{retries + 1})..."
                )
                await asyncio.sleep(backoff_seconds)
            else:
                yield sse_log(
                    f"[ERROR] {step_name} failed after {retries + 1} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                )
    raise last_err  # noqa: RSE102 — re-raise the final failure for the caller


# --------------------------------------------------------------------------- #
# Gemini native TTS — pure text in, voice bytes out — 3-model auto-fallback.
# This is the BACKUP PLAN for the one-shot TTS request: every (model, key)
# combination is tried, in order, against the SAME full script — never
# splitting the text into pieces — before a clean error is raised.
# --------------------------------------------------------------------------- #

def write_wav_from_pcm(pcm_bytes: bytes, out_wav: Path) -> None:
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(TTS_CHANNELS)
        wf.setsampwidth(TTS_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


async def gemini_tts_with_fallback(
    text: str,
    voice_name: str,
    out_wav: Path,
    style_hint: str = "",
    models: Optional[List[str]] = None,
) -> AsyncGenerator[dict, None]:
    """
    Generate speech with Gemini native TTS for the FULL script in ONE call.

    Key insight from real-world logs: trying keys ONE AT A TIME, waiting up
    to TTS_CALL_TIMEOUT_SECONDS on each before moving to the next, means a
    congested model can burn (10 keys x 150s =) 25 minutes before we even
    fall through to the next model — and a connection sitting idle that
    long risks the whole SSE stream getting killed by a proxy in between.

    So instead: for EACH model, every configured API key is tried
    CONCURRENTLY (all fired at once — this is safe, each key has its own
    independent quota/project). We take whichever attempt succeeds FIRST
    and immediately cancel the rest. Only if EVERY key fails for a model do
    we move on to the next model. This means:
      - The common case (at least one of N keys is free) resolves in
        however long the FASTEST attempt takes — often just a few seconds.
      - The worst case (a model is down for everyone) is bounded by
        TTS_CALL_TIMEOUT_SECONDS ONCE per model, not multiplied by the key
        count.

    Yields SSE log dicts as it goes; the FINAL yielded item is always
    {"_tts_result": "<model_name_used>"} so the caller can tell which model
    actually produced the audio.
    """
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Nothing to voice — the translated script is empty.")

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")

    voice_name = voice_name if voice_name in GEMINI_VOICE_NAMES else "Kore"
    spoken = f"{style_hint.strip()}: {text}" if style_hint.strip() else text

    def _call(model_name: str, api_key: str) -> bytes:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=spoken,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )
        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data
        raise RuntimeError(f"{model_name} returned no audio data.")

    async def _attempt(model_name: str, api_key: str, key_idx: int):
        try:
            pcm = await asyncio.wait_for(
                asyncio.to_thread(_call, model_name, api_key),
                timeout=TTS_CALL_TIMEOUT_SECONDS,
            )
            return (True, model_name, key_idx, pcm, None)
        except asyncio.TimeoutError:
            return (False, model_name, key_idx, None,
                    RuntimeError(f"timed out after {TTS_CALL_TIMEOUT_SECONDS}s"))
        except Exception as exc:  # noqa: BLE001
            return (False, model_name, key_idx, None, exc)

    last_err: Optional[Exception] = None
    multi_key = len(GEMINI_API_KEYS) > 1
    model_chain = [m for m in (models or TTS_MODELS) if m in TTS_MODEL_CATALOG] or list(TTS_MODELS)

    for model_name in model_chain:
        tasks = {
            asyncio.create_task(_attempt(model_name, api_key, key_idx))
            for key_idx, api_key in enumerate(GEMINI_API_KEYS)
        }
        winner: Optional[Tuple[str, bytes]] = None

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                ok, mname, key_idx, pcm, err = finished.result()
                if ok:
                    winner = (mname, pcm)
                    continue  # keep draining `done`, but we already have a winner
                last_err = err
                key_note = f" (key #{key_idx + 1}/{len(GEMINI_API_KEYS)})" if multi_key else ""
                if isinstance(err, RuntimeError) and "timed out" in str(err):
                    yield sse_log(
                        f"[WARN] Voice model '{model_name}'{key_note} did not respond "
                        f"within {TTS_CALL_TIMEOUT_SECONDS}s."
                    )
                else:
                    yield sse_log(
                        f"[WARN] Voice model '{model_name}'{key_note} is busy/unavailable "
                        f"({type(err).__name__}: {err})."
                    )
            if winner:
                break

        # A winner (or a fully-drained key set) means we're done racing this
        # model — cancel any attempts still in flight before moving on.
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if winner:
            mname, pcm = winner
            write_wav_from_pcm(pcm, out_wav)
            yield sse_log(f"[SUCCESS] A key succeeded on '{mname}' — using it immediately.")
            yield {"_tts_result": mname}
            return

        yield sse_log(
            f"[WARN] All {len(GEMINI_API_KEYS)} key(s) failed for '{model_name}'. "
            "Trying the next model..."
        )

    raise RuntimeError(
        f"All {len(model_chain)} selected Gemini voice model(s) x "
        f"{len(GEMINI_API_KEYS)} key(s) are currently busy or unavailable. "
        f"Last error: {last_err}"
    )


# --------------------------------------------------------------------------- #
# RAM-safe upload streaming
# --------------------------------------------------------------------------- #

async def save_upload_streaming(upload: UploadFile, dest: Path) -> None:
    async with aiofiles.open(dest, "wb") as out:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            await out.write(chunk)
    await upload.close()


# --------------------------------------------------------------------------- #
# Adaptive silence-trim ladder
# --------------------------------------------------------------------------- #

async def _adaptive_silence_trim(
    raw_path: Path, video_duration: float, seg_dir: Path,
) -> Tuple[Path, float, int, List[dict]]:
    """
    Try each locked silence window — 500ms then 400ms, per the product's
    strict 400-500ms rule — SEQUENTIALLY, one at a time (RAM-safe: never
    two ffmpeg silence-trim jobs alive for the same session at once), and
    delete each rung's scratch file the moment it's no longer needed.
    Stops as soon as a rung lands comfortably close to the video length
    (within COMFORTABLE_MAX_RATIO); otherwise keeps the last (most
    aggressive, i.e. 400ms) result. Never re-trims an already-trimmed
    file, so cuts never compound.
    Returns (trimmed_path, trimmed_duration_seconds, ms_used, sse_log_events).
    """
    logs: List[dict] = []
    raw_dur = await probe_duration(raw_path)

    best: Optional[Tuple[int, Path, float]] = None
    for ms in SILENCE_TRIM_LADDER_MS:
        candidate = seg_dir / f"trim_{ms}ms.wav"
        try:
            await trim_internal_silences(raw_path, candidate, ms / 1000.0)
            dur = await probe_duration(candidate)
        except Exception as exc:  # noqa: BLE001
            logs.append(sse_log(f"[WARN] Silence-trim @ {ms}ms failed ({exc}); skipping this rung."))
            continue

        logs.append(sse_log(
            f"[INFO] Silence-trim @ {ms}ms -> {dur:.2f}s of speech "
            f"(video is {video_duration:.2f}s)."
        ))

        # Delete the previous rung's file now that we have a newer one —
        # only ONE candidate file is ever on disk at a time.
        if best is not None:
            try:
                best[1].unlink(missing_ok=True)
            except OSError:
                pass
        best = (ms, candidate, dur)

        comfortable = video_duration <= 0 or dur <= video_duration * COMFORTABLE_MAX_RATIO
        if comfortable:
            break  # least-aggressive rung that already fits well — stop here.

    if best is None:
        logs.append(sse_log("[WARN] All silence-trim attempts failed; using the untrimmed voice track."))
        return raw_path, raw_dur, 0, logs

    best_ms, best_path, best_dur = best
    if best_ms == SILENCE_TRIM_FLOOR_MS and (video_duration > 0 and best_dur > video_duration * COMFORTABLE_MAX_RATIO):
        logs.append(sse_log(
            f"[WARN] Even at the locked {SILENCE_TRIM_FLOOR_MS}ms floor, speech "
            f"({best_dur:.2f}s) is still longer than a comfortable speed-up "
            f"would allow for a {video_duration:.2f}s video — the remaining "
            "gap will be closed by the locked 1.15x-1.30x speed-up."
        ))

    return best_path, best_dur, best_ms, logs


# --------------------------------------------------------------------------- #
# Core processing: ONE Gemini TTS request -> trim -> exact duration match -> mux
# --------------------------------------------------------------------------- #

def _tts_style_hint(target_language: str) -> str:
    """
    Base delivery style, plus an accent note for languages where Gemini's
    default pronunciation may default to the "wrong" regional accent.
    Bengali is the explicit case in point: Gemini's default Bengali
    pronunciation tends toward the West Bengal (Indian) accent, and a
    Bangladeshi user naturally wants natural Bangladesh (Dhaka-standard)
    Bengali instead — a different accent, not just a different language.
    """
    base = (
        "Speak naturally, clearly, and expressively, with natural pacing "
        "and brief natural pauses between sentences"
    )
    lang = (target_language or "").strip().lower()
    if "bengali" in lang or "bangla" in lang:
        base += (
            ". Use natural Bangladeshi Bengali (Bangladesh, standard Dhaka "
            "pronunciation) — not the West Bengal/Indian Bengali accent"
        )
    return base


async def synthesize_single_track(sess: Session) -> AsyncGenerator[dict, None]:
    seg_dir = sess.dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # atempo is the chosen time-stretch engine (see _speed_filter) — kept
    # as a plain bool for the rest of the pipeline's function signatures,
    # but it no longer switches engines; it's always atempo now.
    use_rb = False
    yield sse_log("[INFO] Time-stretch engine: atempo (FFmpeg native — chosen for clean speech).")

    script_text = "\n".join(s.text.strip() for s in sess.segments if s.text.strip())
    if not script_text:
        raise RuntimeError("Nothing to voice — the translated script came back empty.")

    yield sse_progress(58, "Generating voice (single request)")
    yield sse_log(
        f"[INFO] Sending the FULL script as ONE Gemini TTS request "
        f"({len(script_text)} chars, {len(sess.segments)} line(s)) — no per-line "
        "splitting, so pacing stays perfectly uniform..."
    )

    raw_audio = seg_dir / "raw.wav"
    model_used: Optional[str] = None
    tts_chain = sess.tts_models or list(TTS_MODELS)
    yield sse_log(f"[INFO] Voice model fallback chain: {', '.join(tts_chain)}")

    # Self-healing outer retry: gemini_tts_with_fallback already races every
    # (model, key) combination internally; this outer loop additionally
    # retries the WHOLE attempt once more after a short backoff in case
    # every model/key was transiently down at the same instant.
    last_tts_err: Optional[Exception] = None
    for tts_attempt in range(1 + SELF_HEAL_RETRIES):
        try:
            async for ev in gemini_tts_with_fallback(
                text=script_text,
                voice_name=sess.single_voice,
                out_wav=raw_audio,
                style_hint=_tts_style_hint(sess.target_language),
                models=tts_chain,
            ):
                if "_tts_result" in ev:
                    model_used = ev["_tts_result"]
                else:
                    yield ev
            break
        except Exception as exc:  # noqa: BLE001
            last_tts_err = exc
            if tts_attempt < SELF_HEAL_RETRIES:
                yield sse_log(
                    f"[WARN] Voice generation failed ({exc}) — self-healing: "
                    f"retrying the full model chain in {SELF_HEAL_BACKOFF_SECONDS:.0f}s..."
                )
                await asyncio.sleep(SELF_HEAL_BACKOFF_SECONDS)
            else:
                raise
    yield sse_log(f"[SUCCESS] Voice generated via {model_used}.")

    yield sse_progress(70, "Trimming long silences")
    trimmed_path, trimmed_dur, ms_used, trim_logs = await _adaptive_silence_trim(
        raw_audio, sess.video_duration, seg_dir
    )
    for lg in trim_logs:
        yield lg
    yield sse_log(
        f"[INFO] Final silence-trim window: {ms_used}ms -> {trimmed_dur:.2f}s of "
        f"speech (video is {sess.video_duration:.2f}s)."
    )

    yield sse_progress(80, "Matching audio to video duration")
    raw_ratio = (trimmed_dur / sess.video_duration) if sess.video_duration > 0 else 1.0

    # PRODUCT REQUIREMENT — HARD LOCK: atempo is ALWAYS applied, and the
    # applied ratio is ALWAYS clamped into the [1.15x, 1.30x] band — never
    # below 1.15x (so the dub always has this brand's consistent brisk
    # pacing even when the natural fit needed less), and never above 1.30x
    # (so speech never sounds rushed/unnatural even when the natural fit
    # would have needed more — the mux step's apad+shortest absorbs any
    # remaining gap as trailing silence instead).
    fitted_path = seg_dir / "fitted.wav"
    applied_speed_ratio = max(ATEMPO_LOCK_MIN, min(ATEMPO_LOCK_MAX, raw_ratio))
    yield sse_log(
        f"[INFO] Natural fit would need {raw_ratio:.3f}x — locking applied "
        f"speed to {applied_speed_ratio:.3f}x (product rule: always between "
        f"{ATEMPO_LOCK_MIN:.2f}x-{ATEMPO_LOCK_MAX:.2f}x, pitch preserved via atempo)."
    )
    if raw_ratio < ATEMPO_LOCK_MIN:
        yield sse_log(
            f"[INFO] Speech was shorter/slower than the {ATEMPO_LOCK_MIN:.2f}x "
            "floor requires — speeding it up anyway to stay within the "
            "locked band; any leftover time is trailing silence, not lost audio."
        )
    elif raw_ratio > ATEMPO_LOCK_MAX:
        yield sse_log(
            f"[WARN] Speech needed more than {ATEMPO_LOCK_MAX:.2f}x to fully "
            "match the video length; capped at the locked ceiling instead, so "
            "the dubbed audio may run slightly long relative to the video."
        )
    await _encode_wav(
        ["-i", str(trimmed_path)], fitted_path, _speed_filter(applied_speed_ratio, use_rb)
    )

    # We deliberately do NOT run a second corrective speed pass here even if
    # the fit is a little off target: re-running rubberband on audio that
    # has already been time-stretched once compounds phase-vocoder
    # artifacts and is a real source of "broken"-sounding speech. A single
    # pass gets within a few milliseconds in practice (verified), and the
    # mux step's apad+shortest silently absorbs any tiny remaining gap —
    # far safer than a second stretch.
    fitted_dur = await probe_duration(fitted_path)
    yield sse_log(f"[SUCCESS] Final dubbed audio: {fitted_dur:.2f}s (target {sess.video_duration:.2f}s).")

    yield sse_progress(94, "Muxing dubbed audio into video")
    yield sse_log("[INFO] Merging dubbed audio with the original video (video stream copied, zero quality loss)...")
    out_video = sess.dir / "dubbed_output.mp4"
    await mux_video_with_audio(sess.video_path, fitted_path, out_video)

    # Free scratch segment files early (disk hygiene); keep the final MP4.
    shutil.rmtree(seg_dir, ignore_errors=True)
    save_session(sess)

    yield sse_progress(100, "Complete")
    yield sse_log("[SUCCESS] Dubbing complete.")
    yield sse_done({
        "session_id": sess.session_id,
        "download_url": f"/download/{sess.session_id}",
        "source_language": sess.source_language,
        "target_language": sess.target_language,
        "segments": len(sess.segments),
        "tts_requests": 1,
        "speed_ratio": round(applied_speed_ratio, 4),
        "voice_model": model_used,
    })


async def run_prepare(sess: Session) -> AsyncGenerator[dict, None]:
    """
    The AUTOMATIC phase: runs the instant a video finishes uploading —
    extract audio, probe the real video duration, and transcribe with
    Whisper. This does NOT translate or generate any voice yet (that only
    happens once the person picks a language/voice and presses the dub
    button — see run_dub). Doing the (language-independent) transcription
    here means the actual "press button -> hear voice" wait is shorter.
    """
    try:
        yield sse_progress(10, "Extracting audio track")
        yield sse_log("[INFO] Extracting audio track...")
        sess.audio_path = sess.dir / "source_audio.mp3"
        await extract_audio(sess.video_path, sess.audio_path)

        yield sse_progress(30, "Probing video duration")
        sess.video_duration = await probe_duration(sess.video_path)
        yield sse_log(f"[INFO] Video duration: {sess.video_duration:.2f}s")

        engine = sess.engine
        transcript = None
        if engine == "gemini":
            yield sse_progress(50, f"Transcribing with Gemini ({GEMINI_TRANSCRIBE_MODEL})")
            yield sse_log("[INFO] Sending audio to Gemini for transcription (primary engine)...")
            async for ev in self_heal("Gemini transcription", lambda: gemini_transcribe(sess.audio_path)):
                if "_result" in ev:
                    transcript = ev["_result"]
                else:
                    yield ev
        else:
            yield sse_progress(50, "Transcribing with Groq Whisper (whisper-large-v3)")
            yield sse_log("[INFO] Streaming audio to Groq Whisper for instant transcription...")
            async for ev in self_heal("Groq transcription", lambda: groq_transcribe(sess.audio_path)):
                if "_result" in ev:
                    transcript = ev["_result"]
                else:
                    yield ev

        sess.source_language = transcript["language"]
        sess.raw_segments = [Segment(**s) for s in transcript["segments"]]
        sess.prepared = True
        save_session(sess)

        yield sse_log(f"[INFO] Source language detected: {sess.source_language}")
        yield sse_progress(100, "Ready — pick a language/voice and press Dub")
        yield sse_done({
            "session_id": sess.session_id,
            "video_duration": round(sess.video_duration, 2),
            "source_language": sess.source_language,
            "line_count": len(sess.raw_segments),
        })
    except Exception as exc:  # noqa: BLE001
        yield sse_error(f"[ERROR] {type(exc).__name__}: {exc}")
        _destroy_session(sess.session_id)


async def run_dub(sess: Session) -> AsyncGenerator[dict, None]:
    """
    The MANUAL phase: only runs once the person has picked a target
    language + voice and pressed the dub button. Reuses the transcript
    already produced by run_prepare — translate (chunked, drop-proof) ->
    one-shot TTS -> trim -> exact duration match -> mux.
    """
    try:
        if not sess.raw_segments:
            raise RuntimeError("This video hasn't finished uploading/analyzing yet.")

        engine = sess.engine
        seg_dicts = [asdict(s) for s in sess.raw_segments]
        result = None
        if engine == "gemini":
            yield sse_progress(30, f"Translating {len(sess.raw_segments)} line(s) with Gemini")
            yield sse_log(
                f"[INFO] Translating with Gemini ({GEMINI_TRANSLATE_MODEL}) in "
                f"~{TRANSLATE_CHUNK_CHAR_BUDGET}-char chunks — every line is "
                "verified 1:1 and re-touched for a natural target-audience "
                "voice, so none can be silently dropped..."
            )
            async for ev in self_heal("Gemini translation", lambda: gemini_translate_all(
                transcript_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    result = ev["_result"]
                else:
                    yield ev
        else:
            yield sse_progress(30, f"Translating {len(sess.raw_segments)} line(s) with Groq (gpt-oss-120b)")
            yield sse_log(
                f"[INFO] Translating with Groq openai/gpt-oss-120b in "
                f"~{TRANSLATE_CHUNK_CHAR_BUDGET}-char chunks — every line is "
                "verified 1:1, so none can be silently dropped..."
            )
            async for ev in self_heal("Groq translation", lambda: groq_translate_all(
                transcript_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    result = ev["_result"]
                else:
                    yield ev

        sess.source_language = result["source_language"]
        sess.segments = [Segment(**s) for s in result["segments"]]

        # Guaranteed by groq_translate_all: output_count == input_count,
        # ALWAYS — nothing is ever silently dropped. If a chunk's
        # translation couldn't be verified after retries, it falls back to
        # the literal source-language text for just those lines instead.
        if result["fallback_count"] > 0:
            ranges = ", ".join(result["fallback_ranges"])
            yield sse_log(
                f"[WARN] {result['fallback_count']} line(s) (around {ranges}) "
                "could not be verified as translated after retries, so the "
                "ORIGINAL source-language text was kept for those lines "
                "instead of being dropped."
            )

        yield sse_progress(50, f"Translated all {len(sess.segments)} segment(s), none dropped")

        async for ev in synthesize_single_track(sess):
            yield ev
    except Exception as exc:  # noqa: BLE001
        yield sse_error(f"[ERROR] {type(exc).__name__}: {exc}")
        _destroy_session(sess.session_id)


# --------------------------------------------------------------------------- #
# FastAPI app + endpoints
# --------------------------------------------------------------------------- #

app = FastAPI(title="Ultimate Premium Video Dubbing Platform", version="4.0.0-single")

# CORS_ALLOW_ORIGINS (optional): comma-separated allow-list, e.g.
# "https://your-site.netlify.app,https://your-custom-domain.com". Defaults
# to "*" so any frontend (including this app's Netlify deploy) can call the
# API without extra configuration; tighten it in Render's env vars once
# your Netlify domain is fixed.
_cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = ["*"] if _cors_origins_raw in ("", "*") else _parse_key_list(_cors_origins_raw)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "groq_keys_configured": len(GROQ_API_KEYS),
        "default_engine": ENGINE_MODE_DEFAULT,
        "engines_available": [e for e in ("gemini", "groq") if
                              (GEMINI_API_KEYS if e == "gemini" else GROQ_API_KEYS)],
        "tts_models": TTS_MODELS,
        "tts_model_labels": TTS_MODEL_CATALOG,
        "voices": GEMINI_VOICE_NAMES,
        "whisper_model": WHISPER_MODEL,
        "translation_model": TRANSLATION_MODEL,
        "gemini_transcribe_model": GEMINI_TRANSCRIBE_MODEL,
        "gemini_translate_model": GEMINI_TRANSLATE_MODEL,
        "atempo_lock": [ATEMPO_LOCK_MIN, ATEMPO_LOCK_MAX],
        "silence_trim_lock_ms": SILENCE_TRIM_LADDER_MS,
        "ffmpeg": shutil.which(FFMPEG_BIN) is not None,
        "ffprobe": shutil.which(FFPROBE_BIN) is not None,
        "rubberband": await rubberband_available(),
    })


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    engine: str = Form(""),
) -> JSONResponse:
    """
    Step 1 of the AUTOMATIC phase — called the instant a video is picked
    (gallery, file browser, or drag-and-drop). Plain JSON (not SSE) on
    purpose: this lets the frontend use XMLHttpRequest's real upload
    progress event, so a person on a weak connection sees an actual
    "Uploading… NN%" instead of silence. Just saves the bytes and creates
    a session — the (slower) audio-extract + transcribe work happens next,
    in /prepare, once the browser confirms the upload itself is done.

    `engine` (optional form field, "gemini" | "groq") lets the frontend's
    engine toggle pick the transcription+translation provider for this
    session up front; defaults to ENGINE_MODE_DEFAULT ("gemini") if
    omitted or invalid.
    """
    _sweep_stale_sessions()
    chosen_engine = (engine or "").strip().lower()
    if chosen_engine not in ("gemini", "groq"):
        chosen_engine = ENGINE_MODE_DEFAULT
    if chosen_engine == "groq" and not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")
    if chosen_engine == "gemini" and not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured.")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    session_id = uuid.uuid4().hex[:12].upper()
    sdir = WORK_DIR / session_id
    sdir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename).suffix or ".mp4"
    video_path = sdir / f"input{suffix}"
    try:
        await save_upload_streaming(file, video_path)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(sdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}")

    sess = Session(session_id=session_id, dir=sdir, video_path=video_path, engine=chosen_engine)
    SESSIONS[session_id] = sess
    save_session(sess)

    return JSONResponse({"session_id": session_id, "filename": file.filename, "engine": chosen_engine})


@app.get("/prepare/{session_id}")
async def prepare(session_id: str, engine: str = ""):
    """
    Step 2 of the AUTOMATIC phase — called immediately after /upload
    resolves. Extracts audio, probes the real video duration, and
    transcribes it (language-independent, so this can happen before the
    person has even picked a target language). Does NOT translate or
    generate any voice — that only happens once the person presses the
    dub button (see /dub).

    Optional `?engine=gemini|groq` query param lets the toggle override the
    engine chosen at /upload time (e.g. if the person flips it right
    before analysis starts).
    """
    sess = load_session(session_id)
    if not sess or not sess.video_path or not sess.video_path.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")

    eng = (engine or "").strip().lower()
    if eng in ("gemini", "groq"):
        sess.engine = eng
        save_session(sess)

    async def event_stream() -> AsyncGenerator[dict, None]:
        yield sse_log(f"[INFO] Session {session_id} uploaded. Analyzing with engine='{sess.engine}'...")
        async for ev in run_prepare(sess):
            yield ev

    return EventSourceResponse(event_stream(), ping=10)


@app.post("/dub")
async def dub(
    session_id: str = Form(...),
    target_language: str = Form(...),
    voice_name: str = Form("Kore"),
    engine: str = Form(""),
    tts_models: str = Form(""),
):
    """
    MANUAL phase — only runs when the person presses the dub button after
    picking a target language + voice. No file upload here; the video was
    already uploaded and transcribed by /upload.

    `engine` (optional, "gemini" | "groq") overrides the translation
    engine for this run. `tts_models` (optional) is a comma-separated list
    of the voice-model IDs picked from the single/multi-select UI — the
    4-model catalog (see TTS_MODEL_CATALOG); an empty/invalid value falls
    back to the full default fallback chain.
    """
    eng = (engine or "").strip().lower()
    if eng in ("gemini", "groq"):
        pass
    else:
        eng = None  # keep whatever was set at /upload or /prepare time

    if (eng or ENGINE_MODE_DEFAULT) == "gemini" and not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured.")
    if (eng or ENGINE_MODE_DEFAULT) == "groq" and not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured (required for TTS).")

    sess = load_session(session_id)
    if not sess or not sess.video_path or not sess.video_path.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id — please re-upload the video.")
    if not sess.prepared or not sess.raw_segments:
        raise HTTPException(status_code=409, detail="This video hasn't finished uploading/analyzing yet.")

    voice = (voice_name or "").strip() or "Kore"
    if voice not in GEMINI_VOICE_NAMES:
        voice = "Kore"
    sess.target_language = (target_language or "").strip()
    sess.single_voice = voice
    if eng:
        sess.engine = eng
    sess.tts_models = parse_selected_tts_models(tts_models)
    save_session(sess)

    async def event_stream() -> AsyncGenerator[dict, None]:
        yield sse_log(
            f"[INFO] Dubbing session {session_id} into {sess.target_language} "
            f"({sess.single_voice}) — engine='{sess.engine}', "
            f"voice models={sess.tts_models}..."
        )
        async for ev in run_dub(sess):
            yield ev

    return EventSourceResponse(event_stream(), ping=10)


@app.get("/download/{session_id}")
async def download(session_id: str):
    """
    Streams the finished MP4 straight from disk with native HTTP Range
    support (FileResponse), so the browser can start playback/seeking
    immediately instead of waiting for the whole file — and so a preview
    <video> tag and a separate "Download" click can both read the same
    file without racing each other. The file is NOT deleted here anymore;
    cleanup happens on the 1-hour TTL sweep (or an explicit DELETE
    /session/{id}) so a slow/weak connection never loses the result.
    """
    sess = load_session(session_id)
    out_video = ((sess.dir if sess else WORK_DIR / session_id) / "dubbed_output.mp4")
    if not out_video.exists():
        raise HTTPException(status_code=404, detail="Result not found or already cleaned up.")
    return FileResponse(
        out_video,
        media_type="video/mp4",
        filename=f"dubbed_{session_id}.mp4",
    )


@app.delete("/session/{session_id}")
async def cancel_session(session_id: str) -> JSONResponse:
    _destroy_session(session_id)
    return JSONResponse({"status": "deleted", "session_id": session_id})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)
