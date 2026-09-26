"""
Ultimate Premium Video Dubbing Platform (Single-Character Master Build)
========================================================================

FastAPI backend, single-character-only, real-time-duration-matching
architecture:

  * Groq (whisper-large-v3)              -> heavy audio transcription
  * Groq (openai/gpt-oss-120b)           -> elite, drop-proof translation
  * Gemini native TTS (3-model fallback,
    multi-key concurrent racing)         -> pure-text voice generation

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
                              voice (single request or several concurrent
                              ones — see below), fits it to the video, and
                              muxes it in. An optional custom ending CTA
                              line can also be added here (ending_cta).
  4. GET  /download/{id}  -> streams the finished MP4 (HTTP Range
                              supported) — never deleted on read; cleanup
                              is via the 1h TTL sweep or DELETE /session.

Voice generation strategy
--------------------------
A short/light script is sent as exactly ONE Gemini TTS request. A longer
or text-heavy one is automatically split into several requests instead
(see "TTS chunking" below) — this was a real production failure mode: a
~5-minute video's 6,385-character script blew a single free-tier key's
~10k-input-tokens/minute quota outright. Whichever path is taken, each
individual TTS request always carries the WHOLE text for its slice in one
shot (never split per-line) — mixing per-line speeds would make some
lines sound faster than others, which is far more noticeable than a
single uniform speed change across one piece.

After a piece's raw audio comes back, the pipeline works like a real
audio engineer instead of guessing:

  1. SILENCE TRIM (adaptive ladder): every internal silent gap longer than
     a threshold is trimmed down (FFmpeg `silenceremove`, real audio-level
     detection — not a timestamp guess). The ladder starts at 500ms and
     only steps down (500 -> 400 -> 280ms, never lower) if needed to keep
     the required speed-up in a comfortable range — going lower starts
     cutting into natural between-sentence pauses, which makes the voice
     sound rushed/mashed-together rather than helping. This only ever
     trims SILENCE, never speech.
  2. DURATION MATCH: the trimmed speech duration is compared against the
     ACTUAL target duration for that piece (the real video, or that
     chunk's own time-span within it — not the sum of Whisper segment
     timings) and a single pitch-preserving `atempo` speed-up is applied
     if (and only if) it's too long — capped HARD at MAX_SPEED_RATIO
     (1.20x). Speech is NEVER slowed below its natural 1.0x pace, and
     never sped up past 1.20x either — both sound unnatural. If 1.20x
     alone wouldn't be enough, the SCRIPT is shortened instead (a rough
     pre-TTS estimate flags this proactively, and the actual measured
     result triggers one more shortened regeneration if still needed —
     see _produce_one_chunk_audio and MAX_SHORTEN_ATTEMPTS). In the rare
     case even that isn't enough, speed still never exceeds 1.20x — any
     tiny remainder is covered by extending the video's ending afterward
     (see "Speed ceiling" below), never by cutting off speech. `atempo`
     (FFmpeg native) is used deliberately over `rubberband`: real
     listening tests on this deployment found atempo noticeably cleaner
     on speech.
  3. MUX: the video stream is copied bit-for-bit (`-c:v copy`) — zero
     re-encoding, zero quality loss — only the audio track is replaced.
     (The one exception is the optional ending-CTA feature, which has to
     re-encode to extend the video — see below.)

TTS chunking (long/text-heavy videos)
----------------------------------------
If a script exceeds TTS_CHUNK_MAX_CHARS characters (NOT video duration —
see _split_into_tts_chunks for why), it's split at natural sentence
boundaries into pieces of up to TTS_CHUNK_MAX_CHARS each. Every piece
still goes through the *exact same* one-request pipeline above — nothing
about it changes — just run once per piece. Pieces are
generated CONCURRENTLY, not sequentially, and each piece gets its OWN
slice of the configured Gemini keys (_partition_keys_for_chunks) so
simultaneous pieces never fight over the same key's per-minute quota. As
each piece finishes, its own SSE log/progress events stream through live
(prefixed "[Part N/M]"), and once every piece is back they're joined into
one continuous track (FFmpeg concat demuxer, lossless) before the final
duration-vs-video sanity check and mux.

Speed ceiling & script-shortening
-------------------------------------
Speed is capped at MAX_SPEED_RATIO (1.20x) — never exceeded. If a chunk's
speech looks (or measures) too long for its slot even after silence-trim,
the SCRIPT is shortened via Groq rather than pushing the voice faster:
once proactively (character-count estimate, before the first TTS call)
and once more reactively (measured result, with a regenerated TTS call)
if still needed — MAX_SHORTEN_ATTEMPTS bounds this to at most 2 tries, no
unbounded loop. If a chunk is still over the cap after that (rare), speed
is held at exactly 1.20x and the small remainder is covered by extending
that chunk's video span afterward — see the overflow safety-net in
synthesize_single_track — never by cutting off speech.

Optional ending CTA
-----------------------
If the person supplies a custom closing line (Session.ending_cta), it's
generated as one more short, natural-pace TTS request, appended after the
main track, and the video is extended by freeze-framing its last frame for
exactly that long (`tpad`) so audio and video stay in sync. This is the
one path that can't use `-c:v copy` (a video filter forces a re-encode,
done at crf 18 / visually lossless) — scoped only to this optional,
off-by-default feature; every other video is untouched as always.

Simple, everyday wording + Bengali polish pass
----------------------------------------------------
The translation prompt itself asks for plain, spoken, everyday vocabulary
rather than formal/literary phrasing for every target language. When the
target is Bengali specifically, there's a SECOND dedicated pass after
translation (groq_polish_all) that reviews the translated script
alongside the original source text and rewrites lines that sound stiff,
unnatural, or mistranslated — natural Bangladeshi (Dhaka-standard)
phrasing, common English loanwords for globally-recognized terms
Bangladeshi audiences already use directly ("সিরিয়াল কিলার" rather than a
stiff literal "ধারাবাহিক হত্যাকারী"), and a gender/pronoun cross-check
against the original (source languages like Mandarin, where 他/她 sound
identical, can otherwise get this wrong). Uses the same chunked,
retry-then-fallback-to-pre-polish guarantee as translation — never drops
or corrupts a line, only changes wording. No-op for any other target
language.

Backup / never-crash design
-----------------------------
* For each model, every key in that request's pool is tried CONCURRENTLY
  (not one-at-a-time) — a busy/slow key doesn't hold up the others, so the
  first success wins and the rest are cancelled. Only if EVERY key in the
  pool fails does it fall through to the next model. A bounded per-attempt
  timeout (TTS_CALL_TIMEOUT_SECONDS) guards against a request that just
  hangs with no response at all (observed in production under a fully
  quota-exhausted account).
* Every processing step is wrapped so a failure ends ONLY that session
  with a clean SSE `error` event and the session's scratch files are
  deleted — the FastAPI process itself, and every other in-flight
  session, is never affected.
* Speed ratios are always clamped to a safe FFmpeg range (0.25x-4.0x) so
  a pathological mismatch (e.g. wildly different script vs. video length)
  degrades gracefully with a [WARN] instead of crashing the render.
* Both SSE endpoints send a keep-alive ping every 10s so a long stretch of
  silent processing (e.g. waiting out a congested model) doesn't get the
  connection killed by an idle-timing-out proxy in between.

TTS model selection (Single vs Multiple) — NEW
------------------------------------------------
Four Gemini native-TTS models are available (TTS_MODEL_CATALOG):
gemini-2.5-flash-preview-tts, gemini-3.1-flash-tts-preview,
gemini-3.8-flash-tts, and gemini-3.8-flash-lite-tts. The person picks, per
dub, either:
  * SINGLE  — one preferred model. Every configured Gemini key races on
    that ONE model first (same key-racing engine as always — nothing about
    it changes). If that model is completely down across every key, the
    pipeline still automatically falls through to the remaining catalog
    models afterward — picking a single model narrows which one goes
    FIRST, it never removes the underlying never-crash safety net.
  * MULTIPLE — several preferred models, in the order picked. A problem
    with one immediately switches to the next PICKED model — no waiting
    on the one that's down. Once every picked model is exhausted, any
    remaining (unpicked) catalog models are still tried as a final safety
    net, exactly like SINGLE mode.
See _resolve_tts_model_order (turns the /dub form fields tts_mode +
tts_models into the ordered list) and the `models=` parameter now accepted
by gemini_tts_with_fallback — the racing/fallback engine itself (see
"Backup / never-crash design" above) is unchanged; it now just walks
whichever ordered model list this particular dub resolved to instead of
always the same hardcoded TTS_MODELS default.

Environment
-----------
GEMINI_API_KEYS (required)   Comma-separated list of Gemini API keys.
                              Rotated automatically on quota/errors.
                              (GEMINI_API_KEY, singular, also still works
                              as a 1-key fallback.)
GROQ_API_KEYS   (required)   Comma-separated list of Groq API keys.
                              Rotated automatically on quota/errors.
                              (GROQ_API_KEY, singular, also still works
                              as a 1-key fallback.)
WORK_DIR        (optional)   Scratch dir. Default: ./_sessions
FFMPEG_BIN      (optional)   Default: "ffmpeg"
FFPROBE_BIN     (optional)   Default: "ffprobe"
"""

from __future__ import annotations

import asyncio
import gc
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

# Real-world evidence: on a small Render instance (512MB RAM), letting every
# ffmpeg/ffprobe call fire off unbounded (e.g. a long/dense video's several
# TTS chunks, each ALSO racing 3 silence-trim-ladder rungs concurrently —
# see SILENCE_TRIM_LADDER_MS below — can mean a dozen+ ffmpeg processes all
# decoding/filtering media in memory AT ONCE) is what actually exceeds the
# container's memory limit and gets it killed/restarted — NOT any local
# transcription model (this pipeline only ever calls Groq's cloud Whisper
# API for transcription; it never loads a model in-process — see
# groq_transcribe). Capping how many ffmpeg/ffprobe subprocesses may run
# AT THE SAME TIME (regardless of how many chunks/rungs WANT to run) keeps
# peak memory bounded on small instances without removing or serializing
# any feature — every ffmpeg call still happens exactly as before, some
# just wait briefly for a free slot instead of all launching at once.
# Raise this (env var, no code change) on a bigger Render plan.
FFMPEG_MAX_CONCURRENT = max(1, int(os.environ.get("FFMPEG_MAX_CONCURRENT", "2")))

# --- Groq: transcription + translation engine -------------------------------
WHISPER_MODEL = "whisper-large-v3"            # heavy audio -> text
TRANSLATION_MODEL = "openai/gpt-oss-120b"     # elite translation
                                               # (llama-3.3-70b-versatile is
                                               # retired on Groq as of 2026-08-16)

# --- Gemini: pure-text native TTS engine, with ordered auto-fallback --------
# One model busy -> auto request goes to the next model. Still busy across
# every model -> auto-rotate to the next GEMINI_API_KEYS entry. Every
# model x key combination exhausted -> a clean error is raised via SSE.
# gemini-2.5-flash-preview-tts is tried FIRST by default: it is the most
# established of the four, and in practice a brand-new preview model tends
# to return 503 "high demand" more often while its own capacity is still
# ramping up. This ordering is a practical hedge based on observed behavior,
# not a guarantee — Google's own server load is outside our control either
# way. gemini-3.8-flash-lite-tts / gemini-3.8-flash-tts are Google's newer
# (Sept-2026) TTS models — the lite one is the drop-in replacement for
# gemini-3.1-flash-tts-preview, the full one is the higher-fidelity
# "creative" tier — both are appended after the two older previews in the
# default order since they're the least production-tested here so far.
#
# TTS MODEL SELECTION (Single vs Multiple) — the person picks this in the
# UI per-dub; see Session.tts_mode / Session.tts_models and
# _resolve_tts_model_order. This list (TTS_MODELS) is only the DEFAULT
# fallback order used when nothing was picked (or for any other
# unspecified caller) — every actual /dub call resolves its own ordered
# list and passes it into gemini_tts_with_fallback via the `models=`
# argument; nothing about the underlying key-racing / auto-fallback engine
# below changes because of this — it just walks whatever ordered list of
# models it's handed.
TTS_MODEL_CATALOG: List[Dict[str, str]] = [
    {"id": "gemini-2.5-flash-preview-tts", "label": "Gemini 2.5 Flash Preview TTS"},
    {"id": "gemini-3.1-flash-tts-preview", "label": "Gemini 3.1 Flash TTS Preview"},
    {"id": "gemini-3.8-flash-tts", "label": "Gemini 3.8 Flash TTS"},
    {"id": "gemini-3.8-flash-lite-tts", "label": "Gemini 3.8 Flash Lite TTS"},
]
TTS_MODEL_IDS: List[str] = [m["id"] for m in TTS_MODEL_CATALOG]

TTS_MODELS: List[str] = [
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.8-flash-tts",
]
# Hard ceiling on a SINGLE (model, key) TTS attempt. Real-world evidence
# (a fully-quota-exhausted account) showed genuinely-stuck requests just
# hang with no response at all rather than promptly erroring, so this
# bounds the worst case; tightened from 150s after that evidence, while
# still staying above the ~90s a legitimately slow-but-working call has
# been observed to take.
TTS_CALL_TIMEOUT_SECONDS = 90

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
TRANSLATE_CHUNK_SIZE = 12
TRANSLATE_CHUNK_RETRIES = 2

# --- Silence-trim ladder ------------------------------------------------- #
# The ENTIRE script is one TTS request, so pacing is uniform by construction
# (see module docstring). The only per-render tuning is how much internal
# silence to trim before computing the final speed ratio. We start at a
# generous 500ms (safe — never eats into a natural mid-sentence breath) and
# only step down if the video is short enough that 500ms isn't enough to
# bring the speed-up into a comfortable range. 280ms is the floor — NOT
# 120ms: trimming a natural between-sentence pause down that far starts
# making the voice-over sound rushed/mashed-together ("broken"), which is
# far more noticeable than a slightly bigger, well-preserved speed-up. All
# rungs are tried CONCURRENTLY (they all read from the same raw source) so
# this ladder costs one round-trip, not five sequential ones.
SILENCE_TRIM_LADDER_MS: List[int] = [500, 400, 280]
SILENCE_TRIM_FLOOR_MS: int = SILENCE_TRIM_LADDER_MS[-1]
SILENCE_DB_THRESHOLD = -32.0  # dB below which audio is treated as silence.
COMFORTABLE_MAX_RATIO = 1.2   # stop shrinking the trim window once we're
                               # within +20% of the video's length — lean on
                               # the (now formant-preserving) speed-up rather
                               # than over-trimming pauses.

# Absolute FFmpeg speed-filter safety clamp (atempo hard limits — a
# low-level safety net for the filter itself, distinct from the
# pipeline-level MAX_SPEED_RATIO business rule enforced below).
HARD_SPEED_MIN = 0.25
HARD_SPEED_MAX = 4.0

# --- TTS chunking for long videos ----------------------------------------- #
# Real-world evidence (a 289 MB / ~5 min video, 6,385 translated
# characters) confirms Gemini's FREE-TIER quota is genuinely tight: each
# key is limited to ~10,000 input tokens PER MINUTE on the flash-tts
# models, and non-Latin scripts like Bengali can tokenize far less
# efficiently than English, so a single big request can blow the quota
# outright and either 429 or (worse) hang until our own timeout.
#
# Split decision is PURELY by character count — NOT video duration. A
# long video with sparse dialogue may need only one request; a short,
# dense one might still need several. Character count of the text
# actually being sent is what drives the per-key token quota, so it's
# what decides the split (see _split_into_tts_chunks).
#
# Every chunk still goes through the EXACT SAME pipeline (one-shot TTS ->
# adaptive silence-trim -> speed match, capped at MAX_SPEED_RATIO) as a
# single-request video — just run once per chunk. Chunks are generated
# CONCURRENTLY, each pinned to its OWN slice of the available API keys
# (see _partition_keys_for_chunks) so simultaneous chunks never fight each
# other for the same key's per-minute quota, before being joined back into
# one continuous track.
TTS_CHUNK_MAX_CHARS = 2000             # PRIMARY (and only) split trigger — a
                                         # conservative, evidence-based size that
                                         # comfortably fits under the tightest
                                         # observed per-key, per-minute token
                                         # quota even for token-hungry scripts
                                         # (Bengali, etc.)

# --- Speed ceiling + script-shortening (never sound sped-up) -------------- #
# Speeding audio up too far starts sounding audibly unnatural. Rather than
# ever exceeding a modest ceiling, the SCRIPT is shortened instead: a
# rough pre-TTS character-count estimate flags a chunk likely to run long
# BEFORE spending a request on it (proactive), and the ACTUAL measured
# result after TTS triggers one more shortening + regeneration pass if
# it's still over (reactive) — see _produce_one_chunk_audio.
MAX_SPEED_RATIO = 1.20                 # hard target ceiling; the pipeline
                                         # actively shortens the script rather
                                         # than exceeding this
MAX_SHORTEN_ATTEMPTS = 2               # 1 proactive + 1 reactive — bounded,
                                         # never an unbounded retry loop
CHARS_PER_SECOND_ESTIMATE = 13.0       # rough, language-agnostic speaking-rate
                                         # estimate, used only to flag a chunk
                                         # PROACTIVELY, before the first real
                                         # TTS call — the authoritative check is
                                         # always the actual generated audio's
                                         # measured duration afterward


def _resolve_tts_model_order(tts_mode: str, raw_models: str) -> List[str]:
    """
    Turns the person's UI choice (Single vs Multiple TTS model selection —
    see index.html's "Voice Model" field) into the ordered model-fallback
    list gemini_tts_with_fallback walks for THIS dub.

      * SINGLE: exactly one model is picked up front. Every configured
        Gemini key races on that ONE model first (existing key-race
        behavior — unchanged). If that model is completely down across
        every key (all keys busy/erroring), the pipeline still
        automatically tries the remaining catalog models afterward — the
        pre-existing "never-crash" guarantee is not weakened by picking a
        single model, it only decides which model goes first.
      * MULTIPLE: every model the person checked is tried, in the exact
        order they picked them — a problem with one switches to the next
        PICKED model immediately (no waiting on the one that's down).
        Once every picked model has been exhausted, the remaining
        (unpicked) catalog models are still tried as a final safety net,
        exactly like SINGLE mode above.
      * Anything malformed, unrecognized, or empty (including requests
        from an older frontend that doesn't send these two fields at all)
        falls back to the original default order (TTS_MODELS), completely
        unchanged from before this feature existed.
    """
    picked: List[str] = []
    valid_ids = set(TTS_MODEL_IDS)
    for raw in re.split(r"[,\n]+", raw_models or ""):
        mid = raw.strip()
        if mid in valid_ids and mid not in picked:
            picked.append(mid)

    mode = (tts_mode or "single").strip().lower()
    if mode not in ("single", "multi", "multiple"):
        mode = "single"
    if mode == "single":
        picked = picked[:1]

    if not picked:
        return list(TTS_MODELS)  # nothing usable picked -> original default behavior

    rest = [m for m in TTS_MODELS if m not in picked]
    return picked + rest


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
    ending_cta: str = ""     # optional custom CTA line, spoken at the very end
    tts_mode: str = "single"                          # "single" | "multi" — UI choice
    tts_models: List[str] = field(default_factory=lambda: list(TTS_MODELS))  # resolved
                                                        # fallback order for THIS dub —
                                                        # see _resolve_tts_model_order


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
        "ending_cta": sess.ending_cta,
        "tts_mode": sess.tts_mode,
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
        ending_cta=data.get("ending_cta", ""),
        tts_mode=data.get("tts_mode", "single"),
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
    gc.collect()  # promptly release large buffers (segments, PCM, etc.) —
                  # cheap insurance on small-RAM instances (e.g. Render free
                  # tier); the real memory-safety fix is FFMPEG_MAX_CONCURRENT
                  # above, this is just a harmless extra nudge.


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

_ffmpeg_semaphore = asyncio.Semaphore(FFMPEG_MAX_CONCURRENT)


async def _run(cmd: List[str]) -> str:
    async with _ffmpeg_semaphore:
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


async def concat_audio_files(paths: List[Path], out_audio: Path) -> None:
    """
    Join multiple already-canonical WAV files (same sample rate/channels/
    codec — guaranteed here since every chunk is produced by _encode_wav)
    into one continuous track, in order, via FFmpeg's concat demuxer.
    Lossless (`-c copy`) since every input shares the exact same format.
    """
    if len(paths) == 1:
        shutil.copy(paths[0], out_audio)
        return
    list_file = out_audio.parent / f"{out_audio.stem}_concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in paths), encoding="utf-8",
    )
    try:
        await _run([
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy",
            str(out_audio),
        ])
    finally:
        list_file.unlink(missing_ok=True)


async def extend_video_with_freeze_frame(video_path: Path, extra_seconds: float,
                                         out_path: Path) -> None:
    """
    Extends a video by freezing its LAST FRAME for `extra_seconds` more
    seconds — used only by the optional "ending CTA" feature, to give a
    custom closing line enough time to be spoken after the main dubbed
    content ends (a freeze-frame + voiceover CTA is standard practice for
    short-form video endings anyway).

    This is the ONE place in the whole pipeline that can't use
    `-c:v copy`: FFmpeg's `tpad` filter has to touch every frame, so the
    video is re-encoded (crf 18 = visually lossless) rather than
    stream-copied. That trade-off is scoped ONLY to this optional feature —
    when it's off, every other video passes through untouched exactly as
    before.
    """
    await _run([
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={extra_seconds:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
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


async def _groq_json_segments_call(
    system_prompt: str,
    user_payload_obj: dict,
    expected_count: int,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """
    Shared low-level helper used by translate/polish/shorten alike: call
    Groq chat completions with a system prompt + JSON user payload,
    expecting a strict JSON response shaped
    {"source_language": <optional>, "segments": [...]}. Rotates across
    every configured Groq key on failure. Returns (detected_language,
    segments) — segments is an EMPTY list (never raises for a bad
    response) if parsing fails or the response doesn't contain exactly
    `expected_count` valid objects, so callers can retry/fallback safely.
    """
    user_payload = json.dumps(user_payload_obj, ensure_ascii=False)

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
            f"Groq call failed on all {len(GROQ_API_KEYS)} configured key(s): {last_err}"
        ) from last_err

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, TypeError):
        return None, []

    out_segments: List[Dict[str, object]] = []
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
    if len(out_segments) != expected_count:
        return (str(detected_lang) if detected_lang else None), []
    return (str(detected_lang) if detected_lang else None), out_segments


async def _run_chunked_rewrite(
    segments: List[Dict[str, object]],
    chunk_fn,
    chunk_size: int,
    retries: int,
) -> Tuple[List[Dict[str, object]], int, List[str], Optional[str]]:
    """
    Generic chunked-rewrite-with-retry-and-fallback driver, shared by
    translate/polish/shorten. Splits `segments` into pieces of at most
    `chunk_size`, calls `chunk_fn(piece) -> (lang_or_None, rewritten_or_[])`
    per piece — retrying up to `retries` extra times if it doesn't come
    back as a strict 1:1 match — and falls back to the ORIGINAL
    (pre-rewrite) text for any piece that still fails after every retry.
    GUARANTEES len(output) == len(segments) always; nothing is ever
    silently dropped. Returns (rewritten_segments, fallback_count,
    fallback_ranges, detected_language_or_None).
    """
    if not segments:
        return [], 0, [], None

    chunk_size = max(1, chunk_size)
    chunks = [segments[i:i + chunk_size] for i in range(0, len(segments), chunk_size)]

    detected_lang: Optional[str] = None
    all_out: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk in chunks:
        result_segments: List[Dict[str, object]] = []
        for _attempt in range(1 + retries):
            try:
                lang, rewritten = await chunk_fn(chunk)
            except Exception:  # noqa: BLE001
                continue
            if lang and detected_lang is None:
                detected_lang = lang
            if len(rewritten) == len(chunk):
                result_segments = rewritten
                break

        if not result_segments:
            fallback_count += len(chunk)
            fallback_ranges.append(f"{chunk[0]['start']:.1f}s-{chunk[-1]['end']:.1f}s")
            result_segments = [dict(s) for s in chunk]  # keep the original text as-is

        all_out.extend(result_segments)

    return all_out, fallback_count, fallback_ranges, detected_lang


def _bengali_target(target_language: str) -> bool:
    lang = (target_language or "").strip().lower()
    return "bengali" in lang or "bangla" in lang


async def _groq_translate_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """Translate ONE small chunk. See _run_chunked_rewrite for the retry/fallback contract."""
    if _bengali_target(target_language):
        simplicity_clause = (
            "SIMPLE, EVERYDAY WORDING (important): Use plain, spoken, "
            "everyday Bangladeshi Bengali — the way people in Bangladesh "
            "actually talk, not textbook/literary Bengali. Prefer common "
            "words over heavy formal or Sanskrit-derived (তৎসম) vocabulary "
            "whenever a simpler word means the same thing. Hook lines and "
            "any punchy opening line especially should sound like popular "
            "Bangladeshi short-video content — casual, direct, easy for "
            "anyone to instantly understand, not stiff or academic. This "
            "changes WORD CHOICE only — never change the meaning, remove "
            "content, or alter the structure of what's being said.\n\n"
        )
    else:
        simplicity_clause = (
            "SIMPLE, EVERYDAY WORDING: Prefer plain, clear, commonly-used "
            "words over formal or literary ones whenever they mean the "
            "same thing, so a general audience instantly understands it. "
            "This changes WORD CHOICE only — never change the meaning or "
            "content.\n\n"
        )

    system_prompt = (
        "You are an elite professional translator inside an automated "
        "single-narrator video-dubbing pipeline. You will be given a JSON "
        "array of timestamped transcript segments (a SMALL CHUNK of a "
        "larger transcript), already transcribed from the original spoken "
        f"language ({source_language}).\n\n"
        "TASK 1 — TRANSLATION: Translate every segment's text PERFECTLY and "
        f"naturally into {target_language}. Preserve tone, intent, idiom, and "
        "natural spoken rhythm — this is for dubbing, not a literal "
        "word-for-word gloss. Never leave a segment untranslated. Pay close "
        "attention to PRONOUN GENDER (he/she, boyfriend/girlfriend, etc.) — "
        "some source languages (e.g. Mandarin 他/她) sound identical for "
        "different genders, so infer the correct gender carefully from "
        "context rather than defaulting or guessing.\n\n"
        + simplicity_clause +
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
    return await _groq_json_segments_call(
        system_prompt,
        {"source_language": source_language, "transcript": chunk_segments},
        expected_count=len(chunk_segments),
    )


async def groq_translate_all(
    transcript_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> dict:
    """
    Translate the WHOLE transcript in small chunks (TRANSLATE_CHUNK_SIZE
    lines at a time) so a single LLM hiccup can only ever affect a few
    lines. Guarantees len(output_segments) == len(transcript_segments)
    ALWAYS — see _run_chunked_rewrite.
    """
    if not transcript_segments:
        raise RuntimeError("Nothing to translate — the transcript is empty.")

    payload = [
        {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
         "text": seg.get("text", "")}
        for seg in transcript_segments
    ]

    async def chunk_fn(chunk):
        return await _groq_translate_chunk(chunk, source_language, target_language)

    all_segments, fallback_count, fallback_ranges, detected_lang = await _run_chunked_rewrite(
        payload, chunk_fn, TRANSLATE_CHUNK_SIZE, TRANSLATE_CHUNK_RETRIES,
    )

    return {
        "source_language": detected_lang or source_language,
        "segments": all_segments,
        "input_count": len(transcript_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


async def _groq_polish_chunk(
    chunk_pairs: List[Dict[str, object]],
    target_language: str,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """
    Polish ONE small chunk of ALREADY-TRANSLATED text for natural
    Bangladeshi phrasing. `chunk_pairs` items are
    {start, end, text (current translation), original (source-language
    text, for meaning-fidelity context)}.
    """
    system_prompt = (
        "You are a native Bangladeshi Bengali script editor for a "
        "short-video dubbing pipeline. You will be given a JSON array of "
        "segments, each with the ORIGINAL source-language line and the "
        "CURRENT Bengali translation, in order.\n\n"
        "TASK: Rewrite ONLY the wording of the Bengali translation, where "
        "it needs it, so it sounds like natural, punchy Bangladeshi "
        "short-video/hook-style Bengali — the way real Bangladeshi "
        "creators actually talk — NOT a stiff, literal, textbook "
        "translation. Specifically:\n"
        "- Prefer common ENGLISH LOANWORDS written in Bengali script for "
        "globally-recognized terms Bangladeshi audiences already use "
        "directly in casual speech (e.g. 'সিরিয়াল কিলার' instead of a "
        "stiff literal equivalent like 'ধারাবাহিক হত্যাকারী', similarly "
        "'থ্রিলার', 'টুইস্ট', etc.) — use judgment, not every word needs "
        "an English loanword, only where it's genuinely more natural.\n"
        "- Cross-check the CURRENT translation against the ORIGINAL for "
        "gender/pronoun accuracy (he/she, boyfriend/girlfriend, etc.) and "
        "fix any mismatch you find.\n"
        "- Keep it simple, conversational, and avoid heavy formal/তৎসম "
        "vocabulary.\n"
        "- Most lines may need NO change at all if they're already "
        "natural and correct — only rewrite lines that genuinely sound "
        "stiff, unnatural, or mistranslated. If a line is already good, "
        "return it completely unchanged.\n\n"
        "CRITICAL RULES:\n"
        "- This is a WORDING polish, not a re-translation — NEVER change "
        "the underlying meaning, remove information, or add new "
        "information.\n"
        "- The output 'segments' array MUST have EXACTLY the same number "
        "of objects, in the same order, with the same start/end values as "
        "the input. Never drop, merge, split, or reorder a segment.\n\n"
        "OUTPUT — strict JSON only, no prose, no markdown fences:\n"
        '{"segments": [{"start": 0.0, "end": 3.2, "text": "<possibly-revised Bengali>"}]}'
    )
    payload = [
        {"start": p["start"], "end": p["end"], "original": p.get("original", ""), "text": p["text"]}
        for p in chunk_pairs
    ]
    return await _groq_json_segments_call(
        system_prompt, {"segments": payload}, expected_count=len(chunk_pairs),
    )


async def groq_polish_all(
    translated_segments: List[Dict[str, object]],
    raw_segments: List[Dict[str, object]],
    target_language: str,
) -> dict:
    """
    Bengali-only "polish" pass: reviews the already-translated script
    (with the original text alongside for context) and rewrites lines
    that sound stiff/unnatural/mistranslated into natural, everyday
    Bangladeshi phrasing. Guarantees len(output) == len(translated_segments)
    ALWAYS (falls back to the pre-polish translation for any chunk that
    can't be verified 1:1 after retries — never drops or corrupts a line).
    No-op (returns the input unchanged) if target_language isn't Bengali.
    """
    if not _bengali_target(target_language) or not translated_segments:
        return {
            "segments": translated_segments, "fallback_count": 0,
            "fallback_ranges": [], "polished": False,
        }

    payload = []
    for i, seg in enumerate(translated_segments):
        original_text = raw_segments[i]["text"] if i < len(raw_segments) else ""
        payload.append({
            "start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", ""), "original": original_text,
        })

    async def chunk_fn(chunk):
        return await _groq_polish_chunk(chunk, target_language)

    all_segments, fallback_count, fallback_ranges, _lang = await _run_chunked_rewrite(
        payload, chunk_fn, TRANSLATE_CHUNK_SIZE, TRANSLATE_CHUNK_RETRIES,
    )

    return {
        "segments": all_segments, "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges, "polished": True,
    }


async def _groq_shorten_chunk(
    chunk_segments: List[Dict[str, object]],
    target_language: str,
    shrink_fraction: float,
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """
    Condense an already-translated chunk so it takes noticeably less time
    to speak, while preserving meaning and the exact 1:1 structure. Used
    when a chunk's speech is estimated/measured to run past MAX_SPEED_RATIO
    of its video time-span (see _produce_one_chunk_audio) — shortening the
    SCRIPT is preferred over speeding the VOICE up further.
    """
    pct = max(5, min(60, int(round(shrink_fraction * 100))))
    system_prompt = (
        "You are editing an already-translated dubbing script that runs "
        "too long for its video time slot. You will get a JSON array of "
        "{start, end, text} segments, in order.\n\n"
        f"TASK: Rewrite the 'text' of each segment to be MORE CONCISE — "
        f"aim to cut roughly {pct}% off the total character count across "
        "all segments combined — while preserving the exact meaning and "
        "tone. Cut filler words and redundant phrasing, pick shorter "
        "synonyms and tighter sentence structure; do NOT remove actual "
        "information or change what is being said. Focus your cuts on "
        "the LONGER/wordier segments — a segment that's already short "
        "and essential can stay as-is if there's nothing safe to trim.\n\n"
        "CRITICAL: the output 'segments' array MUST have EXACTLY the same "
        "number of objects, in the same order, with the same start/end "
        "values as the input. Never drop, merge, split, or reorder a "
        "segment.\n\n"
        "OUTPUT — strict JSON only, no prose, no markdown fences:\n"
        '{"segments": [{"start": 0.0, "end": 3.2, "text": "<shortened text>"}]}'
    )
    return await _groq_json_segments_call(
        system_prompt, {"segments": chunk_segments}, expected_count=len(chunk_segments),
    )



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


def _write_tts_audio(data: bytes, out_wav: Path) -> None:
    """
    Gemini's native TTS models don't all return the same container by
    default: gemini-2.5-flash-preview-tts and gemini-3.1-flash-tts-preview
    return headerless raw PCM (audio/l16), which is why write_wav_from_pcm
    (the `wave` module) wraps it in a WAV header below. Google's newer
    (Sept-2026) gemini-3.8-flash-tts / gemini-3.8-flash-lite-tts instead
    return a FULLY-FORMED WAV file (audio/wav, real RIFF header) by
    default for unary requests. Re-wrapping an already-WAV file in another
    WAV header would corrupt it, so this is detected here (by sniffing the
    RIFF/WAVE magic bytes) rather than assumed from the model name — this
    keeps every downstream step (silence-trim, atempo, mux) working on a
    normal on-disk WAV file exactly as before, unchanged, regardless of
    which of the four models produced it.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        out_wav.write_bytes(data)
    else:
        write_wav_from_pcm(data, out_wav)


async def gemini_tts_with_fallback(
    text: str,
    voice_name: str,
    out_wav: Path,
    style_hint: str = "",
    api_keys: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
) -> AsyncGenerator[dict, None]:
    """
    Generate speech with Gemini native TTS for the FULL script in ONE call.

    `api_keys` lets a caller hand this a SUBSET of the configured keys
    (used when multiple chunks run concurrently — see
    _partition_keys_for_chunks — so simultaneous chunks don't all fight
    over the same key's per-minute quota). Defaults to every configured
    key when not given (the original single-request behavior).

    `models` lets a caller hand this a SUBSET/reordering of TTS_MODEL_IDS
    (the person's Single/Multiple TTS model choice for this dub — see
    Session.tts_models / _resolve_tts_model_order). Defaults to the module
    default TTS_MODELS order when not given, so every pre-existing caller
    keeps behaving exactly as before this feature was added.

    Key insight from real-world logs: trying keys ONE AT A TIME, waiting up
    to TTS_CALL_TIMEOUT_SECONDS on each before moving to the next, means a
    congested model can burn many minutes before we even fall through to
    the next model — and a connection sitting idle that long risks the
    whole SSE stream getting killed by a proxy in between.

    So instead: for EACH model, every key in this call's pool is tried
    CONCURRENTLY (all fired at once — this is safe, each key has its own
    independent quota/project). We take whichever attempt succeeds FIRST
    and immediately cancel the rest. Only if EVERY key in the pool fails
    for a model do we move on to the next model. This means:
      - The common case (at least one key in the pool is free) resolves in
        however long the FASTEST attempt takes — often just a few seconds.
      - The worst case (a model is down for the whole pool) is bounded by
        TTS_CALL_TIMEOUT_SECONDS ONCE per model, not multiplied by the key
        count.

    Yields SSE log dicts as it goes; the FINAL yielded item is always
    {"_tts_result": "<model_name_used>"} so the caller can tell which model
    actually produced the audio.
    """
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Nothing to voice — the translated script is empty.")

    keys = api_keys if api_keys is not None else GEMINI_API_KEYS
    if not keys:
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
    multi_key = len(keys) > 1
    model_order = models if models else TTS_MODELS

    for model_name in model_order:
        tasks = {
            asyncio.create_task(_attempt(model_name, api_key, key_idx))
            for key_idx, api_key in enumerate(keys)
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
                key_note = f" (key #{key_idx + 1}/{len(keys)})" if multi_key else ""
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
            _write_tts_audio(pcm, out_wav)
            yield sse_log(f"[SUCCESS] A key succeeded on '{mname}' — using it immediately.")
            yield {"_tts_result": mname}
            return

        yield sse_log(
            f"[WARN] All {len(keys)} key(s) failed for '{model_name}'. "
            "Trying the next model..."
        )

    raise RuntimeError(
        f"All {len(model_order)} Gemini voice model(s) x {len(keys)} "
        f"key(s) are currently busy or unavailable. Last error: {last_err}"
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
# TTS chunk splitting (long videos -> multiple requests)
# --------------------------------------------------------------------------- #

def _split_into_tts_chunks(
    segments: List[Segment], video_duration: float,
) -> List[Tuple[float, float, List[Segment]]]:
    """
    Decide how many separate TTS requests are needed, and which segments +
    which ORIGINAL-VIDEO TIME-SPAN belong to each.

    Split PURELY by character count (TTS_CHUNK_MAX_CHARS) — NOT by video
    duration. Video length isn't a reliable proxy for how much has to be
    said: a long video with sparse dialogue can safely be one request,
    while a short, dense video can still blow a single request's quota.
    Character count (of the text actually being sent) is what really
    drives the per-key token quota, so it's what decides the split.

    Splits happen at natural segment boundaries — never mid-sentence — the
    moment the accumulated character count would cross the budget. The
    returned chunks always partition [0, video_duration] end to end with
    no gaps or overlaps, so each chunk's own duration-matching step (see
    _produce_one_chunk_audio) keeps that piece of audio aligned with the
    right part of the video, and concatenating them afterward reproduces
    the full video's length. The per-chunk TIME SPAN computed here is only
    ever used as an ESTIMATE/target for fitting that piece's audio — never
    as a reason to split in the first place.
    """
    if not segments:
        return []
    total_chars = sum(len(s.text) for s in segments)
    if total_chars <= TTS_CHUNK_MAX_CHARS:
        return [(0.0, video_duration, list(segments))]

    chunks: List[Tuple[float, float, List[Segment]]] = []
    current: List[Segment] = []
    current_start = 0.0
    current_chars = 0

    for seg in segments:
        would_chars = current_chars + len(seg.text)
        if current and would_chars > TTS_CHUNK_MAX_CHARS:
            chunk_end = current[-1].end  # cut at the last natural pause, not mid-sentence
            chunks.append((current_start, chunk_end, current))
            current = []
            current_start = chunk_end
            current_chars = 0
        current.append(seg)
        current_chars += len(seg.text)

    if current:
        # The final chunk always reaches the TRUE end of the video (not
        # just the last segment's timestamp), so the chunks' spans always
        # sum to exactly video_duration with no trailing gap.
        chunks.append((current_start, video_duration, current))

    # Avoid a wasteful tiny trailing chunk (a handful of leftover
    # characters getting their own full TTS request): fold a short tail
    # into the previous chunk instead, as long as that doesn't blow the
    # char budget by much.
    MIN_TAIL_CHARS = 200
    if len(chunks) >= 2:
        _, last_end, last_segs = chunks[-1]
        last_chars = sum(len(s.text) for s in last_segs)
        if last_chars < MIN_TAIL_CHARS:
            prev_start, _, prev_segs = chunks[-2]
            merged_chars = sum(len(s.text) for s in prev_segs) + last_chars
            if merged_chars <= TTS_CHUNK_MAX_CHARS * 1.15:
                chunks[-2] = (prev_start, last_end, prev_segs + last_segs)
                chunks.pop()

    return chunks


def _partition_keys_for_chunks(all_keys: List[str], n_chunks: int) -> List[List[str]]:
    """
    Spread the available Gemini keys across N concurrently-running chunks
    so simultaneous chunks don't all compete for the SAME key's per-minute
    quota (the exact failure mode seen in real logs — a big request blew
    one key's 10k-tokens/minute limit, and racing across the SAME 10 keys
    per chunk would just recreate that contention across chunks). Simple
    round-robin: chunk i gets all_keys[i], all_keys[i+n_chunks], ... If
    there are more chunks than keys, some chunks share a (smaller, still
    non-empty) pool — less redundancy for those, but never zero keys.
    """
    if n_chunks <= 0:
        return []
    if not all_keys:
        return [[] for _ in range(n_chunks)]
    pools: List[List[str]] = [[] for _ in range(n_chunks)]
    for i, key in enumerate(all_keys):
        pools[i % n_chunks].append(key)
    for i, pool in enumerate(pools):
        if not pool:
            pools[i] = [all_keys[i % len(all_keys)]]
    return pools


# --------------------------------------------------------------------------- #
# Adaptive silence-trim ladder
# --------------------------------------------------------------------------- #

async def _adaptive_silence_trim(
    raw_path: Path, video_duration: float, seg_dir: Path,
) -> Tuple[Path, float, int, List[dict]]:
    """
    Try every minimum-silence window (500 -> 280ms) CONCURRENTLY — they all
    read from the same original raw audio independently, so there's no
    reason to run them one after another. Picks the LEAST aggressive
    (largest ms) rung whose trimmed duration is comfortably close to the
    video length (within COMFORTABLE_MAX_RATIO); if none qualify, falls
    back to the most aggressively trimmed (smallest ms / shortest) result.
    Never re-trims an already-trimmed file, so cuts never compound.
    Returns (trimmed_path, trimmed_duration_seconds, ms_used, sse_log_events).
    """
    logs: List[dict] = []
    raw_dur = await probe_duration(raw_path)

    async def _try_rung(ms: int) -> Tuple[int, Optional[Path], Optional[float], Optional[Exception]]:
        candidate = seg_dir / f"trim_{ms}ms.wav"
        try:
            await trim_internal_silences(raw_path, candidate, ms / 1000.0)
            dur = await probe_duration(candidate)
            return ms, candidate, dur, None
        except Exception as exc:  # noqa: BLE001
            return ms, None, None, exc

    results = await asyncio.gather(*[_try_rung(ms) for ms in SILENCE_TRIM_LADDER_MS])

    ok: List[Tuple[int, Path, float]] = []
    for ms, path, dur, exc in results:
        if exc is not None:
            logs.append(sse_log(f"[WARN] Silence-trim @ {ms}ms failed ({exc}); skipping this rung."))
            continue
        target_note = f"video is {video_duration:.2f}s" if video_duration > 0 else "no fixed target"
        logs.append(sse_log(
            f"[INFO] Silence-trim @ {ms}ms -> {dur:.2f}s of speech ({target_note})."
        ))
        ok.append((ms, path, dur))

    if not ok:
        logs.append(sse_log("[WARN] All silence-trim attempts failed; using the untrimmed voice track."))
        return raw_path, raw_dur, 0, logs

    ok.sort(key=lambda t: t[0], reverse=True)  # largest ms (least aggressive) first
    comfortable = [t for t in ok if video_duration <= 0 or t[2] <= video_duration * COMFORTABLE_MAX_RATIO]

    if comfortable:
        best_ms, best_path, best_dur = comfortable[0]
    else:
        best_ms, best_path, best_dur = min(ok, key=lambda t: t[2])  # most aggressive / shortest
        logs.append(sse_log(
            f"[WARN] Even at the {SILENCE_TRIM_FLOOR_MS}ms floor, speech "
            f"({best_dur:.2f}s) is still longer than a comfortable speed-up "
            f"would allow for a {video_duration:.2f}s video — the remaining "
            "gap will be closed with a slightly faster playback speed."
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
    if _bengali_target(target_language):
        base += (
            ". Use natural Bangladeshi Bengali (Bangladesh, standard Dhaka "
            "pronunciation) — not the West Bengal/Indian Bengali accent"
        )
    return base


def _estimate_speaking_seconds(text: str) -> float:
    """
    Rough, language-agnostic estimate of spoken duration from character
    count — used only to PROACTIVELY flag a chunk that's likely to need
    shortening, before spending a real TTS request on it. Not exact; the
    real, authoritative check is always the ACTUAL generated audio's
    measured duration after TTS (see _produce_one_chunk_audio).
    """
    return max(0.1, len(text) / CHARS_PER_SECOND_ESTIMATE)


async def _shorten_segments(
    segments: List[Dict[str, object]], target_language: str, shrink_fraction: float,
) -> List[Dict[str, object]]:
    """
    Best-effort wrapper around _groq_shorten_chunk with the SAME
    retry/fallback guarantee as translate/polish (via _run_chunked_rewrite —
    treated as a single piece here since it's already a small, pre-chunked
    slice). Falls back to the UNSHORTENED text if the shorten call can't be
    verified 1:1 even after a retry, so a failed attempt never corrupts or
    drops a line — it just leaves that text as long as it was.
    """
    async def chunk_fn(chunk):
        return await _groq_shorten_chunk(chunk, target_language, shrink_fraction)

    out, _fb, _ranges, _lang = await _run_chunked_rewrite(
        segments, chunk_fn, chunk_size=max(1, len(segments)), retries=1,
    )
    return out


async def _produce_one_chunk_audio(
    chunk_segments: List[Segment],
    target_seconds: float,
    voice_name: str,
    target_language: str,
    seg_dir: Path,
    tag: str,
    api_keys: Optional[List[str]] = None,
    tts_models: Optional[List[str]] = None,
) -> AsyncGenerator[dict, None]:
    """
    Runs the FULL per-chunk voice pipeline for one piece of script:
    Gemini TTS -> adaptive silence-trim ladder -> speed match against
    `target_seconds`, capped hard at MAX_SPEED_RATIO (1.20x) — speech is
    NEVER slowed below 1.0x, and never sped up past the cap either. If a
    PROACTIVE character-count estimate suggests the script is too long
    for the slot, it's shortened BEFORE the first TTS call; if the
    ACTUAL measured result still comes back over the cap, it's shortened
    again and regenerated once more (REACTIVE) — bounded to
    MAX_SHORTEN_ATTEMPTS total tries. If it's still over after that
    (rare), speed is capped at exactly MAX_SPEED_RATIO and a small
    residual overflow may remain — the caller (synthesize_single_track)
    handles that by extending the video's ending rather than ever cutting
    off speech. Yields SSE events throughout; the FINAL yielded item is
    always
    {"_chunk_result": {"path": Path, "model": str, "applied_ratio": float}}.
    """
    working_segments: List[Dict[str, object]] = [
        {"start": s.start, "end": s.end, "text": s.text} for s in chunk_segments
    ]

    # PROACTIVE: a rough pre-TTS estimate -> shorten before spending a real
    # TTS request if the script looks likely to need more than the cap.
    est_seconds = sum(_estimate_speaking_seconds(s["text"]) for s in working_segments)
    est_ratio = (est_seconds / target_seconds) if target_seconds > 0 else 1.0
    if est_ratio > MAX_SPEED_RATIO:
        shrink = 1.0 - (MAX_SPEED_RATIO / est_ratio)
        yield sse_log(
            f"[INFO] Estimated speaking time (~{est_seconds:.1f}s) looks like it "
            f"would need ~{est_ratio:.2f}x speed for this {target_seconds:.1f}s "
            f"slot (above the {MAX_SPEED_RATIO}x ceiling) -> shortening the "
            f"script by ~{int(shrink * 100)}% in advance, before generating voice..."
        )
        working_segments = await _shorten_segments(working_segments, target_language, shrink)

    model_used: Optional[str] = None
    fitted_path = seg_dir / f"fitted_{tag}.wav"
    applied_ratio = 1.0
    trimmed_path = raw_audio = None  # noqa: F841 (assigned in loop, used after)

    for attempt in range(1, MAX_SHORTEN_ATTEMPTS + 1):
        chunk_text = "\n".join(
            str(s["text"]).strip() for s in working_segments if str(s["text"]).strip()
        )
        raw_audio = seg_dir / f"raw_{tag}_{attempt}.wav"
        async for ev in gemini_tts_with_fallback(
            text=chunk_text,
            voice_name=voice_name,
            out_wav=raw_audio,
            style_hint=_tts_style_hint(target_language),
            api_keys=api_keys,
            models=tts_models,
        ):
            if "_tts_result" in ev:
                model_used = ev["_tts_result"]
            else:
                yield ev
        yield sse_log(f"[SUCCESS] Voice generated via {model_used}.")

        trimmed_path, trimmed_dur, ms_used, trim_logs = await _adaptive_silence_trim(
            raw_audio, target_seconds, seg_dir
        )
        for lg in trim_logs:
            yield lg
        yield sse_log(
            f"[INFO] Silence-trim window: {ms_used}ms -> {trimmed_dur:.2f}s of "
            f"speech (target {target_seconds:.2f}s)."
        )

        ratio = (trimmed_dur / target_seconds) if target_seconds > 0 else 1.0

        # HARD RULE #1: dubbed speech is NEVER slowed below its natural
        # (1.0x) pace — a slowed-down voice sounds unnatural/robotic,
        # which is worse than a bit of trailing silence (filled in later
        # by the mux step's apad+shortest).
        if target_seconds <= 0 or ratio <= 1.02:
            yield sse_log(
                "[INFO] Speech fits within this segment's length -> kept at "
                "natural (1.0x) speed. Speed is never reduced below normal — "
                "any leftover time is silent, not slowed-down speech."
            )
            await _encode_wav(["-i", str(trimmed_path)], fitted_path)
            applied_ratio = 1.0
            break

        # HARD RULE #2: speed is never pushed past MAX_SPEED_RATIO — if a
        # single pass gets there, apply it and we're done.
        if ratio <= MAX_SPEED_RATIO:
            applied_ratio = ratio
            yield sse_log(f"[INFO] Speech is longer than this segment -> speeding audio up {ratio:.3f}x (pitch preserved).")
            await time_stretch_to_duration(trimmed_path, target_seconds, fitted_path, False)
            break

        # Over the cap: prefer shortening the SCRIPT over speeding the
        # VOICE up further.
        if attempt < MAX_SHORTEN_ATTEMPTS:
            shrink = 1.0 - (MAX_SPEED_RATIO / ratio)
            yield sse_log(
                f"[WARN] Measured speech ({trimmed_dur:.2f}s) still needs "
                f"{ratio:.3f}x for this {target_seconds:.1f}s slot — above the "
                f"{MAX_SPEED_RATIO}x ceiling. Shortening the script by "
                f"~{int(shrink * 100)}% and regenerating the voice (attempt "
                f"{attempt + 1}/{MAX_SHORTEN_ATTEMPTS})..."
            )
            working_segments = await _shorten_segments(working_segments, target_language, shrink)
            continue

        # Out of shortening attempts: cap speed at EXACTLY the ceiling —
        # never higher, per hard rule. A residual overflow may remain;
        # the orchestrator extends the video's ending to cover it rather
        # than ever cutting off speech.
        applied_ratio = MAX_SPEED_RATIO
        yield sse_log(
            f"[WARN] Even after {MAX_SHORTEN_ATTEMPTS} shortening attempt(s), "
            f"this segment still needs {ratio:.3f}x. Speed is capped at "
            f"{MAX_SPEED_RATIO}x (never higher) — the audio may run a little "
            "past its slot; the video's ending will be extended slightly to "
            "cover it rather than cutting any words off."
        )
        await time_stretch_to_duration(trimmed_path, trimmed_dur / MAX_SPEED_RATIO, fitted_path, False)
        break

    fitted_dur = await probe_duration(fitted_path)
    yield sse_log(f"[SUCCESS] Segment audio ready: {fitted_dur:.2f}s (target {target_seconds:.2f}s).")

    yield {"_chunk_result": {"path": fitted_path, "model": model_used, "applied_ratio": applied_ratio}}


async def _maybe_append_ending_cta(
    sess: Session, main_audio_path: Path, seg_dir: Path,
    base_video_path: Optional[Path] = None,
) -> AsyncGenerator[dict, None]:
    """
    Optional feature: if the person typed a custom "ending CTA" line and
    switched it on, generate it as ONE MORE short TTS request (same voice,
    natural 1.0x pace — there's no fixed target duration to hit here, it's
    brand-new content, not a translation of anything in the source video),
    append it after the main dubbed track, and freeze-frame-extend the
    video by exactly that much so audio and video stay in sync. Off by
    default; only runs when sess.ending_cta is non-empty (checked by the
    caller before this is invoked). `base_video_path` lets the caller pass
    an already-extended video (e.g. from the speed-overflow safety net) as
    the starting point instead of the original upload.

    Yields SSE events throughout; the FINAL yielded item is always
    {"_cta_result": {"audio_path": Path, "video_path": Path}} — the new
    (longer) audio and video paths the caller should mux together instead
    of the originals.
    """
    base_video_path = base_video_path or sess.video_path
    cta_text = sess.ending_cta.strip()
    yield sse_progress(92, "Adding ending CTA")
    yield sse_log(f"[INFO] Ending CTA enabled: generating \"{cta_text[:80]}\" as one extra spoken line...")

    raw_cta = seg_dir / "raw_cta.wav"
    model_used: Optional[str] = None
    async for ev in gemini_tts_with_fallback(
        text=cta_text,
        voice_name=sess.single_voice,
        out_wav=raw_cta,
        style_hint=_tts_style_hint(sess.target_language),
        models=sess.tts_models,
    ):
        if "_tts_result" in ev:
            model_used = ev["_tts_result"]
        else:
            yield ev
    yield sse_log(f"[SUCCESS] CTA voice generated via {model_used}.")

    # Trim any leading/trailing dead air, but there's no target duration to
    # match against — the CTA always plays at its natural, un-sped pace.
    trimmed_cta, trimmed_dur, ms_used, trim_logs = await _adaptive_silence_trim(
        raw_cta, 0.0, seg_dir,
    )
    for lg in trim_logs:
        yield lg
    yield sse_log(f"[INFO] CTA audio: {trimmed_dur:.2f}s, added after the main content at natural speed.")

    yield sse_log(f"[INFO] Extending the video by {trimmed_dur:.2f}s (freeze-frame on the last frame) to fit the CTA...")
    extended_video = seg_dir / "extended_for_cta.mp4"
    await extend_video_with_freeze_frame(base_video_path, trimmed_dur, extended_video)

    combined_audio = seg_dir / "with_cta.wav"
    await concat_audio_files([main_audio_path, trimmed_cta], combined_audio)

    yield {"_cta_result": {"audio_path": combined_audio, "video_path": extended_video}}


async def synthesize_single_track(sess: Session) -> AsyncGenerator[dict, None]:
    """
    Orchestrates voice generation for the whole video. Splits the
    transcript into one or more chunks (see _split_into_tts_chunks — a
    short, light video is always exactly ONE chunk, unchanged from
    before). Multiple chunks are generated CONCURRENTLY — each pinned to
    its own slice of the available API keys (_partition_keys_for_chunks)
    so simultaneous chunks never fight over the same key's per-minute
    quota — with their live SSE events fanned in through a shared queue so
    progress from every part streams to the client in real time as it
    happens, not only once each part fully finishes. The resulting audio
    pieces are then joined back into one continuous track, an optional
    ending CTA line is appended (see _maybe_append_ending_cta), and the
    final mux step runs exactly as always.
    """
    seg_dir = sess.dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    yield sse_log("[INFO] Time-stretch engine: atempo (FFmpeg native — chosen for clean speech).")

    chunks = _split_into_tts_chunks(sess.segments, sess.video_duration)
    if not chunks:
        raise RuntimeError("Nothing to voice — the translated script came back empty.")

    total_chars = sum(len(s.text) for s in sess.segments)
    multi = len(chunks) > 1

    if not multi:
        yield sse_log(
            f"[INFO] {sess.video_duration:.0f}s video, {total_chars} chars -> "
            "fits comfortably in ONE Gemini TTS request; no splitting needed."
        )
        key_pools = [GEMINI_API_KEYS]
    else:
        key_pools = _partition_keys_for_chunks(GEMINI_API_KEYS, len(chunks))
        pool_note = (
            f"each part gets its own ~{len(key_pools[0])}-key pool so they never "
            "compete for the same key's quota"
        ) if GEMINI_API_KEYS else "no API keys configured"
        yield sse_log(
            f"[INFO] {sess.video_duration:.0f}s video, {total_chars} chars -> too "
            f"much for one safe TTS request (real-world evidence: a single key's "
            f"~10k-token/minute free-tier quota gets blown by scripts this size). "
            f"Auto-splitting into {len(chunks)} part(s) of up to "
            f"{TTS_CHUNK_MAX_CHARS} chars each, generated CONCURRENTLY "
            f"({pool_note}) — once every part's voice is back, they're joined "
            "into one continuous track."
        )

    yield sse_progress(55, f"Generating voice ({len(chunks)} part(s) in parallel)" if multi else "Generating voice")

    queue: "asyncio.Queue[dict]" = asyncio.Queue()
    chunk_results: Dict[int, dict] = {}
    worker_errors: Dict[int, Exception] = {}

    async def _worker(idx: int, c_start: float, c_end: float, c_segments: List[Segment]) -> None:
        n = idx + 1
        target_seconds = max(0.01, c_end - c_start)
        total_chars = sum(len(s.text) for s in c_segments)
        label = f"Part {n}/{len(chunks)}" if multi else "Voice"
        await queue.put(sse_log(
            f"[INFO] {label}: {total_chars} chars ({len(c_segments)} line(s), "
            f"{target_seconds:.1f}s of video) queued for voice generation..."
        ))
        try:
            async for ev in _produce_one_chunk_audio(
                c_segments, target_seconds, sess.single_voice, sess.target_language,
                seg_dir, tag=f"c{idx}", api_keys=key_pools[idx],
                tts_models=sess.tts_models,
            ):
                if "_chunk_result" in ev:
                    chunk_results[idx] = ev["_chunk_result"]
                else:
                    if multi and ev.get("event") == "log":
                        ev = {"event": "log", "data": f"[{label}] {ev['data']}"}
                    await queue.put(ev)
        except Exception as exc:  # noqa: BLE001
            worker_errors[idx] = exc
        finally:
            await queue.put({"_worker_done": idx})

    tasks = [
        asyncio.create_task(_worker(idx, c_start, c_end, c_segments))
        for idx, (c_start, c_end, c_segments) in enumerate(chunks)
    ]

    remaining = len(tasks)
    finished = 0
    progress_lo, progress_hi = 55, 90
    while remaining > 0:
        item = await queue.get()
        if "_worker_done" in item:
            remaining -= 1
            finished += 1
            step = progress_lo + (progress_hi - progress_lo) * finished / len(chunks)
            yield sse_progress(
                int(step),
                f"Finished {finished}/{len(chunks)} part(s)" if multi else "Voice generated",
            )
            continue
        yield item

    await asyncio.gather(*tasks, return_exceptions=True)

    if worker_errors:
        first_idx = min(worker_errors)
        raise RuntimeError(
            f"Part {first_idx + 1}/{len(chunks)} failed: {worker_errors[first_idx]}"
        ) from worker_errors[first_idx]

    chunk_paths = [chunk_results[i]["path"] for i in range(len(chunks))]
    models_used = [chunk_results[i]["model"] for i in range(len(chunks)) if chunk_results[i].get("model")]
    ratios = [chunk_results[i]["applied_ratio"] for i in range(len(chunks))]

    if len(chunk_paths) == 1:
        fitted_path = chunk_paths[0]
    else:
        yield sse_progress(91, "Joining voice parts together")
        yield sse_log(f"[INFO] Joining {len(chunk_paths)} voice part(s) into one continuous track...")
        fitted_path = seg_dir / "joined.wav"
        await concat_audio_files(chunk_paths, fitted_path)

    fitted_dur = await probe_duration(fitted_path)
    yield sse_log(f"[SUCCESS] Final dubbed audio: {fitted_dur:.2f}s (target {sess.video_duration:.2f}s).")

    # Safety net: if some chunk(s) still ran over MAX_SPEED_RATIO even
    # after every shortening attempt (rare), the joined audio can end up
    # slightly longer than the video. Rather than let the final mux's
    # apad+shortest silently cut off the tail of the dubbed speech, extend
    # the video's ending (freeze-frame on the last frame) by that much —
    # same technique as the ending-CTA feature.
    mux_video_path = sess.video_path
    overflow_seconds = max(0.0, fitted_dur - sess.video_duration) if sess.video_duration > 0 else 0.0
    if overflow_seconds > 0.2:
        yield sse_log(
            f"[WARN] Even with script-shortening, the combined dubbed audio "
            f"ran {overflow_seconds:.2f}s long — extending the video's ending "
            "(freeze-frame on the last frame) by that much rather than "
            "cutting off any spoken words."
        )
        extended_for_overflow = seg_dir / "extended_for_overflow.mp4"
        await extend_video_with_freeze_frame(sess.video_path, overflow_seconds, extended_for_overflow)
        mux_video_path = extended_for_overflow

    # Optional ending CTA: appends one more spoken line after the main
    # content and freeze-frame-extends the video (from mux_video_path,
    # which may already be overflow-extended above) to give it room. Off
    # by default; only runs when the person typed CTA text and enabled it.
    if sess.ending_cta.strip():
        async for ev in _maybe_append_ending_cta(sess, fitted_path, seg_dir, base_video_path=mux_video_path):
            if "_cta_result" in ev:
                res = ev["_cta_result"]
                fitted_path = res["audio_path"]
                mux_video_path = res["video_path"]
            else:
                yield ev

    yield sse_progress(94, "Muxing dubbed audio into video")
    yield sse_log("[INFO] Merging dubbed audio with the original video (video stream copied, zero quality loss)...")
    out_video = sess.dir / "dubbed_output.mp4"
    await mux_video_with_audio(mux_video_path, fitted_path, out_video)

    # Free scratch segment files early (disk hygiene); keep the final MP4.
    shutil.rmtree(seg_dir, ignore_errors=True)
    save_session(sess)

    overall_ratio = (sum(ratios) / len(ratios)) if ratios else 1.0
    unique_models = sorted(set(m for m in models_used if m))

    yield sse_progress(100, "Complete")
    yield sse_log("[SUCCESS] Dubbing complete.")
    yield sse_done({
        "session_id": sess.session_id,
        "download_url": f"/download/{sess.session_id}",
        "source_language": sess.source_language,
        "target_language": sess.target_language,
        "segments": len(sess.segments),
        "tts_requests": len(chunks) + (1 if sess.ending_cta.strip() else 0),
        "speed_ratio": round(overall_ratio, 4),
        "voice_model": ", ".join(unique_models) if unique_models else None,
        "ending_cta_added": bool(sess.ending_cta.strip()),
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

        yield sse_progress(50, "Transcribing with Groq Whisper (whisper-large-v3)")
        yield sse_log("[INFO] Streaming audio to Groq Whisper for instant transcription...")
        try:
            transcript = await groq_transcribe(sess.audio_path)
        except Exception as exc:  # noqa: BLE001
            yield sse_log(f"[WARN] Groq transcription problem: {exc}")
            raise

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

        yield sse_progress(30, f"Translating {len(sess.raw_segments)} line(s) with Groq (gpt-oss-120b)")
        yield sse_log(
            f"[INFO] Translating with Groq openai/gpt-oss-120b in small chunks "
            f"of {TRANSLATE_CHUNK_SIZE} lines — every line is verified 1:1, "
            "so none can be silently dropped..."
        )
        try:
            result = await groq_translate_all(
                transcript_segments=[asdict(s) for s in sess.raw_segments],
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )
        except Exception as exc:  # noqa: BLE001
            yield sse_log(f"[WARN] Groq translation problem: {exc}")
            raise

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

        yield sse_progress(45, f"Translated all {len(sess.segments)} segment(s), none dropped")

        if _bengali_target(sess.target_language):
            yield sse_log(
                "[INFO] Polishing the Bengali script for natural Bangladeshi "
                "phrasing (common loanwords for globally-recognized terms, "
                "gender/pronoun double-check, casual hook-style wording)..."
            )
            try:
                polish_result = await groq_polish_all(
                    translated_segments=[asdict(s) for s in sess.segments],
                    raw_segments=[asdict(s) for s in sess.raw_segments],
                    target_language=sess.target_language,
                )
            except Exception as exc:  # noqa: BLE001
                yield sse_log(f"[WARN] Polish pass failed, keeping the plain translation: {exc}")
                polish_result = None

            if polish_result is not None:
                sess.segments = [Segment(**s) for s in polish_result["segments"]]
                if polish_result["fallback_count"] > 0:
                    ranges = ", ".join(polish_result["fallback_ranges"])
                    yield sse_log(
                        f"[WARN] {polish_result['fallback_count']} line(s) (around "
                        f"{ranges}) could not be safely polished after retries, so "
                        "the plain (pre-polish) translation was kept for those "
                        "lines instead of being dropped or corrupted."
                    )
                yield sse_log("[SUCCESS] Script polished.")

        yield sse_progress(50, "Script ready")

        async for ev in synthesize_single_track(sess):
            yield ev
    except Exception as exc:  # noqa: BLE001
        yield sse_error(f"[ERROR] {type(exc).__name__}: {exc}")
        _destroy_session(sess.session_id)


# --------------------------------------------------------------------------- #
# FastAPI app + endpoints
# --------------------------------------------------------------------------- #

app = FastAPI(title="Ultimate Premium Video Dubbing Platform", version="4.0.0-single")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to your frontend origin in production.
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "groq_keys_configured": len(GROQ_API_KEYS),
        "tts_models": TTS_MODELS,
        "tts_model_catalog": TTS_MODEL_CATALOG,
        "voices": GEMINI_VOICE_NAMES,
        "whisper_model": WHISPER_MODEL,
        "translation_model": TRANSLATION_MODEL,
        "ffmpeg": shutil.which(FFMPEG_BIN) is not None,
        "ffprobe": shutil.which(FFPROBE_BIN) is not None,
        "rubberband": await rubberband_available(),
    })


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    """
    Step 1 of the AUTOMATIC phase — called the instant a video is picked
    (gallery, file browser, or drag-and-drop). Plain JSON (not SSE) on
    purpose: this lets the frontend use XMLHttpRequest's real upload
    progress event, so a person on a weak connection sees an actual
    "Uploading… NN%" instead of silence. Just saves the bytes and creates
    a session — the (slower) audio-extract + transcribe work happens next,
    in /prepare, once the browser confirms the upload itself is done.
    """
    _sweep_stale_sessions()
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")
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

    sess = Session(session_id=session_id, dir=sdir, video_path=video_path)
    SESSIONS[session_id] = sess
    save_session(sess)

    return JSONResponse({"session_id": session_id, "filename": file.filename})


@app.get("/prepare/{session_id}")
async def prepare(session_id: str):
    """
    Step 2 of the AUTOMATIC phase — called immediately after /upload
    resolves. Extracts audio, probes the real video duration, and
    transcribes it (language-independent, so this can happen before the
    person has even picked a target language). Does NOT translate or
    generate any voice — that only happens once the person presses the
    dub button (see /dub).
    """
    sess = load_session(session_id)
    if not sess or not sess.video_path or not sess.video_path.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")

    async def event_stream() -> AsyncGenerator[dict, None]:
        yield sse_log(f"[INFO] Session {session_id} uploaded. Analyzing...")
        async for ev in run_prepare(sess):
            yield ev

    return EventSourceResponse(event_stream(), ping=10)


@app.post("/dub")
async def dub(
    session_id: str = Form(...),
    target_language: str = Form(...),
    voice_name: str = Form("Kore"),
    ending_cta: str = Form(""),
    tts_mode: str = Form("single"),
    tts_models: str = Form(""),
):
    """
    MANUAL phase — only runs when the person presses the dub button after
    picking a target language + voice. No file upload here; the video was
    already uploaded and transcribed by /upload. `ending_cta`, when
    non-empty, adds one extra spoken line at the very end (see
    _maybe_append_ending_cta) — an optional, off-by-default feature.

    `tts_mode` ("single" | "multi") + `tts_models` (comma-separated model
    ids, in the order the person picked them) select which of the four
    catalog TTS models (TTS_MODEL_CATALOG) this dub prefers — see
    _resolve_tts_model_order for exactly how these turn into the ordered
    fallback list. Omitting both keeps the original default behavior
    (every model tried in the module's default order), so older frontends
    that don't send these fields are unaffected.
    """
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured.")
    if not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")

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
    sess.ending_cta = (ending_cta or "").strip()[:500]  # sane upper bound on a CTA line
    sess.tts_mode = (tts_mode or "single").strip().lower()
    sess.tts_models = _resolve_tts_model_order(tts_mode, tts_models)
    save_session(sess)

    async def event_stream() -> AsyncGenerator[dict, None]:
        yield sse_log(
            f"[INFO] Dubbing session {session_id} into {sess.target_language} "
            f"({sess.single_voice}) — voice-model priority: "
            f"{' -> '.join(sess.tts_models)}."
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
