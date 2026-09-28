"""
Ultimate Premium Video Dubbing Platform (Single-Character Master Build)
========================================================================

FastAPI backend, single-character-only, real-time-duration-matching
architecture:

  * Groq (whisper-large-v3, +backup model)  -> heavy audio transcription
  * Groq (openai/gpt-oss-120b, +backup model) -> elite translation
  * Gemini (gemini-3.5-flash, +backup models) -> transcription/translation
    when ENGINE_MODE="gemini" (the default)
  * Gemini native TTS (4-model fallback,
    multi-key rotation)                     -> pure-text voice generation
  (every model above runs inside its own ordered fallback chain — see
  "Backup / never-crash design" below.)

ARCHITECTURE NOTE (job-queue + polling build):
-----------------------------------------------
This build replaces the earlier SSE (Server-Sent Events) streaming model
with an async background-job + polling model, specifically to survive
Render/browser idle-connection timeouts on long dubs:

  1. POST /upload          -> plain JSON. Saves the bytes, creates a
                               session, and immediately QUEUES the
                               "prepare" (extract+probe+transcribe) job in
                               the background. Returns session_id right
                               away — no connection is held open.
  2. GET  /status/{id}    -> polled by the frontend every 3-5s. Returns
                               the job's current status, percent, a
                               human message, and a bounded list of recent
                               log lines (for the UI's live console) —
                               everything the old SSE stream used to push,
                               now pulled on demand instead of held open.
  3. POST /dub              -> ONLY runs once the person has picked a
                               target language + voice and pressed the dub
                               button. Optionally takes an email address.
                               Immediately QUEUES the "dub" job and returns
                               session_id — again, no held connection.
  4. GET  /status/{id}    -> polled again during dubbing the same way.
  5. GET  /download/{id}  -> streams the finished MP4 (HTTP Range
                               supported). Optional ?cleanup=true deletes
                               the session's scratch/output the moment the
                               download finishes streaming.

Everything else — the transcription/translation/TTS pipeline itself,
every fallback chain, the chunking, the silence-trim ladder, the
locked atempo band — is unchanged.

Voice generation strategy
--------------------------
The translated script is sent to Gemini TTS in SEQUENTIAL, bounded
chunks — never one single giant request, and never several requests
fired at once:

  * The script is split into blocks of at most TTS_CHUNK_CHAR_BUDGET
    (1500) characters, breaking only on line boundaries so a sentence is
    never cut mid-word. A short remainder (e.g. 700 characters) simply
    becomes its own final chunk — it is never merged into a bigger one
    and never dropped.
  * This keeps peak RAM low (no single oversized generation held in
    memory at once) and cuts latency, because each chunk is a small,
    fast Gemini TTS call instead of one very long one.
  * Chunking also protects voice quality: very long single-shot TTS
    generations are where Gemini's native voice model is most prone to
    drifting/robotic artifacts near the end of the take. Bounding every
    request to <=1500 characters keeps each individual generation well
    inside the model's comfortable range, so the voice stays natural and
    clear all the way through — including the last chunk.
  * Chunks are generated ONE AT A TIME, in order, and their raw PCM audio
    is concatenated back-to-back (with a short, LOCKED silence pad between
    chunks — see TTS_CHUNK_SILENCE_PAD_MS) into a single continuous
    track before the rest of the pipeline (trim / duration-match / mux)
    ever sees it. Downstream, it is still treated as one seamless voice
    track — nothing else in the pipeline needs to know it was chunked.

Timeline-anchored fitting (how the voice is matched to the video)
-------------------------------------------------------------------
Every chunk of consecutive lines keeps the video timestamp where its first
line begins, and its voice is placed on a silent timeline AT THAT TIME — so
the dub can never drift out of sync and no global speed-up is needed:

  1. PAUSES are capped, never deleted: silence is detected at a very low
     threshold (soft word endings are never mistaken for silence); only pauses
     longer than ~0.9s are shortened to ~500ms; speech is never touched.
  2. If a chunk's voice is LONGER than its window, the first remedy is to have
     the LLM condense that chunk's wording (same meaning, fewer words) and
     re-voice it; only then a small speed-up (soft limit 1.12x, hard ceiling
     1.20x, per chunk, pitch preserved). A chunk that fits is left at natural
     speed (no forced speed-up).
  3. COMPLETENESS guard: if the TTS model skipped text (voice far too short for
     the script), the chunk is regenerated.
  4. MUX: video copied bit-for-bit; the voice is NEVER cut — a longer voice is
     kept whole, a shorter one is padded with silence.

Backup / never-crash design
-----------------------------
* Every Gemini TTS chunk tries TTS_MODELS in order, and for EACH model
  tries every configured GEMINI_API_KEYS entry, ONE AT A TIME, before
  moving to the next model. This is a strict, sequential failover chain:
  the primary (model, key) combination is always tried first; on any
  failure or timeout it fails over IMMEDIATELY to the next key, and once
  every key is exhausted for a model it fails over to the next model —
  with a [WARN] log at every step so it's visible via GET /status. Keys/
  models are never fired concurrently ("spammed") against the same chunk;
  only one request is ever in flight per chunk at a time, which keeps
  provider-side rate limits and quotas healthy across chunks and
  sessions.
* Every processing step is wrapped so a failure ends ONLY that session
  with a clean error status and the session's scratch files are
  deleted — the FastAPI process itself, and every other in-flight
  session, is never affected.
* Speed ratios are always clamped to a safe FFmpeg range (0.25x-4.0x) so
  a pathological mismatch (e.g. wildly different script vs. video length)
  degrades gracefully with a [WARN] instead of crashing the render; the
  product-locked 1.15x-1.30x band is enforced on top of that clamp.
* Transcription and translation ("script retouch") each run their OWN
  ordered model-fallback chain too (GEMINI_TRANSCRIBE_MODELS /
  GEMINI_TRANSLATE_MODELS for the Gemini engine, WHISPER_MODELS /
  TRANSLATION_MODELS for the Groq engine) — independent of, and using the
  same proven sequential-failover pattern as, the TTS fallback above:
  every model in the chain is tried against every configured API key
  before that step is considered failed, so a single model being
  deprecated, rate-limited, or briefly down never takes the whole
  pipeline down with it.

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
GEMINI_TRANSCRIBE_MODELS (optional) Comma-separated ordered fallback chain
                              used for Gemini transcription. Default:
                              "gemini-3.5-flash,gemini-2.5-flash,
                              gemini-3.1-flash-lite" (all three are current,
                              verified-working Gemini text models at the time
                              of this build). The old singular
                              GEMINI_TRANSCRIBE_MODEL still works and is
                              promoted to the front of the chain if set.
GEMINI_TRANSLATE_MODELS  (optional) Same shape/default/back-compat as
                              GEMINI_TRANSCRIBE_MODELS above, used for the
                              translation ("script retouch") step.
WHISPER_MODELS  (optional)   Comma-separated ordered fallback chain for Groq
                              transcription. Default:
                              "whisper-large-v3,whisper-large-v3-turbo".
                              The old singular WHISPER_MODEL still works.
TRANSLATION_MODELS (optional) Comma-separated ordered fallback chain for
                              Groq translation. Default:
                              "openai/gpt-oss-120b,qwen/qwen3.6-27b". The old
                              singular TRANSLATION_MODEL still works.
CORS_ALLOW_ORIGINS (optional) Comma-separated allow-list. Default: "*".
WORK_DIR        (optional)   Scratch dir. Default: ./_sessions
FFMPEG_BIN      (optional)   Default: "ffmpeg"
FFPROBE_BIN     (optional)   Default: "ffprobe"

SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM (all optional)
                              If SMTP_HOST is set, a completion email with
                              the download link is sent to the email
                              address supplied on /dub once the job
                              finishes. If unset, email sending is
                              silently skipped (logged, never fatal).
PUBLIC_API_BASE (optional)   Public base URL used to build the download
                              link inside the completion email (e.g.
                              "https://video-dubing-site.onrender.com").
                              Falls back to a relative path if unset.
JOB_LOG_HISTORY_LIMIT (optional) How many recent log lines GET /status
                              keeps/returns per session. Default 300.
"""

from __future__ import annotations

import asyncio
import collections
import difflib
import json
import os
import re
import shutil
import smtplib
import statistics
import time
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
import uuid
import wave
from dataclasses import asdict, dataclass, field
from email.mime.text import MIMEText
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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

# --- Email notification (optional) ------------------------------------------
# If SMTP_HOST is configured, a completion email with the download link is
# sent automatically once a /dub job finishes muxing. Every send is wrapped
# so a broken/missing SMTP config never fails the dubbing job itself — it
# just logs a [WARN] into the session's own log history (visible via
# GET /status) and moves on.
#
# IMPORTANT (Render): free-tier Render web services BLOCK outbound SMTP
# ports 25/465/587, so plain SMTP can never work there. Three providers
# are supported, tried in this order — the first one that is configured
# wins:
#   1. BREVO_API_KEY   (HTTPS API, port 443 — works on Render free tier;
#                       free plan = 300 emails/day, needs a verified sender)
#   2. RESEND_API_KEY  (HTTPS API, port 443 — works on Render free tier)
#   3. SMTP_HOST ...   (only works where outbound SMTP is allowed, e.g. a
#                       paid Render instance, a VPS, or local dev)
# EMAIL_FROM is the verified sender address used by the two HTTPS providers.
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "").strip()
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587") or "587")
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip() or SMTP_USER
EMAIL_FROM = os.environ.get("EMAIL_FROM", "").strip() or SMTP_FROM
EMAIL_FROM_NAME = os.environ.get("EMAIL_FROM_NAME", "Aurora Dub").strip()
# Public base URL the emailed download link should point at. Falls back to
# Render's own RENDER_EXTERNAL_URL (set automatically on Render), then to a
# relative path.
PUBLIC_API_BASE = (
    os.environ.get("PUBLIC_API_BASE", "").strip()
    or os.environ.get("RENDER_EXTERNAL_URL", "").strip()
).rstrip("/")

# --- Keep-alive / restart-recovery (Render free tier sleeps after ~15 min
# without INBOUND HTTP traffic — background jobs alone do not count, so a
# user going offline mid-dub let the whole instance spin down and killed the
# job). While any session is queued/running/awaiting-dub the server pings
# its own public URL so the instance is not put to sleep.
KEEPALIVE_INTERVAL_SECONDS = int(os.environ.get("KEEPALIVE_INTERVAL_SECONDS", "240") or "240")

# --- Job log history (per session, for GET /status polling) ---------------
JOB_LOG_HISTORY_LIMIT = int(os.environ.get("JOB_LOG_HISTORY_LIMIT", "300") or "300")

# --- Transcription + translation engine toggle -------------------------------
# ENGINE_MODE picks the DEFAULT engine used for transcription + translation:
#   "gemini" (default, primary per product requirement) -> Gemini text model
#             does both transcription (via audio understanding) and
#             translation/localization in one coherent pass.
#   "groq"   -> original Whisper (transcribe) + gpt-oss-120b (translate) path.
#   "hybrid" -> (NEW, recommended default) Groq Whisper does the heavy,
#             accurate TRANSCRIPTION; Groq then produces an initial
#             TRANSLATION pass; Gemini then RETOUCHES that translation for
#             natural fluency (never re-translating from scratch, never
#             restructuring, and never dropping a line — see
#             gemini_retouch_all). This exists because pure single-model
#             transcription/translation was found to be less accurate than
#             combining Groq's dedicated ASR strength with Gemini's
#             language polish.
# The frontend can override this per-request (?engine= on /prepare, and an
# `engine` form field on /dub) without restarting the server, so a person
# can flip the toggle live if one provider is degraded.
ENGINE_MODE_DEFAULT = os.environ.get("ENGINE_MODE", "hybrid").strip().lower()
if ENGINE_MODE_DEFAULT not in ("gemini", "groq", "hybrid"):
    ENGINE_MODE_DEFAULT = "hybrid"

def _parse_model_chain(singular_env: str, plural_env: str, default_chain: List[str]) -> List[str]:
    """
    Build an ordered, de-duplicated model fallback chain for a
    transcription/translation step. Reads the plural, comma-separated env
    var first (e.g. GEMINI_TRANSCRIBE_MODELS); falls back to the built-in
    default chain if unset. The old singular env var (e.g.
    GEMINI_TRANSCRIBE_MODEL) still works for back-compat: if set, it is
    promoted to the FRONT of the chain so it stays the top-priority model,
    exactly like before this change for anyone who already configured it.
    """
    chain = _parse_key_list(os.environ.get(plural_env, "")) or list(default_chain)
    legacy = os.environ.get(singular_env, "").strip()
    if legacy:
        chain = [legacy] + [m for m in chain if m != legacy]
    return chain


# --- Gemini: transcription + translation ("script retouch") model chains ---
GEMINI_TRANSCRIBE_MODELS: List[str] = _parse_model_chain(
    "GEMINI_TRANSCRIBE_MODEL", "GEMINI_TRANSCRIBE_MODELS",
    ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"],
)
GEMINI_TRANSLATE_MODELS: List[str] = _parse_model_chain(
    "GEMINI_TRANSLATE_MODEL", "GEMINI_TRANSLATE_MODELS",
    ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"],
)
GEMINI_TRANSCRIBE_MODEL = GEMINI_TRANSCRIBE_MODELS[0]
GEMINI_TRANSLATE_MODEL = GEMINI_TRANSLATE_MODELS[0]

# --- Groq: fallback/alternate transcription + translation engine ------------
WHISPER_MODELS: List[str] = _parse_model_chain(
    "WHISPER_MODEL", "WHISPER_MODELS",
    ["whisper-large-v3", "whisper-large-v3-turbo"],
)
WHISPER_MODEL = WHISPER_MODELS[0]

TRANSLATION_MODELS: List[str] = _parse_model_chain(
    "TRANSLATION_MODEL", "TRANSLATION_MODELS",
    ["openai/gpt-oss-120b", "qwen/qwen3.6-27b"],
)
TRANSLATION_MODEL = TRANSLATION_MODELS[0]

# --- Gemini: pure-text native TTS engine, with ordered auto-fallback --------
TTS_MODEL_CATALOG: Dict[str, str] = {
    "gemini-2.5-flash-preview-tts": "Gemini 2.5 Flash Preview TTS",
    "gemini-3.1-flash-tts-preview": "Gemini 3.1 Flash TTS Preview",
    "gemini-3.8-flash-tts": "Gemini 3.8 Flash TTS",
    "gemini-3.8-flash-tts-lite": "Gemini 3.8 Flash Lite TTS",
}
TTS_MODELS: List[str] = list(TTS_MODEL_CATALOG.keys())


def parse_selected_tts_models(raw: str) -> List[str]:
    if not raw:
        return list(TTS_MODELS)
    chosen: List[str] = []
    for part in raw.split(","):
        mid = part.strip()
        if mid and mid in TTS_MODEL_CATALOG and mid not in chosen:
            chosen.append(mid)
    return chosen or list(TTS_MODELS)


TTS_CALL_TIMEOUT_SECONDS = 150

# --- TTS chunking (RAM protection + voice-quality guard) --------------------
TTS_CHUNK_CHAR_BUDGET = int(os.environ.get("TTS_CHUNK_CHAR_BUDGET", "900") or "900")
# (<=1500 by product rule; lowering it via env makes each generation shorter,
#  which reduces the model's tendency to rush near the end of a long take.)
TTS_CHUNK_CHAR_BUDGET = max(300, min(1500, TTS_CHUNK_CHAR_BUDGET))
# LOCKED silence pad between consecutively generated TTS chunks. Product
# requirement: this must stay locked around 500ms — never arbitrarily lower
# (which would risk splicing chunks together so tightly it sounds like one
# word ran into the next) and never arbitrarily higher (which would
# introduce an audible dead-air gap that breaks the video's sync/pacing).
TTS_CHUNK_SILENCE_PAD_MS = 500

# --- Timeline-anchored voice fitting (replaces the old "one global speed-up") --
# OLD design: one continuous voice track, silence removed wholesale, then ONE
# global speed-up locked to 1.15-1.30x, then the mux cut the audio at the video
# end. That is why sentences vanished (mux truncation + pauses deleted) and the
# voice sounded rushed (a forced speed-up even when none was needed).
# NEW design (how professional dubbing tools work):
#   * the script is split into chunks of consecutive lines that keep their
#     ORIGINAL video timestamps;
#   * every chunk is voiced, then placed on a silent timeline AT ITS OWN START
#     TIME, so the dub can never drift out of sync;
#   * if a chunk's voice is longer than its time window, the FIRST remedy is to
#     have the LLM condense that chunk's wording (same meaning, fewer words) and
#     re-voice it — speeding up is the last, small resort (never above
#     FIT_SPEED_HARD_MAX);
#   * pauses are only CAPPED (long ones shortened to ~500ms), never deleted, and
#     speech is never touched; nothing is ever cut off at the end.
TTS_CHUNK_MAX_SOURCE_SECONDS = float(os.environ.get("TTS_CHUNK_MAX_SOURCE_SECONDS", "30") or "30")
FIT_SPEED_SOFT_MAX = float(os.environ.get("FIT_SPEED_SOFT_MAX", "1.12") or "1.12")  # condense above this
FIT_SPEED_HARD_MAX = float(os.environ.get("FIT_SPEED_HARD_MAX", "1.20") or "1.20")  # never faster than this
CONDENSE_ATTEMPTS = int(os.environ.get("CONDENSE_ATTEMPTS", "2") or "2")
MIN_CHUNK_GAP_SECONDS = 0.25          # breathing room if a chunk overflows into the next window
PAUSE_CAP_SECONDS = TTS_CHUNK_SILENCE_PAD_MS / 1000.0   # long pauses are capped to ~500ms
PAUSE_CAP_MIN_PAUSE_SECONDS = 0.9     # only pauses longer than this are shortened
PAUSE_DETECT_DB = -45.0               # deliberately low so quiet word endings are never seen as "silence"
TTS_TRUNCATION_RATIO = 0.5            # voice shorter than half of what the text needs => text was skipped
# Natural speaking speed (characters per second incl. spaces) per target language,
# used to give the translator a per-line length budget (max_chars).
TARGET_CPS: Dict[str, float] = {
    "english": 15, "spanish": 16, "french": 15, "german": 14, "italian": 15,
    "portuguese": 15, "hindi": 13, "bengali": 13, "arabic": 13, "japanese": 8,
    "korean": 9, "chinese": 5.5, "russian": 14, "turkish": 14, "dutch": 15,
    "indonesian": 15, "vietnamese": 14, "thai": 12, "polish": 14, "ukrainian": 14,
}

GEMINI_VOICE_NAMES: List[str] = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

TTS_SAMPLE_RATE = 24000
TTS_SAMPLE_WIDTH = 2
TTS_CHANNELS = 1

UPLOAD_CHUNK = 1024 * 1024  # 1 MiB streaming chunk (RAM-safe)

# --- Translation: chunked + drop-proof ------------------------------------ #
TRANSLATE_CHUNK_CHAR_BUDGET = 1500
TRANSLATE_CHUNK_SIZE = 12
TRANSLATE_CHUNK_RETRIES = 2

# --- Hook handling (opening line(s) ONLY) ----------------------------------
HOOK_CONTEXT_CHAR_LIMIT = 6000
SELF_HEAL_RETRIES = 2
SELF_HEAL_BACKOFF_SECONDS = 3.0

# --- Silence-trim ladder ------------------------------------------------- #
# Product requirement: silence removal is STRICTLY locked between 400ms and
# 500ms — never lower, never higher — and it only ever removes measured
# SILENCE (peak-level detection below SILENCE_DB_THRESHOLD), never speech,
# so it can never mistakenly cut into an actual word. The ladder tries the
# least-aggressive rung (500ms) first and only steps down to the locked
# floor (400ms) if the video is short and 500ms alone doesn't get speech
# comfortably close to the video length.
SILENCE_TRIM_LADDER_MS: List[int] = [500, 400]
SILENCE_TRIM_FLOOR_MS: int = SILENCE_TRIM_LADDER_MS[-1]
SILENCE_DB_THRESHOLD = -32.0
COMFORTABLE_MAX_RATIO = 1.2

HARD_SPEED_MIN = 0.25
HARD_SPEED_MAX = 4.0
SOFT_SPEED_MAX = 1.3

# Speed limits now apply PER CHUNK and only when really needed (no forced
# speed-up): 1.0 = natural speed, FIT_SPEED_HARD_MAX = absolute ceiling.
ATEMPO_LOCK_MIN = 1.0
ATEMPO_LOCK_MAX = FIT_SPEED_HARD_MAX

# --- Voice "energy consistency" filters (FFmpeg) ------------------------------
# Problem: Gemini TTS starts each take well but tends to RUSH and lose energy
# near the END of a chunk. A prompt can't fully fix that, so it is corrected
# deterministically in audio, per chunk, right after each chunk is generated
# (before the chunks are stitched together):
#   1. tail-pace fix  -> if the last part of the chunk (after its last natural
#                        pause) is rushed, that region is slowed slightly
#                        (atempo, pitch preserved) — see _slow_chunk_tail().
#   2. energy polish  -> highpass (rumble) + 2x dynaudnorm (evens loudness
#                        from first word to last, lifts a fading tail; the
#                        low threshold keeps quiet pauses/breaths from being
#                        boosted) + gentle compressor + presence EQ (clarity/
#                        "energetic" tone) + limiter (never shouts/clips).
TTS_POLISH_ENABLED = os.environ.get("TTS_POLISH_ENABLED", "1").strip() not in ("0", "false", "no")
TTS_ENERGY_POLISH_FILTER = (
    "highpass=f=60,"
    "dynaudnorm=f=100:g=5:p=0.9:m=10:t=0.005,"
    "dynaudnorm=f=100:g=5:p=0.9:m=10:t=0.005,"
    "acompressor=threshold=0.1:ratio=3:attack=5:release=120:makeup=1,"
    "equalizer=f=140:t=q:w=1:g=1.2,"
    "equalizer=f=3200:t=q:w=1:g=2,"
    "alimiter=limit=0.9:level=disabled"
)
# Tail-pace correction (set TTS_TAIL_SLOWDOWN=1.0 to disable).
TTS_TAIL_SLOWDOWN = float(os.environ.get("TTS_TAIL_SLOWDOWN", "0.94") or "0.94")
TTS_TAIL_REGION_FRACTION = float(os.environ.get("TTS_TAIL_REGION_FRACTION", "0.35") or "0.35")
TTS_TAIL_MIN_PAUSE_SECONDS = 0.12
TTS_TAIL_PAUSE_DB = -35.0

# Final whole-track pass (after the locked 1.15-1.30x atempo): even out the
# loudness envelope one more time, add presence, then set ONE consistent
# broadcast-style loudness (-16 LUFS) so the dub is energetic but never
# shouting, from first second to last.
TTS_LOUDNESS_NORMALIZE_FILTER = (
    "equalizer=f=3200:t=q:w=1:g=1.5,"
    "dynaudnorm=f=250:g=15:p=0.95:m=8:t=0.005,"
    "loudnorm=I=-16:TP=-1.5:LRA=9,"
    "alimiter=limit=0.95:level=disabled"
)

SEQUENTIAL_PROCESSING = True

WORK_DIR.mkdir(parents=True, exist_ok=True)

_gemini_clients: Dict[str, "genai.Client"] = {}
_groq_clients: Dict[str, "Groq"] = {}


def get_client(api_key: str) -> "genai.Client":
    if not api_key:
        raise RuntimeError("Empty Gemini API key.")
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]


def get_groq_client(api_key: str) -> "Groq":
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
    raw_segments: List[Segment] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    video_duration: float = 0.0
    prepared: bool = False
    engine: str = ENGINE_MODE_DEFAULT
    tts_models: List[str] = field(default_factory=lambda: list(TTS_MODELS))
    # status flows: uploaded -> queued_prepare -> preparing -> prepared ->
    # queued_dub -> dubbing -> done  (or -> error at any stage). A client
    # polls GET /status/{id} to follow this instead of holding a
    # connection open — the actual work runs in the background worker
    # independent of any one HTTP request.
    status: str = "uploaded"
    job_percent: int = 0
    job_message: str = ""
    job_error: Optional[str] = None
    # Bounded recent-log buffer, newest last — what GET /status returns for
    # the frontend's live console. Kept small and capped so RAM/disk never
    # grow unbounded across a long-running job.
    job_logs: List[str] = field(default_factory=list)
    notify_email: Optional[str] = None
    email_sent: bool = False


SESSIONS: Dict[str, Session] = {}
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", str(60 * 60)) or 3600)  # abandoned/unfinished
DONE_SESSION_TTL_SECONDS = int(os.environ.get("DONE_SESSION_TTL_SECONDS", str(6 * 60 * 60)) or 21600)  # finished dubs stay downloadable longer
_ACTIVE_SESSION_ID: Optional[str] = None  # session the worker is processing right now


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
        "status": sess.status,
        "job_percent": sess.job_percent,
        "job_message": sess.job_message,
        "job_error": sess.job_error,
        "job_logs": sess.job_logs[-JOB_LOG_HISTORY_LIMIT:],
        "notify_email": sess.notify_email,
        "email_sent": sess.email_sent,
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
        status=data.get("status", "uploaded"),
        job_percent=data.get("job_percent", 0),
        job_message=data.get("job_message", ""),
        job_error=data.get("job_error"),
        job_logs=data.get("job_logs", []),
        notify_email=data.get("notify_email"),
        email_sent=data.get("email_sent", False),
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
    """Delete expired sessions (disk + RAM). Finished dubs get the longer
    DONE_SESSION_TTL_SECONDS so the download link/email stays usable; anything
    unfinished uses SESSION_TTL_SECONDS. Never touches a session that is
    running or waiting in the queue."""
    now = time.time()
    if not WORK_DIR.exists():
        return
    queued_ids = {sid for _, sid in JOB_QUEUE}
    for child in WORK_DIR.iterdir():
        if not child.is_dir():
            continue
        if child.name == _ACTIVE_SESSION_ID or child.name in queued_ids:
            continue
        try:
            age = now - child.stat().st_mtime
        except OSError:
            continue
        ttl = DONE_SESSION_TTL_SECONDS if (child / "dubbed_output.mp4").exists() else SESSION_TTL_SECONDS
        if age > ttl:
            _destroy_session(child.name)


# --------------------------------------------------------------------------- #
# Log/progress helpers (the pipeline generators below still yield these
# small event dicts internally — the outer layer now folds them into
# Session.job_logs/job_percent for polling instead of streaming them over
# SSE).
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


def _append_job_log(sess: Session, line: str) -> None:
    sess.job_logs.append(line)
    if len(sess.job_logs) > JOB_LOG_HISTORY_LIMIT:
        sess.job_logs = sess.job_logs[-JOB_LOG_HISTORY_LIMIT:]


async def _apply_event_to_session(sess: Session, ev: dict) -> None:
    """
    Fold one pipeline event (log/progress/error/done) into the session's
    polled state: job_percent/job_message stay in sync, and every event is
    also rendered into a plain log line appended to job_logs so GET
    /status can hand the frontend a live-console feed without ever holding
    a connection open. Persisted immediately so a polling client (or a
    server restart mid-job) always sees near-real-time state.
    """
    kind = ev.get("event")
    if kind == "progress":
        try:
            data = json.loads(ev["data"])
            sess.job_percent = int(data.get("percent", sess.job_percent))
            msg = str(data.get("message", sess.job_message))
            sess.job_message = msg
            if msg:
                _append_job_log(sess, f"[PROGRESS {sess.job_percent}%] {msg}")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    elif kind == "log":
        _append_job_log(sess, str(ev.get("data", "")))
    elif kind == "error":
        sess.job_error = str(ev.get("data", ""))
        _append_job_log(sess, sess.job_error)
    elif kind == "done":
        _append_job_log(sess, "[SUCCESS] Job finished.")
    if sess.dir.exists():
        save_session(sess)


# --------------------------------------------------------------------------- #
# Email notification (optional — no-op if SMTP_HOST isn't configured)
# --------------------------------------------------------------------------- #

def _email_provider() -> str:
    """Which email transport is configured ('' if none)."""
    if BREVO_API_KEY and EMAIL_FROM:
        return "brevo"
    if RESEND_API_KEY and EMAIL_FROM:
        return "resend"
    if SMTP_HOST:
        return "smtp"
    return ""


def _http_json_post(url: str, headers: Dict[str, str], payload: dict, timeout: int = 25) -> None:
    """POST JSON over HTTPS (port 443 — never blocked by Render's free tier).
    Raises RuntimeError with the provider's error text on any non-2xx."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "AuroraDub/5.1")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {exc.code} from email provider: {detail}") from exc


def _send_completion_email_sync(to_email: str, session_id: str, download_url: str,
                                 target_language: str) -> str:
    """
    Synchronous send (run via asyncio.to_thread so it never blocks the event
    loop). Returns the provider name used. Raises on failure — the caller
    catches it and logs a [WARN] instead of failing the job.
    """
    provider = _email_provider()
    if not provider:
        raise RuntimeError(
            "No email provider configured on the server. Render's free tier "
            "blocks SMTP, so set BREVO_API_KEY + EMAIL_FROM (or RESEND_API_KEY "
            "+ EMAIL_FROM) in the Render environment variables."
        )
    subject = "Your Aurora Dub video is ready"
    text_body = (
        f"Good news — your dubbed video (into {target_language}) has finished "
        f"processing.\n\nDownload it here:\n{download_url}\n\n"
        f"Session: {session_id}\n\nThis link stays available for a limited "
        "time, so download it soon."
    )
    html_body = (
        f"<p>Good news — your dubbed video (into <b>{target_language}</b>) has "
        f"finished processing.</p><p><a href=\"{download_url}\">Download your "
        f"dubbed video</a></p><p style=\"color:#888\">Session: {session_id}<br>"
        "This link stays available for a limited time, so download it soon.</p>"
    )

    if provider == "brevo":
        _http_json_post(
            "https://api.brevo.com/v3/smtp/email",
            {"api-key": BREVO_API_KEY},
            {
                "sender": {"name": EMAIL_FROM_NAME, "email": EMAIL_FROM},
                "to": [{"email": to_email}],
                "subject": subject,
                "htmlContent": html_body,
                "textContent": text_body,
            },
        )
    elif provider == "resend":
        _http_json_post(
            "https://api.resend.com/emails",
            {"Authorization": f"Bearer {RESEND_API_KEY}"},
            {
                "from": f"{EMAIL_FROM_NAME} <{EMAIL_FROM}>",
                "to": [to_email],
                "subject": subject,
                "html": html_body,
                "text": text_body,
            },
        )
    else:  # smtp
        msg = MIMEText(text_body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM or "no-reply@aurora-dub.local"
        msg["To"] = to_email
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(msg["From"], [to_email], msg.as_string())
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
                server.starttls()
                if SMTP_USER and SMTP_PASSWORD:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                server.sendmail(msg["From"], [to_email], msg.as_string())
    return provider


async def _maybe_send_completion_email(sess: Session) -> None:
    """
    Fire the completion email for a just-finished dub, if the person
    supplied an address on /dub. Never raises — any failure (SMTP not
    configured, bad address, network issue) is logged as a [WARN] into the
    session's own polled log history and the job is still considered
    successful, since the video itself already finished and is
    downloadable regardless of whether the email goes out.
    """
    if not sess.notify_email or sess.email_sent:
        return
    download_url = f"{PUBLIC_API_BASE}/download/{sess.session_id}" if PUBLIC_API_BASE \
        else f"/download/{sess.session_id}"
    try:
        used = await asyncio.to_thread(
            _send_completion_email_sync, sess.notify_email, sess.session_id,
            download_url, sess.target_language,
        )
        sess.email_sent = True
        _append_job_log(sess, f"[SUCCESS] Completion email sent to {sess.notify_email} (via {used}).")
    except Exception as exc:  # noqa: BLE001
        _append_job_log(sess, f"[WARN] Could not send completion email: {type(exc).__name__}: {exc}")
    if sess.dir.exists():
        save_session(sess)


# --------------------------------------------------------------------------- #
# Background job queue
# --------------------------------------------------------------------------- #
# Processing (prepare/dub) runs in a SINGLE persistent background worker task,
# completely decoupled from any HTTP request. This means:
#   * A person closing the tab, losing connection, or their browser/Render's
#     load balancer timing out an idle request NEVER stops a job that's
#     already running or queued — nothing here depends on a live connection
#     at all anymore (no SSE, no long-held request); the client simply polls
#     GET /status/{id} whenever it wants an update.
#   * Uploading several videos queues several jobs; the single worker
#     processes them strictly ONE AT A TIME, in the order they arrived —
#     never two heavy ffmpeg/TTS jobs running at once, which is exactly the
#     RAM-safety rule the rest of this pipeline already follows internally.
#   * A finished session's result (and its /download/{id} link) is left
#     alone until the person explicitly deletes it (DELETE /session/{id}),
#     downloads it with ?cleanup=true, or the 1h TTL sweep reclaims an
#     abandoned one.
JOB_QUEUE: "collections.deque[Tuple[str, str]]" = collections.deque()
_JOB_QUEUE_EVENT = asyncio.Event()
_worker_task: Optional[asyncio.Task] = None


def _queue_position(session_id: str) -> int:
    for i, (_, sid) in enumerate(JOB_QUEUE):
        if sid == session_id:
            return i + 1
    return 0


def _ensure_worker_started() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.get_running_loop().create_task(_job_worker())


def _enqueue_job(kind: str, session_id: str) -> None:
    JOB_QUEUE.append((kind, session_id))
    _JOB_QUEUE_EVENT.set()
    _ensure_worker_started()


async def _process_job(kind: str, session_id: str) -> None:
    """Run exactly one queued job (prepare or dub) to completion, folding
    every event it produces into the session's polled state as it goes.
    Runs entirely inside the background worker — nothing here depends on
    any HTTP request being open."""
    sess = load_session(session_id)
    if sess is None or not sess.video_path or not sess.video_path.exists():
        return

    sess.status = "preparing" if kind == "prepare" else "dubbing"
    save_session(sess)
    await _apply_event_to_session(sess, sse_log(
        f"[INFO] Starting {kind} for session {session_id} (engine='{sess.engine}')..."
    ))

    gen = run_prepare(sess) if kind == "prepare" else run_dub(sess)
    try:
        async for ev in gen:
            await _apply_event_to_session(sess, ev)
    except Exception as exc:  # noqa: BLE001 — belt-and-braces; run_prepare/run_dub
        # already catch and report their own errors internally. This only
        # fires on a truly unexpected crash outside that handling.
        if sess.dir.exists():
            sess.status = "error"
            sess.job_error = str(exc)
            _append_job_log(sess, f"[ERROR] Unexpected worker failure: {type(exc).__name__}: {exc}")
            save_session(sess)
        return

    if not sess.dir.exists():
        # run_prepare/run_dub already hit a fatal error, logged it, and
        # destroyed the session's scratch files — nothing further to do.
        return

    if kind == "prepare":
        sess.status = "prepared" if sess.prepared else "error"
        save_session(sess)
    else:
        finished_ok = (sess.dir / "dubbed_output.mp4").exists()
        sess.status = "done" if finished_ok else "error"
        save_session(sess)
        if finished_ok:
            await _maybe_send_completion_email(sess)
            # Free every remaining scratch file now that the final MP4 is
            # produced — only the finished output + session.json stay on
            # disk, keeping this session's disk/RAM footprint minimal
            # while it waits to be downloaded or TTL-swept.
            for child in sess.dir.iterdir():
                if child.name not in ("dubbed_output.mp4", "session.json"):
                    try:
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                    except OSError:
                        pass


async def _job_worker() -> None:
    """The single persistent background worker: pulls jobs off JOB_QUEUE
    strictly one at a time, forever. Started lazily on first enqueue and
    stays alive for the lifetime of the process."""
    while True:
        if not JOB_QUEUE:
            _JOB_QUEUE_EVENT.clear()
            await _JOB_QUEUE_EVENT.wait()
            continue
        kind, session_id = JOB_QUEUE.popleft()
        global _ACTIVE_SESSION_ID
        _ACTIVE_SESSION_ID = session_id
        try:
            await _process_job(kind, session_id)
        except Exception:  # noqa: BLE001 — never let one bad job kill the worker loop
            sess = load_session(session_id)
            if sess is not None and sess.dir.exists():
                sess.status = "error"
                save_session(sess)
        finally:
            _ACTIVE_SESSION_ID = None


# --------------------------------------------------------------------------- #
# Restart recovery + keep-alive (Render free tier sleeps / restarts)
# --------------------------------------------------------------------------- #

def _recover_sessions_on_startup() -> int:
    """
    If the process (re)starts while sessions from a previous run are still on
    disk (e.g. a crash/redeploy with a persistent disk, or a quick restart
    before the container's disk was recycled), pick up every session that was
    mid-job and re-queue it so the work continues by itself — the person
    doesn't have to re-upload. Returns how many jobs were re-queued.
    """
    if not WORK_DIR.exists():
        return 0
    resumed = 0
    children = sorted(
        (c for c in WORK_DIR.iterdir() if c.is_dir() and _session_file(c).exists()),
        key=lambda c: _session_file(c).stat().st_mtime,
    )
    for child in children:
        sess = load_session(child.name)
        if sess is None or not sess.video_path or not sess.video_path.exists():
            continue
        if sess.status in ("queued_prepare", "preparing"):
            sess.status = "queued_prepare"
            _append_job_log(sess, "[SYS] Server restarted — resuming analysis automatically.")
            save_session(sess)
            _enqueue_job("prepare", sess.session_id)
            resumed += 1
        elif sess.status in ("queued_dub", "dubbing") and sess.prepared and sess.raw_segments:
            sess.status = "queued_dub"
            _append_job_log(sess, "[SYS] Server restarted — resuming dubbing automatically.")
            save_session(sess)
            _enqueue_job("dub", sess.session_id)
            resumed += 1
    return resumed


def _has_pending_work() -> bool:
    """True while the instance should be kept awake: something is running,
    queued, or a prepared video is waiting for the person to press Dub."""
    if JOB_QUEUE or _ACTIVE_SESSION_ID:
        return True
    for sess in list(SESSIONS.values()):
        if sess.status in ("queued_prepare", "preparing", "queued_dub", "dubbing", "prepared"):
            return True
    return False


def _self_ping(url: str) -> None:
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "AuroraDub-KeepAlive/1.0")
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read(64)


async def _keepalive_loop() -> None:
    """Every KEEPALIVE_INTERVAL_SECONDS: sweep expired sessions, and — only
    while there is real work — hit our own public /ping so Render counts it as
    inbound traffic and does not spin the instance down mid-job."""
    while True:
        await asyncio.sleep(max(60, KEEPALIVE_INTERVAL_SECONDS))
        try:
            await asyncio.to_thread(_sweep_stale_sessions)
            if PUBLIC_API_BASE and _has_pending_work():
                await asyncio.to_thread(_self_ping, f"{PUBLIC_API_BASE}/ping")
        except Exception:  # noqa: BLE001 — a failed ping must never crash the loop
            pass


# --------------------------------------------------------------------------- #
# FFmpeg / FFprobe helpers (async, disk-based)
# --------------------------------------------------------------------------- #

def _lower_priority() -> None:
    """Run ffmpeg at lower CPU priority so the API (/status polling every
    second) stays responsive on Render's tiny CPU share."""
    try:
        os.nice(10)
    except Exception:  # noqa: BLE001
        pass


_SUBPROC_KW = {"preexec_fn": _lower_priority} if hasattr(os, "nice") else {}


async def _run(cmd: List[str]) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_SUBPROC_KW,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", "ignore").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd[:2])}...): {err[:800]}")
    return (stdout or b"").decode("utf-8", "ignore")


async def _run_stderr(cmd: List[str]) -> str:
    """Like _run() but returns STDERR (where ffmpeg's silencedetect prints)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **_SUBPROC_KW,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", "ignore").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd[:2])}...): {err[-500:]}")
    return (stderr or b"").decode("utf-8", "ignore")


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
    atempo (pitch-preserving) is the sole time-stretch engine (see module
    docstring). `dynaudnorm` (TTS_LOUDNESS_NORMALIZE_FILTER) is always
    chained on afterward — this is the FFmpeg-side safety net that evens
    out any quiet/fading stretch anywhere in the track (start, middle, or
    end) regardless of how well the TTS model followed the "stay
    high-energy, no fading" delivery instruction.
    """
    return f"{_atempo_chain(ratio)},{TTS_LOUDNESS_NORMALIZE_FILTER}"


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
                                 min_silence_seconds: float = PAUSE_CAP_MIN_PAUSE_SECONDS,
                                 threshold_db: float = PAUSE_DETECT_DB) -> Tuple[float, float]:
    """
    SAFE pause handling (replaces the old blanket `silenceremove`, which deleted
    every pause completely — gluing words/sentences together — and could eat
    quiet word endings). Here:
      * silence is DETECTED first (very low threshold, so soft speech is never
        classified as silence);
      * a pause longer than `min_silence_seconds` is SHORTENED to
        PAUSE_CAP_SECONDS (~500ms) — never deleted; shorter, natural pauses are
        left exactly as the voice produced them;
      * leading/trailing silence is trimmed but a small natural margin is kept;
      * audio between pauses (the speech) is copied through untouched.
    Returns (duration_before, duration_after).
    """
    dur = await probe_duration(src_audio)
    log = await _run_stderr([
        FFMPEG_BIN, "-hide_banner", "-i", str(src_audio),
        "-af", f"silencedetect=noise={threshold_db:.0f}dB:d=0.15",
        "-f", "null", "-",
    ])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log)]
    keep: List[Tuple[float, float]] = []
    cur: Optional[float] = 0.0
    for i, st in enumerate(starts):
        en = ends[i] if i < len(ends) else dur
        st, en = max(0.0, st), min(dur, en)
        if st <= 0.02:                      # leading silence: keep a small margin
            cur = max(0.0, en - 0.08)
            continue
        if en >= dur - 0.02:                # trailing silence: keep a small margin
            keep.append((cur, min(dur, st + 0.12)))
            cur = None
            break
        if (en - st) >= min_silence_seconds:  # long pause: shorten, never delete
            keep.append((cur, st + PAUSE_CAP_SECONDS))
            cur = en
    if cur is not None:
        keep.append((cur, dur))
    keep = [(a, b) for a, b in keep if b - a > 0.02]
    if not keep or (len(keep) == 1 and keep[0][0] <= 0.001 and keep[0][1] >= dur - 0.001):
        await _encode_wav(["-i", str(src_audio)], out_audio)
        return dur, dur
    keep = keep[:200]  # keep the filter graph bounded
    parts = [f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[p{i}]" for i, (a, b) in enumerate(keep)]
    graph = ";".join(parts) + ";" + "".join(f"[p{i}]" for i in range(len(keep))) + \
        f"concat=n={len(keep)}:v=0:a=1[o]"
    await _run([
        FFMPEG_BIN, "-y", "-i", str(src_audio), "-filter_complex", graph, "-map", "[o]",
        "-ar", str(TTS_SAMPLE_RATE), "-ac", str(TTS_CHANNELS), "-c:a", "pcm_s16le", str(out_audio),
    ])
    return dur, await probe_duration(out_audio)


async def time_stretch_to_duration(src_audio: Path, target_seconds: float,
                                   out_audio: Path, use_rubberband: bool) -> Dict[str, float]:
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
    Replace the video's audio without EVER cutting speech. (The old
    `apad` + `-shortest` combination ended the file at the video's end, which
    silently deleted the last sentences whenever the voice ran longer than the
    video.) `apad=whole_dur` only pads a SHORTER voice with silence up to the
    video length; a longer voice is kept in full. Video stream is copied
    bit-for-bit (zero quality loss).
    """
    vdur = await probe_duration(video_path)
    await _run([
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-af", f"apad=whole_dur={vdur:.3f}",
        "-movflags", "+faststart",
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
# Repeated-segment de-duplication (fixes ASR "stutter"/hallucination, where
# the model outputs the same line — or a near-identical variant — several
# times in a row, most commonly around silence, music, or noisy audio).
# --------------------------------------------------------------------------- #

_DEDUPE_NORMALIZE_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_for_dedupe(text: str) -> str:
    """Lowercase + strip punctuation/extra whitespace so 'Hello!!' and
    'hello' compare as the same line for repeat detection."""
    t = _DEDUPE_NORMALIZE_RE.sub("", (text or "").strip().lower())
    return re.sub(r"\s+", " ", t).strip()


def _dedupe_repeated_segments(
    segments: List[Dict[str, object]],
    similarity_threshold: float = 0.90,
) -> Tuple[List[Dict[str, object]], int]:
    """
    Collapse consecutive segments that are the same line repeated (or a
    near-identical restatement of it) — the classic Whisper/Gemini
    hallucination failure mode on silence, background music, or noisy
    audio, where the same sentence gets transcribed 3-10x in a row. Only
    ADJACENT segments are ever merged (a line that legitimately recurs
    much later in the video, e.g. a repeated catchphrase, is left alone).

    When a run of near-duplicate segments is found, only the FIRST one's
    text is kept (it's usually the cleanest read), but the merged
    segment's end-time is stretched to cover the whole run, so no audio
    time is lost — this keeps the transcript's total duration coverage
    intact for the downstream duration-matching step. Returns
    (deduped_segments, number_of_segments_removed).
    """
    if not segments:
        return segments, 0
    out: List[Dict[str, object]] = [dict(segments[0])]
    removed = 0
    for seg in segments[1:]:
        prev = out[-1]
        prev_norm = _normalize_for_dedupe(str(prev.get("text", "")))
        cur_norm = _normalize_for_dedupe(str(seg.get("text", "")))
        is_repeat = False
        if prev_norm and cur_norm:
            if prev_norm == cur_norm:
                is_repeat = True
            else:
                ratio = difflib.SequenceMatcher(None, prev_norm, cur_norm).ratio()
                if ratio >= similarity_threshold:
                    is_repeat = True
        if is_repeat:
            try:
                prev["end"] = max(float(prev.get("end", 0.0)), float(seg.get("end", 0.0)))
            except (TypeError, ValueError):
                pass
            removed += 1
            continue
        out.append(dict(seg))
    return out, removed


# --------------------------------------------------------------------------- #
# Groq helpers: whisper-large-v3 transcription + openai/gpt-oss-120b
# translation
# --------------------------------------------------------------------------- #

async def groq_transcribe(audio_path: Path) -> dict:
    if not GROQ_API_KEYS:
        raise RuntimeError("No GROQ_API_KEY(s) configured.")

    def _do(model_name: str, api_key: str):
        client = get_groq_client(api_key)
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        resp = client.audio.transcriptions.create(
            file=(audio_path.name, audio_bytes),
            model=model_name,
            response_format="verbose_json",
            temperature=0.0,
        )
        return resp

    resp = None
    model_used: Optional[str] = None
    last_err: Optional[Exception] = None
    for model_name in WHISPER_MODELS:
        for key in GROQ_API_KEYS:
            try:
                resp = await asyncio.to_thread(_do, model_name, key)
                model_used = model_name
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if resp is not None:
            break
    if resp is None:
        raise RuntimeError(
            f"Groq Whisper transcription failed on all {len(WHISPER_MODELS)} "
            f"model(s) x {len(GROQ_API_KEYS)} configured key(s): {last_err}"
        ) from last_err

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

    if not segments:
        whole_text = str(data.get("text") or "").strip()
        if whole_text:
            segments = [{"start": 0.0, "end": 0.0, "text": whole_text}]

    if not segments:
        raise RuntimeError("Groq Whisper returned no usable transcript segments.")

    return {"language": language, "segments": segments, "asr_model": model_used}


def _translation_style_notes(target_language: str) -> str:
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


def _hook_handling_section(target_language: str, full_script_context: str) -> str:
    if not full_script_context:
        return ""
    context = full_script_context[:HOOK_CONTEXT_CHAR_LIMIT]
    if len(full_script_context) > HOOK_CONTEXT_CHAR_LIMIT:
        context += " …[script truncated for length]"
    return (
        "HOOK HANDLING (applies ONLY to the very first segment(s) of the "
        "ENTIRE video — this chunk's opening line(s), however many there "
        "are; every segment after them still follows TASK 1 / NO "
        "RESTRUCTURING above exactly as written):\n"
        "First, using the FULL SOURCE SCRIPT given below purely as "
        "read-only context, decide whether those opening line(s) are a "
        "deliberate attention-grabbing HOOK (a teaser line, a bold claim, a "
        "provocative question meant to hook the viewer BEFORE the actual "
        "scene/action starts) — as opposed to the video simply starting "
        "directly into the scene/action with no separate attention-grab.\n"
        "- If the opening IS a hook (whether it is 1, 2, or more lines): "
        "discard its literal wording entirely and write a NEW, punchy, "
        f"high-energy ACTION HOOK in {target_language}, informed by the "
        "whole script's content below, that grabs attention immediately. "
        "Distribute this new hook naturally across the SAME opening "
        "segment(s) the source hook used — never add or remove segments to "
        "do this.\n"
        "- If the opening is NOT a hook (the video starts directly into "
        "the scene/action): do NOT invent one. Translate those opening "
        "line(s) directly and naturally, exactly like every other "
        "segment — no embellishment.\n\n"
        "FULL SOURCE SCRIPT (read-only context ONLY, to judge/write the "
        "hook — this is not itself part of the segments to translate):\n"
        f"{context}\n\n"
    )


async def gemini_transcribe(audio_path: Path) -> dict:
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

    def _do(model_name: str, api_key: str) -> str:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
                prompt,
            ],
        )
        return resp.text

    raw = None
    model_used: Optional[str] = None
    last_err: Optional[Exception] = None
    for model_name in GEMINI_TRANSCRIBE_MODELS:
        for key in GEMINI_API_KEYS:
            try:
                raw = await asyncio.to_thread(_do, model_name, key)
                model_used = model_name
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError(
            f"Gemini transcription failed on all {len(GEMINI_TRANSCRIBE_MODELS)} "
            f"model(s) x {len(GEMINI_API_KEYS)} configured key(s): {last_err}"
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
    return {"language": language, "segments": segments, "asr_model": model_used}


async def _gemini_translate_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
    is_first_chunk: bool = False,
    full_script_context: str = "",
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    hook_section = (
        _hook_handling_section(target_language, full_script_context)
        if is_first_chunk else ""
    )
    system_prompt = (
        "You are an elite professional translator inside an automated "
        "single-narrator video-dubbing pipeline. You will be given a JSON "
        "array of timestamped transcript segments (a SMALL CHUNK of a "
        "larger transcript), already transcribed from the original spoken "
        f"language ({source_language}).\n\n"
        "TASK 1 — TRANSLATION: Translate every segment's text PERFECTLY and "
        f"naturally into {target_language}, staying ENTIRELY WITHIN that same "
        "segment. Preserve tone, intent, and idiom — this is for dubbing, not "
        "a literal word-for-word gloss — but the translation of a segment "
        "must cover exactly what that segment's source text covers: never "
        "shorten it into a bare fragment, never pad or lengthen it with "
        "content that wasn't in that line, and never borrow words from or "
        "lend words to a neighboring segment. Never leave a segment "
        "untranslated.\n\n"
        f"{_translation_style_notes(target_language)}\n\n"
        "NO RESTRUCTURING (CRITICAL): translate each segment on its own, in "
        "place, exactly where it is. Never merge two segments together, "
        "never split one segment's content across two, never reorder "
        "segments, and never move a phrase from one line into another to "
        "make the pacing 'flow' better — that breaks the sync between the "
        "dubbed voice and the original video. If a segment's direct, "
        "natural translation is already correct, leave it exactly as that "
        "plain translation — do not embellish it. Only retouch a segment "
        "when its own translation is actually inaccurate, unnatural, or "
        "grammatically wrong, and any such fix must stay entirely inside "
        "that same segment. (The ONLY exception to any of this is the "
        "opening hook rule below, if it applies to this chunk.)\n\n"
        f"{hook_section}"
        f"{_length_rule_text()}"
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

    def _do(model_name: str, api_key: str) -> str:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=[system_prompt, user_payload],
        )
        return resp.text

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")

    raw = None
    last_err: Optional[Exception] = None
    for model_name in GEMINI_TRANSLATE_MODELS:
        for key in GEMINI_API_KEYS:
            try:
                raw = await asyncio.to_thread(_do, model_name, key)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError(
            f"Gemini translation failed on all {len(GEMINI_TRANSLATE_MODELS)} "
            f"model(s) x {len(GEMINI_API_KEYS)} configured key(s): {last_err}"
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
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    if not transcript_segments:
        raise RuntimeError("Nothing to translate — the transcript is empty.")

    chunks = _chunk_by_char_budget(transcript_segments)
    budgets = _segment_max_chars(transcript_segments, target_language)

    full_script_context = "\n".join(
        str(seg.get("text", "")).strip() for seg in transcript_segments if seg.get("text")
    )

    detected_source_language: Optional[str] = None
    all_segments: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        payload_chunk = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
                **({"max_chars": budgets[id(seg)]} if id(seg) in budgets else {}),
            }
            for seg in chunk
        ]
        result_segments: List[Dict[str, object]] = []
        for attempt in range(1 + TRANSLATE_CHUNK_RETRIES):
            try:
                lang, translated = await _gemini_translate_chunk(
                    payload_chunk, source_language, target_language,
                    is_first_chunk=(chunk_idx == 0),
                    full_script_context=full_script_context if chunk_idx == 0 else "",
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
        if progress_cb:
            try:
                progress_cb(chunk_idx + 1, len(chunks))
            except Exception:  # noqa: BLE001
                pass

    return {
        "source_language": detected_source_language or source_language,
        "segments": all_segments,
        "input_count": len(transcript_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


async def _gemini_retouch_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
) -> List[Dict[str, object]]:
    """
    ONE chunk of the hybrid-engine retouch pass: each segment already
    carries a Groq-produced draft translation (`text`) plus the original
    source line (`source_text`) for reference. Gemini's ONLY job here is
    to polish that draft for natural, native-sounding fluency — it is
    explicitly NOT asked to re-translate from scratch, restructure, merge,
    split, reorder, or drop anything. Returns the polished segment list,
    or an EMPTY list (never raises) if the response doesn't come back as a
    strict 1:1 match — the caller (gemini_retouch_all) then falls back to
    keeping the Groq draft for that chunk, so a retouch hiccup can never
    cost a line.
    """
    system_prompt = (
        "You are a senior native-language editor doing the FINAL polish "
        "pass on an already-translated video-dubbing script. You will be "
        "given a JSON array of segments; each has the ORIGINAL source-"
        f"language ({source_language}) line and a DRAFT {target_language} "
        "translation of it (already produced by another translator).\n\n"
        "YOUR ONLY JOB: retouch each draft translation for natural, "
        "native, everyday spoken fluency in the target language — fix any "
        "awkward phrasing, stiff/literal wording, or grammar issues, and "
        "make idioms and word choice sound like a native speaker wrote it. "
        "You are NOT re-translating from scratch: use the source line only "
        "as meaning-reference to confirm the draft is accurate, and start "
        "from the draft itself.\n\n"
        "STRICT RULES (CRITICAL):\n"
        "1. NEVER merge two segments, split one into two, reorder segments, "
        "or move words/phrases from one segment into another — each "
        "segment's retouch must stay entirely within that same segment. "
        "This is a dubbing script; breaking segment boundaries breaks the "
        "sync between the dubbed voice and the video.\n"
        "2. NEVER drop, skip, or leave a segment untouched-but-missing from "
        "the output — every input segment MUST have a corresponding output "
        "segment, same order, same count.\n"
        "3. If a draft translation is already natural and correct, keep it "
        "exactly as-is — do not change wording just to change it.\n"
        "4. Do not change the MEANING of any line versus its source.\n"
        "5. Each segment has 'max_chars' (the longest its spoken line may be "
        "to fit its moment in the video). Never make a line LONGER than "
        "max_chars; if the draft is longer, tighten the wording (same "
        "meaning, fewer words).\n\n"
        "Respond with ONLY raw JSON (no markdown fences), in exactly this "
        'shape: {"segments": [{"start": 0.0, "end": 3.2, '
        '"text": "<retouched translation>"}]}'
    )
    user_payload = json.dumps({"segments": chunk_segments}, ensure_ascii=False)

    def _do(model_name: str, api_key: str) -> str:
        client = get_client(api_key)
        resp = client.models.generate_content(
            model=model_name,
            contents=[system_prompt, user_payload],
        )
        return resp.text

    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")

    raw = None
    last_err: Optional[Exception] = None
    for model_name in GEMINI_TRANSLATE_MODELS:
        for key in GEMINI_API_KEYS:
            try:
                raw = await asyncio.to_thread(_do, model_name, key)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError(
            f"Gemini retouch failed on all {len(GEMINI_TRANSLATE_MODELS)} "
            f"model(s) x {len(GEMINI_API_KEYS)} configured key(s): {last_err}"
        ) from last_err

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, TypeError):
        return []

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

    if len(out_segments) != len(chunk_segments):
        return []
    return out_segments


async def gemini_retouch_all(
    draft_segments: List[Dict[str, object]],
    raw_source_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Hybrid-engine retouch pass: takes Groq's already-complete draft
    translation (`draft_segments`, guaranteed 1:1 with the transcript by
    groq_translate_all) and asks Gemini to polish each line's fluency,
    chunk by chunk, in the same char-budgeted, drop-proof pattern used
    everywhere else in this pipeline. Critically: if Gemini's retouch of a
    chunk fails or doesn't come back 1:1 after retries, that chunk falls
    back to the GROQ DRAFT (already a complete, valid translation) rather
    than the raw source text — so a retouch failure only costs polish, or
    never costs a line, only ever costs polish on a few lines, never drops
    or blanks them.
    """
    if not draft_segments:
        raise RuntimeError("Nothing to retouch — the draft translation is empty.")

    chunks = _chunk_by_char_budget(draft_segments)
    source_chunks = _chunk_by_char_budget(raw_source_segments) if raw_source_segments else []
    budgets = _segment_max_chars(draft_segments, target_language)

    all_segments: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        source_chunk = source_chunks[chunk_idx] if chunk_idx < len(source_chunks) else [None] * len(chunk)
        payload_chunk = []
        for i, seg in enumerate(chunk):
            src_text = ""
            if i < len(source_chunk) and source_chunk[i] is not None:
                src_text = str(source_chunk[i].get("text", ""))
            item = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "source_text": src_text,
                "text": str(seg.get("text", "")),
            }
            if id(seg) in budgets:
                item["max_chars"] = budgets[id(seg)]
            payload_chunk.append(item)

        result_segments: List[Dict[str, object]] = []
        for attempt in range(1 + TRANSLATE_CHUNK_RETRIES):
            try:
                retouched = await _gemini_retouch_chunk(payload_chunk, source_language, target_language)
            except Exception:  # noqa: BLE001
                continue
            if len(retouched) == len(chunk):
                result_segments = retouched
                break

        if not result_segments:
            # Retouch failed for this chunk after every retry — fall back
            # to the Groq draft translation for just these lines. The
            # draft is already a complete, correct translation (it went
            # through groq_translate_all's own drop-proof guarantee), so
            # this NEVER drops a line — it only skips the extra polish.
            fallback_count += len(chunk)
            fallback_ranges.append(
                f"{payload_chunk[0]['start']:.1f}s-{payload_chunk[-1]['end']:.1f}s"
            )
            result_segments = [
                {"start": float(seg.get("start", 0.0)), "end": float(seg.get("end", 0.0)),
                 "text": str(seg.get("text", "")).strip()}
                for seg in chunk
            ]

        all_segments.extend(result_segments)
        if progress_cb:
            try:
                progress_cb(chunk_idx + 1, len(chunks))
            except Exception:  # noqa: BLE001
                pass

    return {
        "segments": all_segments,
        "input_count": len(draft_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


async def _groq_translate_chunk(
    chunk_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
    is_first_chunk: bool = False,
    full_script_context: str = "",
) -> Tuple[Optional[str], List[Dict[str, object]]]:
    hook_section = (
        _hook_handling_section(target_language, full_script_context)
        if is_first_chunk else ""
    )
    system_prompt = (
        "You are an elite professional translator inside an automated "
        "single-narrator video-dubbing pipeline. You will be given a JSON "
        "array of timestamped transcript segments (a SMALL CHUNK of a "
        "larger transcript), already transcribed from the original spoken "
        f"language ({source_language}).\n\n"
        "TASK 1 — TRANSLATION: Translate every segment's text PERFECTLY and "
        f"naturally into {target_language}, staying ENTIRELY WITHIN that same "
        "segment. Preserve tone, intent, and idiom — this is for dubbing, not "
        "a literal word-for-word gloss — but the translation of a segment "
        "must cover exactly what that segment's source text covers: never "
        "shorten it into a bare fragment, never pad or lengthen it with "
        "content that wasn't in that line, and never borrow words from or "
        "lend words to a neighboring segment. Never leave a segment "
        "untranslated.\n\n"
        f"{_translation_style_notes(target_language)}\n\n"
        "NO RESTRUCTURING (CRITICAL): translate each segment on its own, in "
        "place, exactly where it is. Never merge two segments together, "
        "never split one segment's content across two (do NOT break a long "
        "source sentence into extra sentences, and do NOT merge short "
        "fragments together — even within the same segment, keep the "
        "translation's overall length matched to the source line, not "
        "restructured for 'better flow'), never reorder segments, and never "
        "move a phrase from one line into another. That kind of "
        "restructuring is exactly what breaks the sync between the dubbed "
        "voice and the original video, so it is never allowed. If a "
        "segment's direct, natural translation is already correct, leave it "
        "exactly as that plain translation — do not embellish it. Only "
        "retouch a segment when its own translation is actually inaccurate, "
        "unnatural, or grammatically wrong, and any such fix must stay "
        "entirely inside that same segment, without lengthening or "
        "shortening its scope. (The ONLY exception to any of this is the "
        "opening hook rule below, if it applies to this chunk.)\n\n"
        f"{hook_section}"
        f"{_length_rule_text()}"
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

    def _do(model_name: str, api_key: str) -> str:
        client = get_groq_client(api_key)
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
            )
        except Exception:
            resp = client.chat.completions.create(
                model=model_name,
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
    for model_name in TRANSLATION_MODELS:
        for key in GROQ_API_KEYS:
            try:
                raw = await asyncio.to_thread(_do, model_name, key)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        if raw is not None:
            break
    if raw is None:
        raise RuntimeError(
            f"Groq translation failed on all {len(TRANSLATION_MODELS)} "
            f"model(s) x {len(GROQ_API_KEYS)} configured key(s): {last_err}"
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


async def groq_translate_all(
    transcript_segments: List[Dict[str, object]],
    source_language: str,
    target_language: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    if not transcript_segments:
        raise RuntimeError("Nothing to translate — the transcript is empty.")

    chunks: List[List[Dict[str, object]]] = _chunk_by_char_budget(transcript_segments)
    budgets = _segment_max_chars(transcript_segments, target_language)

    full_script_context = "\n".join(
        str(seg.get("text", "")).strip() for seg in transcript_segments if seg.get("text")
    )

    detected_source_language: Optional[str] = None
    all_segments: List[Dict[str, object]] = []
    fallback_count = 0
    fallback_ranges: List[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        payload_chunk = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
                **({"max_chars": budgets[id(seg)]} if id(seg) in budgets else {}),
            }
            for seg in chunk
        ]

        result_segments: List[Dict[str, object]] = []
        last_err: Optional[Exception] = None
        for attempt in range(1 + TRANSLATE_CHUNK_RETRIES):
            try:
                lang, translated = await _groq_translate_chunk(
                    payload_chunk, source_language, target_language,
                    is_first_chunk=(chunk_idx == 0),
                    full_script_context=full_script_context if chunk_idx == 0 else "",
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
        if progress_cb:
            try:
                progress_cb(chunk_idx + 1, len(chunks))
            except Exception:  # noqa: BLE001
                pass

    return {
        "source_language": detected_source_language or source_language,
        "segments": all_segments,
        "input_count": len(transcript_segments),
        "output_count": len(all_segments),
        "fallback_count": fallback_count,
        "fallback_ranges": fallback_ranges,
    }


def _target_cps(target_language: str) -> float:
    lang = (target_language or "").strip().lower()
    for key, cps in TARGET_CPS.items():
        if key in lang:
            return cps
    if "bangla" in lang:
        return TARGET_CPS["bengali"]
    return 13.0


def _segment_max_chars(segments: List[Dict[str, object]], target_language: str) -> Dict[int, int]:
    """
    Per-line length budget for the translator: how many characters a line may
    have so its VOICE fits the video moment it belongs to at natural speed.
    Slot = time from this line's start to the next line's start (the gap after
    a line is free time the voice may use). Returns {id(segment): max_chars}
    (empty if the transcript has no usable timestamps).
    """
    out: Dict[int, int] = {}
    if len(segments) < 2:
        return out
    usable = sum(1 for sg in segments if float(sg.get("end", 0) or 0) > float(sg.get("start", 0) or 0))
    if usable < max(2, len(segments) // 2):
        return out
    cps = _target_cps(target_language)
    for i, sg in enumerate(segments):
        st = float(sg.get("start", 0) or 0)
        en = float(sg.get("end", 0) or 0)
        nxt = float(segments[i + 1].get("start", 0) or 0) if i + 1 < len(segments) else en + 0.8
        slot = max(en - st, nxt - st, 0.6)
        out[id(sg)] = max(10, int(slot * cps * 0.95))
    return out


def _length_rule_text() -> str:
    return (
        "LENGTH RULE (dubbing timing — very important): every input segment "
        "has a 'max_chars' number = the longest its spoken translation may be "
        "so the voice fits that exact moment of the video at a natural, "
        "comfortable speaking pace. Keep each translation at or under "
        "max_chars by choosing concise, natural wording — NEVER by dropping "
        "meaning, and never by merging/splitting/moving text between "
        "segments. If a faithful translation truly cannot fit, get as close "
        "to max_chars as you can.\n\n"
    )


def _chunk_by_char_budget(
    transcript_segments: List[Dict[str, object]],
) -> List[List[Dict[str, object]]]:
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


def _chunk_text_by_budget(lines: List[str], budget: int = TTS_CHUNK_CHAR_BUDGET) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        line_len = len(line)
        would_overflow = current and (current_chars + line_len + 1 > budget)
        if would_overflow:
            chunks.append("\n".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_len + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# --------------------------------------------------------------------------- #
# Self-healing retry wrapper
# --------------------------------------------------------------------------- #

async def self_heal(
    step_name: str,
    coro_factory,
    retries: int = SELF_HEAL_RETRIES,
    backoff_seconds: float = SELF_HEAL_BACKOFF_SECONDS,
):
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
# Gemini native TTS — pure text in, voice bytes out — chunked, strictly
# sequential model/key failover (never fires more than one request at a
# time; the primary (model, key) is always tried first, and any failure
# fails over IMMEDIATELY to the next key, then the next model).
# --------------------------------------------------------------------------- #

async def _slow_chunk_tail(src_wav: Path, out_wav: Path) -> bool:
    """
    Fix "the voice rushes at the end of the chunk": find the last natural pause
    in the final TTS_TAIL_REGION_FRACTION of the chunk and slow ONLY the speech
    after that pause by TTS_TAIL_SLOWDOWN (pitch preserved via atempo). The cut
    is placed in the middle of a real silence, so it can never slice a word.
    Returns True if a correction was applied (out_wav written), False if there
    was nothing to fix (caller keeps src_wav).
    """
    if not (0.85 <= TTS_TAIL_SLOWDOWN < 0.999):
        return False
    dur = await probe_duration(src_wav)
    if dur < 3.0:
        return False
    log = await _run_stderr([
        FFMPEG_BIN, "-hide_banner", "-i", str(src_wav),
        "-af", f"silencedetect=noise={TTS_TAIL_PAUSE_DB:.0f}dB:d={TTS_TAIL_MIN_PAUSE_SECONDS}",
        "-f", "null", "-",
    ])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*(-?[\d.]+)", log)]
    region_start = dur * (1.0 - TTS_TAIL_REGION_FRACTION)
    cut = None
    for i, st in enumerate(starts):
        en = ends[i] if i < len(ends) else dur
        mid = (st + en) / 2.0
        # a pause inside the tail region, with real speech left after it
        if mid >= region_start and en < dur - 0.25:
            cut = mid  # keep the LAST such pause
    if cut is None or cut <= 0.5 or cut >= dur - 0.3:
        return False
    tempo = TTS_TAIL_SLOWDOWN
    await _run([
        FFMPEG_BIN, "-y", "-i", str(src_wav),
        "-filter_complex",
        f"[0:a]atrim=0:{cut:.3f},asetpts=PTS-STARTPTS[a];"
        f"[0:a]atrim=start={cut:.3f},asetpts=PTS-STARTPTS,atempo={tempo:.3f}[b];"
        "[a][b]concat=n=2:v=0:a=1[o]",
        "-map", "[o]", "-ar", str(TTS_SAMPLE_RATE), "-ac", str(TTS_CHANNELS),
        "-c:a", "pcm_s16le", str(out_wav),
    ])
    return True


async def _process_chunk_audio(pcm: bytes, work_dir: Path, idx: int) -> Tuple[Path, str]:
    """
    Turn one raw TTS chunk into its final clean WAV (file on disk, so RAM stays
    low): cap over-long pauses (never delete speech) -> fix a rushed tail ->
    energy polish. Every step falls back to the previous step's audio on
    failure, so a processing problem can never lose or damage a chunk.
    Returns (final_wav_path, short_note). Temp files are deleted immediately.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    raw = work_dir / f"c{idx}_raw.wav"
    capped = work_dir / f"c{idx}_cap.wav"
    tail = work_dir / f"c{idx}_tail.wav"
    final = work_dir / f"c{idx}_final.wav"
    notes: List[str] = []
    write_wav_from_pcm(pcm, raw)
    src = raw
    try:
        before, after = await trim_internal_silences(src, capped)
        if abs(before - after) > 0.05:
            notes.append(f"pauses capped (-{before - after:.1f}s)")
        src = capped
    except Exception as exc:  # noqa: BLE001
        notes.append(f"pause-cap skipped ({type(exc).__name__})")
    try:
        if await _slow_chunk_tail(src, tail):
            src = tail
            notes.append(f"tail eased x{TTS_TAIL_SLOWDOWN:.2f}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"tail-fix skipped ({type(exc).__name__})")
    if TTS_POLISH_ENABLED:
        try:
            await _run([
                FFMPEG_BIN, "-y", "-i", str(src), "-filter:a", TTS_ENERGY_POLISH_FILTER,
                "-ar", str(TTS_SAMPLE_RATE), "-ac", str(TTS_CHANNELS),
                "-c:a", "pcm_s16le", str(final),
            ])
            if await probe_duration(final) > 0.2:
                notes.append("energy polish")
                src = final
        except Exception as exc:  # noqa: BLE001
            notes.append(f"polish skipped ({type(exc).__name__})")
    if src != final:
        shutil.copyfile(src, final)
    for f in (raw, capped, tail):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    return final, ", ".join(notes) or "no changes"


def write_wav_from_pcm(pcm_bytes: bytes, out_wav: Path) -> None:
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(TTS_CHANNELS)
        wf.setsampwidth(TTS_SAMPLE_WIDTH)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def _silence_pcm(duration_ms: int) -> bytes:
    """Raw zeroed PCM silence of `duration_ms` at the TTS sample format."""
    num_samples = max(0, int(TTS_SAMPLE_RATE * duration_ms / 1000))
    return b"\x00" * (num_samples * TTS_SAMPLE_WIDTH * TTS_CHANNELS)


def _gemini_tts_call(model_name: str, api_key: str, voice_name: str, spoken_text: str) -> bytes:
    client = get_client(api_key)
    resp = client.models.generate_content(
        model=model_name,
        contents=spoken_text,
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


async def _tts_one_chunk(
    chunk_text: str,
    idx: int,
    total: int,
    voice_name: str,
    style_hint: str,
    model_chain: List[str],
) -> AsyncGenerator[dict, None]:
    """
    Voice ONE chunk with the strict sequential failover chain (primary model x
    key first; on any error/timeout the very next key, then the next model —
    one request in flight at a time). Yields log dicts; the final item is
    {"_pcm": bytes, "_model": name}.
    """
    if not GEMINI_API_KEYS:
        raise RuntimeError("No GEMINI_API_KEY(s) configured.")
    voice_name = voice_name if voice_name in GEMINI_VOICE_NAMES else "Kore"
    spoken = f"{style_hint.strip()}: {chunk_text}" if style_hint.strip() else chunk_text
    pcm: Optional[bytes] = None
    used: Optional[str] = None
    last_err: Optional[Exception] = None
    for model_name in model_chain:
        for key_idx, api_key in enumerate(GEMINI_API_KEYS):
            try:
                pcm = await asyncio.wait_for(
                    asyncio.to_thread(_gemini_tts_call, model_name, api_key, voice_name, spoken),
                    timeout=TTS_CALL_TIMEOUT_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                last_err = RuntimeError(f"timed out after {TTS_CALL_TIMEOUT_SECONDS}s")
                yield sse_log(
                    f"[WARN] Chunk {idx}/{total}: model '{model_name}' key "
                    f"#{key_idx + 1}/{len(GEMINI_API_KEYS)} timed out — next key immediately."
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                yield sse_log(
                    f"[WARN] Chunk {idx}/{total}: model '{model_name}' key "
                    f"#{key_idx + 1}/{len(GEMINI_API_KEYS)} failed "
                    f"({type(exc).__name__}: {exc}) — next key immediately."
                )
        if pcm is not None:
            used = model_name
            break
        yield sse_log(
            f"[WARN] Chunk {idx}/{total}: all {len(GEMINI_API_KEYS)} key(s) failed for "
            f"'{model_name}' — switching to the next model."
        )
    if pcm is None:
        raise RuntimeError(
            f"Chunk {idx}/{total} failed on all {len(model_chain)} model(s) x "
            f"{len(GEMINI_API_KEYS)} key(s). Last error: {last_err}"
        ) from last_err
    yield {"_pcm": pcm, "_model": used}


async def _llm_json_text(system_prompt: str, user_payload: str) -> Optional[str]:
    """One JSON-answering LLM call using whatever is configured: Gemini chain
    first (every model x key, sequential), then the Groq chain. Returns the raw
    text, or None if nothing answered."""
    def _gem(model_name: str, api_key: str) -> str:
        return get_client(api_key).models.generate_content(
            model=model_name, contents=[system_prompt, user_payload]
        ).text

    def _groq(model_name: str, api_key: str) -> str:
        client = get_groq_client(api_key)
        try:
            r = client.chat.completions.create(
                model=model_name, temperature=0.2,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_payload}],
                response_format={"type": "json_object"},
            )
        except Exception:  # noqa: BLE001
            r = client.chat.completions.create(
                model=model_name, temperature=0.2,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_payload}],
            )
        return r.choices[0].message.content

    for model_name in GEMINI_TRANSLATE_MODELS:
        for key in GEMINI_API_KEYS:
            try:
                return await asyncio.to_thread(_gem, model_name, key)
            except Exception:  # noqa: BLE001
                continue
    for model_name in TRANSLATION_MODELS:
        for key in GROQ_API_KEYS:
            try:
                return await asyncio.to_thread(_groq, model_name, key)
            except Exception:  # noqa: BLE001
                continue
    return None


async def _condense_lines(lines: List[str], ratio: float, target_language: str) -> Optional[List[str]]:
    """
    Ask the LLM to make each line SHORTER (same meaning, tighter wording) so its
    voice fits the video window — the professional first remedy instead of
    speeding the voice up. `ratio` < 1 is the target length relative to now.
    Strict 1:1: returns None (caller keeps the original) unless every line comes
    back non-empty and the total is genuinely shorter.
    """
    payload = [{"i": i, "text": t, "max_chars": max(6, int(len(t) * ratio))} for i, t in enumerate(lines)]
    system_prompt = (
        f"You are a dubbing script editor. Each item is a {target_language} line "
        "that is TOO LONG to be spoken in its time slot. Rewrite EVERY line "
        "shorter — same meaning, natural spoken "
        f"{target_language}, tighter wording (drop filler, use shorter "
        "phrasing) — so that each line is at or under its 'max_chars'. Keep it "
        "a complete, natural sentence; never cut mid-thought; never merge, split, "
        "reorder or drop lines. The output array MUST have exactly the same "
        "number of items, same order.\n"
        'Respond with ONLY raw JSON: {"lines": [{"i": 0, "text": "<shorter line>"}]}'
    )
    raw = await _llm_json_text(system_prompt, json.dumps({"lines": payload}, ensure_ascii=False))
    if not raw:
        return None
    try:
        data = _extract_json(raw)
        items = data.get("lines", [])
        by_i = {int(it["i"]): str(it["text"]).strip() for it in items if isinstance(it, dict)}
        out = [by_i.get(i, "") for i in range(len(lines))]
    except Exception:  # noqa: BLE001
        return None
    if any(not t for t in out):
        return None
    if sum(len(t) for t in out) >= sum(len(t) for t in lines) * 0.98:
        return None
    return out


def _plan_tts_chunks(segments: List[Segment], video_duration: float) -> List[dict]:
    """
    Group consecutive script lines into voice chunks (<= TTS_CHUNK_CHAR_BUDGET
    characters AND <= TTS_CHUNK_MAX_SOURCE_SECONDS of source video), breaking
    ONLY on line boundaries. Each chunk keeps the video timestamp where its
    first line begins — that is where its voice will be placed. If the
    transcript has no usable timestamps, starts are spread proportionally by
    text length across the video.
    """
    lines = [(i, sg.text.strip(), float(sg.start), float(sg.end)) for i, sg in enumerate(segments) if sg.text.strip()]
    if not lines:
        return []
    usable = sum(1 for (_, _, a, b) in lines if b > a)
    timed = usable >= max(2, len(lines) // 2) and video_duration > 0
    if not timed:
        total_chars = sum(len(t) for _, t, _, _ in lines) or 1
        acc = 0
        fixed = []
        for (i, t, _, _) in lines:
            st = (acc / total_chars) * max(video_duration, 1.0)
            acc += len(t)
            fixed.append((i, t, st, (acc / total_chars) * max(video_duration, 1.0)))
        lines = fixed
    # monotonic starts, clamped inside the video
    last = 0.0
    mono = []
    for (i, t, a, b) in lines:
        a = max(a, last)
        if video_duration > 0:
            a = min(a, max(0.0, video_duration - 1.0))
        b = max(b, a)
        mono.append((i, t, a, b))
        last = a
    lines = mono

    chunks: List[dict] = []
    cur: List[Tuple[int, str, float, float]] = []
    cur_chars = 0
    for ln in lines:
        _, t, a, b = ln
        over_chars = cur and (cur_chars + len(t) + 1 > TTS_CHUNK_CHAR_BUDGET)
        over_span = cur and (b - cur[0][2] > TTS_CHUNK_MAX_SOURCE_SECONDS)
        if over_chars or over_span:
            chunks.append({"lines": cur})
            cur, cur_chars = [], 0
        cur.append(ln)
        cur_chars += len(t) + 1
    if cur:
        chunks.append({"lines": cur})
    for ch in chunks:
        ch["start"] = ch["lines"][0][2]
        ch["seg_idx"] = [x[0] for x in ch["lines"]]
        ch["texts"] = [x[1] for x in ch["lines"]]
    return chunks


class _TimelineWriter:
    """Streams the final voice track straight to a WAV file: silence up to a
    chunk's start time, then the chunk's audio. Constant, tiny RAM use."""

    def __init__(self, path: Path) -> None:
        self.wf = wave.open(str(path), "wb")
        self.wf.setnchannels(TTS_CHANNELS)
        self.wf.setsampwidth(TTS_SAMPLE_WIDTH)
        self.wf.setframerate(TTS_SAMPLE_RATE)
        self.samples = 0

    @property
    def seconds(self) -> float:
        return self.samples / float(TTS_SAMPLE_RATE)

    def pad_to(self, t: float) -> None:
        need = int(t * TTS_SAMPLE_RATE) - self.samples
        block = b"\x00" * (TTS_SAMPLE_WIDTH * TTS_CHANNELS * 24000)
        while need > 0:
            n = min(need, 24000)
            self.wf.writeframes(block[: n * TTS_SAMPLE_WIDTH * TTS_CHANNELS])
            self.samples += n
            need -= n

    def append_wav(self, path: Path) -> None:
        with wave.open(str(path), "rb") as r:
            while True:
                frames = r.readframes(65536)
                if not frames:
                    break
                self.wf.writeframes(frames)
                self.samples += len(frames) // (TTS_SAMPLE_WIDTH * TTS_CHANNELS)

    def close(self) -> None:
        self.wf.close()


async def _render_chunk(
    sess: "Session", text: str, idx: int, total: int, work_dir: Path,
    model_chain: List[str], rates: List[float],
) -> AsyncGenerator[dict, None]:
    """
    Voice + clean one chunk. Includes a COMPLETENESS guard: if the voice is far
    shorter than the text needs (the TTS model skipped sentences), the chunk is
    regenerated once. Final item: {"_wav", "_dur", "_model", "_note"}.
    """
    style = _tts_style_hint(sess.target_language)
    truncated_retry = 0
    while True:
        pcm: Optional[bytes] = None
        model: Optional[str] = None
        for attempt in range(1 + SELF_HEAL_RETRIES):
            try:
                async for ev in _tts_one_chunk(text, idx, total, sess.single_voice, style, model_chain):
                    if "_pcm" in ev:
                        pcm, model = ev["_pcm"], ev["_model"]
                    else:
                        yield ev
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < SELF_HEAL_RETRIES:
                    yield sse_log(f"[WARN] Chunk {idx}/{total} voice failed ({exc}) — retrying in "
                                  f"{SELF_HEAL_BACKOFF_SECONDS:.0f}s...")
                    await asyncio.sleep(SELF_HEAL_BACKOFF_SECONDS)
                else:
                    raise
        raw_dur = len(pcm) / float(TTS_SAMPLE_RATE * TTS_SAMPLE_WIDTH * TTS_CHANNELS)
        ref_cps = statistics.median(rates) if rates else _target_cps(sess.target_language) * 1.3
        expected = len(text) / max(ref_cps, 1.0)
        if raw_dur < TTS_TRUNCATION_RATIO * expected and truncated_retry < 1:
            truncated_retry += 1
            yield sse_log(
                f"[WARN] Chunk {idx}/{total}: voice is only {raw_dur:.1f}s but this text needs "
                f"about {expected:.1f}s — the model probably skipped part of it. Regenerating once."
            )
            continue
        if raw_dur >= TTS_TRUNCATION_RATIO * expected:
            rates.append(len(text) / max(raw_dur, 0.2))
        break
    final_wav, note = await _process_chunk_audio(pcm, work_dir, idx)
    dur = await probe_duration(final_wav)
    yield {"_wav": final_wav, "_dur": dur, "_model": model, "_note": note}


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
# Core processing: chunked Gemini TTS -> trim -> exact duration match -> mux
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


def _live_progress(sess: "Session", pct: int, msg: str) -> None:
    """Update the polled progress immediately (used from inside long steps that
    are not generators, e.g. per-chunk translation)."""
    sess.job_percent = max(0, min(100, int(pct)))
    sess.job_message = msg
    _append_job_log(sess, f"[PROGRESS {sess.job_percent}%] {msg}")
    if sess.dir.exists():
        save_session(sess)


async def synthesize_single_track(sess: Session) -> AsyncGenerator[dict, None]:
    seg_dir = sess.dir / "segments"
    work_dir = seg_dir / "work"
    seg_dir.mkdir(exist_ok=True)

    plan = _plan_tts_chunks(sess.segments, sess.video_duration)
    if not plan:
        raise RuntimeError("Nothing to voice — the translated script came back empty.")
    total = len(plan)
    vdur = sess.video_duration
    tts_chain = sess.tts_models or list(TTS_MODELS)
    total_chars = sum(len(t) for ch in plan for t in ch["texts"])

    yield sse_log(
        f"[INFO] Voice plan: {total} chunk(s), {total_chars} characters, "
        f"{len(sess.segments)} line(s), video {vdur:.1f}s. Each chunk is voiced on its "
        "own and placed at its original video time (no drift, no global speed-up)."
    )
    yield sse_log(f"[INFO] Voice model fallback chain: {', '.join(tts_chain)}")

    timeline_path = seg_dir / "timeline.wav"
    writer = _TimelineWriter(timeline_path)
    rates: List[float] = []
    models_used: List[str] = []
    max_stretch = 1.0
    condensed_chunks = 0
    overflow_chunks = 0

    try:
        for n, ch in enumerate(plan):
            idx = n + 1
            yield sse_progress(50 + int(42 * n / total), f"Voicing chunk {idx}/{total}")

            next_start = plan[n + 1]["start"] if n + 1 < total else (vdur if vdur > 0 else None)
            start_actual = max(ch["start"], writer.seconds + (MIN_CHUNK_GAP_SECONDS if n > 0 else 0.0))
            window = (next_start - start_actual) if next_start is not None else 1e9
            tight = window < 1.0          # already pushed past its slot by earlier overflow
            texts = list(ch["texts"])
            result: Optional[dict] = None
            r = 0.0

            for attempt in range(1 + CONDENSE_ATTEMPTS):
                result = None
                async for ev in _render_chunk(sess, "\n".join(texts), idx, total, work_dir, tts_chain, rates):
                    if "_wav" in ev:
                        result = ev
                    else:
                        yield ev
                dur = result["_dur"]
                r = dur / window if not tight else 0.0
                if tight or r <= FIT_SPEED_SOFT_MAX or attempt >= CONDENSE_ATTEMPTS:
                    break
                target = min(0.95, (window * FIT_SPEED_SOFT_MAX * 0.97) / dur)
                yield sse_log(
                    f"[INFO] Chunk {idx}/{total}: voice {dur:.1f}s > window {window:.1f}s "
                    f"(x{dur / window:.2f}) — asking the editor to tighten the wording "
                    f"(attempt {attempt + 1}/{CONDENSE_ATTEMPTS}) instead of speeding up."
                )
                shorter = await _condense_lines(texts, target, sess.target_language)
                if shorter is None:
                    yield sse_log(f"[WARN] Chunk {idx}/{total}: could not shorten the wording safely — "
                                  "keeping the original text.")
                    break
                try:
                    result["_wav"].unlink(missing_ok=True)
                except OSError:
                    pass
                texts = shorter
                condensed_chunks += 1 if attempt == 0 else 0
                for k, seg_i in enumerate(ch["seg_idx"]):
                    sess.segments[seg_i].text = texts[k]

            models_used.append(result["_model"])
            wav: Path = result["_wav"]
            dur = result["_dur"]

            # Small, last-resort speed-up — never above FIT_SPEED_HARD_MAX, and none at all
            # when the chunk already fits (no forced speed-up).
            final_dur = dur
            if not tight and r > 1.02:
                ratio = min(r, FIT_SPEED_HARD_MAX)
                fitted = work_dir / f"c{idx}_fit.wav"
                await _encode_wav(["-i", str(wav)], fitted, _atempo_chain(ratio))
                try:
                    wav.unlink(missing_ok=True)
                except OSError:
                    pass
                wav = fitted
                final_dur = await probe_duration(wav)
                max_stretch = max(max_stretch, ratio)
                speed_note = f"sped up x{ratio:.2f}"
            else:
                speed_note = "natural speed"

            writer.pad_to(start_actual)
            writer.append_wav(wav)
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass

            over = (next_start is not None) and (start_actual + final_dur > next_start + 0.05)
            if over:
                overflow_chunks += 1
            yield sse_log(
                f"[SUCCESS] Chunk {idx}/{total} placed at {start_actual:.1f}s: voice {final_dur:.1f}s "
                f"in a {min(window, 9999):.1f}s window, {speed_note} ({result['_note']})"
                + (" — runs slightly into the next window (nothing is cut)." if over else ".")
            )

        if vdur > 0:
            writer.pad_to(vdur)
    finally:
        writer.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    model_used = ", ".join(dict.fromkeys(models_used))
    yield sse_log(f"[SUCCESS] Voice generated via {model_used} across {total} chunk(s); "
                  f"{condensed_chunks} chunk(s) needed tighter wording.")

    yield sse_progress(93, "Finishing voice track (loudness)")
    fitted_path = seg_dir / "final_voice.wav"
    await _encode_wav(["-i", str(timeline_path)], fitted_path, TTS_LOUDNESS_NORMALIZE_FILTER)
    try:
        timeline_path.unlink(missing_ok=True)
    except OSError:
        pass
    fitted_dur = await probe_duration(fitted_path)
    yield sse_log(f"[SUCCESS] Final dubbed audio: {fitted_dur:.2f}s (video {vdur:.2f}s).")
    if vdur > 0 and fitted_dur > vdur + 0.3:
        yield sse_log(
            f"[WARN] The voice is {fitted_dur - vdur:.1f}s longer than the video. Nothing was cut; "
            "to avoid this, use a shorter/more concise target language or split the video."
        )

    yield sse_progress(95, "Muxing dubbed audio into video")
    yield sse_log("[INFO] Merging dubbed audio with the original video (video stream copied, zero quality loss)...")
    out_video = sess.dir / "dubbed_output.mp4"
    await mux_video_with_audio(sess.video_path, fitted_path, out_video)

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
        "tts_requests": total,
        "speed_ratio": round(max_stretch, 3),
        "condensed_chunks": condensed_chunks,
        "overflow_chunks": overflow_chunks,
        "voice_model": model_used,
    })


async def run_prepare(sess: Session) -> AsyncGenerator[dict, None]:
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
            yield sse_progress(50, f"Transcribing with Gemini ({GEMINI_TRANSCRIBE_MODELS[0]})")
            yield sse_log(
                "[INFO] Sending audio to Gemini for transcription (primary engine) "
                f"— model fallback chain: {', '.join(GEMINI_TRANSCRIBE_MODELS)}..."
            )
            async for ev in self_heal("Gemini transcription", lambda: gemini_transcribe(sess.audio_path)):
                if "_result" in ev:
                    transcript = ev["_result"]
                else:
                    yield ev
        else:
            # Both the pure "groq" engine and the "hybrid" engine (Groq
            # transcribe -> Groq translate -> Gemini retouch) use Groq
            # Whisper here — it is the most accurate transcription engine
            # available in this pipeline, which is why "hybrid" is the
            # recommended default.
            yield sse_progress(50, f"Transcribing with Groq Whisper ({WHISPER_MODELS[0]})")
            yield sse_log(
                "[INFO] Streaming audio to Groq Whisper for accurate transcription "
                f"— model fallback chain: {', '.join(WHISPER_MODELS)}..."
            )
            async for ev in self_heal("Groq transcription", lambda: groq_transcribe(sess.audio_path)):
                if "_result" in ev:
                    transcript = ev["_result"]
                else:
                    yield ev

        sess.source_language = transcript["language"]

        # Collapse hallucinated repeated lines (the model saying the same
        # sentence several times in a row, most often around silence,
        # music, or noisy stretches) BEFORE anything downstream ever sees
        # them — this is the fix for repeated-line transcripts. Only
        # adjacent, near-identical repeats are merged; a genuinely
        # recurring line elsewhere in the video is untouched.
        deduped_segments, removed_count = _dedupe_repeated_segments(transcript["segments"])
        if removed_count > 0:
            yield sse_log(
                f"[WARN] Collapsed {removed_count} repeated/hallucinated line(s) "
                "detected in the raw transcript (same sentence transcribed "
                "back-to-back) — kept one copy per run, stretched to cover "
                "the full repeated span."
            )
        sess.raw_segments = [Segment(**s) for s in deduped_segments]
        sess.prepared = True
        save_session(sess)

        # Audio extracted purely for transcription is no longer needed once
        # we have the transcript — free it now instead of waiting for the
        # whole session to finish (RAM/disk hygiene for long queues).
        if sess.audio_path and sess.audio_path.exists():
            try:
                sess.audio_path.unlink(missing_ok=True)
            except OSError:
                pass

        yield sse_log(f"[INFO] Source language detected: {sess.source_language}")
        yield sse_progress(100, "Ready — pick a language/voice and press Dub")
        yield sse_done({
            "session_id": sess.session_id,
            "video_duration": round(sess.video_duration, 2),
            "source_language": sess.source_language,
            "line_count": len(sess.raw_segments),
        })
    except Exception as exc:  # noqa: BLE001
        # Keep the session (video stays) so the person can press Retry instead
        # of re-uploading; only scratch files are removed now, the rest is
        # reclaimed by the TTL sweep.
        yield sse_error(f"[ERROR] {type(exc).__name__}: {exc}")
        if sess.audio_path and sess.audio_path.exists():
            try:
                sess.audio_path.unlink(missing_ok=True)
            except OSError:
                pass


async def run_dub(sess: Session) -> AsyncGenerator[dict, None]:
    try:
        if not sess.raw_segments:
            raise RuntimeError("This video hasn't finished uploading/analyzing yet.")

        engine = sess.engine
        seg_dicts = [asdict(s) for s in sess.raw_segments]
        result = None
        if engine == "gemini":
            yield sse_progress(30, f"Translating {len(sess.raw_segments)} line(s) with Gemini")
            yield sse_log(
                f"[INFO] Translating with Gemini ({GEMINI_TRANSLATE_MODELS[0]}; backup "
                f"chain: {', '.join(GEMINI_TRANSLATE_MODELS[1:]) or 'none'}) in "
                f"~{TRANSLATE_CHUNK_CHAR_BUDGET}-char chunks — every line is "
                "verified 1:1 and re-touched for a natural target-audience "
                "voice, so none can be silently dropped..."
            )
            async for ev in self_heal("Gemini translation", lambda: gemini_translate_all(
                progress_cb=lambda d, t: _live_progress(sess, 30 + int(20 * d / t), f"Translating chunk {d}/{t}"),
                transcript_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    result = ev["_result"]
                else:
                    yield ev
        elif engine == "hybrid":
            # HYBRID PIPELINE (recommended default): Groq Whisper already
            # transcribed accurately in run_prepare; now Groq produces the
            # initial translation (its own drop-proof, chunked, retried
            # pass — identical guarantee to the pure "groq" engine below),
            # and THEN Gemini retouches every line for natural fluency
            # without ever re-translating from scratch, restructuring, or
            # dropping a line (see gemini_retouch_all).
            yield sse_progress(25, f"Translating {len(sess.raw_segments)} line(s) with Groq ({TRANSLATION_MODELS[0]})")
            yield sse_log(
                f"[INFO] Step 1/2 — Draft translation via Groq "
                f"({TRANSLATION_MODELS[0]}; backup chain: "
                f"{', '.join(TRANSLATION_MODELS[1:]) or 'none'}) in "
                f"~{TRANSLATE_CHUNK_CHAR_BUDGET}-char chunks — every line "
                "verified 1:1, so none can be silently dropped..."
            )
            draft_result = None
            async for ev in self_heal("Groq draft translation", lambda: groq_translate_all(
                progress_cb=lambda d, t: _live_progress(sess, 25 + int(15 * d / t), f"Draft translation chunk {d}/{t}"),
                transcript_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    draft_result = ev["_result"]
                else:
                    yield ev

            if draft_result["fallback_count"] > 0:
                ranges = ", ".join(draft_result["fallback_ranges"])
                yield sse_log(
                    f"[WARN] {draft_result['fallback_count']} line(s) (around "
                    f"{ranges}) could not be verified in the draft pass after "
                    "retries, so the ORIGINAL source-language text was kept "
                    "for those lines instead of being dropped."
                )

            sess.source_language = draft_result["source_language"]

            yield sse_progress(40, f"Retouching {len(draft_result['segments'])} line(s) with Gemini")
            yield sse_log(
                f"[INFO] Step 2/2 — Fluency retouch via Gemini "
                f"({GEMINI_TRANSLATE_MODELS[0]}; backup chain: "
                f"{', '.join(GEMINI_TRANSLATE_MODELS[1:]) or 'none'}) — "
                "polishing each line for natural, native phrasing without "
                "re-translating, restructuring, or dropping anything. A "
                "retouch hiccup on any chunk falls back to the already-"
                "correct Groq draft for just those lines, never a blank..."
            )
            async for ev in self_heal("Gemini retouch", lambda: gemini_retouch_all(
                progress_cb=lambda d, t: _live_progress(sess, 40 + int(10 * d / t), f"Retouching chunk {d}/{t}"),
                draft_segments=draft_result["segments"],
                raw_source_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    result = ev["_result"]
                else:
                    yield ev
            result["source_language"] = sess.source_language
            if result["fallback_count"] > 0:
                ranges = ", ".join(result["fallback_ranges"])
                yield sse_log(
                    f"[WARN] {result['fallback_count']} line(s) (around "
                    f"{ranges}) could not be retouched after retries, so the "
                    "Groq draft translation was kept for those lines "
                    "(still fully translated, just without the extra polish)."
                )
        else:
            yield sse_progress(30, f"Translating {len(sess.raw_segments)} line(s) with Groq ({TRANSLATION_MODELS[0]})")
            yield sse_log(
                f"[INFO] Translating with Groq ({TRANSLATION_MODELS[0]}; backup "
                f"chain: {', '.join(TRANSLATION_MODELS[1:]) or 'none'}) in "
                f"~{TRANSLATE_CHUNK_CHAR_BUDGET}-char chunks — every line is "
                "verified 1:1, so none can be silently dropped..."
            )
            async for ev in self_heal("Groq translation", lambda: groq_translate_all(
                progress_cb=lambda d, t: _live_progress(sess, 30 + int(20 * d / t), f"Translating chunk {d}/{t}"),
                transcript_segments=seg_dicts,
                source_language=sess.source_language or "Unknown",
                target_language=sess.target_language,
            )):
                if "_result" in ev:
                    result = ev["_result"]
                else:
                    yield ev

        sess.source_language = result.get("source_language") or sess.source_language
        sess.segments = [Segment(**s) for s in result["segments"]]

        if engine != "hybrid" and result["fallback_count"] > 0:
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
        # Keep the uploaded video + transcript so "Retry Dub" works without a
        # re-upload; free the heavy scratch audio right away.
        yield sse_error(f"[ERROR] {type(exc).__name__}: {exc}")
        shutil.rmtree(sess.dir / "segments", ignore_errors=True)


# --------------------------------------------------------------------------- #
# FastAPI app + endpoints
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Startup: sweep old sessions, resume anything that was mid-job, start the
    # keep-alive loop. Shutdown: stop it cleanly.
    try:
        _sweep_stale_sessions()
        _recover_sessions_on_startup()
    except Exception:  # noqa: BLE001
        pass
    ka_task = asyncio.get_running_loop().create_task(_keepalive_loop())
    try:
        yield
    finally:
        ka_task.cancel()


app = FastAPI(title="Ultimate Premium Video Dubbing Platform", version="5.1.0-polling",
              lifespan=_lifespan)

_cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = ["*"] if _cors_origins_raw in ("", "*") else _parse_key_list(_cors_origins_raw)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping() -> JSONResponse:
    """Ultra-light liveness endpoint (used by the self keep-alive and safe to
    point an external uptime monitor at, e.g. every 5 minutes)."""
    return JSONResponse({"ok": True, "active": bool(_ACTIVE_SESSION_ID), "queued": len(JOB_QUEUE)})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "groq_keys_configured": len(GROQ_API_KEYS),
        "default_engine": ENGINE_MODE_DEFAULT,
        "engines_available": [e for e in ("gemini", "groq", "hybrid") if
                              (bool(GEMINI_API_KEYS) if e == "gemini" else
                               bool(GROQ_API_KEYS) if e == "groq" else
                               bool(GEMINI_API_KEYS) and bool(GROQ_API_KEYS))],
        "tts_models": TTS_MODELS,
        "tts_model_labels": TTS_MODEL_CATALOG,
        "tts_chunk_char_budget": TTS_CHUNK_CHAR_BUDGET,
        "tts_chunk_silence_pad_ms": TTS_CHUNK_SILENCE_PAD_MS,
        "voices": GEMINI_VOICE_NAMES,
        "whisper_model": WHISPER_MODEL,
        "whisper_models": WHISPER_MODELS,
        "translation_model": TRANSLATION_MODEL,
        "translation_models": TRANSLATION_MODELS,
        "gemini_transcribe_model": GEMINI_TRANSCRIBE_MODEL,
        "gemini_transcribe_models": GEMINI_TRANSCRIBE_MODELS,
        "gemini_translate_model": GEMINI_TRANSLATE_MODEL,
        "gemini_translate_models": GEMINI_TRANSLATE_MODELS,
        "atempo_lock": [ATEMPO_LOCK_MIN, ATEMPO_LOCK_MAX],
        "silence_trim_lock_ms": SILENCE_TRIM_LADDER_MS,
        "email_enabled": bool(_email_provider()),
        "email_provider": _email_provider() or None,
        "keepalive_url": bool(PUBLIC_API_BASE),
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
    Saves the uploaded video and immediately QUEUES the automatic
    "prepare" job (extract audio -> probe duration -> transcribe) in the
    background worker, then returns right away with the session_id. No
    connection is held open — poll GET /status/{session_id} to follow
    progress; this is what prevents idle-connection timeouts on slow
    uploads/long videos on Render.
    """
    _sweep_stale_sessions()
    chosen_engine = (engine or "").strip().lower()
    if chosen_engine not in ("gemini", "groq", "hybrid"):
        chosen_engine = ENGINE_MODE_DEFAULT
    if chosen_engine in ("groq", "hybrid") and not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")
    if chosen_engine in ("gemini", "hybrid") and not GEMINI_API_KEYS:
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
    sess.status = "queued_prepare"
    _append_job_log(sess, f"[SYS] Upload received ({file.filename}). Queued for analysis (engine='{chosen_engine}').")
    SESSIONS[session_id] = sess
    save_session(sess)
    _enqueue_job("prepare", session_id)

    return JSONResponse({
        "session_id": session_id,
        "job_id": session_id,
        "filename": file.filename,
        "engine": chosen_engine,
        "status": sess.status,
    })


@app.get("/prepare/{session_id}")
async def prepare_kickoff(session_id: str, engine: str = "") -> JSONResponse:
    """
    Legacy-compatible trigger: /upload already auto-queues the prepare job,
    so this just re-queues it if it isn't already queued/running/done
    (e.g. a client that wants to force a re-analysis), and always returns
    immediately — poll GET /status/{session_id} for progress.
    """
    sess = load_session(session_id)
    if not sess or not sess.video_path or not sess.video_path.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")

    eng = (engine or "").strip().lower()
    if eng in ("gemini", "groq", "hybrid"):
        sess.engine = eng

    if sess.status in ("uploaded",) or (sess.status == "error" and not sess.prepared):
        sess.status = "queued_prepare"
        save_session(sess)
        _enqueue_job("prepare", session_id)

    return JSONResponse({"session_id": session_id, "status": sess.status})


@app.post("/dub")
async def dub(
    session_id: str = Form(...),
    target_language: str = Form(...),
    voice_name: str = Form("Kore"),
    engine: str = Form(""),
    tts_models: str = Form(""),
    email: str = Form(""),
) -> JSONResponse:
    """
    Queues the MANUAL dubbing job (translate -> chunked TTS -> trim ->
    duration-match -> mux) and returns immediately with session_id — no
    connection is held open for the whole run. Poll GET /status/{id} for
    progress and logs.

    `email` (optional): if provided and SMTP is configured on the server,
    a completion email with the download link is sent automatically once
    the job finishes.
    """
    eng = (engine or "").strip().lower()
    eng = eng if eng in ("gemini", "groq", "hybrid") else None
    effective_eng = eng or ENGINE_MODE_DEFAULT

    if effective_eng in ("gemini", "hybrid") and not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured.")
    if effective_eng in ("groq", "hybrid") and not GROQ_API_KEYS:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEY(s) configured.")
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="No GEMINI_API_KEY(s) configured (required for TTS).")

    sess = load_session(session_id)
    if not sess or not sess.video_path or not sess.video_path.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired session_id — please re-upload the video.")
    if not sess.prepared or not sess.raw_segments:
        raise HTTPException(status_code=409, detail="This video hasn't finished uploading/analyzing yet.")
    if sess.status in ("queued_dub", "dubbing"):
        # Already queued/running — don't double-enqueue; just report status.
        return JSONResponse({"session_id": session_id, "job_id": session_id, "status": sess.status})

    voice = (voice_name or "").strip() or "Kore"
    if voice not in GEMINI_VOICE_NAMES:
        voice = "Kore"
    sess.target_language = (target_language or "").strip()
    sess.single_voice = voice
    if eng:
        sess.engine = eng
    sess.tts_models = parse_selected_tts_models(tts_models)
    clean_email = (email or "").strip()
    sess.notify_email = clean_email or None
    sess.email_sent = False
    sess.status = "queued_dub"
    sess.job_percent = 0
    sess.job_error = None
    _append_job_log(
        sess,
        f"[SYS] Queued dub -> {sess.target_language} ({sess.single_voice}) "
        f"— engine='{sess.engine}', voice models={sess.tts_models}"
        + (f", notify={clean_email}" if clean_email else ""),
    )
    save_session(sess)
    _enqueue_job("dub", session_id)

    return JSONResponse({"session_id": session_id, "job_id": session_id, "status": sess.status})


@app.get("/status/{session_id}")
async def status(session_id: str, since: int = 0) -> JSONResponse:
    """
    Polling endpoint — call this every 3-5s while status is one of
    queued_prepare/preparing/queued_dub/dubbing. Returns the current
    status, percent, message, queue position (if still waiting behind
    other jobs), and a bounded list of recent log lines for the UI's live
    console.

    `since` (optional): pass the `log_count` you last saw to receive only
    the log lines appended after that point (cheaper polling on a slow
    connection); omit or pass 0 to get the full recent buffer.
    """
    sess = load_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")

    all_logs = sess.job_logs
    new_logs = all_logs[since:] if 0 <= since <= len(all_logs) else all_logs

    payload = {
        "session_id": session_id,
        "status": sess.status,
        "percent": sess.job_percent,
        "message": sess.job_message,
        "logs": new_logs,
        "log_count": len(all_logs),
        "queue_position": _queue_position(session_id),
        "prepared": sess.prepared,
        "source_language": sess.source_language,
        "video_duration": round(sess.video_duration, 2) if sess.video_duration else None,
        "line_count": len(sess.raw_segments) if sess.raw_segments else 0,
        "target_language": sess.target_language or None,
        "error": sess.job_error,
        "server_time": time.time(),
    }
    if sess.status == "done":
        payload["download_url"] = f"/download/{session_id}"
        payload["segments"] = len(sess.segments)
        payload["email_sent"] = sess.email_sent

    return JSONResponse(payload)


@app.get("/download/{session_id}")
async def download(session_id: str, background_tasks: BackgroundTasks, cleanup: bool = False):
    """
    Streams the finished MP4 straight from disk with native HTTP Range
    support (FileResponse). The file is normally left in place until the
    1-hour TTL sweep or an explicit DELETE /session/{id} — but passing
    ?cleanup=true deletes the session's scratch/output the moment this
    download finishes streaming, for callers that want disk freed
    immediately rather than waiting on the TTL sweep.
    """
    sess = load_session(session_id)
    out_video = ((sess.dir if sess else WORK_DIR / session_id) / "dubbed_output.mp4")
    if not out_video.exists():
        raise HTTPException(status_code=404, detail="Result not found or already cleaned up.")
    if cleanup:
        background_tasks.add_task(_destroy_session, session_id)
    return FileResponse(
        out_video,
        media_type="video/mp4",
        filename=f"dubbed_{session_id}.mp4",
        background=background_tasks,
    )


@app.delete("/session/{session_id}")
async def cancel_session(session_id: str) -> JSONResponse:
    _destroy_session(session_id)
    return JSONResponse({"status": "deleted", "session_id": session_id})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1)
