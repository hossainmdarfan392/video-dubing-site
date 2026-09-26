# Aurora Dub backend — slim Python + FFmpeg, tuned for Render.
FROM python:3.11-slim

# --- System deps -------------------------------------------------------- #
# ffmpeg (+ ffprobe, bundled with it) is required for every audio/video
# step in app.py (extract, silence-trim, atempo speed-lock, mux). Installed
# from Debian's repo so the binary is guaranteed present at container start
# — this is the exact failure Docker is here to prevent ("missing ffmpeg
# binary" on a bare Render Python runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---------------------------------------------------------#
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code -------------------------------------------------------------#
COPY app.py .

# Scratch dir for in-flight sessions (see WORK_DIR in app.py). Render's
# filesystem is ephemeral per-deploy, which is fine — sessions are meant to
# be short-lived (1h TTL sweep) and are never relied on to survive restarts.
RUN mkdir -p /app/_sessions
ENV WORK_DIR=/app/_sessions
ENV FFMPEG_BIN=ffmpeg
ENV FFPROBE_BIN=ffprobe
ENV PYTHONUNBUFFERED=1

# Render provides $PORT at runtime; app.py already reads it (default 8000).
EXPOSE 8000

# Single worker: the whole pipeline is async + one-session-at-a-time
# sequential by design (see SEQUENTIAL_PROCESSING in app.py) to stay inside
# Render's RAM limits — running multiple uvicorn workers would defeat that
# by letting several heavy ffmpeg jobs run in parallel on one instance.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
