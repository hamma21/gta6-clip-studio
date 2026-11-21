"""
GTA 6 AI Clip Studio
====================
Online AI clipping studio (Streamlit). Pipeline:

 1. Download ONLY lightweight audio + English auto-subs (+ optional live-chat replay) via yt-dlp.
 2. Score 5-second blocks: audio energy (pydub dBFS above mean) + keyword hits (+40 each)
    + optional chat-message spikes.
 3. Pick the best non-overlapping windows and download ONLY those ranges
    (download_range_func + force_keyframes_at_cuts).
 4. Render with FFmpeg (9:16 / 16:9 / 1:1 / 4:5, 4 layouts for 9:16), optional
    faster-whisper captions burned in (bold yellow, Alignment=10).
 5. Delete every temp file afterwards; old outputs are purged automatically.

Optional environment variables:
  APP_PASSWORD, MAX_VIDEO_MIN, MAX_PARALLEL_JOBS, FILE_TTL_MIN, YT_COOKIES_FILE, CLIP_DATA_DIR
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path(os.environ.get("CLIP_DATA_DIR", Path(tempfile.gettempdir()) / "gta6_clip_studio"))
WORK_ROOT = BASE_DIR / "work"   # temp_ui_* style scratch space (deleted after every job)
OUT_ROOT = BASE_DIR / "out"     # finished clips (purged after FILE_TTL_MIN)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_VIDEO_MIN = int(os.environ.get("MAX_VIDEO_MIN", "240"))
MAX_PARALLEL_JOBS = int(os.environ.get("MAX_PARALLEL_JOBS", "2"))
FILE_TTL_MIN = int(os.environ.get("FILE_TTL_MIN", "60"))
ENV_COOKIES = os.environ.get("YT_COOKIES_FILE", "")

BLOCK_SEC = 5
KEYWORD_BOOST = 40.0
CHAT_WEIGHT = 30.0

RATIOS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
LAYOUTS = {
    "Standard Center Crop": "center",
    "Full Gameplay Fit (no side cut)": "fit",
    "Split-Screen: Top-Left Facecam (35%) + Gameplay (65%)": "split_left",
    "Split-Screen: Top-Right Facecam (35%) + Gameplay (65%)": "split_right",
}
AUDIO_EXTS = {".webm", ".m4a", ".mp4", ".opus", ".mp3", ".ogg", ".aac", ".wav", ".mka"}


# ----------------------------------------------------------------------------
# Shared server state (job limiter + janitor)
# ----------------------------------------------------------------------------
@st.cache_resource
def server_state() -> dict:
    return {"sem": threading.BoundedSemaphore(MAX_PARALLEL_JOBS), "last_clean": 0.0}


def janitor() -> None:
    """Delete work/output folders older than FILE_TTL_MIN (runs at most every 5 minutes)."""
    state = server_state()
    now = time.time()
    if now - state["last_clean"] < 300:
        return
    state["last_clean"] = now
    for root in (WORK_ROOT, OUT_ROOT):
        for d in root.glob("*"):
            try:
                if now - d.stat().st_mtime > FILE_TTL_MIN * 60:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass


@st.cache_resource
def get_whisper():
    from faster_whisper import WhisperModel
    return WhisperModel("tiny", device="cpu", compute_type="int8")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def parse_ts(text: str) -> float | None:
    """'MM:SS', 'HH:MM:SS' or plain seconds -> seconds. Empty -> None."""
    text = (text or "").strip()
    if not text:
        return None
    parts = [float(p) for p in text.split(":")]
    if len(parts) > 3:
        raise ValueError("Use MM:SS or HH:MM:SS")
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def fmt_ts(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("FFmpeg failed: " + proc.stderr[-900:])


def media_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError("Could not read video duration (is the file valid?)")


def extract_audio(src: Path, dst: Path, sample_rate: int, start: float | None = None,
                  end: float | None = None) -> None:
    """Mono PCM wav. Optional [start, end] range."""
    args: list[str] = []
    if start:
        args += ["-ss", f"{start:.2f}"]
    args += ["-i", str(src)]
    if end is not None:
        args += ["-t", f"{end - (start or 0):.2f}"]
    args += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    run_ffmpeg(args)


# ----------------------------------------------------------------------------
# YouTube (yt-dlp) helpers
# ----------------------------------------------------------------------------
def ydl_base(cookies: str | None) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "retries": 3}
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies
    return opts


def yt_info(url: str, cookies: str | None) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({**ydl_base(cookies), "skip_download": True}) as ydl:
        return ydl.extract_info(url, download=False)


def yt_download_audio(url: str, wd: Path, cookies: str | None) -> Path:
    import yt_dlp
    opts = {**ydl_base(cookies), "format": "worstaudio/bestaudio/worst",
            "outtmpl": str(wd / "audio.%(ext)s")}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    for f in wd.glob("audio.*"):
        if f.suffix.lower() in AUDIO_EXTS:
            return f
    raise RuntimeError("Audio download produced no file.")


def yt_download_subs(url: str, wd: Path, cookies: str | None, want_chat: bool) -> None:
    """Equivalent of: --write-auto-subs --sub-langs en --ignore-errors (+ live_chat)."""
    import yt_dlp
    base = {**ydl_base(cookies), "skip_download": True, "ignoreerrors": True,
            "outtmpl": str(wd / "audio.%(ext)s"), "subtitlesformat": "vtt"}
    try:
        with yt_dlp.YoutubeDL({**base, "writesubtitles": True, "writeautomaticsub": True,
                               "subtitleslangs": ["en"]}) as ydl:
            ydl.download([url])
    except Exception:
        pass
    if want_chat:
        try:
            with yt_dlp.YoutubeDL({**base, "writesubtitles": True, "writeautomaticsub": False,
                                   "subtitleslangs": ["live_chat"]}) as ydl:
                ydl.download([url])
        except Exception:
            pass


def yt_download_clip(url: str, start: float, end: float, wd: Path, idx: int,
                     cookies: str | None) -> Path:
    """Download ONLY [start, end] of the video."""
    import yt_dlp
    from yt_dlp.utils import download_range_func
    opts = {
        **ydl_base(cookies),
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(wd / f"raw_{idx}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [f for f in wd.glob(f"raw_{idx}.*") if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("Clip download produced no file.")
    return files[0]


# ----------------------------------------------------------------------------
# Transcripts / chat parsing
# ----------------------------------------------------------------------------
_VTT_TIME = re.compile(
    r"(?:(\d+):)?(\d+):(\d+)[\.,](\d+)\s+-->\s+(?:(\d+):)?(\d+):(\d+)[\.,](\d+)")


def parse_vtt(path: Path) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    cur: tuple[float, float] | None = None
    buf: list[str] = []
    last_text = ""

    def flush():
        nonlocal last_text
        if cur and buf:
            text = " ".join(buf).strip()
            if text and text != last_text:
                cues.append((cur[0], cur[1], text))
                last_text = text

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = _VTT_TIME.search(line)
        if m:
            flush()
            g = m.groups()
            cur = (int(g[0] or 0) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 10 ** len(g[3]),
                   int(g[4] or 0) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 10 ** len(g[7]))
            buf = []
        elif line.strip() == "":
            flush()
            cur, buf = None, []
        elif cur is not None:
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean:
                buf.append(clean)
    flush()
    return cues


def parse_chat(path: Path) -> list[float]:
    """Offsets (seconds) of every chat message in a yt-dlp live_chat.json file."""
    times: list[float] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                action = json.loads(line).get("replayChatItemAction", {})
                ms = action.get("videoOffsetTimeMsec")
                if ms is not None:
                    times.append(int(ms) / 1000.0)
            except (ValueError, AttributeError):
                continue
    return times


def transcribe(path: Path, offset: float = 0.0, words: bool = False):
    """faster-whisper. Returns (segments, words) with `offset` added to timestamps."""
    model = get_whisper()
    segs, _ = model.transcribe(str(path), language="en", beam_size=1,
                               vad_filter=True, word_timestamps=words)
    out_segs, out_words = [], []
    for s in segs:
        out_segs.append((s.start + offset, s.end + offset, s.text.strip()))
        if words and s.words:
            for w in s.words:
                out_words.append((w.start + offset, w.end + offset, w.word.strip()))
    return out_segs, out_words


def write_caption_srt(words: list[tuple[float, float, str]], path: Path, per_cue: int = 4) -> bool:
    """Group words into short, punchy UPPERCASE caption cues."""
    def t(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines, n = [], 1
    for i in range(0, len(words), per_cue):
        chunk = words[i:i + per_cue]
        start, end = chunk[0][0], max(chunk[-1][1], chunk[0][0] + 0.25)
        text = " ".join(w[2] for w in chunk).upper()
        lines.append(f"{n}\n{t(start)} --> {t(end)}\n{text}\n")
        n += 1
    if not lines:
        return False
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


# ----------------------------------------------------------------------------
# AI moment detection
# ----------------------------------------------------------------------------
def find_moments(wav: Path, rs: float, re_: float, keywords: list[str],
                 cues: list[tuple[float, float, str]], chat: list[float],
                 n_clips: int, clip_len: float) -> list[dict]:
    from pydub import AudioSegment

    seg = AudioSegment.from_wav(str(wav))
    nb = int((re_ - rs) // BLOCK_SEC)
    if nb < 1:
        raise RuntimeError("Scan range is shorter than one 5-second block.")

    # --- Audio energy spikes (dBFS above the mean) ---
    db = np.empty(nb)
    for k in range(nb):
        a = int((rs + k * BLOCK_SEC) * 1000)
        db[k] = max(seg[a:a + BLOCK_SEC * 1000].dBFS, -90.0)
    energy = np.maximum(0.0, db - db.mean())

    # --- Transcript keyword matches (+40 per distinct keyword per block) ---
    texts: list[list[str]] = [[] for _ in range(nb)]
    for s, e, t in cues:
        lo, hi = int((s - rs) // BLOCK_SEC), int((e - rs) // BLOCK_SEC)
        for b in range(max(0, lo), min(nb - 1, hi) + 1):
            texts[b].append(t)
    hits: list[set[str]] = []
    for b in range(nb):
        joined = " ".join(texts[b]).lower()
        hits.append({k for k in keywords if k in joined})
    kw_count = np.array([len(h) for h in hits], dtype=float)

    # --- Optional live-chat spikes ---
    chat_n = np.zeros(nb)
    for ts in chat:
        b = int((ts - rs) // BLOCK_SEC)
        if 0 <= b < nb:
            chat_n[b] += 1
    chat_score = (chat_n / chat_n.max()) * CHAT_WEIGHT if chat_n.max() > 0 else chat_n

    score = energy + KEYWORD_BOOST * kw_count + chat_score

    # --- Sliding window + greedy non-overlapping selection ---
    n_win = max(1, min(nb, math.ceil(clip_len / BLOCK_SEC)))
    cs = np.concatenate([[0.0], np.cumsum(score)])
    win = cs[n_win:] - cs[:-n_win]
    kcs = np.concatenate([[0.0], np.cumsum(kw_count)])
    kwin = kcs[n_win:] - kcs[:-n_win]
    win = win.astype(float)
    pool = win.copy()          # unfiltered windows (fallback once keyword matches run out)
    if keywords and kwin.max() > 0:
        win = np.where(kwin > 0, win, -np.inf)   # prefer windows that contain a keyword

    picks: list[dict] = []
    for _ in range(n_clips):
        i = int(np.argmax(win))
        if not np.isfinite(win[i]):
            i = int(np.argmax(pool))             # fill remaining slots with top-energy moments
            if not np.isfinite(pool[i]):
                break
        lo, hi = max(0, i - n_win + 1), min(len(win), i + n_win)
        win[lo:hi] = -np.inf
        pool[lo:hi] = -np.inf
        start = rs + i * BLOCK_SEC
        end = min(start + clip_len, re_)
        found = sorted(set().union(*hits[i:i + n_win])) if keywords else []
        parts = []
        if found:
            parts.append("🎯 " + ", ".join(found))
        parts.append(f"🔊 +{energy[i:i + n_win].max():.0f} dB peak")
        if chat_n[i:i + n_win].sum() > 0:
            parts.append(f"💬 {int(chat_n[i:i + n_win].sum())} chat msgs")
        picks.append({"start": start, "end": end, "score": float(win_orig(cs, i, n_win)),
                      "reason": "  ·  ".join(parts) if parts else "⚡ High-energy moment"})
    return picks


def win_orig(cs: np.ndarray, i: int, n: int) -> float:
    return float(cs[i + n] - cs[i])


# ----------------------------------------------------------------------------
# Rendering (FFmpeg)
# ----------------------------------------------------------------------------
def build_filter(ratio: str, layout: str, srt_name: str | None) -> str:
    W, H = RATIOS[ratio]
    if ratio == "9:16" and layout == "fit":
        graph = (f"[0:v]split=2[a][b];"
                 f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:3[bg];"
                 f"[b]scale={W}:-2[fg];"
                 f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[v]")
    elif ratio == "9:16" and layout in ("split_left", "split_right"):
        top_h = int(H * 0.35) // 2 * 2
        bot_h = H - top_h
        x = "0" if layout == "split_left" else "iw-ow"
        graph = (f"[0:v]split=2[f][g];"
                 f"[f]crop=w=iw*0.35:h=iw*0.35*{top_h}/{W}:x={x}:y=0,scale={W}:{top_h}[top];"
                 f"[g]scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h}[bot];"
                 f"[top][bot]vstack=inputs=2[v]")
    else:  # standard center crop (also used for 16:9, 1:1, 4:5)
        graph = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}[v]"

    tail = "setsar=1"
    if srt_name:
        size = 13 if H >= 1350 else 15
        style = (f"Alignment=10,Bold=1,Fontsize={size},PrimaryColour=&H0000FFFF,"
                 f"OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,MarginL=40,MarginR=40")
        tail += f",subtitles={srt_name}:force_style='{style}'"
    return f"{graph};[v]{tail}[vout]"


def render_clip(raw: Path, out: Path, ratio: str, layout: str, srt_name: str | None) -> None:
    run_ffmpeg([
        "-i", raw.name, "-filter_complex", build_filter(ratio, layout, srt_name),
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
    ], cwd=raw.parent)   # cwd = work dir so the subtitle path stays relative (Windows-safe)


# ----------------------------------------------------------------------------
# The full job
# ----------------------------------------------------------------------------
def run_job(*, url: str, upload, keywords: list[str], rng: tuple[float | None, float | None],
            n_clips: int, clip_len: int, ratio: str, layout: str, captions: bool,
            use_chat: bool, cookie_upload, sid: str, prog, note) -> list[dict]:
    wd = WORK_ROOT / f"temp_ui_{sid}"
    od = OUT_ROOT / sid
    shutil.rmtree(wd, ignore_errors=True)
    shutil.rmtree(od, ignore_errors=True)
    wd.mkdir(parents=True)
    od.mkdir(parents=True)

    cookies = ENV_COOKIES or None
    if cookie_upload is not None:
        cp = wd / "cookies.txt"
        cp.write_bytes(cookie_upload.getvalue())
        cookies = str(cp)

    results: list[dict] = []
    try:
        cues: list[tuple[float, float, str]] = []
        chat: list[float] = []
        src_file: Path | None = None

        # ---------- 1. Source + lightweight scan assets ----------
        prog.progress(0.03, text="Reading video info…")
        if upload is not None:
            src_file = wd / ("upload" + Path(upload.name).suffix.lower())
            src_file.write_bytes(upload.getvalue())
            duration = media_duration(src_file)
            title = Path(upload.name).stem
            audio = src_file
        else:
            info = yt_info(url, cookies)
            if info.get("is_live"):
                raise RuntimeError("This is a live stream that is still running. Use the VOD after it ends.")
            duration = float(info.get("duration") or 0)
            title = info.get("title", "video")
            if duration <= 0:
                raise RuntimeError("Could not determine video length.")
            if duration > MAX_VIDEO_MIN * 60:
                raise RuntimeError(f"Video is longer than the {MAX_VIDEO_MIN}-minute limit on this server.")
            prog.progress(0.08, text=f"Downloading audio + subtitles only: {title[:60]}")
            audio = yt_download_audio(url, wd, cookies)
            yt_download_subs(url, wd, cookies, use_chat)
            vtts = sorted(wd.glob("audio*.vtt"))
            if vtts:
                cues = parse_vtt(vtts[0])
            chats = list(wd.glob("*live_chat*"))
            if use_chat and chats:
                chat = parse_chat(chats[0])

        if upload is not None and duration > MAX_VIDEO_MIN * 60:
            raise RuntimeError(f"Video is longer than the {MAX_VIDEO_MIN}-minute limit on this server.")

        rs = rng[0] or 0.0
        re_ = min(rng[1] if rng[1] is not None else duration, duration)
        if rs >= re_:
            raise RuntimeError("Start time must be before end time (and inside the video).")
        clip_len = int(min(clip_len, re_ - rs))

        # ---------- 2. Audio energy + transcript analysis ----------
        prog.progress(0.22, text="Analyzing audio energy…")
        scan_wav = wd / "scan.wav"
        extract_audio(audio, scan_wav, 8000)

        if keywords and not cues:
            prog.progress(0.28, text="No subtitles found; transcribing with Whisper for keyword search…")
            wav16 = wd / "scan16.wav"
            extract_audio(audio, wav16, 16000, rs, re_)
            cues, _ = transcribe(wav16, offset=rs)
            wav16.unlink(missing_ok=True)
            if not cues:
                note("No speech detected, using audio energy only.")

        prog.progress(0.38, text="Scoring 5-second blocks…")
        moments = find_moments(scan_wav, rs, re_, keywords, cues, chat, n_clips, clip_len)
        scan_wav.unlink(missing_ok=True)
        if keywords and not any("🎯" in m["reason"] for m in moments):
            note("No keyword matches found, showing the highest-energy moments instead.")
        if not moments:
            raise RuntimeError("No moments found in the selected range.")

        # ---------- 3. Download / cut + render each clip ----------
        for i, mo in enumerate(moments):
            base = 0.42 + 0.56 * i / len(moments)
            span = 0.56 / len(moments)
            label = f"Clip {i + 1}/{len(moments)} [{fmt_ts(mo['start'])}–{fmt_ts(mo['end'])}]"

            prog.progress(base, text=f"{label}: downloading exact range…")
            if upload is not None:
                raw = wd / f"raw_{i}.mp4"
                run_ffmpeg(["-ss", f"{mo['start']:.2f}", "-i", str(src_file),
                            "-t", f"{mo['end'] - mo['start']:.2f}",
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                            "-c:a", "aac", str(raw)])
            else:
                raw = yt_download_clip(url, mo["start"], mo["end"], wd, i, cookies)

            srt_name = None
            if captions:
                prog.progress(base + span * 0.4, text=f"{label}: generating captions…")
                _, words = transcribe(raw, words=True)
                srt_path = wd / f"cap_{i}.srt"
                if write_caption_srt(words, srt_path):
                    srt_name = srt_path.name

            prog.progress(base + span * 0.65, text=f"{label}: rendering {ratio}…")
            out = od / f"clip_{i + 1:02d}_{int(mo['start'])}s_{ratio.replace(':', 'x')}.mp4"
            render_clip(raw, out, ratio, layout, srt_name)
            raw.unlink(missing_ok=True)   # free disk immediately

            results.append({"path": str(out), "rank": i + 1, "start": mo["start"],
                            "end": mo["end"], "reason": mo["reason"], "title": title})

        prog.progress(1.0, text="Done!")
        return results
    finally:
        shutil.rmtree(wd, ignore_errors=True)   # auto-cleanup of all temp files


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
CSS = """
<style>
.block-container{padding-top:2rem;max-width:1200px}
.hero{padding:2.2rem 1.5rem;border-radius:22px;margin-bottom:1.2rem;text-align:center;
  background:radial-gradient(1200px 300px at 50% -20%,#4b2fd6 0%,rgba(11,13,20,0) 70%),#10131f;
  border:1px solid #232a44}
.hero h1{font-size:2.6rem;margin:0;background:linear-gradient(90deg,#9b8cff,#ff7ad9);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#9aa4c7;margin:.5rem 0 0}
.badge{display:inline-block;padding:.25rem .65rem;border-radius:999px;font-size:.78rem;
  background:#1b2140;border:1px solid #2d366b;color:#cfd6ff;margin:0 .35rem .35rem 0}
.badge.rank{background:#4b2fd6;border-color:#6b52ff;color:#fff}
div.stButton>button[kind="primary"]{background:linear-gradient(90deg,#7c5cff,#ff5cc8);border:0;
  font-weight:700;padding:.7rem 1rem;border-radius:12px}
</style>
"""


def password_gate() -> None:
    if not APP_PASSWORD or st.session_state.get("authed"):
        return
    st.markdown("<div class='hero'><h1>🔒 GTA 6 AI Clip Studio</h1><p>Enter the access password.</p></div>",
                unsafe_allow_html=True)
    pw = st.text_input("Password", type="password")
    if st.button("Unlock", type="primary"):
        if pw == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def render_gallery(results: list[dict]) -> None:
    st.subheader("🎞️ Your clips")
    results = [r for r in results if Path(r["path"]).exists()]
    if not results:
        st.info("Clips expired and were cleaned up. Generate new ones.")
        return
    cols = st.columns(3)
    for r in results:
        with cols[(r["rank"] - 1) % 3]:
            with st.container(border=True):
                st.markdown(
                    f"<span class='badge rank'>#{r['rank']}</span>"
                    f"<span class='badge'>⏱ {fmt_ts(r['start'])} – {fmt_ts(r['end'])}</span><br>"
                    f"<span class='badge'>{r['reason']}</span>", unsafe_allow_html=True)
                st.video(r["path"])
                st.download_button("⬇️ Download MP4", data=Path(r["path"]).read_bytes(),
                                   file_name=Path(r["path"]).name, mime="video/mp4",
                                   key=f"dl_{r['rank']}", use_container_width=True)
    st.caption(f"Clips are auto-deleted from the server after {FILE_TTL_MIN} minutes. Download what you need.")


def main() -> None:
    st.set_page_config(page_title="GTA 6 AI Clip Studio", page_icon="🎬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    janitor()
    password_gate()
    sid = st.session_state.setdefault("sid", uuid.uuid4().hex[:12])

    st.markdown("<div class='hero'><h1>🎬 GTA 6 AI Clip Studio</h1>"
                "<p>Paste a YouTube VOD or upload a video. AI finds the viral moments and exports "
                "ready-to-post clips.</p></div>", unsafe_allow_html=True)

    # ---- Source ----
    tab_url, tab_up = st.tabs(["🔗 YouTube URL", "📁 Upload video"])
    with tab_url:
        url = st.text_input("YouTube video / live-stream VOD URL", placeholder="https://www.youtube.com/watch?v=…")
    with tab_up:
        upload = st.file_uploader("Drag & drop a video", type=["mp4", "mov", "mkv"])

    # ---- Settings ----
    with st.container(border=True):
        keywords_raw = st.text_input("🔎 Smart scene / topic search (comma-separated)",
                                     placeholder="car crash, police chase, funny, shootout   (blank = auto-detect energy peaks)")
        c1, c2 = st.columns(2)
        use_range = c1.toggle("Custom time range", value=False)
        if use_range:
            start_txt = c1.text_input("Start time", value="00:00", help="MM:SS or HH:MM:SS")
            end_txt = c2.text_input("End time", placeholder="e.g. 45:00 (blank = end of video)")
        else:
            start_txt, end_txt = "", ""

        c3, c4 = st.columns(2)
        n_clips = c3.slider("Number of clips", 1, 15, 5)
        clip_len = c4.slider("Clip duration (seconds)", 10, 90, 30)

        c5, c6 = st.columns(2)
        ratio = c5.selectbox("Aspect ratio", list(RATIOS), format_func=lambda r: f"{r}  ({RATIOS[r][0]}×{RATIOS[r][1]})")
        if ratio == "9:16":
            layout_label = c6.selectbox("9:16 layout", list(LAYOUTS))
            layout = LAYOUTS[layout_label]
        else:
            c6.caption("Layout modes are available for 9:16. Other ratios use a centered crop.")
            layout = "center"

        c7, c8 = st.columns(2)
        captions = c7.toggle("Auto-captions (bold yellow, Whisper)", value=False)
        use_chat = c8.toggle("Use YouTube live-chat replay spikes (if available)", value=False)

        with st.expander("Advanced: YouTube cookies (fixes 'Sign in to confirm you're not a bot')"):
            cookie_upload = st.file_uploader("cookies.txt (Netscape format)", type=["txt"])
            st.caption("Cloud server IPs are often blocked by YouTube. Cookies from a throwaway account help.")

    go = st.button("✨ Generate clips", type="primary", use_container_width=True)

    if go:
        if upload is None and not url.strip():
            st.error("Enter a YouTube URL or upload a video.")
            st.stop()
        try:
            rng = (parse_ts(start_txt), parse_ts(end_txt))
        except ValueError as e:
            st.error(f"Invalid time: {e}")
            st.stop()

        sem = server_state()["sem"]
        if not sem.acquire(blocking=False):
            st.warning("Server is busy with other jobs. Please try again in a minute.")
            st.stop()
        prog = st.progress(0.0, text="Starting…")
        try:
            kws = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
            st.session_state["results"] = run_job(
                url=url.strip(), upload=upload, keywords=kws, rng=rng, n_clips=n_clips,
                clip_len=clip_len, ratio=ratio, layout=layout, captions=captions,
                use_chat=use_chat, cookie_upload=cookie_upload, sid=sid, prog=prog,
                note=lambda m: st.info(m))
        except Exception as e:  # show a friendly message for the common YouTube block
            msg = str(e)
            if "Sign in" in msg or "bot" in msg.lower():
                msg += "\n\nYouTube blocked this server. Upload a cookies.txt in 'Advanced' or use the Upload tab."
            st.error(msg)
            prog.empty()
        finally:
            sem.release()

    if st.session_state.get("results"):
        render_gallery(st.session_state["results"])


main()
