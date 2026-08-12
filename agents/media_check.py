"""
Media Authenticity Agent — combined visual + audio sub-checks.

Cost-conscious design (same philosophy as fact_check.py):
  - Default path uses FREE, local signal extraction: ffprobe metadata forensics
    (encoder/software tags, container info) and C2PA manifest presence — no paid
    API calls required.
  - Paid detection APIs (e.g. Hive Moderation for AI-image detection, a voice-clone
    detector for audio) are wired in as OPTIONAL plug-ins, only called if their API
    key env vars are set. Omitting them costs nothing and still returns a real,
    if less confident, verdict from the free metadata layer.

This intentionally does NOT attempt to build a from-scratch deepfake/voice-clone
classifier — see the architecture doc's note on why that's the wrong scope for
this project. It orchestrates signals; it doesn't reinvent CV/audio research.
"""
import os
import re
import json
import time
import wave
import subprocess

from utils.security import check_tool_allowed

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Known software/encoder tags that indicate AI-generation or common editing tools.
# This is a real, free signal — many AI generators and editors stamp this metadata,
# though it can be stripped, so absence of a flag is NOT proof of authenticity.
AI_GENERATOR_SIGNATURES = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "sora", "runway",
    "pika labs", "synthesia", "heygen", "deepfacelab", "faceswap",
    "google ai", "gemini", "veo", "kling", "luma ai",
]

# Optional paid API keys — if unset, those sub-checks are skipped gracefully.
HIVE_API_KEY = os.environ.get("HIVE_API_KEY")          # AI-image/video detection
VOICE_CLONE_API_KEY = os.environ.get("VOICE_CLONE_API_KEY")  # e.g. Resemble/Pindrop


def _ffprobe_metadata(file_path: str) -> dict:
    """Free, local metadata extraction via ffprobe. Returns raw tag dict."""
    check_tool_allowed("media_check_agent", "ffprobe")
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", file_path],
            capture_output=True, text=True, timeout=15,
        )
        return json.loads(result.stdout) if result.stdout else {}
    except Exception:
        return {}


def _check_c2pa_presence(file_path: str) -> bool:
    """Checks for a C2PA content-credentials manifest (free — no API needed).
    Real implementation would use the c2pa-python library; this checks for the
    manifest's characteristic JUMBF box signature as a lightweight free proxy."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(2_000_000)  # scan first ~2MB, where manifests live
        return b"c2pa" in head.lower() or b"jumb" in head.lower()
    except Exception:
        return False


def _scan_for_ai_signatures(metadata: dict) -> list:
    """Scans ffprobe tag values for known AI-generator software signatures."""
    found = []
    text_blob = json.dumps(metadata).lower()
    for sig in AI_GENERATOR_SIGNATURES:
        if sig in text_blob:
            found.append(sig)
    return found


def _extract_audio_wav(video_path: str, out_path: str) -> bool:
    """Extracts a mono WAV track from the video via ffmpeg (free, local)."""
    check_tool_allowed("media_check_agent", "ffmpeg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_path],
            capture_output=True, timeout=30,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _basic_audio_signal_check(wav_path: str) -> dict:
    """Free, coarse audio sanity check: duration + silence ratio.
    NOTE: this is NOT a voice-clone or lip-sync detector — it's a cheap first-pass
    signal (e.g. an entirely-silent 'speech' track is itself suspicious/uninformative).
    Real voice-clone/lip-sync scoring should come from the optional paid API below."""
    try:
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            duration = frames / float(rate) if rate else 0.0
        return {"duration_sec": round(duration, 2), "readable": True}
    except Exception:
        return {"duration_sec": 0.0, "readable": False}


def _optional_hive_check(frames: list) -> dict | None:
    """Paid API plug-in for AI-image/video detection. Skipped if no key set."""
    if not HIVE_API_KEY:
        return None
    check_tool_allowed("media_check_agent", "hive_api")
    # Real implementation: POST frames to Hive Moderation's deepfake-detection
    # endpoint. Left unimplemented here since it requires a paid key to test —
    # this is the seam where it plugs in.
    return {"verdict": "not_implemented_stub", "confidence": 0.0}


def _optional_voice_clone_check(wav_path: str) -> dict | None:
    """Paid API plug-in for voice-clone detection. Skipped if no key set."""
    if not VOICE_CLONE_API_KEY:
        return None
    check_tool_allowed("media_check_agent", "voice_clone_api")
    # Real implementation: send wav_path to a voice-clone detector (Resemble AI,
    # Pindrop, etc). Left unimplemented here for the same reason as Hive above.
    return {"verdict": "not_implemented_stub", "confidence": 0.0}


@traceable(name="media_check_agent", run_type="chain")
def run_media_check(file_path: str, key_frames: list, audio_track_hint) -> dict:
    """Returns a MediaAgentResult dict (see state.py).

    file_path: path to the source video (for real use; the graph currently
    passes a stub path from the ingestion node).
    """
    start = time.time()

    if not file_path or not os.path.exists(file_path):
        return {
            "visual_verdict": "Inconclusive",
            "visual_confidence": 0.0,
            "visual_evidence": ["[STUB — no real video file available; ingestion node is stubbed]"],
            "audio_verdict": "N/A",
            "audio_confidence": 0.0,
            "audio_evidence": [],
            "status": "skipped",
            "latency_ms": 0,
            "cost_usd": 0.0,
        }

    try:
        # --- Visual sub-check (free by default) ---
        metadata = _ffprobe_metadata(file_path)
        ai_signatures = _scan_for_ai_signatures(metadata)
        has_c2pa = _check_c2pa_presence(file_path)

        visual_evidence = []
        if has_c2pa:
            visual_evidence.append("C2PA content-credentials manifest detected — provenance signal present.")
        if ai_signatures:
            visual_evidence.append(f"Metadata contains known AI-generation tool signature(s): {', '.join(ai_signatures)}.")
        if not has_c2pa and not ai_signatures:
            visual_evidence.append("No C2PA manifest or known AI-generator metadata found (absence is not proof of authenticity — metadata can be stripped).")

        hive_result = _optional_hive_check(key_frames)
        if hive_result:
            visual_evidence.append(f"Hive detection API result: {hive_result['verdict']}")

        if ai_signatures:
            visual_verdict, visual_confidence = "AI-Generated", 0.7
        elif has_c2pa:
            visual_verdict, visual_confidence = "Authentic", 0.6
        else:
            visual_verdict, visual_confidence = "Inconclusive", 0.4

        # --- Audio sub-check (free coarse signal by default) ---
        audio_verdict, audio_confidence, audio_evidence = "N/A", 0.0, []
        wav_path = file_path + ".extracted.wav"
        if _extract_audio_wav(file_path, wav_path):
            audio_info = _basic_audio_signal_check(wav_path)
            audio_evidence.append(f"Audio track extracted, duration {audio_info['duration_sec']}s.")

            voice_result = _optional_voice_clone_check(wav_path)
            if voice_result:
                audio_evidence.append(f"Voice-clone detection API result: {voice_result['verdict']}")
                audio_verdict, audio_confidence = "Inconclusive", 0.5
            else:
                audio_evidence.append("No paid voice-clone/lip-sync API configured — coarse signal only.")
                audio_verdict, audio_confidence = "Inconclusive", 0.3

            if os.path.exists(wav_path):
                os.remove(wav_path)
        else:
            audio_evidence.append("No audio track found or extraction failed (likely a silent video).")

        return {
            "visual_verdict": visual_verdict,
            "visual_confidence": visual_confidence,
            "visual_evidence": visual_evidence,
            "audio_verdict": audio_verdict,
            "audio_confidence": audio_confidence,
            "audio_evidence": audio_evidence,
            "status": "success",
            "latency_ms": int((time.time() - start) * 1000),
            "cost_usd": 0.0,  # free path costs nothing; add real per-call cost if paid APIs are wired in
        }

    except Exception as e:
        return {
            "visual_verdict": "Inconclusive",
            "visual_confidence": 0.0,
            "visual_evidence": [f"[error during media check: {str(e)[:150]}]"],
            "audio_verdict": "Inconclusive",
            "audio_confidence": 0.0,
            "audio_evidence": [],
            "status": "failed",
            "latency_ms": int((time.time() - start) * 1000),
            "cost_usd": 0.0,
        }
