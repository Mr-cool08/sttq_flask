"""Huvudfilen för Flask-appen. Hanterar routes, filuppladdning och kopplar ihop tjänsterna (STT, diarization, writers)."""

from __future__ import annotations
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

from flask import Flask, request, render_template, redirect, url_for, flash, send_file, after_this_request, jsonify, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import math

from services import ensure_wav_16k_mono, transcribe as stt_transcribe, diarize, assign_speakers, write_txt, write_srt, write_json

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

ALLOWED_EXT = {"wav", "mp3", "m4a", "aac", "flac", "ogg", "wma", "mp4", "mkv", "mov", "webm"}

def _bool(form_value: Optional[str]) -> bool:
    return form_value == "1" or form_value == "on" or form_value is True


def _segments_to_txt(segments) -> str:
    return "\n".join(f"[{seg.speaker}] {seg.text}" for seg in segments)


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(f"{seg.speaker}: {seg.text}")
        lines.append("")
    return "\n".join(lines).strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health() -> tuple[str, int]:
    """Simple health check for the web app."""
    return "OK", 200

@app.route("/api/health")
def api_health() -> tuple[Response, int]:
    """JSON health check for API clients."""
    return jsonify({"status": "ok"}), 200

@app.route("/transcribe", methods=["POST"])
def transcribe_route():
    if "file" not in request.files:
        flash("Ingen fil mottagen.")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("Ingen fil vald.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        flash(f"Otillåten filtyp: .{ext}")
        return redirect(url_for("index"))

    # Läs in formulärparametrar
    language = (request.form.get("language") or None) or None
    model = request.form.get("model", "large-v3")
    chunk_length = int(request.form.get("chunk_length", 45) or 45)
    assign_strategy = request.form.get("assign_strategy", "primary")
    merge_gap = float(request.form.get("merge_gap", 0.5) or 0.5)
    hotwords = request.form.get("hotwords") or None
    loudnorm = _bool(request.form.get("loudnorm"))
    denoise = _bool(request.form.get("denoise"))
    do_diarize = _bool(request.form.get("do_diarize"))
    min_speakers = request.form.get("min_speakers")
    max_speakers = request.form.get("max_speakers")
    min_speakers = int(min_speakers) if (min_speakers or "").strip() else None
    max_speakers = int(max_speakers) if (max_speakers or "").strip() else None

    # Tempdir per jobb
    workdir = Path(tempfile.mkdtemp(prefix="sttq_web_"))

    @after_this_request
    def cleanup(response):
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
        return response

    try:
        # Spara uppladdad fil
        in_path = workdir / filename
        file.save(str(in_path))

        # Konvertera/förbehandla → 16k mono WAV
        wav_path, tmp_dir = ensure_wav_16k_mono(in_path, loudnorm=loudnorm, denoise=denoise)

        # STT
        stt_segments, total_dur = stt_transcribe(
            wav_path,
            model_size=model,
            language=language,
            initial_prompt=None,
            chunk_length=chunk_length,
            hotwords=hotwords,
        )

        # Diarisering (valfri)
        speaker_turns: List[Tuple[float, float, str]] = []
        if do_diarize:
            try:
                speaker_turns = diarize(
                    wav_path,
                    diarize_model="pyannote/speaker-diarization-3.1",
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            except Exception as e:
                # Fortsätt utan diarization
                flash(f"Varning: Diarization misslyckades ({e}); fortsätter med en talare.")

        labeled_segments = assign_speakers(
            stt_segments,
            speaker_turns,
            strategy=assign_strategy,
            merge_gap=merge_gap,
        )

        # Skriv utdata
        outdir = workdir / "stt_out"
        outdir.mkdir(parents=True, exist_ok=True)
        txt_path = outdir / "transcript.txt"
        srt_path = outdir / "transcript.srt"
        json_path = outdir / "transcript.json"

        write_txt(txt_path, labeled_segments)
        write_srt(srt_path, labeled_segments)
        write_json(json_path, labeled_segments)

        # Skapa ZIP för nedladdning
        zip_base = workdir / "result"
        shutil.make_archive(str(zip_base), "zip", root_dir=str(outdir))
        zip_path = Path(f"{zip_base}.zip")

        return send_file(str(zip_path), as_attachment=True, download_name="transcript_results.zip", mimetype="application/zip")

    except Exception as e:
        flash(f"Fel: {e}")
        return redirect(url_for("index"))


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = {}
    if request.form:
        data.update(request.form)
    if request.is_json:
        data.update(request.get_json() or {})

    if "file" not in request.files:
        return jsonify({"error": "Ingen fil mottagen."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Ingen fil vald."}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Otillåten filtyp: .{ext}"}), 400

    language = (data.get("language") or None) or None
    model = data.get("model", "large-v3")
    chunk_length = int(data.get("chunk_length", 45) or 45)
    assign_strategy = data.get("assign_strategy", "primary")
    merge_gap = float(data.get("merge_gap", 0.5) or 0.5)
    hotwords = data.get("hotwords") or None
    loudnorm = _bool(data.get("loudnorm"))
    denoise = _bool(data.get("denoise"))
    do_diarize = _bool(data.get("do_diarize"))
    min_speakers = data.get("min_speakers")
    max_speakers = data.get("max_speakers")
    min_speakers = int(min_speakers) if (min_speakers or "").strip() else None
    max_speakers = int(max_speakers) if (max_speakers or "").strip() else None

    workdir = Path(tempfile.mkdtemp(prefix="sttq_api_"))
    try:
        in_path = workdir / filename
        file.save(str(in_path))
        wav_path, tmp_dir = ensure_wav_16k_mono(in_path, loudnorm=loudnorm, denoise=denoise)
        stt_segments, total_dur = stt_transcribe(
            wav_path,
            model_size=model,
            language=language,
            initial_prompt=None,
            chunk_length=chunk_length,
            hotwords=hotwords,
        )

        speaker_turns: List[Tuple[float, float, str]] = []
        if do_diarize:
            try:
                speaker_turns = diarize(
                    wav_path,
                    diarize_model="pyannote/speaker-diarization-3.1",
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
            except Exception:
                speaker_turns = []

        labeled_segments = assign_speakers(
            stt_segments,
            speaker_turns,
            strategy=assign_strategy,
            merge_gap=merge_gap,
        )

        segments_js = [
            {
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [{"start": w.start, "end": w.end, "text": w.text} for w in seg.words],
            }
            for seg in labeled_segments
        ]

        result = {
            "segments": segments_js,
            "text": _segments_to_txt(labeled_segments),
            "srt": _segments_to_srt(labeled_segments),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
