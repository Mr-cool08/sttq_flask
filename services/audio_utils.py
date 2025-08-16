"""Hanterar ljudkonvertering och förbehandling via ffmpeg (t.ex. loudnorm, denoise, resampling till 16 kHz mono)."""

from __future__ import annotations
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

# -------------------- ffmpeg helpers --------------------

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")

def _parse_ffmpeg_duration(line: str) -> Optional[float]:
    """Parsar duration från ffmpeg-loggrad och returnerar sekunder."""
    m = _DURATION_RE.search(line)
    if not m:
        return None
    h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + s

def _parse_ffmpeg_time(line: str) -> Optional[float]:
    """Parsar aktuell tid från ffmpeg-loggrad och returnerar sekunder."""
    m = _TIME_RE.search(line)
    if not m:
        return None
    h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + s

def _run_ffmpeg_with_progress(cmd: List[str], desc: str = "ffmpeg") -> int:
    """Kör ett ffmpeg-kommando och visar progress bar."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, universal_newlines=True, bufsize=1,
    )
    total = None
    last_time = 0.0
    pbar = None
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            line = line.strip()
            if total is None:
                maybe_total = _parse_ffmpeg_duration(line)
                if maybe_total:
                    total = maybe_total
                    pbar = tqdm(total=total, unit="s", desc=desc, leave=False)
                    continue
            t = _parse_ffmpeg_time(line)
            if t is not None and total:
                delta = max(0.0, t - last_time)
                last_time = t
                if pbar:
                    pbar.update(delta)
        if pbar is None:
            pbar = tqdm(total=1, desc=desc, leave=False)
        if total and last_time < total:
            pbar.update(total - last_time)
        else:
            pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()
    proc.wait()
    return proc.returncode

def _require_ffmpeg():
    """Kontrollerar att ffmpeg finns installerat i PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg saknas i PATH. Installera ffmpeg för högkvalitativ konvertering.")

def _is_exact_16k_mono_wav_pcm(path: Path) -> bool:
    """Kollar om en fil redan är exakt 16kHz mono WAV PCM."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return (
            info.format == "WAV" and
            info.channels == 1 and
            info.samplerate == 16000 and
            info.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
        )
    except Exception:
        return False

def ensure_wav_16k_mono(input_path: Path, loudnorm: bool = False, denoise: bool = False) -> Tuple[Path, Optional[Path]]:
    """Säkerställer att input konverteras till 16kHz mono WAV (med ev. förbehandling)."""
    """
    Returnerar (wav_path, tmp_dir). tmp_dir kan raderas efter användning.
    - Om input redan är exakt 16kHz mono WAV PCM och ingen förbehandling önskas → returnera originalet, tmp_dir=None.
    - Annars förbehandla (valfritt) och resampla med ffmpeg SOXR och returnera dess sökväg + tempdir.
    """
    if _is_exact_16k_mono_wav_pcm(input_path) and not (loudnorm or denoise):
        return input_path, None

    _require_ffmpeg()
    tmp_dir = Path(tempfile.mkdtemp(prefix="sttq_"))
    mid = tmp_dir / "preproc.wav"
    out = tmp_dir / "audio_16k_mono.wav"

    pre_filters = []
    if loudnorm:
        pre_filters.append("loudnorm=I=-23:TP=-2:LRA=7")
    if denoise:
        pre_filters.append("afftdn=nr=12")
    pre_filters.extend(["highpass=f=80", "lowpass=f=12000"])

    if pre_filters:
        cmd1 = ["ffmpeg","-y","-i",str(input_path),"-vn","-af",",".join(pre_filters),str(mid)]
        _run_ffmpeg_with_progress(cmd1, desc="Förbehandling (ffmpeg)")
        src_for_resample = mid
    else:
        src_for_resample = input_path

    cmd2 = ["ffmpeg","-y","-i",str(src_for_resample),"-ac","1","-ar","16000","-vn","-af","aresample=resampler=soxr:precision=33", str(out)]
    _run_ffmpeg_with_progress(cmd2, desc="Konverterar (ffmpeg)")

    return out, tmp_dir