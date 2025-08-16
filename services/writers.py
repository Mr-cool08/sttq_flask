"""Hanterar export av transkript till TXT, SRT och JSON."""

import json
import math
from pathlib import Path
from typing import List

from tqdm import tqdm

from .types import Segment

def srt_timestamp(seconds: float) -> str:
    """Konverterar sekunder till SRT-format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000.0))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_txt(path: Path, segments: List[Segment]):
    """Skriver transkript till en .txt-fil."""
    with open(path, "w", encoding="utf-8") as f, tqdm(total=len(segments), desc="Skriver TXT", leave=False) as pbar:
        for seg in segments:
            f.write(f"[{seg.speaker}] {seg.text}\n")
            pbar.update(1)

def write_srt(path: Path, segments: List[Segment]):
    """Skriver transkript till en .srt-fil (undertexter)."""
    with open(path, "w", encoding="utf-8") as f, tqdm(total=len(segments), desc="Skriver SRT", leave=False) as pbar:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n")
            f.write(f"{seg.speaker}: {seg.text}\n\n")
            pbar.update(1)

def write_json(path: Path, segments: List[Segment]):
    """Skriver transkript till en .json-fil med detaljerad struktur."""
    with tqdm(total=len(segments), desc="Skriver JSON", leave=False) as pbar:
        js = []
        for seg in segments:
            js.append({
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "text": seg.text,
                "words": [{"start": w.start, "end": w.end, "text": w.text} for w in seg.words],
            })
            pbar.update(1)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)