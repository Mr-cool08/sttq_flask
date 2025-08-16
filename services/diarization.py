"""Diarization (talardetektering) och tilldelning av talare till STT-segment.

Filen innehåller:
- diarize(...): kör pyannote.audio på en wav-fil och returnerar en lista med (start, slut, "SPEAKER_XX")
- assign_speakers(...): tilldelar talare till transkriberade Segment baserat på diarization-resultat
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from .types import Segment, Word


def _lazy_imports() -> None:
    """Importerar pyannote.audio först när det behövs."""
    global Pipeline  # type: ignore
    from pyannote.audio import Pipeline  # type: ignore


def diarize(
    wav_path,
    diarize_model: str = "pyannote/speaker-diarization-3.1",
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Tuple[float, float, str]]:
    """
    Kör diarization på en wav-fil och returnerar en lista med talarturer.

    Parametrar:
      wav_path      : Sökväg till ljudfil (helst 16 kHz mono).
      diarize_model : HuggingFace-modellnamn för diarization.
      hf_token      : HuggingFace access token. Om None, hämtas från env-var 'HUGGINGFACE_TOKEN'.
      min_speakers  : (valfritt) minsta antal talare.
      max_speakers  : (valfritt) högsta antal talare.

    Return:
      List[Tuple[start_sec, end_sec, speaker_id]]
      där speaker_id är strängar som "SPEAKER_01", "SPEAKER_02", ...
    """
    _lazy_imports()

    token = hf_token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN saknas. Sätt env-variabeln eller skicka in 'hf_token' för diarization."
        )

    pipeline = Pipeline.from_pretrained(diarize_model, use_auth_token=token)  # type: ignore

    params = {}
    if min_speakers is not None:
        params["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        params["max_speakers"] = int(max_speakers)

    diarization = pipeline({"audio": str(wav_path)}, **params) if params else pipeline({"audio": str(wav_path)})

    # Mappa modellens etiketter till SPEAKER_01, SPEAKER_02, ...
    turns: List[Tuple[float, float, str]] = []
    speaker_map: Dict[str, int] = {}
    counter = 1

    for turn, track, label in diarization.itertracks(yield_label=True):  # type: ignore
        if label not in speaker_map:
            speaker_map[label] = counter
            counter += 1
        spk = f"SPEAKER_{speaker_map[label]:02d}"
        turns.append((float(turn.start), float(turn.end), spk))

    turns.sort(key=lambda x: (x[0], x[1]))
    return turns


# ---------------- Hjälpfunktioner för överlappning ----------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Returnerar överlapp (sekunder) mellan intervall [a0, a1] och [b0, b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speakers_for_interval(start: float, end: float, turns: List[Tuple[float, float, str]]) -> Dict[str, float]:
    """Summerar överlapp per talare inom [start, end] och returnerar {speaker_id: sekunder}."""
    acc: Dict[str, float] = {}
    for t0, t1, spk in turns:
        ov = _overlap(start, end, t0, t1)
        if ov > 0:
            acc[spk] = acc.get(spk, 0.0) + ov
    return acc


def assign_speakers(
    stt_segments: List[Segment],
    speaker_turns: List[Tuple[float, float, str]],
    strategy: str = "primary",
    merge_gap: float = 0.5,
) -> List[Segment]:
    """
    Tilldelar talare till STT-segment baserat på diarization-resultat.

    Parametrar:
      stt_segments : Segment med text/ord och tidsstämplar (utan talare).
      speaker_turns: Lista från diarize(): [(start, end, "SPEAKER_XX"), ...].
      strategy     : "primary" (standard) = välj dominerande talare per segment,
                     "split" = dela upp segmentet om olika talare talar olika delar.
      merge_gap    : Slå ihop angränsande segment från samma talare om gapet ≤ merge_gap (sek).

    Return:
      Lista Segment där 'speaker' är ifylld. Ordlistor behålls.
    """
    if not speaker_turns:
        # Ingen diarization → allt blir SPEAKER_01
        for s in stt_segments:
            s.speaker = "SPEAKER_01"
        return stt_segments

    labeled: List[Segment] = []

    for seg in stt_segments:
        # Om inga ordtidsstämplar finns, estimera talare via mitten av segmentet
        if not seg.words:
            mid = 0.5 * (seg.start + seg.end)
            eps = 0.2
            overlaps_mid = _speakers_for_interval(mid - eps, mid + eps, speaker_turns)
            if overlaps_mid:
                spk = max(overlaps_mid, key=overlaps_mid.get)
            else:
                overlaps = _speakers_for_interval(seg.start, seg.end, speaker_turns)
                spk = max(overlaps, key=overlaps.get) if overlaps else "SPEAKER_01"

            labeled.append(
                Segment(start=seg.start, end=seg.end, speaker=spk, text=seg.text, words=list(seg.words))
            )
            continue

        # Med ordnivå: välj talare per ord och dela ev. upp (strategy="split")
        current_spk: Optional[str] = None
        current_words: List[Word] = []

        def flush(curr_spk: Optional[str], words_acc: List[Word]) -> None:
            if not words_acc:
                return
            txt = "".join([w.text for w in words_acc]).strip()
            if not txt:
                return
            labeled.append(
                Segment(
                    start=words_acc[0].start,
                    end=words_acc[-1].end,
                    speaker=(curr_spk or "SPEAKER_01"),
                    text=txt,
                    words=list(words_acc),
                )
            )

        for w in seg.words:
            overlaps = _speakers_for_interval(w.start, w.end, speaker_turns)
            best_spk = max(overlaps, key=overlaps.get) if overlaps else (current_spk or "SPEAKER_01")

            if current_spk is None:
                current_spk = best_spk

            if best_spk != current_spk and strategy == "split":
                # Avsluta nuvarande del och starta ny för ny talare
                flush(current_spk, current_words)
                current_words = []
                current_spk = best_spk

            # "primary" → håll ihop segment men uppdatera löpande aktuell talare
            if strategy == "primary":
                current_spk = best_spk

            current_words.append(w)

        flush(current_spk, current_words)

    # Slå ihop korta mellanrum mellan segment från samma talare
    merged: List[Segment] = []
    for seg in labeled:
        if (
            merged
            and seg.speaker == merged[-1].speaker
            and (seg.start - merged[-1].end) <= merge_gap
        ):
            merged[-1].end = seg.end
            merged[-1].text = (merged[-1].text + " " + seg.text).strip()
            merged[-1].words.extend(seg.words)
        else:
            merged.append(seg)

    return merged
