"""Hanterar transkribering med faster-whisper. Tar in ljud (WAV 16 kHz mono rekommenderat)
och returnerar en lista av Segment-objekt samt den totala ljudlängden i sekunder."""

from __future__ import annotations
from typing import List, Optional, Tuple

from tqdm import tqdm

from .types import Segment, Word


def _lazy_imports() -> None:
    """
    Importerar tunga beroenden först när de behövs för att snabba upp importtiden.
    Gör Model (faster-whisper.WhisperModel) och torch tillgängliga globalt.
    """
    global Model, torch
    from faster_whisper import WhisperModel as Model  # type: ignore
    import torch  # type: ignore  # noqa: F401


def _pick_device_and_compute(
    user_device: Optional[str],
    user_compute: Optional[str],
) -> Tuple[str, str]:
    """
    Bestämmer device (CPU/GPU) och compute-precision.
    - Om CUDA finns → device='cuda', compute='float16' (default)
    - Annars → device='cpu', compute='float32' (default)
    Användarens val (user_device/user_compute) vinner alltid.
    """
    _lazy_imports()
    import torch  # type: ignore

    device = user_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if user_compute:
        return device, user_compute
    return (device, "float16") if device == "cuda" else (device, "float32")


def transcribe(
    wav_path,
    model_size: str = "large-v3",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    chunk_length: int = 45,
    hotwords: Optional[str] = None,
) -> Tuple[List[Segment], float]:
    """
    Kör transkribering av ett ljud (wav) och returnerar:
      - segments: List[Segment]  (med ord-tidsstämplar när tillgängligt)
      - total_duration_sec: float

    Parametrar:
      wav_path        : Sökväg till ljudfil (helst 16 kHz mono WAV för bäst resultat).
      model_size      : Whisper-modell (t.ex. 'large-v3', 'medium', 'small', 'base', 'tiny').
      device          : 'cpu' eller 'cuda'. Om None autodetekteras.
      compute_type    : t.ex. 'float16' (GPU), 'float32' (CPU). Om None väljs per device.
      language        : ISO-kod (t.ex. 'sv'). None → auto-detect.
      initial_prompt  : Kontext/frö-text som kan styra första segmenten (valfritt).
      chunk_length    : Längd på chunk i sekunder för streaming/inkrementell transkribering.
      hotwords        : Komma-separerad sträng med ord/frases att prioritera.

    Return:
      (segments, total_duration_sec)
    """
    _lazy_imports()
    device, compute_type = _pick_device_and_compute(device, compute_type)
    model = Model(model_size, device=device, compute_type=compute_type)  # type: ignore

    # Förbered hotwords-lista om angivet
    hotwords_list = None
    if hotwords:
        hotwords_list = [w.strip() for w in hotwords.split(",") if w.strip()]

    # Kör transkriberingen
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,              # begär ordnivå-tidsstämplar
        vad_filter=True,                   # enklare VAD-filter
        vad_parameters={"min_silence_duration_ms": 200},
        beam_size=10,
        best_of=5,
        patience=2.0,
        length_penalty=1.05,
        temperature=[0.0, 0.2, 0.4, 0.6],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.45,
        condition_on_previous_text=True,
        initial_prompt=initial_prompt,
        chunk_length=chunk_length,
        max_new_tokens=None,
        prepend_punctuations="“¿([{ -",
        append_punctuations=".”!?)]}%",
        hotwords=hotwords_list,
    )

    total_sec = float(getattr(info, "duration", 0.0) or 0.0)
    pbar = tqdm(
        total=total_sec if total_sec > 0 else None,
        unit="s",
        desc="STT (transkriberar)",
        leave=False,
    )

    segments: List[Segment] = []
    last_end = 0.0

    # Läs ut segmenten och bygg vår interna struktur
    for seg in segments_iter:
        # Bygg ordlistan (kan saknas för vissa segment)
        words: List[Word] = []
        if getattr(seg, "words", None):
            words = [Word(float(w.start), float(w.end), w.word) for w in list(seg.words)]

        segments.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                speaker="",  # talare sätts senare av diarization/assign_speakers
                text=(seg.text or "").strip(),
                words=words,
            )
        )

        # Progress
        current_end = float(seg.end)
        if pbar is not None and total_sec > 0:
            delta = max(0.0, current_end - last_end)
            pbar.update(delta)
        last_end = current_end

    # Fyll upp progressen om den inte kom hela vägen
    if pbar is not None:
        if total_sec > 0 and last_end < total_sec:
            pbar.update(total_sec - last_end)
        pbar.close()

    # Om info.duration saknas, använd sista segmentets end som total längd
    total_duration = float(getattr(info, "duration", None) or last_end)
    return segments, total_duration
