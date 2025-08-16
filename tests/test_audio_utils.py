import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from services.audio_utils import (
    _parse_ffmpeg_duration,
    _parse_ffmpeg_time,
    _is_exact_16k_mono_wav_pcm,
)


def test_parse_ffmpeg_duration():
    line = "Duration: 00:01:23.45, start: 0.000000, bitrate: 128 kb/s"
    assert _parse_ffmpeg_duration(line) == pytest.approx(83.45)


def test_parse_ffmpeg_time():
    line = "size=       0kB time=00:00:10.50 bitrate=   0.0kbits/s"
    assert _parse_ffmpeg_time(line) == pytest.approx(10.50)


def test_is_exact_16k_mono_wav_pcm_true(tmp_path):
    data = np.zeros(16000, dtype=np.float32)
    file_path = tmp_path / "mono16k.wav"
    sf.write(file_path, data, 16000, subtype="PCM_16")
    assert _is_exact_16k_mono_wav_pcm(file_path) is True


def test_is_exact_16k_mono_wav_pcm_false(tmp_path):
    data = np.zeros(8000, dtype=np.float32)
    file_path = tmp_path / "mono8k.wav"
    sf.write(file_path, data, 8000, subtype="PCM_16")
    assert _is_exact_16k_mono_wav_pcm(file_path) is False
