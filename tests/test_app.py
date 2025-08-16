import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytest
from app import app, _srt_timestamp, _segments_to_srt, _segments_to_txt
from services.types import Segment

@pytest.fixture
def sample_segments():
    return [
        Segment(start=0.0, end=1.0, speaker="SPEAKER_01", text="Hello", words=[]),
        Segment(start=1.0, end=2.5, speaker="SPEAKER_02", text="World", words=[]),
    ]

def test_srt_timestamp():
    assert _srt_timestamp(3661.234) == "01:01:01,234"

def test_segments_to_srt(sample_segments):
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "SPEAKER_01: Hello\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "SPEAKER_02: World"
    )
    assert _segments_to_srt(sample_segments) == expected

def test_segments_to_txt(sample_segments):
    assert _segments_to_txt(sample_segments) == "[SPEAKER_01] Hello\n[SPEAKER_02] World"

def test_health_endpoints():
    client = app.test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.data == b"OK"

    resp_json = client.get("/api/health")
    assert resp_json.status_code == 200
    assert resp_json.get_json() == {"status": "ok"}
